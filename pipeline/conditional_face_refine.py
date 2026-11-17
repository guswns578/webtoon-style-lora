from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import request

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from imgutils.detect import detect_faces
from imgutils.metrics import ccip_batch_differences, ccip_extract_feature

from lineart_normalize import normalize_file
from pose_metrics import detect_pose, pose_metrics, pose_source_quality

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "conditional_face_refine_config.json"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

COMMON_STYLE = (
    "webtoon_lineart_style, lineart, monochrome, manga_panel, greyscale, "
    "clean lineart, thin controlled ink lines, white background"
)
NEGATIVE = (
    "color, full color, cel shading, sepia, gray skin, pale lines, faint lines, low contrast, "
    "sketch, pencil sketch, thick lines, heavy shading, screentone, blurry, bad anatomy, "
    "deformed face, round face, childish face, soft expression, extra eyes, duplicate face, "
    "crowded background, speed lines, impact lines, motion blur, dramatic background, "
    "text, subtitle, caption, speech bubble, watermark"
)


@dataclass
class FaceResult:
    crop: Image.Image | None
    box: tuple[int, int, int, int] | None
    confidence: float


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def api_json(base_url: str, endpoint: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = request.Request(
        base_url.rstrip("/") + endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with request.urlopen(req, timeout=1800) as response:
        return json.loads(response.read().decode("utf-8"))


def image_to_base64(image: Image.Image | Path) -> str:
    if isinstance(image, Path):
        image = Image.open(image).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_api_image(value: str) -> Image.Image:
    encoded = value.split(",", 1)[-1]
    return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")


def set_checkpoint(config: dict[str, Any]) -> None:
    api_json(
        config["forge_api"],
        "/sdapi/v1/options",
        {"sd_model_checkpoint": config["checkpoint"], "CLIP_stop_at_last_layers": 1},
    )


def controlnet_args(config: dict[str, Any]) -> list[dict[str, Any]]:
    cn = config["controlnet"]
    unit = {
        "enabled": True,
        "module": cn["module"],
        "model": cn["model"],
        "weight": cn["weight"],
        "image": image_to_base64(Path(config["pose_image"])),
        "resize_mode": cn["resize_mode"],
        "processor_res": -1,
        "threshold_a": -1,
        "threshold_b": -1,
        "guidance_start": cn["guidance_start"],
        "guidance_end": cn["guidance_end"],
        "pixel_perfect": cn["pixel_perfect"],
        "control_mode": cn["control_mode"],
        "type_filter": "OpenPose",
        "save_detected_map": True,
    }
    return [unit]


def build_prompt(config: dict[str, Any], character: dict[str, str]) -> str:
    return ", ".join(
        [
            config["style_adapter"],
            config["character_adapter"],
            character["trigger"],
            COMMON_STYLE,
            character["positive"],
        ]
    )


def generate_candidate(
    config: dict[str, Any], prompt: str, seed: int, negative_prompt: str = NEGATIVE
) -> tuple[Image.Image, str]:
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "sampler_name": config["sampler"],
        "scheduler": config["scheduler"],
        "steps": config["steps"],
        "cfg_scale": config["cfg"],
        "width": config["width"],
        "height": config["height"],
        "batch_size": 1,
        "n_iter": 1,
        "restore_faces": False,
        "enable_hr": False,
        "override_settings": {
            "sd_model_checkpoint": config["checkpoint"],
            "CLIP_stop_at_last_layers": 1,
        },
        "override_settings_restore_afterwards": False,
        "alwayson_scripts": {"ControlNet": {"args": controlnet_args(config)}},
    }
    result = api_json(config["forge_api"], "/sdapi/v1/txt2img", payload)
    return decode_api_image(result["images"][0]), result.get("info", "")


def refine_with_masked_inpaint(
    config: dict[str, Any],
    character: dict[str, str],
    source: Image.Image,
    face_box: tuple[int, int, int, int],
    seed: int,
    denoising_strength: float,
) -> tuple[Image.Image, str]:
    refine = config["refinement"]
    left, top, right, bottom = face_box
    mask = Image.new("L", source.size, 0)
    draw = ImageDraw.Draw(mask)
    # The saved face box already includes hair/forehead padding. An ellipse avoids
    # needlessly redrawing nearby costume or background details.
    draw.ellipse((left, top, right, bottom), fill=255)
    refine_prompt = ", ".join(
        [
            refine["style_adapter"],
            refine["character_adapter"],
            character["trigger"],
            COMMON_STYLE,
            character["face_positive"],
        ]
    )
    payload = {
        "init_images": [image_to_base64(source)],
        "mask": image_to_base64(mask.convert("RGB")),
        "prompt": refine_prompt,
        "negative_prompt": character["face_negative"] + ", " + NEGATIVE,
        "seed": seed,
        "sampler_name": config["sampler"],
        "scheduler": config["scheduler"],
        "steps": config["steps"],
        "cfg_scale": config["cfg"],
        "width": source.width,
        "height": source.height,
        "resize_mode": 0,
        "denoising_strength": denoising_strength,
        "mask_blur": refine["mask_blur"],
        "inpainting_fill": 1,
        "inpaint_full_res": True,
        "inpaint_full_res_padding": refine["mask_padding"],
        "inpainting_mask_invert": 0,
        "batch_size": 1,
        "n_iter": 1,
        "override_settings": {
            "sd_model_checkpoint": config["checkpoint"],
            "CLIP_stop_at_last_layers": 1,
        },
        "override_settings_restore_afterwards": False,
    }
    result = api_json(config["forge_api"], "/sdapi/v1/img2img", payload)
    return decode_api_image(result["images"][0]), result.get("info", "")


def detect_largest_face(image: Image.Image, confidence: float) -> FaceResult:
    detections = detect_faces(image, level="s", conf_threshold=confidence)
    if not detections:
        return FaceResult(None, None, 0.0)
    (x1, y1, x2, y2), _, score = max(
        detections, key=lambda item: (item[0][2] - item[0][0]) * (item[0][3] - item[0][1])
    )
    width, height = x2 - x1, y2 - y1
    padding = int(max(width, height) * 0.32)
    left = max(0, x1 - padding)
    top = max(0, y1 - padding)
    right = min(image.width, x2 + padding)
    bottom = min(image.height, y2 + padding)
    return FaceResult(
        image.crop((left, top, right, bottom)), (left, top, right, bottom), float(score)
    )


def reference_features(
    config: dict[str, Any], cache_dir: Path
) -> tuple[list[np.ndarray], list[str], list[str]]:
    ref_dir = Path(config["reference_face_dir"])
    cache_file = cache_dir / "reference_features.npz"
    refs = sorted(
        path
        for path in ref_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    signature = [f"{p.name}:{p.stat().st_size}:{p.stat().st_mtime_ns}" for p in refs]
    if cache_file.exists():
        cached = np.load(cache_file, allow_pickle=True)
        if list(cached["signature"]) == signature:
            return list(cached["features"]), list(cached["labels"]), list(cached["paths"])

    features: list[np.ndarray] = []
    labels: list[str] = []
    paths: list[str] = []
    for index, path in enumerate(refs, 1):
        label = path.stem.split("_", 1)[0]
        print(f"[reference {index:02d}/{len(refs):02d}] {path.name}", flush=True)
        features.append(ccip_extract_feature(path))
        labels.append(label)
        paths.append(str(path))
    np.savez_compressed(
        cache_file,
        features=np.stack(features),
        labels=np.array(labels),
        paths=np.array(paths),
        signature=np.array(signature),
    )
    return features, labels, paths


def score_identity(
    face: Image.Image | None,
    target: str,
    reference_data: tuple[list[np.ndarray], list[str], list[str]],
    nearest_count: int,
) -> dict[str, Any]:
    if face is None:
        return {
            "face_detected": False,
            "target_distance": math.inf,
            "predicted_character": "none",
            "predicted_distance": math.inf,
            "second_character": "none",
            "margin": -math.inf,
        }
    features, labels, paths = reference_data
    query = ccip_extract_feature(face)
    matrix = ccip_batch_differences([query, *features])
    distances = matrix[0, 1:]
    per_character: dict[str, float] = {}
    nearest_refs: dict[str, list[dict[str, Any]]] = {}
    for label in sorted(set(labels)):
        indices = [idx for idx, value in enumerate(labels) if value == label]
        ranked = sorted(indices, key=lambda idx: float(distances[idx]))
        selected = ranked[: max(1, min(nearest_count, len(ranked)))]
        per_character[label] = float(np.median([distances[idx] for idx in selected]))
        nearest_refs[label] = [
            {"path": paths[idx], "distance": float(distances[idx])} for idx in selected
        ]
    ranking = sorted(per_character.items(), key=lambda item: item[1])
    predicted, predicted_distance = ranking[0]
    second, second_distance = ranking[1]
    target_distance = per_character[target]
    return {
        "face_detected": True,
        "target_distance": target_distance,
        "predicted_character": predicted,
        "predicted_distance": predicted_distance,
        "second_character": second,
        "second_distance": second_distance,
        "margin": second_distance - target_distance,
        "per_character": per_character,
        "nearest_target_references": nearest_refs[target],
    }


def identity_pass(score: dict[str, Any], target: str, config: dict[str, Any]) -> tuple[bool, str]:
    rules = config["identity"]
    if not score["face_detected"]:
        return False, "face_not_detected"
    if score["predicted_character"] != target:
        return False, "wrong_nearest_character"
    if score["target_distance"] <= rules["threshold"]:
        return True, "strict_threshold"
    if (
        score["target_distance"] <= rules["soft_threshold"]
        and score["margin"] >= rules["minimum_margin"]
    ):
        return True, "soft_threshold_with_margin"
    return False, "distance_or_margin_failed"


def save_face(face_result: FaceResult, path: Path) -> None:
    if face_result.crop is not None:
        face_result.crop.save(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "stage",
        "seed",
        "image",
        "face_image",
        "face_detected",
        "face_confidence",
        "face_box",
        "face_width",
        "face_height",
        "face_area_ratio",
        "target_distance",
        "predicted_character",
        "predicted_distance",
        "second_character",
        "second_distance",
        "margin",
        "passed",
        "pass_reason",
        "pose_passed",
        "balanced_pass",
        "failure_mode",
        "recommended_action",
        "refined_from",
        "refine_strength",
        "pose_detected",
        "pose_people",
        "pose_common_joints",
        "pose_coverage",
        "pose_aligned_rmse",
        "pose_composition_rmse",
        "pose_score",
        "identity_score",
        "identity_pose_score",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def label_font(size: int = 22) -> ImageFont.ImageFont:
    for font_path in [Path("C:/Windows/Fonts/malgun.ttf"), Path("C:/Windows/Fonts/arial.ttf")]:
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size)
    return ImageFont.load_default()


def fit_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    copy = image.convert("RGB")
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(copy, ((size[0] - copy.width) // 2, (size[1] - copy.height) // 2))
    return canvas


def make_contact_sheet(config: dict[str, Any], rows: list[dict[str, Any]], out_dir: Path) -> Path:
    pose = Image.open(config["pose_image"]).convert("RGB")
    display_rows = sorted(
        rows,
        key=lambda row: (
            row["seed"],
            0 if row["stage"] == "before" else 1,
            float(row.get("refine_strength") or 0.0),
        ),
    )
    cell_w, cell_h, label_h = 300, 386, 74
    columns = 3
    items: list[tuple[str, Image.Image]] = [("ControlNet pose", pose)]
    for row in display_rows:
        image = Image.open(row["image"]).convert("RGB")
        distance = row["target_distance"]
        distance_text = "inf" if not math.isfinite(float(distance)) else f"{float(distance):.4f}"
        label = (
            f"{row['stage']} seed {row['seed']}\n"
            f"CCIP {distance_text} / {row['predicted_character']}\n"
            f"{'PASS' if row['passed'] else 'FAIL'} ({row['pass_reason']})"
        )
        items.append((label, image))
    rows_count = math.ceil(len(items) / columns)
    sheet = Image.new("RGB", (cell_w * columns, (cell_h + label_h) * rows_count), "white")
    draw = ImageDraw.Draw(sheet)
    font = label_font(18)
    for idx, (label, image) in enumerate(items):
        x = (idx % columns) * cell_w
        y = (idx // columns) * (cell_h + label_h)
        sheet.paste(fit_image(image, (cell_w, cell_h)), (x, y))
        draw.multiline_text((x + 8, y + cell_h + 4), label, font=font, fill="black", spacing=2)
        draw.rectangle((x, y, x + cell_w - 1, y + cell_h + label_h - 1), outline=(160, 160, 160))
    path = out_dir / "comparison_contact_sheet.png"
    sheet.save(path)
    return path


def evaluate_and_record(
    config: dict[str, Any],
    image: Image.Image,
    image_path: Path,
    face_path: Path,
    stage: str,
    seed: int,
    target: str,
    reference_data: tuple[list[np.ndarray], list[str], list[str]],
    refined_from: str = "",
    refine_strength: float | str = "",
    source_pose: dict[str, Any] | None = None,
) -> dict[str, Any]:
    face = detect_largest_face(image, config["identity"]["face_detection_confidence"])
    save_face(face, face_path)
    score = score_identity(
        face.crop,
        target,
        reference_data,
        config["identity"]["nearest_reference_count"],
    )
    passed, reason = identity_pass(score, target, config)
    if source_pose is not None:
        candidate_pose = detect_pose(config["forge_api"], image_path)
        pose = pose_metrics(source_pose, candidate_pose)
    else:
        pose = {
            "pose_detected": False,
            "pose_people": 0,
            "pose_common_joints": 0,
            "pose_coverage": 0.0,
            "pose_aligned_rmse": math.inf,
            "pose_composition_rmse": math.inf,
            "pose_score": 0.0,
        }
    target_distance = float(score["target_distance"])
    id_score = 0.0 if not math.isfinite(target_distance) else max(0.0, 1.0 - target_distance / 0.18)
    if score["predicted_character"] != target:
        id_score *= 0.45
    weights = config.get("pose_evaluation", {"identity_weight": 0.55, "pose_weight": 0.45})
    identity_pose_score = (
        weights["identity_weight"] * id_score + weights["pose_weight"] * pose["pose_score"]
    )
    minimum_pose = float(weights.get("minimum_score_for_direct_accept", 0.45))
    pose_passed = bool(pose["pose_detected"] and pose["pose_score"] >= minimum_pose)
    balanced_pass = bool(passed and pose_passed)
    if balanced_pass:
        failure_mode = "pass"
        recommended_action = "accept"
    elif not score["face_detected"] and not pose["pose_detected"]:
        failure_mode = "identity_and_pose_unavailable"
        recommended_action = "manual_review_or_regenerate"
    elif not score["face_detected"]:
        failure_mode = "identity_unavailable"
        recommended_action = "face_refine_or_higher_resolution"
    elif not pose["pose_detected"]:
        failure_mode = "pose_unavailable"
        recommended_action = "manual_pose_review"
    elif not passed and not pose_passed:
        failure_mode = "identity_and_pose_failed"
        recommended_action = "regenerate_with_controlnet_then_identity_check"
    elif not passed:
        failure_mode = "identity_failed"
        recommended_action = "character_weight_or_face_refine"
    else:
        failure_mode = "pose_failed"
        recommended_action = "controlnet_or_canvas_retry"
    face_width = int(face.box[2] - face.box[0]) if face.box is not None else 0
    face_height = int(face.box[3] - face.box[1]) if face.box is not None else 0
    face_area_ratio = (face_width * face_height) / max(1, image.width * image.height)
    return {
        "stage": stage,
        "seed": seed,
        "image": str(image_path),
        "face_image": str(face_path) if face.crop is not None else "",
        "face_detected": score["face_detected"],
        "face_confidence": face.confidence,
        "face_box": list(face.box) if face.box is not None else None,
        "face_width": face_width,
        "face_height": face_height,
        "face_area_ratio": face_area_ratio,
        "target_distance": score["target_distance"],
        "predicted_character": score["predicted_character"],
        "predicted_distance": score["predicted_distance"],
        "second_character": score["second_character"],
        "second_distance": score.get("second_distance", math.inf),
        "margin": score["margin"],
        "passed": passed,
        "pass_reason": reason,
        "pose_passed": pose_passed,
        "balanced_pass": balanced_pass,
        "failure_mode": failure_mode,
        "recommended_action": recommended_action,
        "refined_from": refined_from,
        "refine_strength": refine_strength,
        **pose,
        "identity_score": id_score,
        "identity_pose_score": identity_pose_score,
        "details": score,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate ControlNet candidates, rank identity with CCIP, and refine only failures."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--character", help="Override character key from config")
    parser.add_argument("--pose", type=Path, help="Override pose image")
    parser.add_argument("--output", type=Path, help="Override output folder")
    parser.add_argument("--seeds", help="Comma-separated seed override, e.g. 43,100")
    parser.add_argument("--width", type=int, help="Override generation width")
    parser.add_argument("--height", type=int, help="Override generation height")
    parser.add_argument("--style-adapter", help="Override style adapter prompt token")
    parser.add_argument("--character-adapter", help="Override character adapter prompt token")
    parser.add_argument("--control-weight", type=float, help="Override ControlNet weight")
    parser.add_argument("--control-end", type=float, help="Override ControlNet guidance end")
    parser.add_argument("--control-mode", help="Override ControlNet mode")
    parser.add_argument(
        "--force-refine",
        action="store_true",
        help="Run refinement sweep even when another first-pass candidate passed",
    )
    parser.add_argument("--no-generate", action="store_true", help="Only rebuild reference cache")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.character:
        config["character"] = args.character
    if args.pose:
        config["pose_image"] = str(args.pose)
    if args.seeds:
        config["seeds"] = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    if args.width:
        config["width"] = args.width
    if args.height:
        config["height"] = args.height
    if args.style_adapter:
        config["style_adapter"] = args.style_adapter
    if args.character_adapter:
        config["character_adapter"] = args.character_adapter
    if args.control_weight is not None:
        config["controlnet"]["weight"] = args.control_weight
    if args.control_end is not None:
        config["controlnet"]["guidance_end"] = args.control_end
    if args.control_mode:
        config["controlnet"]["control_mode"] = args.control_mode
    target = config["character"]
    if target not in config["characters"]:
        raise KeyError(f"Unknown character: {target}")
    if not Path(config["pose_image"]).exists():
        raise FileNotFoundError(config["pose_image"])

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = (
        args.output
        or Path(config["output_root"]) / f"{target}_{Path(config['pose_image']).stem}_{stamp}"
    )
    images_dir = out_dir / "images"
    faces_dir = out_dir / "faces"
    cache_dir = Path(config["output_root"]) / "_cache"
    for directory in (out_dir, images_dir, faces_dir, cache_dir):
        directory.mkdir(parents=True, exist_ok=True)

    print("Building/loading CCIP reference features...", flush=True)
    references = reference_features(config, cache_dir)
    if args.no_generate:
        print(cache_dir / "reference_features.npz")
        return 0

    print("Connecting to Forge and selecting checkpoint...", flush=True)
    set_checkpoint(config)
    character = config["characters"][target]
    prompt = build_prompt(config, character)
    initial_negative = character.get("face_negative", "") + ", " + NEGATIVE
    print("Detecting source pose for quantitative pose evaluation...", flush=True)
    source_pose = detect_pose(config["forge_api"], Path(config["pose_image"]))
    source_quality = pose_source_quality(source_pose)
    if not source_quality["valid"]:
        raise ValueError(
            "Unusable ControlNet pose source: "
            f"only {source_quality['visible_joints']} visible joints detected; require at least 8"
        )
    rows: list[dict[str, Any]] = []
    source_images: dict[int, Image.Image] = {}

    for index, seed in enumerate(config["seeds"], 1):
        print(f"[generate {index}/{len(config['seeds'])}] seed={seed}", flush=True)
        image, info = generate_candidate(config, prompt, int(seed), initial_negative)
        image_path = images_dir / f"before_seed{seed}.png"
        face_path = faces_dir / f"before_seed{seed}_face.png"
        image.save(image_path)
        source_images[int(seed)] = image
        row = evaluate_and_record(
            config,
            image,
            image_path,
            face_path,
            "before",
            int(seed),
            target,
            references,
            source_pose=source_pose,
        )
        row["forge_info"] = info
        rows.append(row)
        print(
            f"  CCIP={row['target_distance']:.4f} predicted={row['predicted_character']} "
            f"margin={row['margin']:.4f} pass={row['passed']}",
            flush=True,
        )

    minimum_pose = config.get("pose_evaluation", {}).get("minimum_score_for_direct_accept", 0.45)
    any_balanced_pass = any(bool(row["balanced_pass"]) for row in rows)
    failures = sorted(
        [row for row in rows if not row["passed"] and row.get("face_box")],
        key=lambda row: (-row["pose_score"], row["target_distance"]),
    )[: config["refinement"]["max_candidates"]]
    if (
        config["refinement"].get("refine_only_if_no_pass", True)
        and any_balanced_pass
        and not args.force_refine
    ):
        failures = []

    if config["refinement"]["enabled"] and failures:
        refinement_strengths = config["refinement"].get(
            "denoising_strengths", [config["refinement"].get("denoising_strength", 0.38)]
        )
        total_refinements = len(failures) * len(refinement_strengths)
        refinement_index = 0
        for before in failures:
            seed = int(before["seed"])
            for strength in refinement_strengths:
                refinement_index += 1
                strength = float(strength)
                print(
                    f"[refine {refinement_index}/{total_refinements}] "
                    f"seed={seed} denoise={strength:.2f}",
                    flush=True,
                )
                refined, info = refine_with_masked_inpaint(
                    config,
                    character,
                    source_images[seed],
                    tuple(before["face_box"]),
                    seed,
                    strength,
                )
                suffix = f"d{int(round(strength * 100)):02d}"
                image_path = images_dir / f"after_seed{seed}_{suffix}.png"
                face_path = faces_dir / f"after_seed{seed}_{suffix}_face.png"
                refined.save(image_path)
                row = evaluate_and_record(
                    config,
                    refined,
                    image_path,
                    face_path,
                    "after",
                    seed,
                    target,
                    references,
                    refined_from=before["image"],
                    refine_strength=strength,
                    source_pose=source_pose,
                )
                row["forge_info"] = info
                rows.append(row)
                print(
                    f"  CCIP={row['target_distance']:.4f} predicted={row['predicted_character']} "
                    f"margin={row['margin']:.4f} pass={row['passed']}",
                    flush=True,
                )
    else:
        print("No failing candidate required refinement.", flush=True)

    write_csv(out_dir / "identity_results.csv", rows)
    settings = {
        "config": config,
        "positive_prompt": prompt,
        "negative_prompt": initial_negative,
        "results": rows,
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8"
    )
    sheet = make_contact_sheet(config, rows, out_dir)

    before_rows = [row for row in rows if row["stage"] == "before"]
    after_rows = [row for row in rows if row["stage"] == "after"]
    target_identity_rows = [
        row for row in rows if row.get("face_detected") and row.get("predicted_character") == target
    ]
    balanced_rows = [row for row in target_identity_rows if bool(row.get("balanced_pass"))]
    selection_pool = balanced_rows or target_identity_rows or rows
    selected = max(selection_pool, key=lambda row: row["identity_pose_score"])
    selected_balanced_pass = bool(
        selected.get("passed")
        and selected.get("pose_detected")
        and float(selected.get("pose_score", 0.0)) >= minimum_pose
    )
    manual_review_required = bool(not selected_balanced_pass)
    selected_postprocessed = out_dir / "selected_postprocessed.png"
    normalize_file(Path(selected["image"]), selected_postprocessed, gamma=0.40)
    summary = {
        "before_pass": sum(bool(row["passed"]) for row in before_rows),
        "before_total": len(before_rows),
        "after_pass": sum(bool(row["passed"]) for row in after_rows),
        "after_total": len(after_rows),
        "selected_result": selected["image"],
        "selected_stage": selected["stage"],
        "selected_seed": selected["seed"],
        "selected_target_distance": selected["target_distance"],
        "selected_predicted_character": selected["predicted_character"],
        "selected_passed": selected["passed"],
        "selected_balanced_pass": selected_balanced_pass,
        "selected_failure_mode": selected["failure_mode"],
        "selected_recommended_action": selected["recommended_action"],
        "selected_pose_score": selected["pose_score"],
        "selected_identity_pose_score": selected["identity_pose_score"],
        "manual_review_required": manual_review_required,
        "selected_postprocessed": str(selected_postprocessed),
        "contact_sheet": str(sheet),
        "output": str(out_dir),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
