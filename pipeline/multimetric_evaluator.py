from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib import request

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lineart_normalize import normalize_file

from conditional_face_refine import (
    DEFAULT_CONFIG,
    detect_largest_face,
    load_config,
    reference_features,
    score_identity,
)

ROOT = Path(__file__).resolve().parent
CSD_HELPER = ROOT / "csd_embed_images.py"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
DEFAULT_STYLE_REFS = Path(r"data/style_references")


def image_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def api_json(base_url: str, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    req = request.Request(
        base_url.rstrip("/") + endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with request.urlopen(req, timeout=600) as response:
        return json.loads(response.read().decode("utf-8"))


def detect_pose(base_url: str, image: Path) -> dict[str, Any]:
    result = api_json(
        base_url,
        "/controlnet/detect",
        {
            "controlnet_module": "dw_openpose_full",
            "controlnet_input_images": [image_b64(image)],
            "controlnet_processor_res": 768,
            "controlnet_threshold_a": -1,
            "controlnet_threshold_b": -1,
        },
    )
    poses = result.get("poses") or []
    if not poses:
        return {"people": [], "canvas_width": 0, "canvas_height": 0}
    return poses[0]


def largest_person(pose: dict[str, Any]) -> dict[str, Any] | None:
    people = pose.get("people") or []
    if not people:
        return None

    def area(person: dict[str, Any]) -> float:
        raw = np.asarray(person.get("pose_keypoints_2d", []), dtype=np.float32).reshape(-1, 3)
        visible = raw[raw[:, 2] > 0]
        if len(visible) < 2:
            return 0.0
        span = visible[:, :2].max(axis=0) - visible[:, :2].min(axis=0)
        return float(span[0] * span[1])

    return max(people, key=area)


def pose_metrics(source_pose: dict[str, Any], candidate_pose: dict[str, Any]) -> dict[str, Any]:
    source_person = largest_person(source_pose)
    candidate_person = largest_person(candidate_pose)
    if source_person is None or candidate_person is None:
        return {
            "pose_detected": False,
            "pose_people": len(candidate_pose.get("people") or []),
            "pose_common_joints": 0,
            "pose_coverage": 0.0,
            "pose_aligned_rmse": math.inf,
            "pose_composition_rmse": math.inf,
            "pose_score": 0.0,
        }

    src = np.asarray(source_person["pose_keypoints_2d"], dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(candidate_person["pose_keypoints_2d"], dtype=np.float64).reshape(-1, 3)
    count = min(len(src), len(dst), 18)
    src, dst = src[:count], dst[:count]
    visible_source = src[:, 2] > 0
    common = visible_source & (dst[:, 2] > 0)
    common_count = int(common.sum())
    source_count = max(1, int(visible_source.sum()))
    coverage = common_count / source_count
    if common_count < 4:
        return {
            "pose_detected": True,
            "pose_people": len(candidate_pose.get("people") or []),
            "pose_common_joints": common_count,
            "pose_coverage": coverage,
            "pose_aligned_rmse": math.inf,
            "pose_composition_rmse": math.inf,
            "pose_score": 0.0,
        }

    src_xy = src[common, :2] / np.array(
        [source_pose["canvas_width"], source_pose["canvas_height"]], dtype=np.float64
    )
    dst_xy = dst[common, :2] / np.array(
        [candidate_pose["canvas_width"], candidate_pose["canvas_height"]], dtype=np.float64
    )
    composition_rmse = float(np.sqrt(np.mean(np.sum((src_xy - dst_xy) ** 2, axis=1))))

    src_centered = src_xy - src_xy.mean(axis=0, keepdims=True)
    dst_centered = dst_xy - dst_xy.mean(axis=0, keepdims=True)
    src_scale = max(float(np.linalg.norm(src_centered)), 1e-8)
    dst_scale = max(float(np.linalg.norm(dst_centered)), 1e-8)
    src_norm = src_centered / src_scale
    dst_norm = dst_centered / dst_scale
    # Preserve screen-space orientation. Rotating a candidate into alignment
    # would reward a sideways or upside-down ControlNet result.
    aligned = dst_norm
    aligned_rmse = float(np.sqrt(np.mean(np.sum((src_norm - aligned) ** 2, axis=1))))
    score = float(coverage * math.exp(-5.0 * aligned_rmse) * math.exp(-4.0 * composition_rmse))
    return {
        "pose_detected": True,
        "pose_people": len(candidate_pose.get("people") or []),
        "pose_common_joints": common_count,
        "pose_coverage": coverage,
        "pose_aligned_rmse": aligned_rmse,
        "pose_composition_rmse": composition_rmse,
        "pose_score": score,
    }


def read_gray(path: Path) -> np.ndarray:
    raw = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Cannot decode image: {path}")
    return image


def line_metrics(path: Path) -> dict[str, float]:
    gray = read_gray(path)
    # Use only genuinely dark pixels for stroke width. A light gray paper veil
    # must not be interpreted as one giant connected ink region.
    ink = (gray < 120).astype(np.uint8)
    edges = cv2.Canny(gray, 80, 160)
    distance = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    ink_values = distance[ink > 0]
    return {
        "black_ratio": float(np.mean(gray < 70)),
        "heavy_ratio": float(np.mean(gray < 30)),
        "gray_ratio": float(np.mean((gray >= 70) & (gray < 235))),
        "white_ratio": float(np.mean(gray > 245)),
        "mean_gray": float(np.mean(gray)),
        "edge_density": float(np.mean(edges > 0)),
        "stroke_width_proxy": float(2.0 * np.median(ink_values)) if len(ink_values) else 0.0,
        "laplacian_var": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    }


LINE_KEYS = [
    "black_ratio",
    "heavy_ratio",
    "gray_ratio",
    "white_ratio",
    "edge_density",
    "stroke_width_proxy",
]


def line_reference_stats(paths: list[Path]) -> tuple[dict[str, float], dict[str, float]]:
    rows = [line_metrics(path) for path in paths]
    means = {key: float(np.mean([row[key] for row in rows])) for key in LINE_KEYS}
    stds = {
        key: max(float(np.std([row[key] for row in rows], ddof=1)), abs(means[key]) * 0.08, 1e-4)
        for key in LINE_KEYS
    }
    return means, stds


def line_similarity(
    metrics: dict[str, float], means: dict[str, float], stds: dict[str, float]
) -> float:
    # The target lineart is defined especially by white-space economy, low gray-tone
    # density, and restrained stroke width. Weight those dimensions more than
    # generic edge density.
    weights = {
        "black_ratio": 1.0,
        "heavy_ratio": 1.1,
        "gray_ratio": 1.8,
        "white_ratio": 1.8,
        "edge_density": 0.7,
        "stroke_width_proxy": 1.5,
    }
    clipped = {key: min(abs(metrics[key] - means[key]) / stds[key], 5.0) for key in LINE_KEYS}
    weighted_z = sum(clipped[key] * weights[key] for key in LINE_KEYS) / sum(weights.values())
    return float(math.exp(-weighted_z / 2.0))


def run_csd(paths: list[Path], output_dir: Path) -> dict[str, np.ndarray]:
    input_json = output_dir / "csd_inputs.json"
    output_npz = output_dir / "csd_embeddings.npz"
    input_json.write_text(
        json.dumps([str(path) for path in paths], ensure_ascii=False), encoding="utf-8"
    )
    env = dict(__import__("os").environ)
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    subprocess.run(
        [
            sys.executable,
            str(CSD_HELPER),
            "--input-json",
            str(input_json),
            "--output",
            str(output_npz),
        ],
        check=True,
        cwd=str(ROOT),
        env=env,
    )
    data = np.load(output_npz)
    return {str(path): embedding for path, embedding in zip(data["paths"], data["embeddings"])}


def csd_similarity(
    embedding: np.ndarray, reference_embeddings: list[np.ndarray]
) -> tuple[float, float]:
    prototype = np.mean(np.stack(reference_embeddings), axis=0)
    prototype /= max(float(np.linalg.norm(prototype)), 1e-8)
    cosine = float(np.dot(embedding, prototype))
    individual = [float(np.dot(embedding, ref)) for ref in reference_embeddings]
    return cosine, float(np.median(individual))


def identity_score(row: dict[str, Any], target: str) -> float:
    distance = float(row["target_distance"])
    if not math.isfinite(distance):
        return 0.0
    score = max(0.0, 1.0 - distance / 0.18)
    if row["predicted_character"] != target:
        score *= 0.45
    return float(score)


def safe_json(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: safe_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [safe_json(item) for item in value]
    return value


def font(size: int = 18) -> ImageFont.ImageFont:
    for path in [Path("C:/Windows/Fonts/malgun.ttf"), Path("C:/Windows/Fonts/arial.ttf")]:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def thumb(path: Path, size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def contact_sheet(rows: list[dict[str, Any]], pose: Path, output: Path) -> None:
    items: list[tuple[str, Path]] = [("POSE SOURCE", pose)]
    for rank, row in enumerate(
        sorted(rows, key=lambda item: item["composite_score"], reverse=True), 1
    ):
        label = (
            f"#{rank} {Path(row['image']).stem}\n"
            f"total {row['composite_score']:.3f} | id {row['identity_score']:.3f}\n"
            f"pose {row['pose_score']:.3f} | style {row['style_score']:.3f}\n"
            f"CCIP {row['target_distance']:.4f} -> {row['predicted_character']}"
        )
        items.append((label, Path(row.get("evaluation_image", row["image"]))))
    cell_w, cell_h, label_h, cols = 300, 380, 92, 3
    rows_count = math.ceil(len(items) / cols)
    sheet = Image.new("RGB", (cell_w * cols, (cell_h + label_h) * rows_count), "white")
    draw = ImageDraw.Draw(sheet)
    label_font = font(17)
    for idx, (label, path) in enumerate(items):
        x = (idx % cols) * cell_w
        y = (idx // cols) * (cell_h + label_h)
        sheet.paste(thumb(path, (cell_w, cell_h)), (x, y))
        draw.multiline_text(
            (x + 7, y + cell_h + 3), label, font=label_font, fill="black", spacing=1
        )
        draw.rectangle((x, y, x + cell_w - 1, y + cell_h + label_h - 1), outline=(150, 150, 150))
    sheet.save(output)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate identity, pose, and webtoon line style")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--character", required=True)
    parser.add_argument(
        "--scene",
        choices=["closeup", "upper_sword", "fullbody", "dynamic_action"],
        default="dynamic_action",
    )
    parser.add_argument("--style-reference-root", type=Path, default=DEFAULT_STYLE_REFS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--skip-csd", action="store_true")
    parser.add_argument(
        "--raw-style-evaluation",
        action="store_true",
        help="Evaluate raw Forge output instead of the normalized deliverable image",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = args.run_dir / "multimetric"
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = sorted(path for path in (args.run_dir / "images").glob("*.png") if path.is_file())
    if not candidates:
        raise FileNotFoundError(f"No candidates in {args.run_dir / 'images'}")
    style_ref_dir = args.style_reference_root / args.scene
    style_refs = sorted(
        path
        for path in style_ref_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and not path.name.startswith("_")
    )
    if not style_refs:
        raise FileNotFoundError(f"No style references in {style_ref_dir}")

    cache_dir = Path(config["output_root"]) / "_cache"
    references = reference_features(config, cache_dir)
    source_pose = detect_pose(config["forge_api"], args.pose)
    line_means, line_stds = line_reference_stats(style_refs)

    evaluation_paths: dict[Path, Path] = {path: path for path in candidates}
    if not args.raw_style_evaluation:
        normalized_dir = output_dir / "normalized"
        normalized_dir.mkdir(parents=True, exist_ok=True)
        for candidate in candidates:
            normalized = normalized_dir / candidate.name
            normalize_file(candidate, normalized, gamma=0.40)
            evaluation_paths[candidate] = normalized

    csd_map: dict[str, np.ndarray] = {}
    if not args.skip_csd:
        csd_map = run_csd([*style_refs, *evaluation_paths.values()], output_dir)

    rows: list[dict[str, Any]] = []
    pose_cache: dict[str, Any] = {"source": source_pose, "candidates": {}}
    for index, candidate in enumerate(candidates, 1):
        print(f"[{index}/{len(candidates)}] {candidate.name}", flush=True)
        image = Image.open(candidate).convert("RGB")
        face = detect_largest_face(image, config["identity"]["face_detection_confidence"])
        identity = score_identity(
            face.crop,
            args.character,
            references,
            config["identity"]["nearest_reference_count"],
        )
        candidate_pose = detect_pose(config["forge_api"], candidate)
        pose_cache["candidates"][str(candidate)] = candidate_pose
        pose = pose_metrics(source_pose, candidate_pose)
        evaluation_path = evaluation_paths[candidate]
        lines = line_metrics(evaluation_path)
        line_score = line_similarity(lines, line_means, line_stds)
        if csd_map:
            csd_proto, csd_median = csd_similarity(
                csd_map[str(evaluation_path)], [csd_map[str(path)] for path in style_refs]
            )
        else:
            csd_proto, csd_median = 0.0, 0.0
        # CSD captures broad artistic similarity, while the hand-crafted line
        # statistics enforce the project's actual production target: thin,
        # economical black lineart on white. Do not let CSD dominate this axis.
        style_score = 0.4 * csd_proto + 0.6 * line_score if csd_map else line_score
        id_score = identity_score(identity, args.character)
        composite = 0.45 * id_score + 0.30 * pose["pose_score"] + 0.25 * style_score
        if identity["predicted_character"] != args.character:
            composite *= 0.75
        row = {
            "image": str(candidate),
            "evaluation_image": str(evaluation_path),
            **identity,
            "identity_score": id_score,
            **pose,
            **lines,
            "line_similarity": line_score,
            "csd_prototype_similarity": csd_proto,
            "csd_median_reference_similarity": csd_median,
            "style_score": style_score,
            "composite_score": float(composite),
        }
        rows.append(row)

    rows.sort(key=lambda row: row["composite_score"], reverse=True)
    fields = [
        "rank",
        "image",
        "evaluation_image",
        "composite_score",
        "identity_score",
        "target_distance",
        "predicted_character",
        "margin",
        "pose_score",
        "pose_detected",
        "pose_people",
        "pose_common_joints",
        "pose_coverage",
        "pose_aligned_rmse",
        "pose_composition_rmse",
        "style_score",
        "csd_prototype_similarity",
        "csd_median_reference_similarity",
        "line_similarity",
        *LINE_KEYS,
        "laplacian_var",
    ]
    with (output_dir / "multimetric_ranking.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(rows, 1):
            output_row = {field: row.get(field, "") for field in fields}
            output_row["rank"] = rank
            writer.writerow(output_row)

    report = {
        "method": {
            "identity_weight": 0.45,
            "pose_weight": 0.30,
            "style_weight": 0.25,
            "identity_metric": "CCIP nearest-3 median distance against 8 character references",
            "pose_metric": "DW OpenPose common-joint coverage plus Procrustes and composition error",
            "style_metric": "CSD ViT-L prototype cosine (40%) plus target-weighted line-distribution similarity (60%)",
            "csd_warning": "Official repository warns uploaded weights may differ from paper numbers; do not use CSD alone.",
        },
        "target_character": args.character,
        "scene": args.scene,
        "pose_source": str(args.pose),
        "style_evaluated_after_lineart_normalization": not args.raw_style_evaluation,
        "style_references": [str(path) for path in style_refs],
        "line_reference_mean": line_means,
        "line_reference_std": line_stds,
        "rows": rows,
    }
    (output_dir / "multimetric_report.json").write_text(
        json.dumps(safe_json(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "pose_keypoints.json").write_text(
        json.dumps(safe_json(pose_cache), ensure_ascii=False), encoding="utf-8"
    )
    contact_sheet(rows, args.pose, output_dir / "multimetric_contact_sheet.png")
    print(
        json.dumps(
            safe_json({"best": rows[0], "output": str(output_dir)}), ensure_ascii=False, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
