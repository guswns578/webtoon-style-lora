from __future__ import annotations

import base64
import csv
import io
import json
import time
from pathlib import Path
from urllib import request

import numpy as np
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin

API = "http://127.0.0.1:7861"
OUTPUT_DIR = Path("outputs/crowd_baseline")
CHECKPOINT = "waiIllustriousSDXL_v160.safetensors"
LORA = "style_locon_e3"
WEIGHTS = (0.0, 0.65)

SEED = 43
WIDTH = 1216
HEIGHT = 832
STEPS = 28
CFG = 5.0
SAMPLER = "DPM++ 2M"
SCHEDULER = "Karras"

PROMPT_BODY = (
    "webtoon_lineart_style, lineart, monochrome, manga panel, wide shot, battlefield, "
    "six anonymous medieval infantry soldiers, mixed faces, varied helmets and light armor, "
    "three soldiers in foreground and three soldiers behind them, marching formation, "
    "overlapping bodies, depth, perspective, full body, spears and swords, "
    "clean thin ink lines, delicate controlled linework, large white negative space"
)
NEGATIVE_PROMPT = (
    "color, full color, cel shading, sepia, gray background, thick lines, bold outline, "
    "heavy ink, heavy shading, dense hatching, screentone, rough sketch, blurry, lowres, "
    "single person, solo, close-up, portrait, identical twins, same face, cloned faces, "
    "duplicate body, fused body, extra limbs, bad anatomy, bad hands, text, speech bubble, watermark"
)


def api_json(endpoint: str, payload: dict | None = None, timeout: int = 1800):
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = request.Request(
        API + endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def decode_image(value: str) -> Image.Image:
    encoded = value.split(",", 1)[-1]
    return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")


def metrics(image: Image.Image) -> dict[str, float]:
    rgb = np.asarray(image, dtype=np.float32)
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    dx = np.abs(np.diff(gray, axis=1))
    dy = np.abs(np.diff(gray, axis=0))
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    return {
        "mean_luma": round(float(gray.mean()), 4),
        "dark_ink_ratio_lt80": round(float(np.mean(gray < 80)), 6),
        "ink_ratio_lt200": round(float(np.mean(gray < 200)), 6),
        "midtone_ratio_80_to230": round(float(np.mean((gray >= 80) & (gray < 230))), 6),
        "near_white_ratio_gt245": round(float(np.mean(gray > 245)), 6),
        "edge_density_delta_gt24": round(float((np.mean(dx > 24) + np.mean(dy > 24)) / 2.0), 6),
        "mean_rgb_chroma": round(float(chroma.mean()), 4),
    }


def save_png(image: Image.Image, path: Path, record: dict) -> None:
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("experiment", json.dumps(record, ensure_ascii=False))
    if record.get("forge_info"):
        pnginfo.add_text("parameters", record["forge_info"])
    image.save(path, pnginfo=pnginfo)


def make_contact_sheet(rows: list[dict], output: Path) -> None:
    preview_width = 608
    preview_height = 416
    label_height = 58
    sheet = Image.new("RGB", (preview_width * len(rows), preview_height + label_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, row in enumerate(rows):
        with Image.open(row["path"]) as image:
            preview = image.convert("RGB").resize(
                (preview_width, preview_height), Image.Resampling.LANCZOS
            )
        x = index * preview_width
        sheet.paste(preview, (x, 0))
        draw.rectangle(
            (x, preview_height, x + preview_width, preview_height + label_height),
            fill=(245, 245, 245),
        )
        label = (
            f"style LoRA weight {row['lora']['weight']:.2f} | "
            f"seed {SEED} | {SAMPLER} {SCHEDULER}"
        )
        draw.text((x + 12, preview_height + 20), label, fill="black", font=font)
    sheet.save(output)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    available_models = api_json("/sdapi/v1/sd-models", timeout=30)
    matching = [m for m in available_models if CHECKPOINT.lower() in m["title"].lower()]
    if not matching:
        raise RuntimeError(f"Checkpoint not exposed by Forge: {CHECKPOINT}")
    selected_checkpoint = matching[0]["title"]
    api_json(
        "/sdapi/v1/options",
        {"sd_model_checkpoint": selected_checkpoint, "CLIP_stop_at_last_layers": 1},
        timeout=900,
    )

    records: list[dict] = []
    started = time.time()
    for index, weight in enumerate(WEIGHTS):
        prompt = f"<lora:{LORA}:{weight:.2f}>, {PROMPT_BODY}"
        payload = {
            "prompt": prompt,
            "negative_prompt": NEGATIVE_PROMPT,
            "seed": SEED,
            "sampler_name": SAMPLER,
            "scheduler": SCHEDULER,
            "steps": STEPS,
            "cfg_scale": CFG,
            "width": WIDTH,
            "height": HEIGHT,
            "batch_size": 1,
            "n_iter": 1,
            "restore_faces": False,
            "override_settings": {
                "sd_model_checkpoint": selected_checkpoint,
                "CLIP_stop_at_last_layers": 1,
            },
            "override_settings_restore_afterwards": False,
        }
        condition_started = time.time()
        result = api_json("/sdapi/v1/txt2img", payload)
        image = decode_image(result["images"][0])
        file_name = f"{index:02d}_style_w{weight:.2f}_seed{SEED}.png"
        output_path = OUTPUT_DIR / file_name
        record = {
            "file": file_name,
            "condition": "style_weight_only_ablation",
            "prompt": prompt,
            "negative_prompt": NEGATIVE_PROMPT,
            "seed": SEED,
            "sampler": SAMPLER,
            "scheduler": SCHEDULER,
            "steps": STEPS,
            "cfg": CFG,
            "width": WIDTH,
            "height": HEIGHT,
            "checkpoint": selected_checkpoint,
            "lora": {"name": LORA, "weight": weight},
            "controlnet": {
                "enabled": False,
                "model": None,
                "preprocessor": None,
                "weight": 0.0,
                "guidance_start": 0.0,
                "guidance_end": 0.0,
            },
            "elapsed_seconds": round(time.time() - condition_started, 3),
            "forge_info": result.get("info", ""),
        }
        record["metrics"] = metrics(image)
        save_png(image, output_path, record)
        record["path"] = str(output_path)
        records.append(record)
        print(f"saved {output_path}", flush=True)

    make_contact_sheet(records, OUTPUT_DIR / "comparison.png")
    manifest = {
        "experiment": "webtoon generic battlefield crowd, style LoRA weight A/B",
        "created_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "single_changed_variable": "style_locon_e3 LoRA weight (0.00 vs 0.65)",
        "total_elapsed_seconds": round(time.time() - started, 3),
        "records": [{k: v for k, v in row.items() if k != "path"} for row in records],
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (OUTPUT_DIR / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fieldnames = ["file", "lora_weight", *next(iter(records))["metrics"].keys()]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(
                {"file": row["file"], "lora_weight": row["lora"]["weight"], **row["metrics"]}
            )
    print(OUTPUT_DIR / "comparison.png", flush=True)


if __name__ == "__main__":
    main()
