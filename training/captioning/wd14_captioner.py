# -*- coding: utf-8 -*-
"""
Prepare and WD14-caption the webtoon lineart dataset.

Default behavior:
  - reads top-level *_cut folders under data/raw_episodes
  - optionally creates a cleaned training copy
  - runs WD14 tagger
  - writes .txt captions next to each training image

Recommended first run:
  python wd14_captioner.py --prepare --caption
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

try:
    from tqdm import tqdm
except ImportError:  # tqdm is only for progress bars.
    def tqdm(iterable, **_kwargs):
        return iterable


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

DEFAULT_INPUT_ROOT = r"data/raw_episodes"
DEFAULT_OUTPUT_ROOT = r"data/lineart_captioned"
DEFAULT_TRIGGER = "webtoon_lineart_style"
DEFAULT_MODEL_ID = "SmilingWolf/wd-v1-4-convnext-tagger-v2"


BASE_TAGS = [
    "lineart",
    "monochrome",
    "manga_panel",
]

KEEP_EVEN_IF_COLORISH = {
    "monochrome",
    "greyscale",
    "grayscale",
    "lineart",
    "black_and_white",
    "white_background",
    "simple_background",
    "transparent_background",
}

DROP_EXACT = {
    # Rating / quality / training-noise tags
    "safe",
    "sensitive",
    "questionable",
    "explicit",
    "masterpiece",
    "best_quality",
    "high_quality",
    "low_quality",
    "normal_quality",
    "lowres",
    "bad_anatomy",
    "bad_hands",
    "watermark",
    "signature",
    "artist_name",
    "username",
    "copyright_name",
    # Color or non-lineart media states
    "color",
    "colorful",
    "full_color",
    "partially_colored",
    "spot_color",
    "limited_palette",
    "watercolor",
    "oil_painting",
    "realistic",
    "photorealistic",
    "photo",
    "3d",
    "render",
    "painting_(medium)",
}

COLOR_WORDS = {
    "black",
    "white",
    "grey",
    "gray",
    "silver",
    "red",
    "blue",
    "green",
    "yellow",
    "purple",
    "pink",
    "orange",
    "brown",
    "blonde",
    "aqua",
    "cyan",
    "teal",
    "magenta",
    "gold",
    "multicolored",
    "two-tone",
    "gradient",
}

COLOR_TARGETS = {
    "hair",
    "eyes",
    "eye",
    "shirt",
    "dress",
    "clothes",
    "clothing",
    "coat",
    "jacket",
    "robe",
    "cape",
    "cloak",
    "pants",
    "skirt",
    "gloves",
    "boots",
    "shoes",
    "armor",
    "weapon",
    "sword",
    "hat",
    "hood",
    "background",
    "skin",
}


@dataclass(frozen=True)
class ImageInfo:
    src: Path
    rel: Path
    width: int
    height: int

    @property
    def aspect(self) -> float:
        return self.width / max(self.height, 1)

    @property
    def low_info_reason(self) -> str:
        if min(self.width, self.height) < 384:
            return "min_side_lt_384"
        if self.aspect > 4.0:
            return "very_wide"
        if self.aspect < 0.18:
            return "very_tall"
        return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare and caption webtoon lineart *_cut folders with WD14."
    )
    parser.add_argument("--input-root", default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--trigger", default=DEFAULT_TRIGGER)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--general-threshold", type=float, default=0.35)
    parser.add_argument("--character-threshold", type=float, default=0.85)
    parser.add_argument("--max-tags", type=int, default=45)
    parser.add_argument("--prepare", action="store_true", help="Create cleaned copy.")
    parser.add_argument("--caption", action="store_true", help="Run WD14 captions.")
    parser.add_argument("--report", action="store_true", help="Write CSV report only/also.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    parser.add_argument("--skip-low-info", action="store_true", help="Skip tiny/extreme aspect images in prepared copy.")
    parser.add_argument("--no-remove-red-ink", action="store_true")
    parser.add_argument("--no-grayscale", action="store_true")
    parser.add_argument("--no-autocontrast", action="store_true")
    parser.add_argument("--caption-ext", default=".txt")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.prepare and not args.caption and not args.report:
        args.report = True
    return args


def find_images(root: Path) -> list[ImageInfo]:
    if not root.exists():
        raise FileNotFoundError(f"Input root does not exist: {root}")

    cut_dirs = [p for p in root.iterdir() if p.is_dir() and p.name.endswith("_cut")]
    infos: list[ImageInfo] = []

    for cut_dir in sorted(cut_dirs, key=lambda p: p.name):
        for src in sorted(cut_dir.rglob("*")):
            if not src.is_file() or src.suffix.lower() not in IMAGE_EXTS:
                continue
            try:
                with Image.open(src) as im:
                    infos.append(ImageInfo(src=src, rel=src.relative_to(root), width=im.width, height=im.height))
            except Exception as exc:
                print(f"[WARN] Cannot read image: {src} ({exc})", file=sys.stderr)

    return infos


def remove_red_ink_rgb(arr: np.ndarray) -> np.ndarray:
    """Replace red review marks with white before grayscale conversion."""
    red = arr[:, :, 0].astype(np.int16)
    green = arr[:, :, 1].astype(np.int16)
    blue = arr[:, :, 2].astype(np.int16)
    mask = (red > 120) & (red > green + 45) & (red > blue + 45)
    arr = arr.copy()
    arr[mask] = [255, 255, 255]
    return arr


def clean_image(
    src: Path,
    dst: Path,
    *,
    remove_red_ink: bool,
    grayscale: bool,
    autocontrast: bool,
    overwrite: bool,
) -> None:
    if dst.exists() and not overwrite:
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in {"RGBA", "LA"} or (im.mode == "P" and "transparency" in im.info):
            rgba = im.convert("RGBA")
            white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            white.alpha_composite(rgba)
            im = white.convert("RGB")
        else:
            im = im.convert("RGB")

        arr = np.array(im)
        if remove_red_ink:
            arr = remove_red_ink_rgb(arr)
        im = Image.fromarray(arr, "RGB")

        if grayscale:
            im = im.convert("L")
            if autocontrast:
                im = ImageOps.autocontrast(im, cutoff=0.2)
            im = im.convert("RGB")
        elif autocontrast:
            im = ImageOps.autocontrast(im, cutoff=0.2)

        im.save(dst)


def prepare_dataset(infos: list[ImageInfo], input_root: Path, output_root: Path, args: argparse.Namespace) -> list[Path]:
    prepared: list[Path] = []
    for info in tqdm(infos, desc="Preparing images"):
        if args.skip_low_info and info.low_info_reason:
            continue
        dst = output_root / info.rel
        # Use PNG for cleaned images to avoid adding JPEG artifacts to lineart.
        dst = dst.with_suffix(".png")
        prepared.append(dst)
        if not args.dry_run:
            clean_image(
                info.src,
                dst,
                remove_red_ink=not args.no_remove_red_ink,
                grayscale=not args.no_grayscale,
                autocontrast=not args.no_autocontrast,
                overwrite=args.overwrite,
            )
    return prepared


def write_report(infos: list[ImageInfo], output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "dataset_report.csv"
    with report_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "width", "height", "aspect", "low_info_reason"])
        for info in infos:
            writer.writerow([str(info.src), info.width, info.height, f"{info.aspect:.4f}", info.low_info_reason])
    return report_path


def hf_download_model(model_id: str) -> tuple[Path, Path]:
    from huggingface_hub import hf_hub_download

    model_path = Path(hf_hub_download(model_id, filename="model.onnx"))
    tags_path = Path(hf_hub_download(model_id, filename="selected_tags.csv"))
    return model_path, tags_path


def make_square_white(im: Image.Image) -> Image.Image:
    im = ImageOps.exif_transpose(im).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    bg.alpha_composite(im)
    im = bg.convert("RGB")
    side = max(im.size)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
    return canvas


def preprocess_for_wd14(path: Path, image_size: int) -> np.ndarray:
    with Image.open(path) as im:
        im = make_square_white(im)
        im = im.resize((image_size, image_size), Image.Resampling.LANCZOS)
        arr = np.asarray(im, dtype=np.float32)
        arr = arr[:, :, ::-1]  # RGB to BGR, WD tagger convention
        return np.expand_dims(arr, axis=0)


def should_drop_tag(tag: str) -> bool:
    tag = tag.strip()
    if not tag or tag in KEEP_EVEN_IF_COLORISH:
        return False
    if tag in DROP_EXACT:
        return True

    parts = tag.split("_")
    if any(part in COLOR_WORDS for part in parts):
        if any(target in parts for target in COLOR_TARGETS):
            return True
        if len(parts) == 1:
            return True

    # Examples: red_dress, blue_shirt, black_gloves, blonde_hair.
    if len(parts) >= 2 and parts[0] in COLOR_WORDS and parts[1] in COLOR_TARGETS:
        return True

    # Examples: colored_skin, colored_inner_hair, multicolored_hair.
    if re.search(r"(colored|multicolored|two-tone|gradient).*(hair|eyes|skin|clothes)", tag):
        return True

    return False


def normalize_tag(tag: str) -> str:
    return tag.strip().replace(" ", "_")


def build_caption(
    scored_tags: Iterable[tuple[str, float]],
    *,
    trigger: str,
    max_tags: int,
) -> str:
    caption: list[str] = [trigger, *BASE_TAGS]
    seen = set(caption)

    for tag, _score in scored_tags:
        tag = normalize_tag(tag)
        if tag in seen or should_drop_tag(tag):
            continue
        caption.append(tag)
        seen.add(tag)
        if len(caption) >= max_tags + 1 + len(BASE_TAGS):
            break

    return ", ".join(caption)


def caption_images(image_paths: list[Path], args: argparse.Namespace) -> None:
    import onnxruntime as ort

    model_path, tags_path = hf_download_model(args.model_id)
    tag_df = pd.read_csv(tags_path)
    tag_names = tag_df["name"].tolist()
    categories = tag_df["category"].tolist() if "category" in tag_df.columns else [0] * len(tag_names)

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(str(model_path), providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    shape = session.get_inputs()[0].shape
    image_size = int(shape[1] if isinstance(shape[1], int) else 448)

    for path in tqdm(image_paths, desc="Captioning"):
        out_txt = path.with_suffix(args.caption_ext)
        if out_txt.exists() and not args.overwrite:
            continue

        tensor = preprocess_for_wd14(path, image_size)
        probs = session.run([output_name], {input_name: tensor})[0][0]

        scored: list[tuple[str, float]] = []
        for name, category, prob in zip(tag_names, categories, probs):
            prob = float(prob)
            if category == 9:
                continue  # rating tag
            if category == 4:
                if prob < args.character_threshold:
                    continue
                # Character tags are intentionally skipped for style LoRA.
                continue
            if prob < args.general_threshold:
                continue
            scored.append((name, prob))

        scored.sort(key=lambda x: x[1], reverse=True)
        caption = build_caption(scored, trigger=args.trigger, max_tags=args.max_tags)
        if args.dry_run:
            print(f"{path}: {caption}")
        else:
            out_txt.write_text(caption + "\n", encoding="utf-8")


def summarize(infos: list[ImageInfo]) -> None:
    if not infos:
        print("No images found.")
        return
    widths = [x.width for x in infos]
    heights = [x.height for x in infos]
    low_info = [x for x in infos if x.low_info_reason]
    print(f"Images: {len(infos)}")
    print(f"Width: min={min(widths)} median={int(np.median(widths))} max={max(widths)}")
    print(f"Height: min={min(heights)} median={int(np.median(heights))} max={max(heights)}")
    print(f"Low-info/extreme aspect candidates: {len(low_info)}")


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    infos = find_images(input_root)
    summarize(infos)

    if args.report:
        report_path = write_report(infos, output_root)
        print(f"Report: {report_path}")

    if args.prepare:
        prepared = prepare_dataset(infos, input_root, output_root, args)
        print(f"Prepared images: {len(prepared)} -> {output_root}")
        target_images = prepared
    else:
        target_images = [x.src for x in infos]

    if args.caption:
        existing = [p for p in target_images if p.exists()]
        if not existing and args.prepare and args.dry_run:
            existing = [x.src for x in infos]
        caption_images(existing, args)


if __name__ == "__main__":
    main()
