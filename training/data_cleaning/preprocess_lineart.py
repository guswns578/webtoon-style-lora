# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


SRC_ROOT = Path(r"data/raw_episodes")
DST_ROOT = Path(r"data/lineart_preprocessed")
REPORT_PATH = Path(r"reports/lineart_preprocess_report.csv")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# Conservative filtering for SDXL buckets: keep normal webtoon panels, skip strips
# that mostly teach blank panel layout or extreme framing.
MIN_SIDE = 384
MIN_AREA = 400_000
MAX_WIDE_RATIO = 3.5
MAX_TALL_RATIO = 5.5


def iter_images() -> list[Path]:
    paths: list[Path] = []
    cut_dirs = sorted(
        [p for p in SRC_ROOT.iterdir() if p.is_dir() and p.name.endswith("_cut")],
        key=lambda p: int(p.name.removesuffix("_cut")) if p.name.removesuffix("_cut").isdigit() else 999999,
    )
    for cut_dir in cut_dirs:
        for path in sorted(cut_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                paths.append(path)
    return paths


def reject_reason(width: int, height: int) -> str:
    ratio = width / max(height, 1)
    if min(width, height) < MIN_SIDE:
        return "min_side_lt_384"
    if width * height < MIN_AREA:
        return "area_lt_400000"
    if ratio > MAX_WIDE_RATIO:
        return "too_wide"
    if (height / max(width, 1)) > MAX_TALL_RATIO:
        return "too_tall"
    return ""


def flatten_to_white(im: Image.Image) -> Image.Image:
    im = ImageOps.exif_transpose(im)
    if im.mode in {"RGBA", "LA"} or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        white.alpha_composite(rgba)
        return white.convert("RGB")
    return im.convert("RGB")


def remove_colored_marks(rgb: np.ndarray) -> np.ndarray:
    arr = rgb.astype(np.float32)
    maxc = arr.max(axis=2)
    minc = arr.min(axis=2)
    sat = (maxc - minc) / np.maximum(maxc, 1)

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    red_mark = (r > 95) & (r > g + 22) & (r > b + 22)
    blue_mark = (b > 95) & (b > r + 18) & (b > g + 8)
    purple_mark = (r > 90) & (b > 90) & (sat > 0.08) & (g < np.maximum(r, b) - 6)
    saturated = (sat > 0.16) & (maxc > 80)

    # Keep near-black ink even if compression introduces tiny color noise.
    dark_ink = maxc < 80
    mask = (red_mark | blue_mark | purple_mark | saturated) & ~dark_ink

    out = rgb.copy()
    out[mask] = [255, 255, 255]
    return out


def normalize_lineart(src: Path, dst: Path) -> None:
    with Image.open(src) as im:
        im = flatten_to_white(im)
        rgb = np.array(im)
        rgb = remove_colored_marks(rgb)

        gray = np.array(Image.fromarray(rgb).convert("L")).astype(np.float32)

        # Push light gray paper / panel background to white.
        gray[gray >= 205] = 255

        # Remove very faint construction lines while keeping anti-aliased ink.
        faint = (gray >= 175) & (gray < 205)
        gray[faint] = np.maximum(gray[faint], 238)

        # Slightly strengthen real ink.
        dark = gray < 165
        gray[dark] = np.clip((gray[dark] - 12) * 0.92, 0, 255)

        out = Image.fromarray(gray.astype(np.uint8), mode="L")
        out = ImageOps.autocontrast(out, cutoff=0.05)
        out = out.convert("RGB")
        dst.parent.mkdir(parents=True, exist_ok=True)
        out.save(dst, format="PNG", compress_level=3)


def safe_output_name(path: Path) -> str:
    rel = path.relative_to(SRC_ROOT)
    stem = rel.with_suffix("").as_posix().replace("/", "_").replace("\\", "_")
    stem = stem.replace(":", "_").replace("*", "_").replace("?", "_").replace('"', "_")
    stem = stem.replace("<", "_").replace(">", "_").replace("|", "_")
    return stem + ".png"


def main() -> None:
    if not SRC_ROOT.exists():
        raise FileNotFoundError(SRC_ROOT)
    DST_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    images = iter_images()
    stats = Counter()

    with REPORT_PATH.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "output", "width", "height", "ratio", "status", "reason"])

        for idx, src in enumerate(images, 1):
            try:
                with Image.open(src) as im:
                    width, height = im.size
            except Exception as exc:
                stats["read_error"] += 1
                writer.writerow([str(src), "", "", "", "", "skipped", f"read_error:{exc}"])
                continue

            reason = reject_reason(width, height)
            if reason:
                stats[f"skipped_{reason}"] += 1
                writer.writerow([str(src), "", width, height, f"{width / max(height, 1):.4f}", "skipped", reason])
                continue

            dst = DST_ROOT / safe_output_name(src)
            if dst.exists() and dst.stat().st_size > 0:
                stats["already_done"] += 1
                writer.writerow([str(src), str(dst), width, height, f"{width / max(height, 1):.4f}", "already_done", ""])
            else:
                normalize_lineart(src, dst)
                stats["processed"] += 1
                writer.writerow([str(src), str(dst), width, height, f"{width / max(height, 1):.4f}", "processed", ""])

            if idx % 500 == 0:
                print(f"{idx}/{len(images)} scanned, processed={stats['processed']}")

    print(f"SCANNED={len(images)}")
    for key in sorted(stats):
        print(f"{key.upper()}={stats[key]}")
    print(f"OUTPUT={DST_ROOT}")
    print(f"REPORT={REPORT_PATH}")


if __name__ == "__main__":
    main()
