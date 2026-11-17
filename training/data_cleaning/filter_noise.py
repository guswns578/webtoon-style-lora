# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SRC = Path(r"data/lineart_preprocessed")
CLEAN = Path(r"data/lineart_clean")
REJECTS = Path(r"data/lineart_rejects")
REPORT = Path(r"reports/lineart_noise_filter_report.csv")
CONTACT = Path(r"reports/lineart_rejects_contact.jpg")


def metrics(path: Path) -> dict[str, float | int]:
    with Image.open(path) as im:
        gray = np.asarray(im.convert("L"), dtype=np.uint8)
    total = gray.size
    return {
        "width": int(gray.shape[1]),
        "height": int(gray.shape[0]),
        "mean": float(gray.mean()),
        "dark_ratio": float((gray < 80).sum() / total),
        "ink_ratio": float((gray < 170).sum() / total),
        "mid_ratio": float(((gray >= 80) & (gray < 235)).sum() / total),
        "white_ratio": float((gray > 245).sum() / total),
        "near_white_ratio": float((gray > 235).sum() / total),
        "std": float(gray.std()),
    }


def reject_reasons(m: dict[str, float | int]) -> list[str]:
    dark = float(m["dark_ratio"])
    ink = float(m["ink_ratio"])
    mid = float(m["mid_ratio"])
    white = float(m["white_ratio"])
    near_white = float(m["near_white_ratio"])
    mean = float(m["mean"])
    std = float(m["std"])

    reasons: list[str] = []

    # Very faint construction-line panels or almost-empty panels.
    if dark < 0.0015 and ink < 0.015:
        reasons.append("almost_blank_or_faint")

    # Mostly white image with only tiny specks/ghost lines.
    if dark < 0.003 and white > 0.985:
        reasons.append("too_white_low_ink")

    # Strong gray texture/background panels: bad for a clean lineart LoRA.
    if mid > 0.38 and near_white < 0.62:
        reasons.append("gray_texture_background")

    # Washed gray panel where the background dominates more than line art.
    if mean < 228 and mid > 0.30 and dark < 0.09:
        reasons.append("gray_background_dominant")

    # Low contrast, low ink panels are usually rough/scan noise after cleanup.
    if std < 18 and ink < 0.035:
        reasons.append("low_contrast_low_ink")

    return reasons


def copy_pair(src_img: Path, dst_dir: Path) -> None:
    dst_img = dst_dir / src_img.name
    dst_txt = dst_dir / src_img.with_suffix(".txt").name
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_img, dst_img)
    src_txt = src_img.with_suffix(".txt")
    if src_txt.exists():
        shutil.copy2(src_txt, dst_txt)


def make_contact(paths: list[Path]) -> None:
    if not paths:
        return
    sample = paths[:48]
    thumb_w, thumb_h = 220, 240
    cols = 6
    rows = (len(sample) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_w, rows * thumb_h), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, path in enumerate(sample):
        r, c = divmod(idx, cols)
        try:
            im = Image.open(path).convert("RGB")
            im.thumbnail((thumb_w, thumb_h - 30), Image.Resampling.LANCZOS)
            x = c * thumb_w + (thumb_w - im.width) // 2
            y = r * thumb_h + (thumb_h - 30 - im.height) // 2
            sheet.paste(im, (x, y))
            draw.text((c * thumb_w + 4, r * thumb_h + thumb_h - 26), path.name[:32], fill=(0, 0, 0))
        except Exception as exc:
            draw.text((c * thumb_w + 4, r * thumb_h + 10), f"ERR {exc}", fill=(255, 0, 0))
    CONTACT.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(CONTACT, quality=92)


def main() -> None:
    if not SRC.exists():
        raise FileNotFoundError(SRC)
    CLEAN.mkdir(parents=True, exist_ok=True)
    REJECTS.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    pngs = sorted(SRC.glob("*.png"))
    counts = Counter()
    rejected_paths: list[Path] = []

    with REPORT.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "file",
                "status",
                "reasons",
                "width",
                "height",
                "mean",
                "dark_ratio",
                "ink_ratio",
                "mid_ratio",
                "white_ratio",
                "std",
            ]
        )
        for idx, img in enumerate(pngs, 1):
            try:
                m = metrics(img)
                reasons = reject_reasons(m)
            except Exception as exc:
                m = {
                    "width": 0,
                    "height": 0,
                    "mean": 0,
                    "dark_ratio": 0,
                    "ink_ratio": 0,
                    "mid_ratio": 0,
                    "white_ratio": 0,
                    "std": 0,
                }
                reasons = [f"read_error:{exc}"]

            if reasons:
                status = "reject"
                rejected_paths.append(img)
                copy_pair(img, REJECTS)
                counts["reject"] += 1
                for reason in reasons:
                    counts[f"reject_{reason}"] += 1
            else:
                status = "keep"
                copy_pair(img, CLEAN)
                counts["keep"] += 1

            writer.writerow(
                [
                    img.name,
                    status,
                    "|".join(reasons),
                    m["width"],
                    m["height"],
                    f"{float(m['mean']):.4f}",
                    f"{float(m['dark_ratio']):.6f}",
                    f"{float(m['ink_ratio']):.6f}",
                    f"{float(m['mid_ratio']):.6f}",
                    f"{float(m['white_ratio']):.6f}",
                    f"{float(m['std']):.4f}",
                ]
            )

            if idx % 1000 == 0:
                print(
                    f"{idx}/{len(pngs)} scanned, keep={counts['keep']}, reject={counts['reject']}"
                )

    make_contact(rejected_paths)

    clean_pngs = len(list(CLEAN.glob("*.png")))
    clean_txts = len(list(CLEAN.glob("*.txt")))
    reject_pngs = len(list(REJECTS.glob("*.png")))
    reject_txts = len(list(REJECTS.glob("*.txt")))

    print(f"SCANNED={len(pngs)}")
    print(f"CLEAN_PNG={clean_pngs}")
    print(f"CLEAN_TXT={clean_txts}")
    print(f"REJECT_PNG={reject_pngs}")
    print(f"REJECT_TXT={reject_txts}")
    for key in sorted(counts):
        print(f"{key.upper()}={counts[key]}")
    print(f"CLEAN_DIR={CLEAN}")
    print(f"REJECTS_DIR={REJECTS}")
    print(f"REPORT={REPORT}")
    print(f"CONTACT={CONTACT}")


if __name__ == "__main__":
    main()
