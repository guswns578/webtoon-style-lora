# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

CLEAN = Path(r"data/lineart_clean")
REJECTS = Path(r"data/lineart_rejects")
REPORT = Path(r"reports/lineart_effect_line_filter_report.csv")
CONTACT = Path(r"reports/lineart_effect_line_rejects_contact.jpg")


def line_features(path: Path) -> dict[str, float | int]:
    gray = np.asarray(Image.open(path).convert("L"))
    h, w = gray.shape

    scale = min(1.0, 1100 / max(h, w))
    if scale < 1.0:
        small = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        small = gray

    sh, sw = small.shape
    dark = small < 170
    very_dark = small < 90

    edges = cv2.Canny(small, 80, 180, apertureSize=3)
    min_len = max(70, int(min(sw, sh) * 0.28))
    max_gap = max(8, int(min(sw, sh) * 0.025))
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=70, minLineLength=min_len, maxLineGap=max_gap
    )

    line_mask = np.zeros_like(small, dtype=np.uint8)
    long_count = 0
    total_len = 0.0
    if lines is not None:
        for line in lines[:, 0]:
            x1, y1, x2, y2 = map(int, line)
            length = float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
            if length < min_len:
                continue
            long_count += 1
            total_len += length
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, thickness=max(5, int(min(sw, sh) * 0.01)))

    line_pixels = line_mask > 0
    residual_dark = dark & ~line_pixels
    residual_very_dark = very_dark & ~line_pixels

    center = small[int(sh * 0.2) : int(sh * 0.8), int(sw * 0.2) : int(sw * 0.8)]
    center_white = float((center > 245).mean()) if center.size else 1.0
    center_dark = float((center < 90).mean()) if center.size else 0.0
    margin_y = max(1, int(sh * 0.12))
    margin_x = max(1, int(sw * 0.12))
    edge_mask = np.zeros_like(dark, dtype=bool)
    edge_mask[:margin_y, :] = True
    edge_mask[-margin_y:, :] = True
    edge_mask[:, :margin_x] = True
    edge_mask[:, -margin_x:] = True
    edge_dark = float(dark[edge_mask].mean()) if edge_mask.any() else 0.0
    edge_very_dark = float(very_dark[edge_mask].mean()) if edge_mask.any() else 0.0

    return {
        "width": w,
        "height": h,
        "dark_ratio": float(dark.mean()),
        "very_dark_ratio": float(very_dark.mean()),
        "line_mask_ratio": float(line_pixels.mean()),
        "long_line_count": int(long_count),
        "line_length_norm": float(total_len / max(sw * sh, 1)),
        "residual_dark_ratio": float(residual_dark.mean()),
        "residual_very_dark_ratio": float(residual_very_dark.mean()),
        "center_white": center_white,
        "center_dark": center_dark,
        "edge_dark": edge_dark,
        "edge_very_dark": edge_very_dark,
    }


def reject_reason(f: dict[str, float | int]) -> str:
    long_count = int(f["long_line_count"])
    line_mask = float(f["line_mask_ratio"])
    residual_dark = float(f["residual_dark_ratio"])
    residual_vdark = float(f["residual_very_dark_ratio"])
    center_white = float(f["center_white"])
    center_dark = float(f["center_dark"])
    edge_dark = float(f["edge_dark"])
    edge_vdark = float(f["edge_very_dark"])
    dark = float(f["dark_ratio"])

    # Pure impact/speed-line panels: many long lines, little non-line drawing left.
    if long_count >= 14 and residual_dark < 0.018 and residual_vdark < 0.006:
        return "effect_lines_only"

    # Border/ending-line panels with a mostly blank center.
    if long_count >= 8 and center_white > 0.94 and center_dark < 0.012 and residual_dark < 0.025:
        return "blank_center_effect_lines"

    # Similar to the above, but catches panels where Hough detects only a few
    # long segments while the center is still almost empty.
    if long_count >= 20 and center_white > 0.94 and center_dark < 0.003 and residual_dark < 0.04:
        return "blank_center_many_end_lines"

    # Sparse faint streaks without useful character/background information.
    if long_count >= 5 and dark < 0.045 and center_white > 0.985 and residual_dark < 0.012:
        return "sparse_end_lines"

    # Very blank center with only border/end-line activity near the edges.
    if (
        center_white > 0.985
        and center_dark < 0.001
        and 0.018 < dark < 0.075
        and edge_dark > 0.035
        and edge_vdark > 0.012
        and residual_dark < 0.05
    ):
        return "blank_center_edge_lines"

    # Wide speedline strips are usually just panel transition effects.
    width = int(f["width"])
    height = int(f["height"])
    if (
        width / max(height, 1) > 1.7
        and long_count >= 10
        and residual_dark < 0.03
        and line_mask > 0.035
    ):
        return "wide_speedline_strip"

    return ""


def move_pair_to_rejects(img: Path) -> None:
    REJECTS.mkdir(parents=True, exist_ok=True)
    dst_img = REJECTS / img.name
    shutil.move(str(img), str(dst_img))
    txt = img.with_suffix(".txt")
    if txt.exists():
        shutil.move(str(txt), str(REJECTS / txt.name))


def make_contact(paths: list[Path]) -> None:
    if not paths:
        return
    sample = paths[:60]
    tw, th = 220, 230
    cols = 6
    rows = (len(sample) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * th), "white")
    draw = ImageDraw.Draw(sheet)
    for i, path in enumerate(sample):
        r, c = divmod(i, cols)
        try:
            im = Image.open(REJECTS / path.name).convert("RGB")
            im.thumbnail((tw, th - 28), Image.Resampling.LANCZOS)
            x = c * tw + (tw - im.width) // 2
            y = r * th + (th - 28 - im.height) // 2
            sheet.paste(im, (x, y))
            draw.text((c * tw + 3, r * th + th - 25), path.name[:31], fill=(0, 0, 0))
        except Exception as exc:
            draw.text((c * tw + 3, r * th + 10), f"ERR {exc}", fill=(255, 0, 0))
    CONTACT.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(CONTACT, quality=92)


def main() -> None:
    pngs = sorted(CLEAN.glob("*.png"))
    moved: list[Path] = []
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    with REPORT.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "file",
                "status",
                "reason",
                "width",
                "height",
                "dark_ratio",
                "line_mask_ratio",
                "long_line_count",
                "line_length_norm",
                "residual_dark_ratio",
                "residual_very_dark_ratio",
                "center_white",
                "center_dark",
                "edge_dark",
                "edge_very_dark",
            ]
        )
        for idx, img in enumerate(pngs, 1):
            fts = line_features(img)
            reason = reject_reason(fts)
            status = "reject" if reason else "keep"
            if reason:
                moved.append(img)
                move_pair_to_rejects(img)

            writer.writerow(
                [
                    img.name,
                    status,
                    reason,
                    fts["width"],
                    fts["height"],
                    f"{float(fts['dark_ratio']):.6f}",
                    f"{float(fts['line_mask_ratio']):.6f}",
                    fts["long_line_count"],
                    f"{float(fts['line_length_norm']):.6f}",
                    f"{float(fts['residual_dark_ratio']):.6f}",
                    f"{float(fts['residual_very_dark_ratio']):.6f}",
                    f"{float(fts['center_white']):.6f}",
                    f"{float(fts['center_dark']):.6f}",
                    f"{float(fts['edge_dark']):.6f}",
                    f"{float(fts['edge_very_dark']):.6f}",
                ]
            )

            if idx % 1000 == 0:
                print(f"{idx}/{len(pngs)} scanned, moved={len(moved)}")

    make_contact(moved)

    print(f"SCANNED={len(pngs)}")
    print(f"MOVED_TO_REJECTS={len(moved)}")
    print(f"CLEAN_PNG={len(list(CLEAN.glob('*.png')))}")
    print(f"CLEAN_TXT={len(list(CLEAN.glob('*.txt')))}")
    print(f"REJECTS_PNG={len(list(REJECTS.glob('*.png')))}")
    print(f"REJECTS_TXT={len(list(REJECTS.glob('*.txt')))}")
    print(f"REPORT={REPORT}")
    print(f"CONTACT={CONTACT}")


if __name__ == "__main__":
    main()
