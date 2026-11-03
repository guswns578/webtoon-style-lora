from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from pose_metrics import detect_pose, pose_source_quality
from conditional_face_refine import load_config


ROOT = Path(__file__).resolve().parent
PIPELINE = ROOT / "conditional_face_refine.py"
DEFAULT_CONFIG = ROOT / "conditional_face_refine_config.json"
DEFAULT_OUTPUT = Path("workspace/conditional_face_refine/benchmark_8char_2pose")
DEFAULT_POSE_ROOT = Path("workspace/pose_src")


def load_font(size: int) -> ImageFont.ImageFont:
    for path in (Path("C:/Windows/Fonts/malgun.ttf"), Path("C:/Windows/Fonts/arial.ttf")):
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def thumbnail(path: Path, size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def write_contact_sheet(rows: list[dict], output: Path) -> None:
    cell_w, image_h, label_h, cols = 300, 390, 100, 4
    total_rows = math.ceil(len(rows) / cols)
    sheet = Image.new("RGB", (cols * cell_w, total_rows * (image_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    label_font = load_font(16)
    for index, row in enumerate(rows):
        x = (index % cols) * cell_w
        y = (index // cols) * (image_h + label_h)
        image_path = Path(row["selected_postprocessed"])
        if image_path.exists():
            sheet.paste(thumbnail(image_path, (cell_w, image_h)), (x, y))
        label = (
            f"{row['character']} / {row['pose']}\n"
            f"seed {row.get('selected_seed', '-')} | {row.get('selected_stage', '-')}\n"
            f"CCIP {row.get('selected_target_distance', math.inf):.4f} -> "
            f"{row.get('selected_predicted_character', 'none')}\n"
            f"pose {row.get('selected_pose_score', 0.0):.3f} | "
            f"joint {row.get('selected_identity_pose_score', 0.0):.3f}"
        )
        draw.multiline_text((x + 6, y + image_h + 4), label, font=label_font, fill="black", spacing=1)
        draw.rectangle((x, y, x + cell_w - 1, y + image_h + label_h - 1), outline=(145, 145, 145))
    sheet.save(output)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the webtoon conditional face-refine benchmark")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pose-root", type=Path, default=DEFAULT_POSE_ROOT)
    parser.add_argument("--characters", default="char_a,char_b,char_c,char_d,char_e,char_f,char_g,char_h")
    parser.add_argument("--poses", default="punch,cheer")
    parser.add_argument("--seeds", default="43,100")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    characters = [item.strip() for item in args.characters.split(",") if item.strip()]
    poses = [item.strip() for item in args.poses.split(",") if item.strip()]
    config = load_config(args.config)
    invalid_poses = []
    for pose_name in poses:
        pose_path = args.pose_root / f"{pose_name}.png"
        quality = pose_source_quality(detect_pose(config["forge_api"], pose_path))
        print(f"pose source {pose_name}: {quality['visible_joints']} visible joints", flush=True)
        if not quality["valid"]:
            invalid_poses.append({"pose": pose_name, **quality})
    if invalid_poses:
        raise ValueError(f"Invalid pose sources; no generation started: {invalid_poses}")
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    failures: list[dict] = []
    total = len(characters) * len(poses)
    current = 0

    for character in characters:
        for pose_name in poses:
            current += 1
            pose_path = args.pose_root / f"{pose_name}.png"
            run_dir = args.output / f"{character}_{pose_name}"
            summary_path = run_dir / "summary.json"
            print(f"[{current}/{total}] {character} / {pose_name}", flush=True)
            if not (args.resume and summary_path.exists()):
                run_dir.mkdir(parents=True, exist_ok=True)
                command = [
                    sys.executable,
                    str(PIPELINE),
                    "--config",
                    str(args.config),
                    "--character",
                    character,
                    "--pose",
                    str(pose_path),
                    "--seeds",
                    args.seeds,
                    "--output",
                    str(run_dir),
                ]
                with (run_dir / "benchmark_stdout.log").open("w", encoding="utf-8") as stdout, (
                    run_dir / "benchmark_stderr.log"
                ).open("w", encoding="utf-8") as stderr:
                    completed = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr)
                if completed.returncode != 0:
                    failures.append({"character": character, "pose": pose_name, "returncode": completed.returncode})
                    print(f"  FAILED ({completed.returncode})", flush=True)
                    continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary.update({"character": character, "pose": pose_name, "run_dir": str(run_dir)})
            rows.append(summary)
            print(
                f"  selected seed={summary['selected_seed']} "
                f"CCIP={summary['selected_target_distance']:.4f} "
                f"pose={summary.get('selected_pose_score', 0.0):.3f}",
                flush=True,
            )

    fields = [
        "character",
        "pose",
        "selected_seed",
        "selected_stage",
        "selected_target_distance",
        "selected_predicted_character",
        "selected_passed",
        "selected_balanced_pass",
        "selected_failure_mode",
        "selected_recommended_action",
        "selected_pose_score",
        "selected_identity_pose_score",
        "manual_review_required",
        "before_pass",
        "before_total",
        "after_pass",
        "after_total",
        "selected_result",
        "selected_postprocessed",
        "run_dir",
    ]
    with (args.output / "benchmark_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    (args.output / "benchmark_failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if rows:
        write_contact_sheet(rows, args.output / "benchmark_contact_sheet.png")
    print(json.dumps({"completed": len(rows), "failed": len(failures), "output": str(args.output)}, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
