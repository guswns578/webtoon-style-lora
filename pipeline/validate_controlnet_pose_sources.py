from __future__ import annotations

import argparse
import base64
import csv
import json
from pathlib import Path
from urllib import request

from pose_metrics import pose_source_quality

DEFAULT_INPUT = Path("workspace/pose_src")


def detect_with_preview(base_url: str, image: Path) -> dict:
    payload = {
        "controlnet_module": "dw_openpose_full",
        "controlnet_input_images": [base64.b64encode(image.read_bytes()).decode("ascii")],
        "controlnet_processor_res": 768,
        "controlnet_threshold_a": -1,
        "controlnet_threshold_b": -1,
    }
    req = request.Request(
        base_url.rstrip("/") + "/controlnet/detect",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with request.urlopen(req, timeout=600) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight ControlNet pose sources")
    parser.add_argument("input", type=Path, nargs="?", default=DEFAULT_INPUT)
    parser.add_argument("--forge", default="http://127.0.0.1:7861")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.input / "pose_preflight"
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    images = [
        path
        for path in sorted(args.input.glob("*.png"))
        if "preview" not in path.stem and path.parent != output
    ]
    for index, path in enumerate(images, 1):
        print(f"[{index}/{len(images)}] {path.name}", flush=True)
        result = detect_with_preview(args.forge, path)
        poses = result.get("poses") or []
        pose = poses[0] if poses else {"people": []}
        quality = pose_source_quality(pose)
        preview = ""
        if result.get("images"):
            preview_path = output / f"{path.stem}_dw_openpose.png"
            preview_path.write_bytes(base64.b64decode(result["images"][0].split(",", 1)[-1]))
            preview = str(preview_path)
        rows.append(
            {
                "source": str(path),
                "valid": quality["valid"],
                "visible_joints": quality["visible_joints"],
                "people": quality["people"],
                "preview": preview,
            }
        )

    report_csv = output / "pose_preflight.csv"
    with report_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "pose_preflight.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(report_csv)
    print(f"valid={sum(row['valid'] for row in rows)}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
