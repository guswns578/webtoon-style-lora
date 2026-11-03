from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = scores >= threshold
    tp = int(np.sum(predicted & labels))
    tn = int(np.sum(~predicted & ~labels))
    fp = int(np.sum(predicted & ~labels))
    fn = int(np.sum(~predicted & labels))
    tpr = tp / max(1, tp + fn)
    tnr = tn / max(1, tn + fp)
    precision = tp / max(1, tp + fp)
    recall = tpr
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "threshold": threshold,
        "balanced_accuracy": (tpr + tnr) / 2,
        "accuracy": (tp + tn) / max(1, len(labels)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate pose-score threshold from human labels")
    parser.add_argument("labels", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = list(csv.DictReader(args.labels.open(encoding="utf-8-sig", newline="")))
    usable = [row for row in rows if row.get("human_label") in {"same_pose", "wrong_pose"}]
    scores = np.asarray([float(row["pose_score"]) for row in usable], dtype=np.float64)
    labels = np.asarray([row["human_label"] == "same_pose" for row in usable], dtype=bool)
    if not len(usable) or len(set(labels.tolist())) < 2:
        raise ValueError("Need at least one same_pose and one wrong_pose label")

    candidates = np.unique(np.r_[0.0, scores, 1.0])
    report_rows = [metrics(scores, labels, float(value)) for value in candidates]
    best = max(report_rows, key=lambda row: (row["balanced_accuracy"], row["f1"]))
    detector_failures = sum(row.get("human_label") == "detector_failure" for row in rows)
    report = {
        "labeled_total": sum(bool(row.get("human_label")) for row in rows),
        "calibration_total": len(usable),
        "same_pose": int(labels.sum()),
        "wrong_pose": int((~labels).sum()),
        "detector_failure": detector_failures,
        "detector_failure_rate": detector_failures / max(1, len(rows)),
        "best": best,
        "all_thresholds": report_rows,
    }
    output = args.output or args.labels.with_name("pose_threshold_calibration.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["best"], ensure_ascii=False, indent=2))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
