from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    accepted = scores <= threshold
    tp = int(np.sum(accepted & labels))
    fn = int(np.sum(~accepted & labels))
    tn = int(np.sum(~accepted & ~labels))
    fp = int(np.sum(accepted & ~labels))
    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
        "sensitivity": tpr,
        "specificity": tnr,
        "balanced_accuracy": (tpr + tnr) / 2.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate production CCIP threshold from human labels")
    parser.add_argument("labels_csv", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.labels_csv.with_name("production_ccip_calibration.json")
    rows = list(csv.DictReader(args.labels_csv.open(encoding="utf-8-sig", newline="")))
    rows = [row for row in rows if row.get("human_label") in {"same", "different"}]
    if len(rows) < 20:
        raise ValueError("At least 20 human-labeled rows are required for a useful calibration")
    labels = np.asarray([row["human_label"] == "same" for row in rows], dtype=bool)
    scores = np.asarray([float(row["target_distance"]) for row in rows], dtype=float)
    finite = np.isfinite(scores)
    labels, scores = labels[finite], scores[finite]
    thresholds = np.unique(np.concatenate([scores, np.linspace(scores.min(), scores.max(), 2001)]))
    curve = [metrics(labels, scores, float(threshold)) for threshold in thresholds]
    best = max(curve, key=lambda row: (row["balanced_accuracy"], -row["threshold"]))
    report = {
        "labeled_count": int(len(labels)),
        "same_count": int(labels.sum()),
        "different_count": int((~labels).sum()),
        "recommended_threshold": best,
        "threshold_0_0633": metrics(labels, scores, 0.0633),
        "threshold_0_09": metrics(labels, scores, 0.09),
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
