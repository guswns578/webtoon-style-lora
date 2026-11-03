from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from imgutils.metrics import ccip_batch_differences


DEFAULT_CACHE = Path(
    "workspace/conditional_face_refine/_cache/reference_features.npz"
)
DEFAULT_OUTPUT = Path("workspace/conditional_face_refine/ccip_calibration")


def confusion(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = scores <= threshold
    tp = int(np.sum(predicted & labels))
    fn = int(np.sum(~predicted & labels))
    tn = int(np.sum(~predicted & ~labels))
    fp = int(np.sum(predicted & ~labels))
    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * precision * tpr / max(precision + tpr, 1e-12)
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
        "tpr": tpr,
        "tnr": tnr,
        "balanced_accuracy": (tpr + tnr) / 2.0,
        "precision": precision,
        "f1": f1,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate CCIP for webtoon monochrome face crops")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--nearest", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    cached = np.load(args.cache, allow_pickle=True)
    features = np.asarray(cached["features"], dtype=np.float32)
    labels = np.asarray(cached["labels"]).astype(str)
    paths = np.asarray(cached["paths"]).astype(str)
    distances = ccip_batch_differences(list(features))

    trials: list[dict] = []
    binary_labels: list[bool] = []
    binary_scores: list[float] = []
    nearest_correct = 0
    for query_index, (query_label, query_path) in enumerate(zip(labels, paths)):
        character_scores: dict[str, float] = {}
        for character in sorted(set(labels)):
            indices = [
                index
                for index, label in enumerate(labels)
                if label == character and index != query_index
            ]
            if not indices:
                continue
            values = sorted(float(distances[query_index, index]) for index in indices)
            character_scores[character] = float(np.median(values[: min(args.nearest, len(values))]))
        ranking = sorted(character_scores.items(), key=lambda item: item[1])
        predicted, predicted_distance = ranking[0]
        nearest_correct += int(predicted == query_label)
        target_distance = character_scores[query_label]
        second_distance = min(value for key, value in character_scores.items() if key != query_label)
        trials.append(
            {
                "path": query_path,
                "label": query_label,
                "predicted": predicted,
                "target_distance": target_distance,
                "predicted_distance": predicted_distance,
                "nearest_other_distance": second_distance,
                "margin": second_distance - target_distance,
                "correct": predicted == query_label,
            }
        )
        for character, score in character_scores.items():
            binary_labels.append(character == query_label)
            binary_scores.append(score)

    y_true = np.asarray(binary_labels, dtype=bool)
    y_score = np.asarray(binary_scores, dtype=float)
    candidates = np.unique(
        np.concatenate(
            [
                np.linspace(float(y_score.min()), float(y_score.max()), 2001),
                y_score,
            ]
        )
    )
    curves = [confusion(y_true, y_score, float(threshold)) for threshold in candidates]
    best = max(curves, key=lambda row: (row["balanced_accuracy"], row["f1"], -row["threshold"]))

    with (args.output / "leave_one_out_trials.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trials[0]))
        writer.writeheader()
        writer.writerows(trials)
    with (args.output / "threshold_curve.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curves[0]))
        writer.writeheader()
        writer.writerows(curves)

    same_scores = np.asarray([row["target_distance"] for row in trials])
    other_scores = np.asarray([row["nearest_other_distance"] for row in trials])
    report = {
        "protocol": "leave-one-out nearest-k median character prototype",
        "reference_count": int(len(labels)),
        "character_count": int(len(set(labels))),
        "nearest_k": args.nearest,
        "nearest_character_accuracy": nearest_correct / len(trials),
        "same_character_distance": {
            "median": float(np.median(same_scores)),
            "p90": float(np.quantile(same_scores, 0.90)),
            "max": float(np.max(same_scores)),
        },
        "nearest_other_distance": {
            "median": float(np.median(other_scores)),
            "p10": float(np.quantile(other_scores, 0.10)),
            "min": float(np.min(other_scores)),
        },
        "recommended_threshold": best,
        "existing_threshold_0_09": confusion(y_true, y_score, 0.09),
        "note": "CCIP is validated for single-character anime images; face cropping is part of this protocol.",
    }
    (args.output / "calibration_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
