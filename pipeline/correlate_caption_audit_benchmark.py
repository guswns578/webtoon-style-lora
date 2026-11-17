from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from scipy.stats import spearmanr

AUDIT = Path("workspace/char_stage/caption_contamination_audit.csv")
BENCHMARK = Path(
    "workspace/conditional_face_refine/benchmark_8char_punch_crossed/benchmark_summary.csv"
)
OUTPUT = Path("workspace/conditional_face_refine/caption_audit_benchmark_correlation.json")


def main() -> int:
    audit = {
        row["character"]: row
        for row in csv.DictReader(AUDIT.open(encoding="utf-8-sig", newline=""))
    }
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in csv.DictReader(BENCHMARK.open(encoding="utf-8-sig", newline="")):
        grouped[row["character"]].append(row)

    rows = []
    for character in sorted(grouped):
        cases = grouped[character]
        top1 = [row["selected_predicted_character"] == character for row in cases]
        production = [
            row["selected_predicted_character"] == character
            and float(row["selected_target_distance"]) <= 0.09
            for row in cases
        ]
        rows.append(
            {
                "character": character,
                "contradiction_rate": float(audit[character]["contradiction_rate"]),
                "mixed_rate": float(audit[character]["mixed_rate"]),
                "top1_rate": sum(top1) / len(top1),
                "production_accept_rate": sum(production) / len(production),
            }
        )

    contradiction = [row["contradiction_rate"] for row in rows]
    mixed = [row["mixed_rate"] for row in rows]
    production = [row["production_accept_rate"] for row in rows]
    report = {
        "warning": "Exploratory only: n=8 characters and two pose cases per character.",
        "rows": rows,
        "spearman_contradiction_vs_accept": {
            "rho": float(spearmanr(contradiction, production).statistic),
            "pvalue": float(spearmanr(contradiction, production).pvalue),
        },
        "spearman_mixed_vs_accept": {
            "rho": float(spearmanr(mixed, production).statistic),
            "pvalue": float(spearmanr(mixed, production).pvalue),
        },
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
