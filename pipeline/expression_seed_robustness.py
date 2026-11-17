from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

DEFAULT_INPUT = Path("workspace/expr_seed/ccip_expression_evaluation.csv")


def samples_for_hit(success_rate: float, target_probability: float) -> int | None:
    if success_rate <= 0:
        return None
    if success_rate >= 1:
        return 1
    return math.ceil(math.log(1 - target_probability) / math.log(1 - success_rate))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize multi-seed expression identity robustness"
    )
    parser.add_argument("input", type=Path, nargs="?", default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = list(csv.DictReader(args.input.open(encoding="utf-8-sig", newline="")))
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["target"], row["expression"])].append(row)

    summary: list[dict] = []
    for (character, expression), items in sorted(groups.items()):
        accepted = sum(row["production_accept"].lower() == "true" for row in items)
        top1 = sum(row["top1_correct"].lower() == "true" for row in items)
        rate = accepted / len(items)
        summary.append(
            {
                "character": character,
                "expression": expression,
                "seeds": len(items),
                "top1_rate": top1 / len(items),
                "production_accept_rate": rate,
                "samples_for_95pct_one_pass": samples_for_hit(rate, 0.95),
                "samples_for_99pct_one_pass": samples_for_hit(rate, 0.99),
            }
        )

    output = args.output or args.input.with_name("expression_seed_robustness.json")
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    md = [
        "# Expression Seed Robustness",
        "",
        "Observed pass rates use target top-1 plus the provisional generated-domain CCIP gate.",
        "The sample-count estimate assumes seeds behave independently and should be treated as an operational heuristic.",
        "",
        "| Character | Expression | Seeds | Top-1 | Accept | N for 95% | N for 99% |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        n95 = row["samples_for_95pct_one_pass"] or "unresolved"
        n99 = row["samples_for_99pct_one_pass"] or "unresolved"
        md.append(
            f"| {row['character']} | {row['expression']} | {row['seeds']} | "
            f"{row['top1_rate']:.0%} | {row['production_accept_rate']:.0%} | {n95} | {n99} |"
        )
    output.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    print(output)
    print(output.with_suffix(".md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
