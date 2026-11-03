from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


DEFAULT_ROOT = Path("workspace/char_stage")
FEMALE = {"char_g", "char_h"}
MALE = {"char_a", "char_e", "char_b", "char_c", "char_d", "char_f"}
AUDIT_TAGS = [
    "1boy",
    "1girl",
    "male_focus",
    "solo",
    "solo_focus",
    "multiple_boys",
    "multiple_girls",
    "2boys",
    "2girls",
    "facial_hair",
    "beard",
    "mustache",
    "pointy_ears",
    "animal_ears",
]


def character_from_folder(folder: Path) -> str:
    return folder.name.split("_", 1)[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit character-trigger caption contamination")
    parser.add_argument("root", type=Path, nargs="?", default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    output = args.output or args.root / "caption_contamination_audit.csv"
    rows: list[dict] = []
    examples: dict[str, dict[str, list[str]]] = {}
    for folder in sorted(path for path in args.root.iterdir() if path.is_dir()):
        character = character_from_folder(folder)
        files = sorted(folder.glob("*.txt"))
        counts = Counter()
        missing_trigger = 0
        contaminated: list[str] = []
        mixed_people: list[str] = []
        for path in files:
            tags = {tag.strip() for tag in path.read_text(encoding="utf-8-sig").split(",") if tag.strip()}
            counts.update(tags)
            if f"{character}_character" not in tags and not (
                character == "char_b" and "char_b_legacy_character" in tags
            ):
                missing_trigger += 1
            is_mixed = bool(tags & {"2boys", "2girls", "multiple_boys", "multiple_girls", "1boy"}) and bool(
                tags & {"1girl", "2girls", "multiple_girls"}
            )
            if is_mixed or "solo_focus" in tags:
                mixed_people.append(str(path))
            contradiction = False
            if character in FEMALE:
                contradiction = bool(tags & {"1boy", "male_focus", "facial_hair", "beard", "mustache"})
            elif character in MALE:
                contradiction = "1girl" in tags
            if contradiction:
                contaminated.append(str(path))

        total = len(files)
        row = {
            "character": character,
            "captions": total,
            "missing_trigger": missing_trigger,
            "sex_or_beard_contradiction": len(contaminated),
            "contradiction_rate": len(contaminated) / max(1, total),
            "mixed_or_solo_focus": len(set(mixed_people)),
            "mixed_rate": len(set(mixed_people)) / max(1, total),
        }
        for tag in AUDIT_TAGS:
            row[tag] = counts[tag]
            row[f"{tag}_rate"] = counts[tag] / max(1, total)
        rows.append(row)
        examples[character] = {
            "contradiction": contaminated[:20],
            "mixed_or_solo_focus": sorted(set(mixed_people))[:20],
        }

    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output.with_suffix(".json").write_text(
        json.dumps({"rows": rows, "examples": examples}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md = [
        "# Character Caption Contamination Audit",
        "",
        "Contradiction is a screening flag, not automatic deletion. Mixed panels need human review because the trigger may bind to the wrong face.",
        "",
        "| Character | Captions | Contradiction | Rate | Mixed/solo-focus | Rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md.append(
            f"| {row['character']} | {row['captions']} | {row['sex_or_beard_contradiction']} | "
            f"{row['contradiction_rate']:.1%} | {row['mixed_or_solo_focus']} | {row['mixed_rate']:.1%} |"
        )
    output.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    print(output)
    print(output.with_suffix(".md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
