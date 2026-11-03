from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def finite(value: str) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize webtoon character-pose benchmark")
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("--pose-threshold", type=float, default=0.45)
    parser.add_argument("--reference-threshold", type=float, default=0.0633)
    parser.add_argument("--production-threshold", type=float, default=0.09)
    args = parser.parse_args()
    csv_path = args.benchmark / "benchmark_summary.csv"
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8-sig", newline="")))
    rescore_path = args.benchmark / "pose_rescore.csv"
    if rescore_path.exists():
        rescored = {
            (row["character"], row["pose"]): row
            for row in csv.DictReader(rescore_path.open(encoding="utf-8-sig", newline=""))
        }
        for row in rows:
            updated = rescored.get((row["character"], row["pose"]))
            if updated:
                row["original_pose_score"] = row["selected_pose_score"]
                row["selected_pose_score"] = updated["pose_score"]

    categorized: list[dict] = []
    by_pose: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        distance = finite(row["selected_target_distance"])
        pose_score = float(row["selected_pose_score"])
        identity_correct = row["character"] == row["selected_predicted_character"]
        issues: list[str] = []
        identity_band = "unscored"
        if distance is None:
            issues.append("face_not_detected")
        elif not identity_correct:
            issues.append("wrong_character")
        elif distance <= args.reference_threshold:
            identity_band = "reference_strict"
        elif distance <= args.production_threshold:
            identity_band = "production_accept"
        else:
            identity_band = "weak"
            issues.append("weak_identity")
        if pose_score == 0:
            issues.append("pose_not_detected")
        elif pose_score < args.pose_threshold:
            issues.append("weak_pose")
        item = {
            "character": row["character"],
            "pose": row["pose"],
            "distance": distance,
            "predicted_character": row["selected_predicted_character"],
            "pose_score": pose_score,
            "identity_pose_score": float(row["selected_identity_pose_score"]),
            "identity_band": identity_band,
            "issues": issues,
            "status": "review" if issues else "strong",
            "selected_result": row["selected_result"],
        }
        categorized.append(item)
        by_pose[row["pose"]].append(item)

    pose_summary: list[dict] = []
    for pose, group in sorted(by_pose.items()):
        detected = [item["pose_score"] for item in group if item["pose_score"] > 0]
        distances = [item["distance"] for item in group if item["distance"] is not None]
        pose_summary.append(
            {
                "pose": pose,
                "count": len(group),
                "pose_detection_rate": len(detected) / len(group),
                "pose_score_mean_all": statistics.mean(item["pose_score"] for item in group),
                "pose_score_median_detected": statistics.median(detected) if detected else 0.0,
                "identity_top1_accuracy": sum(
                    item["character"] == item["predicted_character"] for item in group
                )
                / len(group),
                "ccip_mean_finite": statistics.mean(distances) if distances else None,
            }
        )

    report = {
        "thresholds": {
            "reference_domain": args.reference_threshold,
            "production_provisional": args.production_threshold,
            "pose_acceptance": args.pose_threshold,
        },
        "total": len(categorized),
        "strong": sum(item["status"] == "strong" for item in categorized),
        "review": sum(item["status"] == "review" for item in categorized),
        "reference_strict_identity": sum(
            item["identity_band"] == "reference_strict" for item in categorized
        ),
        "production_identity_accept": sum(
            item["identity_band"] in {"reference_strict", "production_accept"}
            for item in categorized
        ),
        "pose_summary": pose_summary,
        "cases": categorized,
    }
    (args.benchmark / "benchmark_analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# 웹툰 캐릭터·포즈 벤치마크 요약",
        "",
        f"- 총 조합: {report['total']}",
        f"- 모든 자동 기준 충족: {report['strong']}",
        f"- 검수 또는 재시도 필요: {report['review']}",
        f"- 원본-domain strict 정체성 통과: {report['reference_strict_identity']}",
        f"- 생성물 운영 정체성 통과: {report['production_identity_accept']}",
        "",
        "## 포즈별 결과",
        "",
        "| 포즈 | 검출률 | 검출된 포즈 중앙값 | 캐릭터 Top-1 | 평균 CCIP |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in pose_summary:
        lines.append(
            f"| {row['pose']} | {row['pose_detection_rate']:.1%} | "
            f"{row['pose_score_median_detected']:.3f} | {row['identity_top1_accuracy']:.1%} | "
            f"{row['ccip_mean_finite']:.4f} |"
        )
    lines.extend(["", "## 재시도·검수 대상", "", "| 캐릭터 | 포즈 | CCIP | 예측 | 포즈 | 사유 |", "|---|---|---:|---|---:|---|"])
    for item in categorized:
        if item["issues"]:
            distance = "inf" if item["distance"] is None else f"{item['distance']:.4f}"
            lines.append(
                f"| {item['character']} | {item['pose']} | {distance} | "
                f"{item['predicted_character']} | {item['pose_score']:.3f} | "
                f"{', '.join(item['issues'])} |"
            )
    (args.benchmark / "benchmark_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"strong": report["strong"], "review": report["review"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
