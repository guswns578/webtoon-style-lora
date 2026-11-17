from __future__ import annotations

import base64
import json
import math
from pathlib import Path
from typing import Any
from urllib import request

import numpy as np


def _api_json(base_url: str, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    req = request.Request(
        base_url.rstrip("/") + endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with request.urlopen(req, timeout=600) as response:
        return json.loads(response.read().decode("utf-8"))


def detect_pose(base_url: str, image: Path) -> dict[str, Any]:
    result = _api_json(
        base_url,
        "/controlnet/detect",
        {
            "controlnet_module": "dw_openpose_full",
            "controlnet_input_images": [base64.b64encode(image.read_bytes()).decode("ascii")],
            "controlnet_processor_res": 768,
            "controlnet_threshold_a": -1,
            "controlnet_threshold_b": -1,
        },
    )
    poses = result.get("poses") or []
    return poses[0] if poses else {"people": [], "canvas_width": 0, "canvas_height": 0}


def _largest_person(pose: dict[str, Any]) -> dict[str, Any] | None:
    people = pose.get("people") or []
    if not people:
        return None

    def area(person: dict[str, Any]) -> float:
        raw = np.asarray(person.get("pose_keypoints_2d", []), dtype=np.float32).reshape(-1, 3)
        visible = raw[raw[:, 2] > 0]
        if len(visible) < 2:
            return 0.0
        span = visible[:, :2].max(axis=0) - visible[:, :2].min(axis=0)
        return float(span[0] * span[1])

    return max(people, key=area)


def pose_source_quality(pose: dict[str, Any]) -> dict[str, Any]:
    person = _largest_person(pose)
    if person is None:
        return {"valid": False, "visible_joints": 0, "people": 0}
    raw = np.asarray(person.get("pose_keypoints_2d", []), dtype=np.float32).reshape(-1, 3)
    visible_joints = int(np.sum(raw[:, 2] > 0))
    return {
        "valid": visible_joints >= 8,
        "visible_joints": visible_joints,
        "people": len(pose.get("people") or []),
    }


def pose_metrics(source_pose: dict[str, Any], candidate_pose: dict[str, Any]) -> dict[str, Any]:
    source_person = _largest_person(source_pose)
    candidate_person = _largest_person(candidate_pose)
    if source_person is None or candidate_person is None:
        return {
            "pose_detected": False,
            "pose_people": len(candidate_pose.get("people") or []),
            "pose_common_joints": 0,
            "pose_coverage": 0.0,
            "pose_aligned_rmse": math.inf,
            "pose_composition_rmse": math.inf,
            "pose_score": 0.0,
        }

    src = np.asarray(source_person["pose_keypoints_2d"], dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(candidate_person["pose_keypoints_2d"], dtype=np.float64).reshape(-1, 3)
    count = min(len(src), len(dst), 18)
    src, dst = src[:count], dst[:count]
    visible_source = src[:, 2] > 0
    common = visible_source & (dst[:, 2] > 0)
    common_count = int(common.sum())
    source_count = max(1, int(visible_source.sum()))
    coverage = common_count / source_count
    if common_count < 4:
        return {
            "pose_detected": True,
            "pose_people": len(candidate_pose.get("people") or []),
            "pose_common_joints": common_count,
            "pose_coverage": coverage,
            "pose_aligned_rmse": math.inf,
            "pose_composition_rmse": math.inf,
            "pose_score": 0.0,
        }

    src_xy = src[common, :2] / np.array(
        [source_pose["canvas_width"], source_pose["canvas_height"]], dtype=np.float64
    )
    dst_xy = dst[common, :2] / np.array(
        [candidate_pose["canvas_width"], candidate_pose["canvas_height"]], dtype=np.float64
    )
    composition_rmse = float(np.sqrt(np.mean(np.sum((src_xy - dst_xy) ** 2, axis=1))))
    src_centered = src_xy - src_xy.mean(axis=0, keepdims=True)
    dst_centered = dst_xy - dst_xy.mean(axis=0, keepdims=True)
    src_norm = src_centered / max(float(np.linalg.norm(src_centered)), 1e-8)
    dst_norm = dst_centered / max(float(np.linalg.norm(dst_centered)), 1e-8)
    # Preserve screen-space orientation. Rotating a candidate into alignment
    # would reward a sideways or upside-down ControlNet result.
    aligned = dst_norm
    aligned_rmse = float(np.sqrt(np.mean(np.sum((src_norm - aligned) ** 2, axis=1))))
    score = float(coverage * math.exp(-5.0 * aligned_rmse) * math.exp(-4.0 * composition_rmse))
    return {
        "pose_detected": True,
        "pose_people": len(candidate_pose.get("people") or []),
        "pose_common_joints": common_count,
        "pose_coverage": coverage,
        "pose_aligned_rmse": aligned_rmse,
        "pose_composition_rmse": composition_rmse,
        "pose_score": score,
    }
