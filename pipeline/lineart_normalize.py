from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_unicode(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Cannot decode image: {path}")
    return image


def write_unicode(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise RuntimeError(f"Cannot encode image: {path}")
    encoded.tofile(str(path))


def normalize_uniform_lineart(
    gray: np.ndarray,
    background_percentile: float = 90.0,
    gamma: float = 0.40,
    white_snap: int = 238,
) -> np.ndarray:
    """Remove Forge's near-uniform gray/sepia veil while preserving black ink.

    Unlike hard connected-component removal, this levels transform does not erase
    large connected silhouettes such as cloaks or speed-line clusters.
    """
    source = gray.astype(np.float32)
    background = max(float(np.percentile(source, background_percentile)), 1.0)
    normalized = np.power(np.clip(source / background, 0.0, 1.0), gamma) * 255.0
    normalized[normalized > white_snap] = 255.0
    return normalized.astype(np.uint8)


def normalize_file(source: Path, destination: Path, gamma: float = 0.40) -> Path:
    output = normalize_uniform_lineart(read_unicode(source), gamma=gamma)
    write_unicode(destination, output)
    return destination
