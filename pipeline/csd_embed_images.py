from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Keep CSD evaluation off the Forge GPU. This must be set before torch import.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import argparse as argparse_module
import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parent
CSD_REPO = ROOT.parent / "third_party" / "CSD"
sys.path.insert(0, str(CSD_REPO))

from CSD.loss_utils import transforms_branch0  # noqa: E402
from CSD.model import CSD_CLIP  # noqa: E402


def load_model() -> CSD_CLIP:
    # The official HF checkpoint contains a few harmless numpy/argparse metadata
    # objects. PyTorch 2.6+ requires these to be explicitly allow-listed.
    from numpy.core.multiarray import scalar

    torch.serialization.add_safe_globals(
        [scalar, np.dtype, np.dtypes.Float64DType, argparse_module.Namespace]
    )
    return CSD_CLIP.from_pretrained("tomg-group-umd/CSD-ViT-L").eval()


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract official CSD ViT-L style descriptors")
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    paths = [Path(item) for item in json.loads(args.input_json.read_text(encoding="utf-8"))]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing CSD inputs: " + ", ".join(missing))

    model = load_model()
    embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(paths), args.batch_size):
            batch_paths = paths[start : start + args.batch_size]
            batch = torch.stack(
                [transforms_branch0(Image.open(path).convert("RGB")) for path in batch_paths]
            )
            _, _, style = model(batch)
            embeddings.extend(style.cpu().numpy().astype(np.float32))
            print(f"CSD {min(start + len(batch_paths), len(paths))}/{len(paths)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        paths=np.array([str(path) for path in paths]),
        embeddings=np.stack(embeddings),
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
