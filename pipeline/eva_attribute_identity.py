from __future__ import annotations

import os

# Keep the large tagger on CPU so it can run beside Forge without consuming VRAM.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import argparse
import json
from pathlib import Path

import numpy as np
from wdtagger import Tagger


DEFAULT_REFERENCES = Path("workspace/face_crops")
MODEL = "SmilingWolf/wd-eva02-large-tagger-v3"
TAGS = [
    "1boy",
    "1girl",
    "male_focus",
    "facial_hair",
    "beard",
    "mustache",
    "stubble",
    "mature_male",
    "pointy_ears",
    "animal_ears",
    "long_hair",
    "short_hair",
    "hair_over_one_eye",
    "asymmetrical_bangs",
    "hair_slicked_back",
    "ponytail",
    "wavy_hair",
    "half_updo",
    "muscular_male",
]


def character_from_reference(path: Path) -> str:
    return path.stem.split("_", 1)[0]


def tag_vectors(tagger: Tagger, paths: list[Path], batch_size: int) -> np.ndarray:
    vectors: list[np.ndarray] = []
    for start in range(0, len(paths), batch_size):
        batch = paths[start : start + batch_size]
        results = tagger.tag(batch, general_threshold=0.0, character_threshold=1.0)
        for result in results:
            vectors.append(np.asarray([result.general_tag_data.get(tag, 0.0) for tag in TAGS]))
        print(f"tagged {min(start + batch_size, len(paths))}/{len(paths)}", flush=True)
    return np.stack(vectors)


def main() -> int:
    parser = argparse.ArgumentParser(description="EVA attribute-prototype adjunct for CCIP")
    parser.add_argument("images", type=Path, nargs="+")
    parser.add_argument("--references", type=Path, default=DEFAULT_REFERENCES)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    references = sorted(args.references.glob("*.png"))
    cache = args.cache or args.references / "eva_attribute_reference_cache.npz"

    tagger = Tagger(model_repo=MODEL)
    if cache.exists():
        data = np.load(cache, allow_pickle=True)
        cached_paths = [str(item) for item in data["paths"]]
        if cached_paths == [str(path) for path in references] and list(data["tags"]) == TAGS:
            reference_vectors = data["vectors"]
        else:
            reference_vectors = tag_vectors(tagger, references, args.batch_size)
    else:
        reference_vectors = tag_vectors(tagger, references, args.batch_size)
    np.savez_compressed(
        cache,
        paths=np.asarray([str(path) for path in references]),
        tags=np.asarray(TAGS),
        vectors=reference_vectors,
    )

    labels = [character_from_reference(path) for path in references]
    characters = sorted(set(labels))
    prototypes = {
        character: reference_vectors[np.asarray(labels) == character].mean(axis=0)
        for character in characters
    }
    prototype_matrix = np.stack([prototypes[character] for character in characters])
    # Emphasize tags that separate reference characters, while limiting unstable amplification.
    scale = np.maximum(prototype_matrix.std(axis=0), 0.08)

    candidate_vectors = tag_vectors(tagger, args.images, args.batch_size)
    rows = []
    for path, vector in zip(args.images, candidate_vectors, strict=True):
        distances = {
            character: float(np.sqrt(np.mean(((vector - prototypes[character]) / scale) ** 2)))
            for character in characters
        }
        ranked = sorted(distances.items(), key=lambda item: item[1])
        rows.append(
            {
                "image": str(path),
                "predicted_character": ranked[0][0],
                "predicted_distance": ranked[0][1],
                "second_character": ranked[1][0],
                "margin": ranked[1][1] - ranked[0][1],
                "distances": distances,
                "attributes": {tag: float(value) for tag, value in zip(TAGS, vector, strict=True)},
            }
        )

    self_test = None
    if args.self_test:
        predictions = []
        for index, (vector, actual) in enumerate(zip(reference_vectors, labels, strict=True)):
            loo_prototypes = {}
            for character in characters:
                indices = [i for i, label in enumerate(labels) if label == character and i != index]
                loo_prototypes[character] = reference_vectors[indices].mean(axis=0)
            distances = {
                character: float(np.sqrt(np.mean(((vector - prototype) / scale) ** 2)))
                for character, prototype in loo_prototypes.items()
            }
            predicted = min(distances, key=distances.get)
            predictions.append({"path": str(references[index]), "actual": actual, "predicted": predicted})
        self_test = {
            "total": len(predictions),
            "correct": sum(row["actual"] == row["predicted"] for row in predictions),
            "accuracy": sum(row["actual"] == row["predicted"] for row in predictions) / len(predictions),
            "rows": predictions,
        }

    report = {
        "warning": "Attribute prototype is an adjunct, not a replacement for CCIP or human review.",
        "model": MODEL,
        "tags": TAGS,
        "reference_self_test": self_test,
        "rows": rows,
    }
    output = args.output or Path("eva_attribute_identity_report.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in rows:
        print(row["image"])
        print(
            f"  {row['predicted_character']} {row['predicted_distance']:.4f} "
            f"margin={row['margin']:.4f}"
        )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
