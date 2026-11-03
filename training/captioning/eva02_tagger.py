# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps
from tqdm import tqdm


DEFAULT_SOURCE = Path(r"data/lineart_clean")
DEFAULT_DESTINATION = Path(r"data/lineart_clean_v2")
DEFAULT_REPORT = Path("reports/eva02_v3_report.csv")

MODEL_ID = "SmilingWolf/wd-eva02-large-tagger-v3"
TRIGGER = "webtoon_lineart_style"

# EVA02 v3 is relatively recall-heavy on drawings. Starting at 0.40 keeps
# useful pose/clothing tags while avoiding much of the low-confidence noise.
GENERAL_THRESHOLD = 0.40
CHARACTER_THRESHOLD = 0.85
MAX_CONTENT_TAGS = 20
DEFAULT_BATCH_SIZE = 4

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

# These describe the dataset globally or are frequent WD false positives.
# Keeping them out lets the trigger learn the common line-art style while the
# remaining tags describe content that should stay prompt-controllable.
DROP_EXACT = {
    "safe",
    "sensitive",
    "questionable",
    "explicit",
    "masterpiece",
    "best_quality",
    "high_quality",
    "great_quality",
    "normal_quality",
    "low_quality",
    "worst_quality",
    "lowres",
    "bad_anatomy",
    "bad_hands",
    "watermark",
    "signature",
    "text",
    "english_text",
    "japanese_text",
    "chinese_text",
    "korean_text",
    "speech_bubble",
    "thought_bubble",
    "letterboxed",
    "pillarboxed",
    "panel_border",
    "border",
    "artist_name",
    "username",
    "copyright_name",
    "meme",
    "ambiguous_gender",
    "androgynous",
    "1other",
    "dual_persona",
    "death",
    "dead",
    "silent_comic",
    "comic",
    "comic_panel",
    "lineart",
    "monochrome",
    "greyscale",
    "grayscale",
    "black_and_white",
    "sketch",
    "manga",
    "manga_panel",
    "color",
    "colorful",
    "full_color",
    "partially_colored",
    "spot_color",
    "limited_palette",
    "watercolor",
    "oil_painting",
    "painting_(medium)",
    "realistic",
    "photorealistic",
    "photo",
    "3d",
    "render",
}

COLOR_WORDS = {
    "black",
    "white",
    "grey",
    "gray",
    "silver",
    "red",
    "blue",
    "green",
    "yellow",
    "purple",
    "pink",
    "orange",
    "brown",
    "blonde",
    "aqua",
    "cyan",
    "teal",
    "magenta",
    "gold",
    "multicolored",
    "two-tone",
    "gradient",
}

COLOR_TARGETS = {
    "hair",
    "eyes",
    "eye",
    "shirt",
    "dress",
    "clothes",
    "clothing",
    "coat",
    "jacket",
    "robe",
    "cape",
    "cloak",
    "pants",
    "skirt",
    "gloves",
    "boots",
    "shoes",
    "armor",
    "weapon",
    "sword",
    "hat",
    "hood",
    "background",
    "skin",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy clean line-art images and caption them with WD EVA02 Large v3."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--threshold", type=float, default=GENERAL_THRESHOLD)
    parser.add_argument("--character-threshold", type=float, default=CHARACTER_THRESHOLD)
    parser.add_argument("--max-tags", type=int, default=MAX_CONTENT_TAGS)
    parser.add_argument(
        "--provider",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="ONNX Runtime provider. auto prefers CUDA and falls back to CPU.",
    )
    parser.add_argument(
        "--overwrite-captions",
        action="store_true",
        help="Replace captions already present in the destination.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N images for a test run. 0 processes all images.",
    )
    parser.add_argument(
        "--clean-existing-only",
        action="store_true",
        help="Apply the current filters to destination captions without running the model.",
    )
    return parser.parse_args()


def normalize_tag(tag: str) -> str:
    return tag.strip().replace(" ", "_")


def should_drop_tag(tag: str) -> bool:
    if not tag or tag in DROP_EXACT:
        return True

    parts = tag.split("_")
    if any(part in COLOR_WORDS for part in parts):
        if len(parts) == 1 or any(target in parts for target in COLOR_TARGETS):
            return True
    if re.search(
        r"(colored|multicolored|two-tone|gradient).*(hair|eyes|skin|clothes|background)",
        tag,
    ):
        return True
    return False


def make_square_white(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGBA")
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    background.alpha_composite(image)
    image = background.convert("RGB")
    side = max(image.size)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    return canvas


def preprocess(path: Path, image_size: int) -> np.ndarray:
    with Image.open(path) as image:
        image = make_square_white(image)
        image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
        array = np.asarray(image, dtype=np.float32)
    return array[:, :, ::-1]  # RGB to BGR, as expected by WD ONNX models.


def choose_providers(requested: str) -> list[str]:
    available = ort.get_available_providers()
    if requested == "cpu":
        return ["CPUExecutionProvider"]
    if requested == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError("CUDAExecutionProvider is not available.")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def register_cuda_dll_directory() -> None:
    """Expose PyTorch's bundled CUDA/cuDNN DLLs to ONNX Runtime on Windows."""
    if os.name != "nt":
        return
    try:
        import torch

        torch_lib = Path(torch.__file__).resolve().parent / "lib"
        if torch_lib.is_dir():
            os.add_dll_directory(str(torch_lib))
            os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        # ONNX Runtime can still use CPU if CUDA dependencies are unavailable.
        pass


def build_caption(
    probabilities: np.ndarray,
    tag_names: list[str],
    categories: list[int],
    general_threshold: float,
    character_threshold: float,
    max_tags: int,
) -> str:
    scored_tags: list[tuple[str, float]] = []
    for name, category, probability in zip(tag_names, categories, probabilities):
        probability = float(probability)

        # WD categories: 9=rating, 4=character, 0=general.
        if category == 9:
            continue
        if category == 4:
            # Named characters/copyright identities are not useful for this
            # single-artist style LoRA, even at high confidence.
            if probability >= character_threshold:
                continue
            continue
        if probability < general_threshold:
            continue

        tag = normalize_tag(name)
        if should_drop_tag(tag):
            continue
        scored_tags.append((tag, probability))

    scored_tags.sort(key=lambda item: item[1], reverse=True)

    # Prefer explicit person-count tags over vague/overlapping alternatives.
    detected = {tag for tag, _score in scored_tags}
    if "6+boys" in detected:
        scored_tags = [
            item for item in scored_tags if item[0] not in {"1boy", "2boys", "3boys", "4boys", "5boys"}
        ]
    elif "5boys" in detected:
        scored_tags = [
            item for item in scored_tags if item[0] not in {"1boy", "2boys", "3boys", "4boys"}
        ]
    elif "4boys" in detected:
        scored_tags = [
            item for item in scored_tags if item[0] not in {"1boy", "2boys", "3boys"}
        ]
    elif "3boys" in detected:
        scored_tags = [item for item in scored_tags if item[0] not in {"1boy", "2boys"}]
    elif "2boys" in detected:
        scored_tags = [item for item in scored_tags if item[0] != "1boy"]

    caption = [TRIGGER]
    seen = {TRIGGER}
    for tag, _score in scored_tags:
        if tag in seen:
            continue
        caption.append(tag)
        seen.add(tag)
        if len(caption) - 1 >= max_tags:
            break
    return ", ".join(caption)


def chunks(items: list[Path], size: int):
    for index in range(0, len(items), size):
        yield items[index : index + size]


def write_caption_atomic(path: Path, caption: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(caption + "\n", encoding="utf-8")
    temporary.replace(path)


def clean_existing_caption(caption: str, max_tags: int) -> str:
    cleaned = [TRIGGER]
    seen = {TRIGGER}
    for raw_tag in caption.split(","):
        tag = normalize_tag(raw_tag)
        if tag in seen or should_drop_tag(tag):
            continue
        cleaned.append(tag)
        seen.add(tag)
        if len(cleaned) - 1 >= max_tags:
            break
    return ", ".join(cleaned)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if not args.source.is_dir():
        raise FileNotFoundError(f"Source directory not found: {args.source}")

    source_images = sorted(
        path
        for path in args.source.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if args.limit > 0:
        source_images = source_images[: args.limit]
    if not source_images:
        raise FileNotFoundError(f"No images found in {args.source}")

    args.destination.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    if args.clean_existing_only:
        captions = sorted(args.destination.glob("*.txt"))
        changed = 0
        for caption_path in tqdm(captions, desc="Cleaning existing captions"):
            original = caption_path.read_text(encoding="utf-8", errors="replace").strip()
            cleaned = clean_existing_caption(original, args.max_tags)
            if cleaned != original:
                write_caption_atomic(caption_path, cleaned)
                changed += 1
        print(f"CAPTIONS={len(captions)}")
        print(f"CHANGED={changed}")
        print(f"DESTINATION={args.destination}")
        return

    print(f"Downloading/loading {MODEL_ID}...")
    model_path = hf_hub_download(MODEL_ID, filename="model.onnx")
    tags_path = hf_hub_download(MODEL_ID, filename="selected_tags.csv")

    tag_frame = pd.read_csv(tags_path)
    tag_names = tag_frame["name"].tolist()
    categories = (
        tag_frame["category"].astype(int).tolist()
        if "category" in tag_frame.columns
        else [0] * len(tag_names)
    )

    register_cuda_dll_directory()
    providers = choose_providers(args.provider)
    print(f"Requested providers: {providers}")
    session = ort.InferenceSession(model_path, providers=providers)
    print(f"Active providers: {session.get_providers()}")

    input_info = session.get_inputs()[0]
    output_name = session.get_outputs()[0].name
    input_name = input_info.name
    shape = input_info.shape
    image_size = int(shape[1] if isinstance(shape[1], int) else 448)

    copied = 0
    written = 0
    skipped = 0
    failed = 0

    report_exists = args.report.exists()
    with args.report.open("a", newline="", encoding="utf-8-sig") as report_file:
        writer = csv.writer(report_file)
        if not report_exists or args.report.stat().st_size == 0:
            writer.writerow(["source", "destination", "caption", "status", "error"])

        for batch_paths in tqdm(
            list(chunks(source_images, args.batch_size)),
            desc="WD EVA02 v3 captioning",
        ):
            todo: list[tuple[Path, Path, Path]] = []
            for source in batch_paths:
                destination_image = args.destination / source.name
                destination_caption = destination_image.with_suffix(".txt")

                if not destination_image.exists():
                    shutil.copy2(source, destination_image)
                    copied += 1

                if destination_caption.exists() and not args.overwrite_captions:
                    skipped += 1
                    continue
                todo.append((source, destination_image, destination_caption))

            if not todo:
                continue

            try:
                batch = np.stack(
                    [preprocess(destination_image, image_size) for _, destination_image, _ in todo],
                    axis=0,
                )
                probability_batch = session.run([output_name], {input_name: batch})[0]
            except Exception as exc:
                # A bad image should not discard progress already completed.
                for source, destination_image, _ in todo:
                    writer.writerow([source, destination_image, "", "failed_batch", repr(exc)])
                    failed += 1
                continue

            for (source, destination_image, destination_caption), probabilities in zip(
                todo, probability_batch
            ):
                try:
                    caption = build_caption(
                        probabilities,
                        tag_names,
                        categories,
                        args.threshold,
                        args.character_threshold,
                        args.max_tags,
                    )
                    write_caption_atomic(destination_caption, caption)
                    writer.writerow([source, destination_image, caption, "written", ""])
                    written += 1
                except Exception as exc:
                    writer.writerow([source, destination_image, "", "failed_write", repr(exc)])
                    failed += 1
            report_file.flush()

    destination_images = [
        path
        for path in args.destination.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    destination_captions = list(args.destination.glob("*.txt"))
    missing_captions = sum(
        not (args.destination / image.name).with_suffix(".txt").exists()
        for image in source_images
    )

    print(f"SOURCE_IMAGES={len(source_images)}")
    print(f"COPIED_IMAGES={copied}")
    print(f"WRITTEN_CAPTIONS={written}")
    print(f"SKIPPED_EXISTING={skipped}")
    print(f"FAILED={failed}")
    print(f"DESTINATION_IMAGES={len(destination_images)}")
    print(f"DESTINATION_CAPTIONS={len(destination_captions)}")
    print(f"MISSING_CAPTIONS_FOR_THIS_RUN={missing_captions}")
    print(f"DESTINATION={args.destination}")
    print(f"REPORT={args.report}")

    if failed or missing_captions:
        sys.exit(1)


if __name__ == "__main__":
    main()
