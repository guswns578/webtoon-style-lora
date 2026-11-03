from __future__ import annotations

import json
import struct
from pathlib import Path


MODEL_DIR = Path(r"C:\Dev\StabilityMatrix\Data\Packages\forge-neo\models\Lora")
MODELS = [
    "lineart_locon_v2.safetensors",
    "lineart_locon_v4_tags3.safetensors",
    "style_locon_e3.safetensors",
    "char_locon_e3.safetensors",
    "char_locon_run2_b4_e3.safetensors",
    "char_b_solo_e3.safetensors",
]
OUTPUT_PATH = Path("outputs/crowd_baseline/model_metadata_audit.json")
KEYS = [
    "ss_network_module",
    "ss_network_dim",
    "ss_network_alpha",
    "ss_network_args",
    "ss_learning_rate",
    "ss_unet_lr",
    "ss_text_encoder_lr",
    "ss_num_epochs",
    "ss_max_train_steps",
    "ss_keep_tokens",
    "ss_batch_size_per_device",
    "ss_gradient_accumulation_steps",
    "ss_optimizer",
    "ss_num_train_images",
    "ss_sd_model_name",
    "ss_base_model_version",
    "ss_clip_skip",
    "ss_min_snr_gamma",
    "ss_output_name",
    "ss_training_comment",
]


def metadata(path: Path) -> dict:
    with path.open("rb") as handle:
        header_length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_length))
    return header.get("__metadata__", {})


def maybe_json(value):
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


def main() -> None:
    report = []
    for name in MODELS:
        path = MODEL_DIR / name
        row = {"name": name, "path": str(path), "exists": path.exists()}
        if path.exists():
            raw = metadata(path)
            row["metadata"] = {key: maybe_json(raw[key]) for key in KEYS if key in raw}
            datasets = maybe_json(raw.get("ss_datasets")) if raw.get("ss_datasets") else None
            if isinstance(datasets, list):
                row["dataset_summary"] = []
                for dataset in datasets:
                    summary = {
                        "num_train_images": dataset.get("num_train_images"),
                        "resolution": dataset.get("resolution"),
                        "batch_size_per_device": dataset.get("batch_size_per_device"),
                        "key_tag_counts": {},
                        "subsets": [],
                    }
                    selected_tags = {
                        "2boys",
                        "3boys",
                        "4boys",
                        "5boys",
                        "6+boys",
                        "multiple_boys",
                        "multiple_others",
                        "soldier",
                        "military",
                        "battle",
                        "full_body",
                    }
                    for tag_counts in dataset.get("tag_frequency", {}).values():
                        for tag in selected_tags:
                            if tag in tag_counts:
                                summary["key_tag_counts"][tag] = (
                                    summary["key_tag_counts"].get(tag, 0) + tag_counts[tag]
                                )
                    for subset in dataset.get("subsets", []):
                        image_dir = subset.get("image_dir")
                        tags = dataset.get("tag_frequency", {}).get(image_dir, {})
                        character_triggers = {
                            key: value
                            for key, value in tags.items()
                            if key.endswith("_character") or key == "webtoon_lineart_style"
                        }
                        summary["subsets"].append(
                            {
                                "image_dir": image_dir,
                                "img_count": subset.get("img_count"),
                                "num_repeats": subset.get("num_repeats"),
                                "keep_tokens": subset.get("keep_tokens"),
                                "triggers": character_triggers,
                            }
                        )
                    row["dataset_summary"].append(summary)
        report.append(row)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(text, encoding="utf-8")
    print(text)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()

