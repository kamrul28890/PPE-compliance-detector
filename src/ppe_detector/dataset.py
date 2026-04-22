from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from datasets import load_dataset
from PIL import Image


IMAGE_KEYS = ("image", "img", "frame", "frames", "picture")
ANNOTATION_KEYS = ("objects", "annotations", "boxes", "bboxes", "labels")
LABEL_KEYS = ("label", "class", "category", "category_id", "name")


def _first_present(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


@dataclass(frozen=True)
class DatasetConfig:
    dataset_name: str = "keremberke/hard-hat-detection"
    dataset_config: str = "full"
    splits: tuple[str, ...] = ("train", "validation", "test")
    sample_size: int = 500
    seed: int = 42


def _pick_key(example: dict[str, Any], candidates: tuple[str, ...]) -> str:
    for key in candidates:
        if key in example:
            return key
    raise KeyError(f"Could not infer one of {candidates} from example keys: {list(example.keys())}")


def _load_split(dataset_name: str, dataset_config: str, split: str, sample_size: int, seed: int):
    dataset = load_dataset(dataset_name, dataset_config, split=split)
    sample_size = min(sample_size, len(dataset))
    indices = np.random.default_rng(seed).choice(len(dataset), size=sample_size, replace=False)
    return dataset.select(sorted(int(index) for index in indices))


def _decode_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path"):
            return Image.open(value["path"]).convert("RGB")
    if isinstance(value, str):
        return Image.open(value).convert("RGB")
    raise TypeError(f"Unsupported image payload type: {type(value)!r}")


def _extract_boxes_and_labels(example: dict[str, Any], annotation_key: str):
    annotation_value = example[annotation_key]
    boxes: list[list[float]] = []
    labels: list[str | int] = []

    if isinstance(annotation_value, dict):
        boxes_source = annotation_value.get("bboxes") or annotation_value.get("boxes") or annotation_value.get("bbox")
        labels_source = _first_present(annotation_value, ("labels", "classes", "category", "category_id"))
        if boxes_source is not None and labels_source is not None:
            for box, label in zip(boxes_source, labels_source):
                boxes.append(list(map(float, box)))
                labels.append(label)
            return boxes, labels

    if isinstance(annotation_value, list):
        for item in annotation_value:
            if isinstance(item, dict):
                box = item.get("bbox") or item.get("box") or item.get("coordinates")
                label = _first_present(item, LABEL_KEYS)
                if box is not None and label is not None:
                    boxes.append(list(map(float, box)))
                    labels.append(label)
        if boxes:
            return boxes, labels

    raise ValueError(f"Could not extract bounding boxes from annotation value: {annotation_value!r}")


def _label_name(features: Any, annotation_key: str, label: str | int) -> str:
    try:
        annotation_feature = features[annotation_key].feature
        category_feature = annotation_feature["category"]
        if hasattr(category_feature, "names") and category_feature.names:
            return str(category_feature.names[int(label)])
    except Exception:
        pass
    return str(label)


def _normalize_box(box: list[float], image_width: int, image_height: int):
    if len(box) != 4:
        raise ValueError(f"Expected 4 box values, got {box}")
    x, y, width, height = box
    if width <= 1.0 and height <= 1.0 and x <= 1.0 and y <= 1.0:
        center_x = x
        center_y = y
        box_width = width
        box_height = height
    else:
        center_x = (x + width / 2.0) / image_width
        center_y = (y + height / 2.0) / image_height
        box_width = width / image_width
        box_height = height / image_height
    return [center_x, center_y, box_width, box_height]


def prepare_yolo_dataset(output_dir: str | Path, config: DatasetConfig | None = None) -> Path:
    config = config or DatasetConfig()
    output_dir = Path(output_dir)
    images_root = output_dir / "images"
    labels_root = output_dir / "labels"
    images_root.mkdir(parents=True, exist_ok=True)
    labels_root.mkdir(parents=True, exist_ok=True)

    class_names: set[str] = set()
    rows: list[dict[str, Any]] = []

    for split in config.splits:
        dataset = _load_split(config.dataset_name, config.dataset_config, split, config.sample_size, config.seed)
        output_split = "valid" if split == "validation" else split
        example = dataset[0]
        image_key = _pick_key(example, IMAGE_KEYS)
        annotation_key = _pick_key(example, ANNOTATION_KEYS)

        split_image_dir = images_root / output_split
        split_label_dir = labels_root / output_split
        split_image_dir.mkdir(parents=True, exist_ok=True)
        split_label_dir.mkdir(parents=True, exist_ok=True)

        for index, row in enumerate(dataset):
            image = _decode_image(row[image_key])
            image_path = split_image_dir / f"{output_split}_{index:05d}.jpg"
            image.save(image_path, quality=95)

            boxes, labels = _extract_boxes_and_labels(row, annotation_key)
            label_path = split_label_dir / f"{output_split}_{index:05d}.txt"
            with label_path.open("w", encoding="utf-8") as handle:
                for box, label in zip(boxes, labels):
                    class_name = _label_name(dataset.features, annotation_key, label)
                    class_names.add(class_name)
                    normalized_box = _normalize_box(box, image.width, image.height)
                    handle.write(f"{class_name} {' '.join(f'{value:.6f}' for value in normalized_box)}\n")

            rows.append(
                {
                    "split": output_split,
                    "image_path": str(image_path),
                    "label_path": str(label_path),
                    "bbox_count": len(boxes),
                }
            )

    mapping = {name: index for index, name in enumerate(sorted(class_names))}
    for row in rows:
        label_path = Path(row["label_path"])
        rewritten_lines = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            class_name, x_center, y_center, width, height = line.split()
            rewritten_lines.append(
                f"{mapping[class_name]} {x_center} {y_center} {width} {height}"
            )
        label_path.write_text("\n".join(rewritten_lines) + ("\n" if rewritten_lines else ""), encoding="utf-8")

    data_yaml = output_dir / "data.yaml"
    yaml_text = "\n".join(
        [
            f"path: {output_dir.as_posix()}",
            "train: images/train",
            "val: images/valid",
            "test: images/test",
            f"names: {json.dumps([name for name, _ in sorted(mapping.items(), key=lambda item: item[1])])}",
        ]
    )
    data_yaml.write_text(yaml_text, encoding="utf-8")

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "image_path", "label_path", "bbox_count"])
        writer.writeheader()
        writer.writerows(rows)

    return data_yaml
