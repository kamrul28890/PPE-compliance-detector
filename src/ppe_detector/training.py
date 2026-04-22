from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _load_yolo(config_root: Path):
    config_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_root))
    from ultralytics import YOLO

    return YOLO


def train_yolo_detector(data_yaml: str | Path, output_dir: str | Path, epochs: int = 1, batch_size: int = 8, imgsz: int = 640) -> dict[str, Any]:
    output_dir = Path(output_dir)
    YOLO = _load_yolo(output_dir / ".config")
    model = YOLO("yolov8n.pt")
    results = model.train(data=str(data_yaml), epochs=epochs, batch=batch_size, imgsz=imgsz, project=str(output_dir), name="yolov8n_ppe", exist_ok=True)
    return {"results_dir": str(Path(output_dir) / "yolov8n_ppe"), "train_results": str(results)}


def evaluate_yolo_detector(weights_path: str | Path, data_yaml: str | Path) -> dict[str, Any]:
    weights_path = Path(weights_path)
    output_dir = weights_path.parents[2]
    YOLO = _load_yolo(output_dir / ".config")
    model = YOLO(str(weights_path))
    metrics = model.val(data=str(data_yaml), project=str(output_dir), name="yolov8n_ppe_val", exist_ok=True)
    return {
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }
