from __future__ import annotations

from pathlib import Path
from typing import Any

from ultralytics import YOLO


def train_yolo_detector(data_yaml: str | Path, output_dir: str | Path, epochs: int = 1, batch_size: int = 8, imgsz: int = 640) -> dict[str, Any]:
    model = YOLO("yolov8n.pt")
    results = model.train(data=str(data_yaml), epochs=epochs, batch=batch_size, imgsz=imgsz, project=str(output_dir), name="yolov8n_ppe", exist_ok=True)
    return {"results_dir": str(Path(output_dir) / "yolov8n_ppe"), "train_results": str(results)}


def evaluate_yolo_detector(weights_path: str | Path, data_yaml: str | Path) -> dict[str, Any]:
    model = YOLO(str(weights_path))
    metrics = model.val(data=str(data_yaml))
    return {
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }
