from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .bayes import score_detection_counts
from .dataset import DatasetConfig, prepare_yolo_dataset
from .reporting import write_report
from .training import evaluate_yolo_detector, train_yolo_detector


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "sample"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"


def command_prepare_data(args: argparse.Namespace) -> None:
    config = DatasetConfig(sample_size=args.sample_size, seed=args.seed)
    data_yaml = prepare_yolo_dataset(DATA_DIR, config)
    print(data_yaml)


def command_train_yolo(args: argparse.Namespace) -> None:
    data_yaml = DATA_DIR / "data.yaml"
    result = train_yolo_detector(data_yaml, ARTIFACTS_DIR / "yolo", epochs=args.epochs, batch_size=args.batch_size, imgsz=args.imgsz)
    (ARTIFACTS_DIR / "yolo_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(ARTIFACTS_DIR / "yolo_result.json")


def command_evaluate_yolo(_: argparse.Namespace) -> None:
    weights = ARTIFACTS_DIR / "yolo" / "yolov8n_ppe" / "weights" / "best.pt"
    metrics = evaluate_yolo_detector(weights, DATA_DIR / "data.yaml")
    (ARTIFACTS_DIR / "yolo_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(ARTIFACTS_DIR / "yolo_metrics.json")


def command_score_sample(_: argparse.Namespace) -> None:
    result = score_detection_counts(hardhat_count=1, no_hardhat_count=1, person_count=2)
    payload = {
        "compliance_probability": result.compliance_probability,
        "risk_probability": result.risk_probability,
    }
    (ARTIFACTS_DIR / "bayes_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(ARTIFACTS_DIR / "bayes_result.json")


def command_write_report(_: argparse.Namespace) -> None:
    manifest_path = DATA_DIR / "manifest.csv"
    class_rows = []
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        class_rows = [
            {"split": split, "images": int((manifest["split"] == split).sum()), "avg_boxes": float(manifest.loc[manifest["split"] == split, "bbox_count"].mean())}
            for split in sorted(manifest["split"].unique())
        ]
    context = {
        "dataset_summary": {
            "source": "keremberke/hard-hat-detection",
            "train_images": int((pd.read_csv(manifest_path)["split"] == "train").sum()) if manifest_path.exists() else "n/a",
            "valid_images": int((pd.read_csv(manifest_path)["split"] == "valid").sum()) if manifest_path.exists() else "n/a",
            "test_images": int((pd.read_csv(manifest_path)["split"] == "test").sum()) if manifest_path.exists() else "n/a",
            "class_names": ["hardhat", "no-hardhat"],
        },
        "yolo_metrics": json.loads((ARTIFACTS_DIR / "yolo_metrics.json").read_text(encoding="utf-8")) if (ARTIFACTS_DIR / "yolo_metrics.json").exists() else {},
        "bayes_result": json.loads((ARTIFACTS_DIR / "bayes_result.json").read_text(encoding="utf-8")) if (ARTIFACTS_DIR / "bayes_result.json").exists() else {},
        "class_rows": class_rows,
    }
    report_path = write_report(REPORTS_DIR / "final_project_report.md", context)
    print(report_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PPE compliance detector CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-data", help="Export a sampled dataset into YOLO format")
    prepare.add_argument("--sample-size", type=int, default=300)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.set_defaults(func=command_prepare_data)

    train = subparsers.add_parser("train-yolo", help="Fine-tune YOLOv8n on the exported dataset")
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--imgsz", type=int, default=640)
    train.set_defaults(func=command_train_yolo)

    evaluate = subparsers.add_parser("evaluate-yolo", help="Validate the YOLO detector")
    evaluate.set_defaults(func=command_evaluate_yolo)

    score = subparsers.add_parser("score-sample", help="Run the Bayesian risk scorer on a sample detection summary")
    score.set_defaults(func=command_score_sample)

    report = subparsers.add_parser("write-report", help="Generate the markdown report")
    report.set_defaults(func=command_write_report)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
