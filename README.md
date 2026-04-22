# PPE Compliance Detector

This project uses a public hard-hat detection dataset to build a small but complete PPE compliance workflow:

- dataset export from Hugging Face into YOLO format
- YOLOv8 training and validation
- a Bayesian compliance layer that converts detections into risk estimates
- a report generator for the final paper-style write-up

## Main commands

```bash
python -m ppe_detector.cli prepare-data
python -m ppe_detector.cli train-yolo
python -m ppe_detector.cli evaluate-yolo
python -m ppe_detector.cli score-sample
python -m ppe_detector.cli write-report
```

## Dataset

The default dataset source is `keremberke/hard-hat-detection`, a public Roboflow-derived dataset with hardhat and no-hardhat labels.

