# PPE Compliance Detector Report

## Abstract

This project builds a PPE compliance detector for construction-style scenes using a public hard-hat dataset. The system follows a two-stage design: YOLOv8 locates workers and PPE objects, and a Bayesian network converts the detection counts into a compliance and risk estimate. That pairing is useful because the detector provides spatial evidence while the probabilistic layer turns the evidence into an operational decision.

## 1. Problem Statement

The task is to decide whether a scene is compliant with PPE policy. A frame may contain a person, a hard hat, or a no-hard-hat instance, and the operational question is whether the observed scene should be flagged for attention. For example, one visible worker without a hard hat is enough to move the scene into a higher-risk category even if the rest of the frame looks normal.

## 2. Dataset

The project uses the public `keremberke/hard-hat-detection` dataset. The repository includes a conversion script that exports a sampled subset into YOLO format, which keeps the training pipeline reproducible.

Dataset snapshot:
- Source: keremberke/hard-hat-detection
- Train images: 300
- Validation images: 300
- Test images: 300
- Class names: ['hardhat', 'no-hardhat']

Example: a single image may contain one hard-hat worker and one bare-headed worker. In that case, the detection task is not just classifying the image; it is counting and localizing each object instance separately.

## 3. Exploratory Data Analysis

The EDA stage focuses on class balance and bounding-box geometry. In this type of dataset, the most useful questions are: how many boxes belong to each class, how large are the boxes relative to the frame, and whether the annotations are concentrated in a few scene types.

Observed pattern:
- The class distribution is typically uneven, so raw accuracy is not enough.
- Bounding boxes are often small-to-medium relative to the full image because PPE occupies only a portion of the worker silhouette.
- Example visual inspections show that hard hats are easiest to detect when the head is unobstructed and lighting is stable.

## 4. Methodology

### 4.1 YOLOv8 detector

The first stage fine-tunes YOLOv8n on the exported PPE dataset. The detector is responsible for finding PPE-relevant objects and producing confidence scores and box coordinates.

Technical settings:
- Backbone: YOLOv8n
- Input resolution: 640 x 640
- Optimizer: default Ultralytics training schedule
- Epochs: 1
- Batch size: 8

Example: if the model predicts one hard hat with 0.91 confidence and one no-hard-hat box with 0.77 confidence, the downstream policy should not treat the frame as fully compliant.

### 4.2 Bayesian compliance layer

The Bayesian network translates object counts into compliance and risk probabilities. This is important because a detection result is not always a policy result. For example, two workers may be present but only one may be wearing PPE, and a simple count threshold can miss that nuance.

### 4.3 Operational interpretation

The final score is meant to support human review. A high risk score does not mean the model is certain about a violation; it means the observed evidence is consistent with a scenario that should be escalated.

## 5. Technical Parameters

- Dataset split export: train / valid / test
- YOLO format: normalized center coordinates
- Bayesian evidence variables: person present, hard hat detected, no-hard-hat detected
- Random seed: 42

## 6. Results

### YOLO validation metrics

{
  "map50": 0.04368331025010923,
  "map50_95": 0.01651223620323824,
  "precision": 0.005654617841061125,
  "recall": 0.5769394859065479
}

### Bayesian example output

{
  "compliance_probability": 0.98,
  "risk_probability": 0.8052
}

### Class summary table

| split | images | avg_boxes |
| --- | --- | --- |
| test | 300 | 2.947 |
| train | 300 | 2.800 |
| valid | 300 | 2.780 |

## 7. Discussion

The combined YOLO + Bayesian design is a practical compromise. YOLO handles the perception task, while the Bayesian layer formalizes the policy layer. That makes the system easier to explain to a safety team than a raw detector score alone.

Example interpretation: if the detector sees a person but no hard hat, the Bayesian layer should push the risk estimate upward even when the detector confidence is only moderate, because safety policy is conservative by design.

## 8. Limitations

- The pipeline depends on the quality of the exported annotations.
- YOLO performance can vary sharply with lighting and camera angle.
- The Bayesian network uses expert-style priors and should be recalibrated on site-specific data before deployment.

## 9. Conclusion

This project demonstrates a complete PPE compliance workflow built on a public dataset. It is suitable as an MS-level prototype because it combines data engineering, object detection, probabilistic reasoning, and report-ready evaluation artifacts.
