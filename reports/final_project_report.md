# PPE Compliance Detector Report

## Abstract

This project builds a practical PPE compliance detector using a public hard-hat detection dataset. The workflow combines object detection and probabilistic reasoning: YOLOv8 identifies hard hats, missing hard hats, and people in the scene, while a Bayesian network converts those detections into a compliance and risk estimate. The result is a pipeline that is not only technically useful but also easy to explain in an operational safety context.

The project is framed as a computer-vision and decision-support task rather than just an object-detection benchmark. That distinction matters because a detector can find objects without necessarily telling you whether the scene is compliant. For example, one worker without head protection is enough to trigger a noncompliance review, even if the rest of the image looks ordinary.

## 1. Introduction

PPE compliance monitoring is a classic applied computer-vision problem. Construction, manufacturing, and industrial sites often need a way to detect whether workers are wearing required safety equipment. The project focuses on hard-hat compliance because it is a clean and well-defined subproblem with public data available.

The system is organized into two layers:

- a perception layer that detects objects in images
- a reasoning layer that converts object counts into safety risk

Example: if an image contains two people and only one hard hat, the image should not be treated as fully compliant. The object detector may find the relevant items, but the compliance logic still needs to interpret the scene.

## 2. Problem Statement

The task is to determine whether a scene satisfies a simple hard-hat PPE policy. The scene can contain multiple people, multiple hard hats, and cases where a person is visible without a hard hat. The output should support both localization and a higher-level compliance judgment.

The project asks three main questions:

- Can a YOLOv8 detector localize hard hats and no-hard-hat instances reliably on a public dataset?
- Can the exported detections be mapped into a transparent compliance decision?
- Does a Bayesian network improve interpretability by turning detection counts into risk probabilities?

Example: a raw detection list might say “1 hard hat, 1 no-hard-hat, 2 people.” The Bayesian layer turns that list into a more actionable statement such as “compliance probability is low and risk probability is high.”

## 3. Dataset

The default dataset is `keremberke/hard-hat-detection`, a public Roboflow-derived dataset that contains hardhat and no-hardhat labels. The repository includes a conversion step that exports a sampled subset into YOLO format so training and evaluation can be repeated locally.

Dataset snapshot:

- source: keremberke/hard-hat-detection
- labels: hardhat and no-hardhat
- splits: train, valid, test
- annotation format: converted to YOLO text files with normalized bounding boxes

Example: an image that contains one worker with a hard hat and another worker without one produces two separate boxes, not a single scene label. This is important because the downstream safety decision depends on the presence or absence of PPE around each person.

## 4. Exploratory Data Analysis

The EDA stage focuses on object detection properties rather than only class counts.

The main checks are:

- how many images exist in each split
- how many boxes each image contains on average
- how large the boxes are relative to the image size
- whether the class distribution is balanced or skewed

A scene-level interpretation is useful here. Example: if most boxes are small, the detector must learn to localize objects that occupy only a limited part of the frame. This is harder than full-image classification because the model must first find the object before it can classify it.

Another example is class skew. If no-hard-hat cases are rare, the model may appear accurate while still missing the most important violations. In that case, recall on the violation class matters more than overall accuracy.

## 5. Methodology

### 5.1 Dataset export to YOLO format

The dataset loader samples the public benchmark and exports images plus annotation text files in the YOLO convention. That means each object is stored as a normalized center coordinate plus width and height.

This step matters because it makes the repository self-contained. Example: if the dataset is re-exported with the same seed, the same sample can be re-used for future experiments without changing the code.

### 5.2 YOLOv8 detector

The detection backbone uses YOLOv8n as the starting point. The model is fine-tuned on the exported PPE data with the standard Ultralytics workflow.

Technical settings:

- backbone: YOLOv8n
- input resolution: 640 x 640
- epochs: typically 1 for a lightweight benchmark, more for a full run
- batch size: configurable
- evaluation: validation metrics from the detector output

Example: if the detector finds a hard hat with high confidence but misses the worker’s bare head, the compliance layer should still mark the frame as potentially risky, because the policy is about the worker’s protection status, not just about whether an object exists somewhere in the image.

### 5.3 Bayesian compliance layer

The Bayesian network adds a reasoning layer on top of detections. The network uses evidence such as whether a person is present, whether a hard hat is detected, and whether a no-hard-hat instance is detected.

This matters because object detection and policy interpretation are different problems. Example: a person can be present without an obvious hard-hat box. A rule-based alarm may work, but a Bayesian network gives a probability that is easier to explain and can be recalibrated later.

The network is intentionally simple:

- person present -> compliance state
- hard hat detected -> compliance state
- no-hard-hat detected -> compliance state
- compliance state -> risk state

### 5.4 Operational interpretation

The final output is a risk estimate rather than a binary alarm alone. That makes the system more useful in practice because it can prioritize human review. Example: a moderate-risk scene may be logged for later inspection, while a high-risk scene can be escalated immediately.

## 6. Technical Parameters

The key implementation choices are:

- dataset source: keremberke/hard-hat-detection
- exported format: YOLO text labels
- detector: YOLOv8n
- image resolution: 640 x 640
- Bayesian evidence variables: person present, hard hat detected, no-hard-hat detected
- class names: hardhat and no-hardhat
- random seed: fixed for reproducibility

Example: increasing the sample size will usually help the detector, but it also makes the export and training steps slower. For a report, the sample size must be stated explicitly so that the results can be interpreted correctly.

## 7. Results and Discussion

The repository stores validation and Bayesian outputs in JSON so the report can include them directly.

A useful way to think about the results is in two layers:

- the detector layer tells you what objects are present
- the Bayesian layer tells you how risky the scene is

Example interpretation: if YOLO predicts one hard hat and one no-hard-hat box, the risk should be higher than if it predicts only hard hats. That is true even if the detector confidence is only moderate, because safety policy is conservative by design.

Another example is the difference between precision and recall. In a PPE setting, missing a no-hard-hat case is often worse than flagging a compliant frame for review. That means recall on the violation class is especially important.

## 8. Limitations

Several limitations should be stated clearly.

- The exported YOLO sample is only as good as the annotation conversion.
- Small image subsets can make the detector look better or worse than it really is.
- The Bayesian network is intentionally simple and should be recalibrated for site-specific policy.
- Scene safety often depends on more than hard hats alone, such as vests, barriers, and camera angle.

Example: a person wearing a hard hat but standing in an unsafe zone might still need review, even though the current dataset only models the hard-hat aspect. That is one reason the Bayesian layer is valuable: it can be extended later with new evidence nodes.

## 9. Conclusion

This project provides a complete PPE compliance pipeline that is easy to explain and extend. It combines public data export, object detection, probabilistic reasoning, and report-ready outputs. That makes it a strong MS-level project because it demonstrates not only model training but also problem framing, data handling, and operational interpretation.

The central takeaway is that PPE compliance is best treated as a decision problem built on top of perception. YOLO identifies what is in the frame, and the Bayesian network decides how that evidence should influence the risk score.
