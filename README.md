# Quadcopter Drowning Detection

An applied computer vision project for detecting swimmers, tread-water behavior, and potential drowning cases from aerial imagery and video. The repository combines dataset mining, YOLO-based training, Raspberry Pi AI Camera deployment utilities, and compact evaluation artifacts.

## Overview

This project focuses on three water-activity classes:

- `swimming`
- `tread water`
- `drowning`

The main goal is to support a safety-oriented detection pipeline that can:

- train on custom aquatic data
- mine additional labeled frames from DJI videos
- test edge cases where `tread water` and `drowning` look visually similar
- support Raspberry Pi AI Camera deployment using IMX500-compatible firmware

## Project Scope

The repository is intentionally lightweight. Large raw assets are kept out of GitHub so the project remains inspectable without exhausting storage or Git LFS.

Included here:

- sample DJI003 dataset assets for inspection
- dataset metadata and training configuration
- fine-tuning results and validation plots
- Raspberry Pi / IMX500 test scripts
- utilities used for mining, inspection, and image-level testing

Not included here:

- full raw DJI videos
- full mined frame datasets
- large training checkpoints
- bulky local experimentation outputs
- the complete original self-made dataset payload

## Repository Structure

- `repo_assets/dji003_samples/`
  Small GitHub-friendly sample subset with images, YOLO labels, and preview detections.

- `datasets/dji003_priority_drowning_v2/`
  Metadata for the DJI003-derived fine-tuning dataset, including `data.yaml` and `summary.csv`.

- `reports/dji003_priority_finetune/`
  Final training artifacts used for reporting, including metrics, plots, and confusion matrices.

- `manual_tests/`
  Utility scripts for IMX500 detection, frame extraction, one-off image testing, ONNX inspection, and dataset mining.

- `drowning-detection-dataset/`
  Legacy project material kept locally for reference. The full original dataset itself is not tracked in GitHub.

## Dataset Pipeline

The DJI003 dataset variant in this repo was created from aerial DJI videos and used for fine-tuning experiments on the three-class drowning detector.

Key characteristics:

- frames sampled at approximately `1 second` intervals
- empty detections removed during mining
- strict class preservation for known `swimming` and `tread water` source videos
- drowning-priority mining used for ambiguous cases in drowning-oriented footage

GitHub includes only the lightweight metadata:

- [datasets/dji003_priority_drowning_v2/data.yaml](/Users/neerad/drowning-detection/datasets/dji003_priority_drowning_v2/data.yaml)
- [datasets/dji003_priority_drowning_v2/summary.csv](/Users/neerad/drowning-detection/datasets/dji003_priority_drowning_v2/summary.csv)
- [datasets/dji003_priority_drowning_v2/README.md](/Users/neerad/drowning-detection/datasets/dji003_priority_drowning_v2/README.md)

GitHub sample subset:

- [repo_assets/dji003_samples/images](/Users/neerad/drowning-detection/repo_assets/dji003_samples/images)
- [repo_assets/dji003_samples/labels](/Users/neerad/drowning-detection/repo_assets/dji003_samples/labels)
- [repo_assets/dji003_samples/previews](/Users/neerad/drowning-detection/repo_assets/dji003_samples/previews)

## Training Result

The reported fine-tuning run used the DJI003 priority-drowning dataset variant and YOLOv8n-based training.

Run details:

- model family: `YOLOv8n`
- image size: `640`
- epochs: `20`
- batch size: `8`
- device used for this run: `CPU`

Best validation metrics from the saved run at epoch `19`:

- Precision: `0.82914`
- Recall: `0.90738`
- mAP@50: `0.89577`
- mAP@50-95: `0.74062`

Artifacts:

- [results.csv](/Users/neerad/drowning-detection/reports/dji003_priority_finetune/results.csv)
- [results.png](/Users/neerad/drowning-detection/reports/dji003_priority_finetune/results.png)
- [confusion_matrix.png](/Users/neerad/drowning-detection/reports/dji003_priority_finetune/confusion_matrix.png)
- [confusion_matrix_normalized.png](/Users/neerad/drowning-detection/reports/dji003_priority_finetune/confusion_matrix_normalized.png)
- [BoxPR_curve.png](/Users/neerad/drowning-detection/reports/dji003_priority_finetune/BoxPR_curve.png)
- [BoxF1_curve.png](/Users/neerad/drowning-detection/reports/dji003_priority_finetune/BoxF1_curve.png)

## Raspberry Pi AI Camera Deployment

The repo also contains scripts for running detection with Raspberry Pi AI Camera workflows using IMX500-compatible converted firmware.

Relevant scripts:

- [imx500_frame_detector.py](/Users/neerad/drowning-detection/manual_tests/imx500_frame_detector.py)
  Frame-by-frame IMX500 detection with logging, optional preview, and optional annotated frame saving.

- [imx500_drowning_servo.py](/Users/neerad/drowning-detection/manual_tests/imx500_drowning_servo.py)
  Servo-trigger version that can actuate hardware after repeated drowning detections.

These scripts assume the model has already been converted for the IMX500 pipeline and uploaded to the camera, or that an `.rpk` path is available.

## Useful Scripts

- [mine_dji_videos.py](/Users/neerad/drowning-detection/manual_tests/mine_dji_videos.py)
  Sample DJI footage and generate pseudo-labeled fine-tuning data.

- [extract_video_frames.py](/Users/neerad/drowning-detection/manual_tests/extract_video_frames.py)
  Extract frames from labeled videos for review or dataset expansion.

- [priority_detect_image.py](/Users/neerad/drowning-detection/manual_tests/priority_detect_image.py)
  Run image inference while surfacing `drowning` as a priority class when present.

- [inspect_onnx_model.py](/Users/neerad/drowning-detection/manual_tests/inspect_onnx_model.py)
  Inspect ONNX model structure and metadata during export/conversion checks.

## Reproducing Dataset Mining

Example:

```bash
python3 manual_tests/mine_dji_videos.py \
  --output-dir /Users/neerad/drowning-detection/datasets/dji003_priority_drowning_v2
```

## Reproducing Fine-Tuning

```bash
/Users/neerad/ai_camv8n/.venv/bin/yolo detect train \
  model=/Users/neerad/ai_camv8n/best.pt \
  data=/Users/neerad/drowning-detection/datasets/dji003_priority_drowning_v2/data.yaml \
  imgsz=640 \
  epochs=20 \
  batch=8 \
  device=cpu \
  workers=4 \
  lr0=0.001 \
  lrf=0.01 \
  freeze=10 \
  close_mosaic=0 \
  project=/Users/neerad/drowning-detection/runs/detect \
  name=dji003_priority_finetune \
  exist_ok=True
```

## Notes

- This repo is meant to document the workflow and results cleanly, not to act as a storage bucket for large raw assets.
- The full local datasets remain on disk outside the GitHub-friendly subset included here.
- If you are evaluating deployment behavior, prefer the normal detector for training metrics and use safety-oriented priority logic at inference time when appropriate.
