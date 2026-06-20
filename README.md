# Quadcopter Drowning Detection

This repository contains the DJI003 drowning-detection dataset workflow, a lightweight sample subset for GitHub, and the final YOLOv8n fine-tuning results used for analysis.

## What Is In This Repo

- `repo_assets/dji003_samples/`
  - Small GitHub-friendly sample subset of images, labels, and boxed previews for the three classes:
    - `swimming`
    - `tread water`
    - `drowning`
- `datasets/dji003_priority_drowning_v2/data.yaml`
  - Dataset configuration used for fine-tuning.
- `datasets/dji003_priority_drowning_v2/summary.csv`
  - Frame-level metadata for the mined DJI003 dataset.
- `reports/dji003_priority_finetune/`
  - Validation plots, confusion matrices, and training CSV for the final fine-tuning run.
- `mine_dji_videos.py`
  - Mines DJI videos into a pseudo-labeled dataset.
- `extract_video_frames.py`
  - Extracts frames from class-organized videos.
- `priority_detect_image.py`
  - Runs single-image inference with drowning-priority reporting.
- `imx500_drowning_servo.py`
  - Raspberry Pi AI Camera servo trigger script.

## What Is Not In This Repo

The following large assets are intentionally kept out of GitHub:

- Full original self-made dataset
- Full DJI003 mined image dataset
- Raw DJI videos
- Model checkpoints and large intermediate artifacts

This keeps the repository clean and avoids GitHub/LFS storage limits.

## DJI003 Dataset Summary

The latest mined DJI003 priority dataset produced:

- Total sampled dataset snapshot: `822` kept frames at one stage of analysis
- Fine-tuning dataset snapshot used for training:
  - `605` train images
  - `138` val images
- Later local dataset snapshot:
  - `823` train images
  - `181` val images

Class distribution in the local priority dataset summary:

- `drowning`: `238`
- `tread water`: `302`
- `swimming`: `282`

The full local dataset remains available outside GitHub at:

- `/Users/neerad/drowning-detection/datasets/dji003_priority_drowning_v2`

## Fine-Tuning Result

Final fine-tuning run:

- Base model: `YOLOv8n`
- Fine-tuned from: local `best.pt`
- Dataset: DJI003 priority-drowning variant
- Epochs: `20`
- Batch size: `8`
- Image size: `640`

Best validation metrics from the final run:

- Precision: `0.82914`
- Recall: `0.90738`
- mAP@50: `0.89577`
- mAP@50-95: `0.74062`

See:

- `reports/dji003_priority_finetune/results.csv`
- `reports/dji003_priority_finetune/results.png`
- `reports/dji003_priority_finetune/confusion_matrix.png`
- `reports/dji003_priority_finetune/confusion_matrix_normalized.png`

## Sample Assets

GitHub includes a small sample subset to make the repository inspectable without pushing the full dataset:

- Sample images: `repo_assets/dji003_samples/images/`
- Sample labels: `repo_assets/dji003_samples/labels/`
- Sample boxed previews: `repo_assets/dji003_samples/previews/`

## Reproducing the Dataset Mining

Example command:

```bash
mct_workspace/venv_mct/bin/python mine_dji_videos.py \
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
