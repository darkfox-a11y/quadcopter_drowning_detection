# DJI003 Priority Drowning Dataset

This directory documents the DJI003-derived pseudo-labeled dataset variant used for fine-tuning the drowning detector.

## Purpose

The dataset was mined from DJI aerial videos and designed to support three detection classes:

- `swimming`
- `tread water`
- `drowning`

It specifically preserves ambiguous drowning cases where a frame may contain both `drowning` and `tread water`, enabling safety-oriented analysis.

## Included in GitHub

Only lightweight metadata is tracked here:

- `data.yaml`
- `summary.csv`
- this `README.md`

The full image, label, and preview directories are intentionally excluded from GitHub because of repository and LFS storage limits.

## Local Full Dataset

The complete local dataset exists at:

- `/Users/neerad/drowning-detection/datasets/dji003_priority_drowning_v2`

Local contents include:

- `images/train`
- `images/val`
- `labels/train`
- `labels/val`
- `previews/train`
- `previews/val`

## Notes

- The dataset was generated using `manual_tests/mine_dji_videos.py`
- Frames were sampled at approximately `1 second` intervals
- Empty detections were removed
- Strict `swimming` and `tread water` video labels were preserved
- Drowning videos were mined with drowning-priority logic for mixed cases
