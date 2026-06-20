## Manual Test Utilities

This folder contains the ad hoc scripts created during dataset validation,
DJI003 pseudo-label mining, ONNX inspection, Raspberry Pi AI Camera checks,
and one-off image testing.

Files:
- `imx500_frame_detector.py`: frame-by-frame IMX500 detection, logging, preview, and optional annotated frame saving
- `mine_dji_videos.py`: sample DJI videos, run detection, and build pseudo-labeled datasets
- `extract_video_frames.py`: extract frames from labeled videos for manual review or fine-tuning
- `priority_detect_image.py`: run one-off image inference with drowning-priority logic
- `inspect_onnx_model.py`: inspect ONNX metadata and outputs
- `imx500_drowning_servo.py`: Raspberry Pi AI Camera + servo trigger script
