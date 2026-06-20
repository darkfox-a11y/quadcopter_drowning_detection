#!/usr/bin/env python3
"""
Run frame-by-frame detection on a Raspberry Pi AI Camera using IMX500 firmware.

This script is meant for the Raspberry Pi AI Camera workflow where the network
has already been converted for the IMX500 and uploaded as camera firmware.
It captures frames, reads IMX500 inference metadata, prints detections, and can
optionally save annotated frames when detections occur.

Example:
    python3 manual_tests/imx500_frame_detector.py \
        --labels /home/pi/ai_camv8n/labels.txt \
        --print-all \
        --save-dir /home/pi/drowning_hits
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from picamera2 import Picamera2
from picamera2.devices import IMX500

try:
    import cv2
except ImportError:  # pragma: no cover - depends on Pi environment
    cv2 = None


DEFAULT_LABELS = ["swimming", "tread water", "drowning"]


@dataclass
class Detection:
    box: np.ndarray
    score: float
    class_id: int
    label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run frame detection on a Raspberry Pi AI Camera with IMX500."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Optional .rpk path. Leave empty if firmware is already loaded on the camera.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Optional labels file, one class name per line.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.25,
        help="Minimum score for normal detections.",
    )
    parser.add_argument(
        "--priority-label",
        default="drowning",
        help="Label to highlight in logs.",
    )
    parser.add_argument(
        "--priority-min-conf",
        type=float,
        default=0.20,
        help="Minimum score for the priority label to count as a priority hit.",
    )
    parser.add_argument(
        "--frame-rate",
        type=int,
        default=30,
        help="Requested camera frame rate.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Preview width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Preview height.",
    )
    parser.add_argument(
        "--print-all",
        action="store_true",
        help="Print all detections for every frame that has at least one hit.",
    )
    parser.add_argument(
        "--debug-metadata",
        action="store_true",
        help="Print metadata keys once to confirm inference output exists.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Optional directory to save annotated frames with detections.",
    )
    parser.add_argument(
        "--save-priority-only",
        action="store_true",
        help="Only save frames when the priority label is present.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Save every Nth matching frame.",
    )
    parser.add_argument(
        "--show-preview",
        action="store_true",
        help="Show an OpenCV preview window if cv2 is installed.",
    )
    return parser.parse_args()


def load_labels(labels_path: Path | None) -> list[str]:
    if labels_path is None:
        return DEFAULT_LABELS.copy()

    labels = [
        line.strip()
        for line in labels_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not labels:
        raise ValueError(f"No labels found in {labels_path}")
    return labels


def make_imx500(model_path: Path | None) -> IMX500:
    if model_path is None:
        return IMX500()
    return IMX500(str(model_path))


def extract_detections(
    metadata: dict,
    imx500: IMX500,
    labels: list[str],
    threshold: float,
) -> list[Detection]:
    outputs = imx500.get_outputs(metadata)
    if not outputs:
        return []

    detections: list[Detection] = []

    if len(outputs) >= 4:
        boxes = outputs[0][0]
        scores = outputs[1][0]
        classes = outputs[2][0]
        valid = outputs[3][0]
        try:
            valid_count = int(valid[0] if np.ndim(valid) else valid)
        except (TypeError, ValueError, IndexError):
            valid_count = len(scores)

        for box, score, class_id in zip(boxes[:valid_count], scores[:valid_count], classes[:valid_count]):
            score = float(score)
            class_id = int(class_id)
            if score < threshold:
                continue
            label = labels[class_id] if 0 <= class_id < len(labels) else f"class_{class_id}"
            detections.append(Detection(box=np.asarray(box), score=score, class_id=class_id, label=label))
        return detections

    if len(outputs) == 3:
        boxes = outputs[0][0]
        scores = outputs[1][0]
        classes = outputs[2][0]

        for box, score, class_id in zip(boxes, scores, classes):
            score = float(score)
            class_id = int(class_id)
            if score < threshold:
                continue
            label = labels[class_id] if 0 <= class_id < len(labels) else f"class_{class_id}"
            detections.append(Detection(box=np.asarray(box), score=score, class_id=class_id, label=label))
        return detections

    if len(outputs) == 1:
        raw = np.asarray(outputs[0])
        if raw.ndim == 3 and raw.shape[0] == 1:
            raw = raw[0]
        if raw.ndim != 2:
            return []
        if raw.shape[0] < raw.shape[1]:
            raw = raw.T
        if raw.shape[1] < 5:
            return []

        for pred in raw:
            class_scores = pred[4:]
            class_id = int(np.argmax(class_scores))
            score = float(class_scores[class_id])
            if score < threshold:
                continue
            label = labels[class_id] if 0 <= class_id < len(labels) else f"class_{class_id}"
            detections.append(Detection(box=np.asarray(pred[:4]), score=score, class_id=class_id, label=label))
        return detections

    return []


def is_priority_hit(detection: Detection, label: str, min_conf: float) -> bool:
    return detection.label.strip().lower() == label.strip().lower() and detection.score >= min_conf


def describe_detections(detections: list[Detection]) -> str:
    return ", ".join(
        f"{det.label}(id={det.class_id}, score={det.score:.2f})" for det in detections
    )


def normalize_box(box: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    values = np.asarray(box, dtype=float).flatten()
    if values.size < 4:
        return 0, 0, 0, 0

    x1, y1, x2, y2 = values[:4]

    # Some pipelines emit normalized coordinates.
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
        x1 *= width
        x2 *= width
        y1 *= height
        y2 *= height

    return (
        max(0, min(width - 1, int(round(x1)))),
        max(0, min(height - 1, int(round(y1)))),
        max(0, min(width - 1, int(round(x2)))),
        max(0, min(height - 1, int(round(y2)))),
    )


def annotate_frame(frame: np.ndarray, detections: list[Detection], priority_label: str) -> np.ndarray:
    if cv2 is None:
        return frame

    annotated = frame.copy()
    height, width = annotated.shape[:2]

    for det in detections:
        x1, y1, x2, y2 = normalize_box(det.box, width, height)
        is_priority = det.label.strip().lower() == priority_label.strip().lower()
        color = (0, 0, 255) if is_priority else (0, 255, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        text = f"{det.label} {det.score:.2f}"
        cv2.putText(
            annotated,
            text,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def main() -> int:
    args = parse_args()
    labels = load_labels(args.labels)
    imx500 = make_imx500(args.model)

    picam2 = Picamera2(camera_num=imx500.camera_num)
    config = picam2.create_preview_configuration(
        main={"size": (args.width, args.height)},
        controls={"FrameRate": args.frame_rate},
    )
    picam2.configure(config)

    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    stop_requested = False

    def handle_signal(signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(f"\nStopping on signal {signum}...")

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print("Starting IMX500 frame detector")
    print(f"Model: {args.model if args.model else 'camera/default configuration'}")
    print(f"Labels: {labels}")
    print(
        f"Threshold={args.threshold}, priority_label='{args.priority_label}', "
        f"priority_min_conf={args.priority_min_conf}"
    )

    frame_index = 0
    saved_count = 0
    start_time = time.monotonic()

    try:
        picam2.start()

        while not stop_requested:
            frame = picam2.capture_array()
            metadata = picam2.capture_metadata()

            if args.debug_metadata:
                print(f"Metadata keys: {sorted(metadata.keys())}")
                args.debug_metadata = False

            detections = extract_detections(metadata, imx500, labels, args.threshold)
            priority_hits = [
                det for det in detections if is_priority_hit(det, args.priority_label, args.priority_min_conf)
            ]

            if detections:
                prefix = "PRIORITY" if priority_hits else "DETECTIONS"
                print(f"[frame {frame_index}] {prefix}: {describe_detections(detections)}")
            elif args.print_all:
                print(f"[frame {frame_index}] no detections")

            should_save = bool(detections)
            if args.save_priority_only:
                should_save = bool(priority_hits)
            if should_save and args.save_every > 1:
                should_save = (saved_count % args.save_every) == 0

            annotated = frame
            if (args.show_preview or should_save) and cv2 is not None:
                annotated = annotate_frame(frame, detections, args.priority_label)

            if should_save and args.save_dir is not None:
                if cv2 is None:
                    print("OpenCV is not installed; cannot save annotated frames.")
                else:
                    output_path = args.save_dir / f"frame_{frame_index:06d}.jpg"
                    cv2.imwrite(str(output_path), annotated)
                    saved_count += 1

            if args.show_preview and cv2 is not None:
                cv2.imshow("IMX500 Frame Detector", annotated)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    stop_requested = True

            frame_index += 1

    finally:
        elapsed = max(0.001, time.monotonic() - start_time)
        fps = frame_index / elapsed
        print(f"Processed {frame_index} frames in {elapsed:.1f}s ({fps:.2f} FPS)")
        if args.save_dir is not None:
            print(f"Saved {saved_count} frames to {args.save_dir}")
        picam2.stop()
        if args.show_preview and cv2 is not None:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
