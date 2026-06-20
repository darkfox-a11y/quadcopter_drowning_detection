#!/usr/bin/env python3
"""
Trigger a servo when the Raspberry Pi AI Camera detects a drowning person.

This script is designed for the IMX500-based Raspberry Pi AI Camera and a
model whose firmware has already been prepared for the sensor. It polls the AI
camera metadata, looks for the configured drowning class, and actuates a servo
after the detection is seen consistently for a few frames.

Example:
    python imx500_drowning_servo.py \
        --model /home/pi/ai_camv8n/network.rpk \
        --labels /home/pi/ai_camv8n/labels.txt
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from gpiozero import Servo
from picamera2 import Picamera2
from picamera2.devices import IMX500


DEFAULT_LABELS = ["swimming", "tread water", "drowning"]


@dataclass
class Detection:
    box: object
    score: float
    class_id: int
    label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trigger a servo when the IMX500 model detects drowning."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to the IMX500 network package (.rpk). Leave empty if your Pi camera setup already handles the network.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Optional labels.txt file, one class name per line.",
    )
    parser.add_argument(
        "--drowning-label",
        default="drowning",
        help="Class name that should trigger the servo.",
    )
    parser.add_argument(
        "--drowning-class-id",
        type=int,
        default=None,
        help="Optional class id override. If set, it is checked in addition to the class name.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.55,
        help="Minimum confidence score required for a detection.",
    )
    parser.add_argument(
        "--confirm-frames",
        type=int,
        default=3,
        help="Number of consecutive frames with drowning detection required before triggering.",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=8.0,
        help="Seconds to wait before another trigger is allowed.",
    )
    parser.add_argument(
        "--servo-gpio",
        type=int,
        default=18,
        help="GPIO pin for the servo signal wire.",
    )
    parser.add_argument(
        "--servo-min-pulse-us",
        type=int,
        default=500,
        help="Servo minimum pulse width in microseconds.",
    )
    parser.add_argument(
        "--servo-max-pulse-us",
        type=int,
        default=2500,
        help="Servo maximum pulse width in microseconds.",
    )
    parser.add_argument(
        "--active-position",
        type=float,
        default=1.0,
        help="Servo value when triggered, between -1.0 and 1.0.",
    )
    parser.add_argument(
        "--rest-position",
        type=float,
        default=0.0,
        help="Servo idle value, between -1.0 and 1.0.",
    )
    parser.add_argument(
        "--hold-time",
        type=float,
        default=1.0,
        help="How long to hold the servo in the active position.",
    )
    parser.add_argument(
        "--settle-time",
        type=float,
        default=0.5,
        help="How long to wait after returning the servo to rest.",
    )
    parser.add_argument(
        "--print-all",
        action="store_true",
        help="Print every parsed detection for debugging.",
    )
    parser.add_argument(
        "--debug-metadata",
        action="store_true",
        help="Print the available camera metadata keys once at startup.",
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
            detections.append(Detection(box=box, score=score, class_id=class_id, label=label))
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
            detections.append(Detection(box=box, score=score, class_id=class_id, label=label))
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

        num_classes = max(0, raw.shape[1] - 4)
        if num_classes == 0:
            return []

        for pred in raw:
            class_scores = pred[4:]
            class_id = int(np.argmax(class_scores))
            score = float(class_scores[class_id])
            if score < threshold:
                continue

            label = labels[class_id] if 0 <= class_id < len(labels) else f"class_{class_id}"
            detections.append(Detection(box=pred[:4], score=score, class_id=class_id, label=label))
        return detections

    return []


def is_drowning_detection(
    detection: Detection,
    drowning_label: str,
    drowning_class_id: int | None,
) -> bool:
    label_matches = detection.label.strip().lower() == drowning_label.strip().lower()
    id_matches = drowning_class_id is not None and detection.class_id == drowning_class_id
    return label_matches or id_matches


def clamp_servo_value(value: float) -> float:
    return max(-1.0, min(1.0, value))


def trigger_servo(servo: Servo, active_position: float, rest_position: float, hold_time: float, settle_time: float) -> None:
    servo.value = clamp_servo_value(active_position)
    time.sleep(max(0.0, hold_time))
    servo.value = clamp_servo_value(rest_position)
    time.sleep(max(0.0, settle_time))


def describe_detections(detections: Iterable[Detection]) -> str:
    return ", ".join(
        f"{det.label}(id={det.class_id}, score={det.score:.2f})" for det in detections
    )


def main() -> int:
    args = parse_args()
    labels = load_labels(args.labels)
    imx500 = make_imx500(args.model)

    picam2 = Picamera2(camera_num=imx500.camera_num)
    config = picam2.create_preview_configuration(controls={"FrameRate": 30})
    picam2.configure(config)

    servo = Servo(
        args.servo_gpio,
        min_pulse_width=args.servo_min_pulse_us / 1_000_000,
        max_pulse_width=args.servo_max_pulse_us / 1_000_000,
    )
    servo.value = clamp_servo_value(args.rest_position)

    stop_requested = False

    def handle_signal(signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(f"\nStopping on signal {signum}...")

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print("Starting IMX500 drowning detection servo monitor")
    print(f"Model: {args.model if args.model else 'camera/default configuration'}")
    print(f"Labels: {labels}")
    print(
        f"Trigger rule: label='{args.drowning_label}', "
        f"class_id={args.drowning_class_id}, threshold={args.threshold}"
    )
    print(f"Servo GPIO: {args.servo_gpio}, rest={args.rest_position}, active={args.active_position}")

    consecutive_hits = 0
    last_trigger_time = 0.0

    try:
        picam2.start()

        while not stop_requested:
            metadata = picam2.capture_metadata()
            if args.debug_metadata:
                print(f"Metadata keys: {sorted(metadata.keys())}")
                args.debug_metadata = False
            detections = extract_detections(metadata, imx500, labels, args.threshold)

            if args.print_all and detections:
                print(describe_detections(detections))

            drowning_detections = [
                det
                for det in detections
                if is_drowning_detection(det, args.drowning_label, args.drowning_class_id)
            ]

            if drowning_detections:
                consecutive_hits += 1
                best = max(drowning_detections, key=lambda det: det.score)
                print(
                    f"Drowning detected: score={best.score:.2f}, "
                    f"frame_hits={consecutive_hits}/{args.confirm_frames}"
                )
            else:
                consecutive_hits = 0

            now = time.monotonic()
            cooldown_elapsed = now - last_trigger_time

            if drowning_detections and consecutive_hits >= args.confirm_frames:
                if cooldown_elapsed >= args.cooldown:
                    print("Triggering servo...")
                    trigger_servo(
                        servo,
                        active_position=args.active_position,
                        rest_position=args.rest_position,
                        hold_time=args.hold_time,
                        settle_time=args.settle_time,
                    )
                    last_trigger_time = time.monotonic()
                    consecutive_hits = 0
                else:
                    remaining = args.cooldown - cooldown_elapsed
                    print(f"Drowning seen but still in cooldown for {remaining:.1f}s")

    finally:
        servo.value = clamp_servo_value(args.rest_position)
        servo.close()
        picam2.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
