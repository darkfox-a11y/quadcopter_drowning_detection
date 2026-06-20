#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO detection on an image while prioritizing a chosen class."
    )
    parser.add_argument("--model", type=Path, required=True, help="Path to a YOLO model file.")
    parser.add_argument("--image", type=Path, required=True, help="Path to the image to test.")
    parser.add_argument(
        "--priority-class",
        default="drowning",
        help="Class name to surface first when present.",
    )
    parser.add_argument(
        "--priority-min-conf",
        type=float,
        default=0.20,
        help="Minimum confidence for the priority class to count as a hit.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Base confidence threshold for normal detections.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=320,
        help="Inference image size.",
    )
    parser.add_argument(
        "--save-name",
        default="priority_test",
        help="Name of the output folder under runs/detect.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model = YOLO(str(args.model), task="detect")
    threshold = min(args.conf, args.priority_min_conf)
    result = model.predict(
        source=str(args.image),
        imgsz=args.imgsz,
        conf=threshold,
        save=True,
        verbose=False,
        project="runs/detect",
        name=args.save_name,
        exist_ok=True,
    )[0]

    names = result.names
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        print("detections: none")
        return 0

    detections = []
    for index in range(len(boxes)):
        class_id = int(boxes.cls[index].item())
        label = names[class_id]
        confidence = float(boxes.conf[index].item())
        xyxy = [round(float(value), 2) for value in boxes.xyxy[index].tolist()]
        detections.append(
            {
                "label": label,
                "confidence": confidence,
                "box": xyxy,
            }
        )

    priority_hits = [
        detection
        for detection in detections
        if detection["label"] == args.priority_class and detection["confidence"] >= args.priority_min_conf
    ]

    if priority_hits:
        best_priority = max(priority_hits, key=lambda detection: detection["confidence"])
        print(
            f"priority_hit: {best_priority['label']} "
            f"conf={best_priority['confidence']:.4f} box={best_priority['box']}"
        )
    else:
        best_detection = max(detections, key=lambda detection: detection["confidence"])
        print(
            f"top_hit: {best_detection['label']} "
            f"conf={best_detection['confidence']:.4f} box={best_detection['box']}"
        )

    print("all_detections:")
    for detection in sorted(detections, key=lambda item: item["confidence"], reverse=True):
        print(
            f"{detection['label']} conf={detection['confidence']:.4f} box={detection['box']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
