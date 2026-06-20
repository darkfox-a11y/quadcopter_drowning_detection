#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
from ultralytics import YOLO


@dataclass(frozen=True)
class VideoSpec:
    path: Path
    target: str
    strict: bool


VIDEO_SPECS = [
    VideoSpec(Path("/Users/neerad/Desktop/DJI_003_BEPROJECT/DJI_20260410103033_0018_D.MP4"), "drowning", False),
    VideoSpec(Path("/Users/neerad/Desktop/DJI_003_BEPROJECT/DJI_20260410102649_0017_D.MP4"), "drowning", False),
    VideoSpec(Path("/Users/neerad/Desktop/DJI_003_BEPROJECT/DJI_20260410102340_0016_D.MP4"), "drowning", False),
    VideoSpec(Path("/Users/neerad/Desktop/DJI_003_BEPROJECT/DJI_20260410101920_0015_D.MP4"), "drowning", False),
    VideoSpec(Path("/Users/neerad/Desktop/DJI_003_BEPROJECT/DJI_20260410101422_0012_D.MP4"), "drowning", False),
    VideoSpec(Path("/Users/neerad/Desktop/DJI_003_BEPROJECT/DJI_20260410101404_0011_D.MP4"), "drowning", False),
    VideoSpec(Path("/Users/neerad/Desktop/DJI_003_BEPROJECT/DJI_20260410101331_0010_D.MP4"), "drowning", False),
    VideoSpec(Path("/Users/neerad/Desktop/DJI_003_BEPROJECT/DJI_20260410101309_0009_D.MP4"), "drowning", False),
    VideoSpec(Path("/Users/neerad/Desktop/DJI_003_BEPROJECT/DJI_20260410101256_0008_D.MP4"), "drowning", False),
    VideoSpec(Path("/Users/neerad/Desktop/DJI_003_BEPROJECT/DJI_20260410101229_0007_D.MP4"), "drowning", False),
    VideoSpec(Path("/Users/neerad/Desktop/DJI_003_BEPROJECT/DJI_20260410094717_0006_D.MP4"), "tread water", True),
    VideoSpec(Path("/Users/neerad/Desktop/DJI_003_BEPROJECT/DJI_20260410094333_0005_D.MP4"), "tread water", True),
    VideoSpec(Path("/Users/neerad/Desktop/DJI_003_BEPROJECT/DJI_20260410093949_0004_D.MP4"), "tread water", True),
    VideoSpec(Path("/Users/neerad/Desktop/DJI_003_BEPROJECT/DJI_20260410092246_0003_D.MP4"), "swimming", True),
    VideoSpec(Path("/Users/neerad/Desktop/DJI_003_BEPROJECT/DJI_20260410091901_0002_D.MP4"), "swimming", True),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample DJI videos every second, pseudo-label with YOLOv8n, and build a fine-tuning dataset."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/Users/neerad/ai_camv8n/best.pt"),
        help="Path to the YOLO model used for pseudo-labeling.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/neerad/drowning-detection/datasets/dji003_pseudolabels"),
        help="Directory where the pseudo-labeled dataset will be written.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=320,
        help="Inference image size.",
    )
    parser.add_argument(
        "--sample-seconds",
        type=float,
        default=1.0,
        help="Sampling interval in seconds.",
    )
    parser.add_argument(
        "--base-conf",
        type=float,
        default=0.25,
        help="Default confidence threshold for detections.",
    )
    parser.add_argument(
        "--drowning-conf",
        type=float,
        default=0.20,
        help="Lower confidence threshold allowed for drowning detections.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of accepted frames written to val.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=None,
        help="Optional subset of targets to process, e.g. drowning or swimming.",
    )
    parser.add_argument(
        "--priority-drowning",
        action="store_true",
        help="For drowning videos, keep mixed drowning+tread frames and preserve all boxes in those frames.",
    )
    parser.add_argument(
        "--priority-floor",
        type=float,
        default=0.001,
        help="Very low confidence floor used when --priority-drowning is enabled for drowning videos.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing images/labels/previews under the output directory before mining.",
    )
    return parser.parse_args()


def ensure_dirs(output_dir: Path) -> None:
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "previews" / split).mkdir(parents=True, exist_ok=True)


def write_dataset_yaml(output_dir: Path) -> None:
    yaml_path = output_dir / "data.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f'train: "{(output_dir / "images" / "train").resolve()}"',
                f'val: "{(output_dir / "images" / "val").resolve()}"',
                "",
                "nc: 3",
                "names:",
                "  - swimming",
                "  - tread water",
                "  - drowning",
                "",
            ]
        ),
        encoding="utf-8",
    )


def choose_split(rng: random.Random, val_ratio: float) -> str:
    return "val" if rng.random() < val_ratio else "train"


def select_detections(
    result,
    target: str,
    strict: bool,
    drowning_conf: float,
    base_conf: float,
    priority_drowning: bool,
    priority_floor: float,
) -> list[tuple[int, float, list[float], list[float]]]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    names = result.names
    selected = []
    for index in range(len(boxes)):
        class_id = int(boxes.cls[index].item())
        label = names[class_id]
        confidence = float(boxes.conf[index].item())
        threshold = drowning_conf if label == "drowning" else base_conf
        if priority_drowning and target == "drowning":
            threshold = priority_floor
        if confidence < threshold:
            continue
        if strict and label != target:
            continue
        if not strict and target == "drowning" and not priority_drowning and label != "drowning":
            continue
        xywhn = [float(value) for value in boxes.xywhn[index].tolist()]
        xyxy = [float(value) for value in boxes.xyxy[index].tolist()]
        selected.append((class_id, confidence, xywhn, xyxy))
    if priority_drowning and target == "drowning":
        labels = {result.names[class_id] for class_id, _confidence, _xywhn, _xyxy in selected}
        if "drowning" not in labels:
            return []
    return selected


def write_label_file(path: Path, detections: list[tuple[int, float, list[float], list[float]]]) -> None:
    lines = [
        f"{class_id} {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f}"
        for class_id, _confidence, xywhn, _xyxy in detections
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def save_preview(path: Path, result, detections: list[tuple[int, float, list[float], list[float]]]) -> None:
    if not detections:
        return

    boxes = result.boxes
    keep_indices = []
    for class_id, confidence, _xywhn, _xyxy in detections:
        for index in range(len(boxes)):
            if int(boxes.cls[index].item()) != class_id:
                continue
            if abs(float(boxes.conf[index].item()) - confidence) < 1e-6:
                keep_indices.append(index)
                break

    plotted = result.plot()
    cv2.imwrite(str(path), plotted)


def process_video(
    spec: VideoSpec,
    model: YOLO,
    output_dir: Path,
    rng: random.Random,
    args: argparse.Namespace,
    summary_writer,
) -> Counter:
    counts = Counter(sampled=0, kept=0, drowning_frames=0)
    capture = cv2.VideoCapture(str(spec.path))
    if not capture.isOpened():
        print(f"Skipping unreadable video: {spec.path}")
        return counts

    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    frame_step = max(1, int(round(fps * args.sample_seconds))) if fps > 0 else 30
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_index = 0

    while total_frames <= 0 or frame_index < total_frames:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            break

        counts["sampled"] += 1
        result = model.predict(source=frame, imgsz=args.imgsz, conf=min(args.base_conf, args.drowning_conf), verbose=False)[0]
        selected = select_detections(
            result=result,
            target=spec.target,
            strict=spec.strict,
            drowning_conf=args.drowning_conf,
            base_conf=args.base_conf,
            priority_drowning=args.priority_drowning,
            priority_floor=args.priority_floor,
        )
        if not selected:
            frame_index += 1
            continue

        split = choose_split(rng, args.val_ratio)
        stem = f"{spec.target.replace(' ', '_')}__{spec.path.stem}__frame_{frame_index:06d}"
        image_path = output_dir / "images" / split / f"{stem}.jpg"
        label_path = output_dir / "labels" / split / f"{stem}.txt"
        preview_path = output_dir / "previews" / split / f"{stem}.jpg"
        cv2.imwrite(str(image_path), frame)
        write_label_file(label_path, selected)
        save_preview(preview_path, result, selected)

        labels = [result.names[class_id] for class_id, _confidence, _xywhn, _xyxy in selected]
        best_label = max(selected, key=lambda item: item[1])
        if "drowning" in labels:
            counts["drowning_frames"] += 1
        counts["kept"] += 1
        summary_writer.writerow(
            {
                "video": spec.path.name,
                "target": spec.target,
                "strict": spec.strict,
                "frame_index": frame_index,
                "split": split,
                "saved_image": str(image_path),
                "saved_label": str(label_path),
                "saved_preview": str(preview_path),
                "num_boxes": len(selected),
                "labels": ",".join(labels),
                "top_label": result.names[best_label[0]],
                "top_conf": f"{best_label[1]:.4f}",
            }
        )
        frame_index += frame_step

    capture.release()
    return counts


def main() -> int:
    args = parse_args()
    ensure_dirs(args.output_dir)
    write_dataset_yaml(args.output_dir)
    rng = random.Random(args.seed)
    model = YOLO(str(args.model), task="detect")

    if args.clean and args.output_dir.exists():
        for name in ("images", "labels", "previews"):
            folder = args.output_dir / name
            if folder.exists():
                shutil.rmtree(folder)
        ensure_dirs(args.output_dir)
        write_dataset_yaml(args.output_dir)

    summary_path = args.output_dir / "summary.csv"
    append_summary = summary_path.exists() and not args.clean
    with summary_path.open("a" if append_summary else "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "video",
                "target",
                "strict",
                "frame_index",
                "split",
                "saved_image",
                "saved_label",
                "saved_preview",
                "num_boxes",
                "labels",
                "top_label",
                "top_conf",
            ],
        )
        if not append_summary:
            writer.writeheader()

        overall = Counter()
        targets = set(args.targets) if args.targets else None
        for spec in VIDEO_SPECS:
            if targets and spec.target not in targets:
                continue
            counts = process_video(spec, model, args.output_dir, rng, args, writer)
            overall.update(counts)
            print(
                f"{spec.path.name}: sampled={counts['sampled']} kept={counts['kept']} "
                f"drowning_frames={counts['drowning_frames']} target={spec.target}"
            )

    print(f"Dataset written to: {args.output_dir}")
    print(
        f"Overall sampled={overall['sampled']} kept={overall['kept']} "
        f"drowning_frames={overall['drowning_frames']}"
    )
    print(f"Summary CSV: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
