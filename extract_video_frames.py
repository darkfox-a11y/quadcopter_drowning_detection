#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
from ultralytics import YOLO


CLASS_NAMES = ("swimming", "tread water", "drowning")
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract labeled frames from class-organized videos for YOLO fine-tuning."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Directory containing one subfolder per class with videos inside.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where train/val images and empty label files will be created.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=list(CLASS_NAMES),
        help="Class folder names to scan under --source-dir.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=15,
        help="Keep every Nth frame from each video.",
    )
    parser.add_argument(
        "--max-frames-per-video",
        type=int,
        default=250,
        help="Upper bound on saved frames from a single video.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of saved frames that should go to val.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=640,
        help="Resize saved frames to a square size. Use 0 to keep original size.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for train/val split.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Optional YOLO model used to auto-label extracted frames.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold used when --model is provided.",
    )
    return parser.parse_args()


def list_videos(class_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def ensure_dataset_dirs(output_dir: Path) -> None:
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def choose_split(rng: random.Random, val_ratio: float) -> str:
    return "val" if rng.random() < val_ratio else "train"


def save_frame(
    frame,
    output_dir: Path,
    split: str,
    class_name: str,
    video_stem: str,
    frame_index: int,
    image_size: int,
) -> tuple[Path, Path]:
    if image_size > 0:
        frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)

    image_name = f"{class_name}__{video_stem}__frame_{frame_index:06d}.jpg"
    image_path = output_dir / "images" / split / image_name
    label_path = output_dir / "labels" / split / image_name.replace(".jpg", ".txt")

    cv2.imwrite(str(image_path), frame)
    label_path.touch(exist_ok=True)
    return image_path, label_path


def write_yolo_labels(result, label_path: Path) -> int:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        label_path.write_text("", encoding="utf-8")
        return 0

    lines = []
    for index in range(len(boxes)):
        class_id = int(boxes.cls[index].item())
        x_center, y_center, width, height = [float(value) for value in boxes.xywhn[index].tolist()]
        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def extract_from_video(
    video_path: Path,
    class_name: str,
    output_dir: Path,
    rng: random.Random,
    frame_step: int,
    max_frames_per_video: int,
    val_ratio: float,
    image_size: int,
    model: YOLO | None,
    conf: float,
) -> int:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print(f"Skipping unreadable video: {video_path}")
        return 0

    saved = 0
    frame_index = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        if frame_index % frame_step == 0:
            split = choose_split(rng, val_ratio)
            image_path, label_path = save_frame(
                frame=frame,
                output_dir=output_dir,
                split=split,
                class_name=class_name,
                video_stem=video_path.stem,
                frame_index=frame_index,
                image_size=image_size,
            )
            if model is not None:
                result = model.predict(source=str(image_path), conf=conf, verbose=False)[0]
                write_yolo_labels(result, label_path)
            saved += 1
            if saved >= max_frames_per_video:
                break

        frame_index += 1

    capture.release()
    return saved


def write_dataset_yaml(output_dir: Path, classes: list[str]) -> Path:
    yaml_path = output_dir / "data.yaml"
    lines = [
        f'train: "{(output_dir / "images" / "train").resolve()}"',
        f'val: "{(output_dir / "images" / "val").resolve()}"',
        "",
        f"nc: {len(classes)}",
        "names:",
    ]
    lines.extend(f"  - {class_name}" for class_name in classes)
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    ensure_dataset_dirs(args.output_dir)
    model = YOLO(str(args.model), task="detect") if args.model else None

    total_saved = 0
    for class_name in args.classes:
        class_dir = args.source_dir / class_name
        if not class_dir.exists():
            print(f"Missing class folder: {class_dir}")
            continue

        videos = list_videos(class_dir)
        if not videos:
            print(f"No videos found for class: {class_name}")
            continue

        print(f"{class_name}: found {len(videos)} video(s)")
        for video_path in videos:
            saved = extract_from_video(
                video_path=video_path,
                class_name=class_name,
                output_dir=args.output_dir,
                rng=rng,
                frame_step=max(1, args.frame_step),
                max_frames_per_video=max(1, args.max_frames_per_video),
                val_ratio=args.val_ratio,
                image_size=max(0, args.image_size),
                model=model,
                conf=args.conf,
            )
            total_saved += saved
            print(f"  {video_path.name}: saved {saved} frame(s)")

    yaml_path = write_dataset_yaml(args.output_dir, args.classes)
    print(f"Saved {total_saved} frame(s) total")
    print(f"Dataset YAML: {yaml_path}")
    if model is None:
        print("Note: label files are empty placeholders. Annotate the frames before training.")
    else:
        print("Auto-labels were generated with the supplied model. Review them before training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
