#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import onnx


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect an ONNX model file.")
    parser.add_argument("model", type=Path, help="Path to the ONNX model")
    args = parser.parse_args()

    model = onnx.load(str(args.model))
    print(f"Path: {args.model}")
    print(f"Size: {args.model.stat().st_size} bytes")
    print(f"SHA256: {sha256sum(args.model)}")
    print(f"IR version: {model.ir_version}")
    print(f"Producer: {model.producer_name} {model.producer_version}")

    print("Inputs:")
    for item in model.graph.input:
        dims = [d.dim_value or d.dim_param or "?" for d in item.type.tensor_type.shape.dim]
        print(f"  - {item.name}: {dims}")

    print("Outputs:")
    for item in model.graph.output:
        dims = [d.dim_value or d.dim_param or "?" for d in item.type.tensor_type.shape.dim]
        print(f"  - {item.name}: {dims}")

    metadata = {item.key: item.value for item in model.metadata_props}
    if metadata:
        print("Metadata:")
        for key, value in metadata.items():
            print(f"  - {key}: {value}")

    print(f"Node count: {len(model.graph.node)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
