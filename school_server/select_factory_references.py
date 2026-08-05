#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def read_classes(label_path: Path) -> list[int]:
    classes: list[int] = []
    if not label_path.is_file():
        return classes
    for line in label_path.read_text(encoding="utf-8-sig").splitlines():
        parts = line.split()
        if parts:
            classes.append(int(parts[0]))
    return classes


def image_feature(path: Path, classes: list[int]) -> tuple[np.ndarray, dict[str, float]]:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    small = ImageOps.fit(image, (64, 64), method=Image.Resampling.BILINEAR)
    array = np.asarray(small, dtype=np.float32) / 255.0
    gray = np.asarray(small.convert("L"), dtype=np.float32) / 255.0
    low_resolution = np.asarray(
        small.convert("L").resize((8, 8), Image.Resampling.BILINEAR),
        dtype=np.float32,
    ).reshape(-1) / 255.0
    histogram, _ = np.histogram(gray, bins=16, range=(0, 1), density=True)
    edges = np.asarray(small.filter(ImageFilter.FIND_EDGES).convert("L"), dtype=np.float32)
    class_counts = np.array([classes.count(index) for index in range(3)], dtype=np.float32)
    class_counts = np.clip(class_counts / np.array([12.0, 12.0, 2.0]), 0, 1)
    feature = np.concatenate(
        [
            array.mean(axis=(0, 1)),
            array.std(axis=(0, 1)),
            np.array([gray.mean(), gray.std(), edges.mean() / 255.0]),
            histogram.astype(np.float32),
            low_resolution,
            class_counts,
        ]
    )
    metadata = {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "edge_energy": float(edges.mean() / 255.0),
    }
    return feature, metadata


def farthest_point_indices(features: np.ndarray, count: int) -> list[int]:
    normalized = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-6)
    center_distance = np.square(normalized).sum(axis=1)
    selected = [int(center_distance.argmax())]
    minimum_distance = np.square(normalized - normalized[selected[0]]).sum(axis=1)
    while len(selected) < min(count, len(normalized)):
        next_index = int(minimum_distance.argmax())
        selected.append(next_index)
        distance = np.square(normalized - normalized[next_index]).sum(axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)
        minimum_distance[selected] = -1
    return selected


def create_contact_sheet(paths: list[Path], output: Path) -> None:
    thumb_size = 180
    label_height = 28
    columns = 5
    rows = (len(paths) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_size, rows * (thumb_size + label_height)), "#111820")
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(paths):
        with Image.open(path) as source:
            thumb = ImageOps.fit(
                ImageOps.exif_transpose(source).convert("RGB"),
                (thumb_size, thumb_size),
                method=Image.Resampling.LANCZOS,
            )
        x = index % columns * thumb_size
        y = index // columns * (thumb_size + label_height)
        sheet.paste(thumb, (x, y))
        draw.text((x + 5, y + thumb_size + 5), path.name[:24], fill="white")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=90)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select diverse train-only factory-generation references.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=25)
    args = parser.parse_args()

    images_root = args.dataset / "train" / "images"
    labels_root = args.dataset / "train" / "labels"
    images = sorted(
        path for path in images_root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise ValueError(f"No train images found under {images_root}")

    rows = []
    features = []
    for image_path in images:
        classes = read_classes(labels_root / f"{image_path.stem}.txt")
        feature, metadata = image_feature(image_path, classes)
        features.append(feature)
        rows.append({"source": image_path.name, "classes": classes, **metadata})

    indices = farthest_point_indices(np.stack(features), args.count)
    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)
    selected_paths: list[Path] = []
    manifest = []
    for rank, index in enumerate(indices, start=1):
        source = images[index]
        destination = args.output / f"reference_{rank:02d}{source.suffix.lower()}"
        shutil.copy2(source, destination)
        selected_paths.append(destination)
        manifest.append({"rank": rank, "file": destination.name, **rows[index]})

    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    create_contact_sheet(selected_paths, args.output / "contact_sheet.jpg")
    print(json.dumps({"selected": len(selected_paths), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
