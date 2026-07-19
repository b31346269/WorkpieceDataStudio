from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from ultralytics import YOLO


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def safe_extract(archive_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        for item in archive.infolist():
            pure = PurePosixPath(item.filename.replace("\\", "/"))
            if pure.is_absolute() or ".." in pure.parts:
                raise ValueError(f"Unsafe ZIP entry: {item.filename}")
        archive.extractall(destination)


def find_split(root: Path, names: tuple[str, ...]) -> tuple[Path, Path] | None:
    wanted = {name.lower() for name in names}
    for images in root.rglob("images"):
        if images.is_dir() and images.parent.name.lower() in wanted:
            labels = images.parent / "labels"
            if labels.is_dir():
                return images, labels
    return None


def copy_split(
    source: tuple[Path, Path],
    destination: Path,
    prefix: str,
) -> int:
    images_source, labels_source = source
    images_target = destination / "images"
    labels_target = destination / "labels"
    images_target.mkdir(parents=True, exist_ok=True)
    labels_target.mkdir(parents=True, exist_ok=True)
    copied = 0
    for image in sorted(images_source.iterdir()):
        if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        stem = f"{prefix}_{image.stem}"
        image_target = images_target / f"{stem}{image.suffix.lower()}"
        label_source = labels_source / f"{image.stem}.txt"
        label_target = labels_target / f"{stem}.txt"
        shutil.copy2(image, image_target)
        if label_source.is_file():
            shutil.copy2(label_source, label_target)
        else:
            label_target.write_text("", encoding="utf-8")
        copied += 1
    return copied


def read_names(root: Path) -> dict[int, str]:
    yaml_files = list(root.rglob("data.yaml"))
    if not yaml_files:
        raise ValueError("The source Roboflow ZIP has no data.yaml.")
    with yaml_files[0].open("r", encoding="utf-8-sig") as handle:
        payload: dict[str, Any] = yaml.safe_load(handle) or {}
    names = payload.get("names")
    if isinstance(names, list):
        return {index: str(name) for index, name in enumerate(names)}
    if isinstance(names, dict):
        return {int(index): str(name) for index, name in names.items()}
    raise ValueError("data.yaml does not contain a valid names list.")


def prepare_dataset(
    source_zip: Path,
    synthetic_zip: Path | None,
    destination: Path,
) -> Path:
    with tempfile.TemporaryDirectory(prefix="workpiece-source-") as source_tmp:
        source_root = Path(source_tmp)
        safe_extract(source_zip, source_root)
        names = read_names(source_root)
        real_train = find_split(source_root, ("train",))
        real_val = find_split(source_root, ("valid", "val", "validation"))
        real_test = find_split(source_root, ("test",))
        if not real_train:
            raise ValueError("The source ZIP has no train/images and train/labels.")
        if not real_val:
            raise ValueError(
                "The source ZIP needs a real validation split. "
                "Synthetic images must not be used for validation."
            )
        copy_split(real_train, destination / "train", "real")
        copy_split(real_val, destination / "valid", "real")
        if real_test:
            copy_split(real_test, destination / "test", "real")

    if synthetic_zip:
        with tempfile.TemporaryDirectory(prefix="workpiece-synthetic-") as synthetic_tmp:
            synthetic_root = Path(synthetic_tmp)
            safe_extract(synthetic_zip, synthetic_root)
            synthetic_train = find_split(synthetic_root, ("train",))
            if not synthetic_train:
                raise ValueError("The synthetic ZIP has no train split.")
            copy_split(synthetic_train, destination / "train", "synthetic")

    data = {
        "path": str(destination.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "names": names,
    }
    if (destination / "test" / "images").is_dir():
        data["test"] = "test/images"
    output = destination / "data.yaml"
    output.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a larger Ultralytics detector on real + approved synthetic data."
    )
    parser.add_argument("--source", type=Path, required=True, help="Original Roboflow ZIP")
    parser.add_argument("--synthetic", type=Path, help="Approved generated YOLOv8 ZIP")
    parser.add_argument("--model", default="yolo26l.pt")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=float, default=-1)
    parser.add_argument(
        "--device",
        default="2,3,6",
        help="Physical GPU ids; only 2, 3 and 6 are authorized.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--name", default="workpiece_yolo26l")
    parser.add_argument("--output", type=Path, default=Path("school_training"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested_devices = {
        item.strip() for item in str(args.device).split(",") if item.strip()
    }
    allowed_devices = {"2", "3", "6"}
    if not requested_devices or not requested_devices <= allowed_devices:
        raise ValueError(
            "Only physical GPU ids 2, 3 and 6 are allowed for this account."
        )
    dataset_root = args.output / "dataset"
    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    dataset_root.mkdir(parents=True)
    data_yaml = prepare_dataset(args.source, args.synthetic, dataset_root)
    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=30,
        cache=False,
        project=str((args.output / "runs").resolve()),
        name=args.name,
    )


if __name__ == "__main__":
    main()
