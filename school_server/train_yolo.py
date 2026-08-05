from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath
from typing import Any

import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
RF_SUFFIX = re.compile(r"\.rf\.[^.]+$", re.IGNORECASE)


def source_id(stem: str) -> str:
    """Return the original capture id before Roboflow's augmentation suffix."""
    return RF_SUFFIX.sub("", stem)


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
    blocked_source_ids: set[str] | None = None,
    blocked_hashes: set[str] | None = None,
    seen_hashes: set[str] | None = None,
    max_items: int | None = None,
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
        if blocked_source_ids and source_id(image.stem) in blocked_source_ids:
            continue
        digest = ""
        if blocked_hashes is not None or seen_hashes is not None:
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
        if blocked_hashes and digest in blocked_hashes:
            continue
        if seen_hashes is not None and digest in seen_hashes:
            continue
        if max_items is not None and copied >= max_items:
            break
        stem = f"{prefix}_{image.stem}"
        image_target = images_target / f"{stem}{image.suffix.lower()}"
        label_source = labels_source / f"{image.stem}.txt"
        label_target = labels_target / f"{stem}.txt"
        shutil.copy2(image, image_target)
        if label_source.is_file():
            shutil.copy2(label_source, label_target)
        else:
            label_target.write_text("", encoding="utf-8")
        if seen_hashes is not None:
            seen_hashes.add(digest)
        copied += 1
    return copied


def collect_source_ids(split: tuple[Path, Path] | None) -> set[str]:
    if not split:
        return set()
    images, _ = split
    return {
        source_id(path.stem)
        for path in images.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def collect_hashes(split: tuple[Path, Path] | None) -> set[str]:
    if not split:
        return set()
    images, _ = split
    return {
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in images.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def find_voc_split(root: Path, names: tuple[str, ...]) -> Path | None:
    wanted = {name.lower() for name in names}
    for folder in root.rglob("*"):
        if folder.is_dir() and folder.name.lower() in wanted and next(folder.glob("*.xml"), None):
            return folder
    return None


def read_voc_names(root: Path) -> set[str]:
    names: set[str] = set()
    for xml_path in root.rglob("*.xml"):
        tree = ET.parse(xml_path)
        names.update(
            node.text.strip()
            for node in tree.findall(".//object/name")
            if node.text and node.text.strip()
        )
    if not names:
        raise ValueError("The auxiliary VOC ZIP contains no object classes.")
    return names


def copy_voc_train(
    folder: Path,
    destination: Path,
    prefix: str,
    name_to_id: dict[str, int],
    blocked_source_ids: set[str],
) -> int:
    images_target = destination / "images"
    labels_target = destination / "labels"
    images_target.mkdir(parents=True, exist_ok=True)
    labels_target.mkdir(parents=True, exist_ok=True)
    copied = 0

    for image in sorted(folder.iterdir()):
        if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if source_id(image.stem) in blocked_source_ids:
            continue
        xml_path = folder / f"{image.stem}.xml"
        if not xml_path.is_file():
            raise ValueError(f"Missing VOC annotation for {image.name}")

        root = ET.parse(xml_path).getroot()
        width = float(root.findtext("size/width", "0"))
        height = float(root.findtext("size/height", "0"))
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image size in {xml_path}")

        lines: list[str] = []
        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip()
            if name not in name_to_id:
                raise ValueError(f"Unknown class {name!r} in {xml_path}")
            box = obj.find("bndbox")
            if box is None:
                continue
            xmin = max(0.0, float(box.findtext("xmin", "0")))
            ymin = max(0.0, float(box.findtext("ymin", "0")))
            xmax = min(width, float(box.findtext("xmax", "0")))
            ymax = min(height, float(box.findtext("ymax", "0")))
            if xmax <= xmin or ymax <= ymin:
                continue
            cx = (xmin + xmax) / 2 / width
            cy = (ymin + ymax) / 2 / height
            box_width = (xmax - xmin) / width
            box_height = (ymax - ymin) / height
            lines.append(
                f"{name_to_id[name]} {cx:.8f} {cy:.8f} {box_width:.8f} {box_height:.8f}"
            )

        stem = f"{prefix}_{image.stem}"
        shutil.copy2(image, images_target / f"{stem}{image.suffix.lower()}")
        (labels_target / f"{stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
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
    auxiliary_voc_zips: list[Path] | None = None,
    auxiliary_yolo_zips: list[Path] | None = None,
    synthetic_max_fraction: float = 0.25,
) -> Path:
    report: dict[str, Any] = {"sources": {}, "leakage_rule": "Roboflow source id"}
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
        locked_source_ids = collect_source_ids(real_val) | collect_source_ids(real_test)
        locked_hashes = collect_hashes(real_val) | collect_hashes(real_test)
        seen_real_hashes = set(locked_hashes)
        real_train_count = copy_split(
            real_train,
            destination / "train",
            "new_real",
            blocked_source_ids=locked_source_ids,
            blocked_hashes=locked_hashes,
            seen_hashes=seen_real_hashes,
        )
        real_val_count = copy_split(real_val, destination / "valid", "new_real")
        real_test_count = copy_split(real_test, destination / "test", "new_real") if real_test else 0
        report["sources"]["new_workpiece"] = {
            "train": real_train_count,
            "valid": real_val_count,
            "test": real_test_count,
            "locked_source_ids": len(locked_source_ids),
            "locked_exact_hashes": len(locked_hashes),
        }

    name_to_id = {name: class_id for class_id, name in names.items()}
    for index, auxiliary_zip in enumerate(auxiliary_voc_zips or []):
        with tempfile.TemporaryDirectory(prefix="workpiece-voc-") as voc_tmp:
            voc_root = Path(voc_tmp)
            safe_extract(auxiliary_zip, voc_root)
            voc_names = read_voc_names(voc_root)
            if voc_names != set(name_to_id):
                raise ValueError(
                    f"VOC classes {sorted(voc_names)} do not exactly match "
                    f"primary classes {sorted(name_to_id)}."
                )
            voc_train = find_voc_split(voc_root, ("train",))
            if not voc_train:
                raise ValueError(f"The auxiliary VOC ZIP has no train folder: {auxiliary_zip}")
            count = copy_voc_train(
                voc_train,
                destination / "train",
                f"tool2_{index}",
                name_to_id,
                locked_source_ids,
            )
            real_train_count += count
            report["sources"][f"aux_voc_{index}"] = {
                "archive": str(auxiliary_zip),
                "train": count,
                "validation_included": False,
            }

    for index, auxiliary_zip in enumerate(auxiliary_yolo_zips or []):
        with tempfile.TemporaryDirectory(prefix="workpiece-aux-yolo-") as yolo_tmp:
            yolo_root = Path(yolo_tmp)
            safe_extract(auxiliary_zip, yolo_root)
            auxiliary_names = read_names(yolo_root)
            if auxiliary_names != names:
                raise ValueError(
                    f"Auxiliary YOLO classes {auxiliary_names} do not exactly match "
                    f"primary classes {names}."
                )
            total = 0
            split_counts: dict[str, int] = {}
            for split_name, aliases in (
                ("train", ("train",)),
                ("valid", ("valid", "val", "validation")),
                ("test", ("test",)),
            ):
                split = find_split(yolo_root, aliases)
                if not split:
                    continue
                count = copy_split(
                    split,
                    destination / "train",
                    f"tool2_{index}_{split_name}",
                    blocked_source_ids=locked_source_ids,
                    blocked_hashes=locked_hashes,
                    seen_hashes=seen_real_hashes,
                )
                split_counts[split_name] = count
                total += count
            if total == 0:
                raise ValueError(f"The auxiliary YOLO ZIP contains no usable images: {auxiliary_zip}")
            real_train_count += total
            report["sources"][f"aux_yolo_{index}"] = {
                "archive": str(auxiliary_zip),
                "train_from_all_auxiliary_splits": total,
                "source_split_counts": split_counts,
                "validation_included": False,
            }

    if synthetic_zip:
        with tempfile.TemporaryDirectory(prefix="workpiece-synthetic-") as synthetic_tmp:
            synthetic_root = Path(synthetic_tmp)
            safe_extract(synthetic_zip, synthetic_root)
            synthetic_train = find_split(synthetic_root, ("train",))
            if not synthetic_train:
                raise ValueError("The synthetic ZIP has no train split.")
            max_synthetic = int(real_train_count * synthetic_max_fraction)
            synthetic_count = copy_split(
                synthetic_train,
                destination / "train",
                "synthetic",
                max_items=max_synthetic,
            )
            report["sources"]["synthetic"] = {
                "archive": str(synthetic_zip),
                "train": synthetic_count,
                "maximum_fraction_of_real": synthetic_max_fraction,
                "validation_included": False,
            }

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
    report["classes"] = names
    report["real_train_images"] = real_train_count
    (destination / "preparation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a larger Ultralytics detector on real + approved synthetic data."
    )
    parser.add_argument("--source", type=Path, required=True, help="Original Roboflow ZIP")
    parser.add_argument(
        "--aux-voc",
        type=Path,
        action="append",
        default=[],
        help="Optional auxiliary Pascal VOC ZIP; may be supplied more than once.",
    )
    parser.add_argument(
        "--aux-yolo",
        type=Path,
        action="append",
        default=[],
        help="Optional auxiliary YOLO ZIP; all of its splits become train-only.",
    )
    parser.add_argument("--synthetic", type=Path, help="Approved generated YOLOv8 ZIP")
    parser.add_argument(
        "--synthetic-max-fraction",
        type=float,
        default=0.25,
        help="Maximum synthetic train images as a fraction of combined real train images.",
    )
    parser.add_argument("--model", default="yolo26l.pt")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument(
        "--export-imgsz",
        type=int,
        default=640,
        help="Fixed ONNX input size for the existing Unity decoder.",
    )
    parser.add_argument("--batch", type=float, default=-1)
    parser.add_argument(
        "--device",
        default="6",
        help="Physical GPU ids; only 6 and 8 are currently authorized.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default="workpiece_yolo26l")
    parser.add_argument("--output", type=Path, default=Path("school_training"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO

    requested_devices = {
        item.strip() for item in str(args.device).split(",") if item.strip()
    }
    allowed_devices = {"6", "8"}
    if not requested_devices or not requested_devices <= allowed_devices:
        raise ValueError(
            "Only physical GPU ids 6 and 8 are allowed for this account."
        )
    if not 0 <= args.synthetic_max_fraction <= 1:
        raise ValueError("--synthetic-max-fraction must be between 0 and 1.")
    dataset_root = args.output / "dataset"
    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    dataset_root.mkdir(parents=True)
    data_yaml = prepare_dataset(
        args.source,
        args.synthetic,
        dataset_root,
        auxiliary_voc_zips=args.aux_voc,
        auxiliary_yolo_zips=args.aux_yolo,
        synthetic_max_fraction=args.synthetic_max_fraction,
    )
    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=30,
        seed=args.seed,
        deterministic=True,
        close_mosaic=10,
        cache=False,
        project=str((args.output / "runs").resolve()),
        name=args.name,
    )
    best_path = Path(model.trainer.best)
    exported = YOLO(str(best_path)).export(
        format="onnx",
        imgsz=args.export_imgsz,
        dynamic=False,
        simplify=True,
        nms=False,
    )
    print(f"Unity-compatible ONNX: {exported}")


if __name__ == "__main__":
    main()
