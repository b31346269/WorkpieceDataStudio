from __future__ import annotations

import argparse
import json
import os
import random
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CLASSES = ("hole", "screw", "tool")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_GPUS = {"2", "3", "6"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit Pascal VOC data, train EfficientDet-Lite2/Lite3, evaluate the "
            "Keras and TFLite models, and export an FP16 mobile model."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("datasets/tool2_roboflow_20260720_voc"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-name",
        choices=("efficientdet_lite2", "efficientdet_lite3"),
        default="efficientdet_lite2",
        help="TensorFlow Lite Model Maker EfficientDet backbone.",
    )
    parser.add_argument(
        "--gpus",
        default="2,3,6",
        help="Physical GPU ids. This project permits only 2, 3 and 6.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=48,
        help="Total batch across all visible GPUs; must be divisible by GPU count.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--anchor-scale", type=float, default=1.5)
    parser.add_argument(
        "--aspect-ratios",
        type=float,
        nargs="+",
        default=(1.0,),
        help="The speed_optimized baseline used only square anchors.",
    )
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--max-instances", type=int, default=300)
    parser.add_argument("--max-detections", type=int, default=30)
    parser.add_argument("--score-threshold", type=float, default=0.20)
    parser.add_argument("--iou-threshold", type=float, default=0.45)
    parser.add_argument(
        "--tflite-filename",
        default=None,
        help="Output filename; defaults to <model-name>_fp16.tflite.",
    )
    parser.add_argument(
        "--checkpoint-every-epochs",
        type=int,
        default=5,
        help="Reduce checkpoint disk use; the legacy script saved every epoch.",
    )
    parser.add_argument(
        "--keep-checkpoints",
        type=int,
        default=3,
        help="After a successful export, retain only the newest checkpoint prefixes.",
    )
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument(
        "--resume-weights",
        type=Path,
        help="Optional TensorFlow checkpoint prefix from an identical stage-1 config.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Validate Pascal VOC files and write dataset_audit.json without TensorFlow.",
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Build the model on all requested GPUs without starting training.",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Load --resume-weights, then evaluate and export without more training.",
    )
    parser.add_argument(
        "--skip-keras-eval",
        action="store_true",
        help="Skip pre-export Keras COCO evaluation; useful when multi-GPU NMS evaluation is unstable.",
    )
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def roboflow_source_id(filename: str) -> str:
    return filename.split(".rf.", 1)[0]


def audit_dataset(root: Path, expected_size: int = 448) -> dict[str, Any]:
    root = root.expanduser().resolve()
    report: dict[str, Any] = {
        "dataset": str(root),
        "expected_classes": list(CLASSES),
        "splits": {},
        "warnings": [],
        "errors": [],
    }
    source_splits: dict[str, set[str]] = defaultdict(set)

    for split in ("train", "valid", "test"):
        split_dir = root / split
        if not split_dir.is_dir():
            report["errors"].append(f"Missing split directory: {split_dir}")
            continue

        images = sorted(
            path
            for path in split_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        xml_files = sorted(split_dir.glob("*.xml"))
        image_stems = {path.stem for path in images}
        xml_stems = {path.stem for path in xml_files}
        class_counts: Counter[str] = Counter()
        class_box_stats: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        dimensions: Counter[tuple[int, int]] = Counter()
        box_areas: list[float] = []
        invalid_boxes: list[dict[str, Any]] = []
        empty_annotations = 0

        for image in images:
            source_splits[roboflow_source_id(image.name)].add(split)

        for xml_path in xml_files:
            try:
                annotation = ET.parse(xml_path).getroot()
            except (ET.ParseError, OSError) as error:
                report["errors"].append(f"Invalid XML {xml_path}: {error}")
                continue

            width = int(float(annotation.findtext("size/width", "0")))
            height = int(float(annotation.findtext("size/height", "0")))
            dimensions[(width, height)] += 1
            object_count = 0

            for object_node in annotation.findall("object"):
                class_name = (object_node.findtext("name") or "").strip()
                class_counts[class_name] += 1
                object_count += 1
                box = object_node.find("bndbox")
                try:
                    xmin = float(box.findtext("xmin"))  # type: ignore[union-attr]
                    ymin = float(box.findtext("ymin"))  # type: ignore[union-attr]
                    xmax = float(box.findtext("xmax"))  # type: ignore[union-attr]
                    ymax = float(box.findtext("ymax"))  # type: ignore[union-attr]
                except (AttributeError, TypeError, ValueError):
                    invalid_boxes.append({"file": xml_path.name, "reason": "parse"})
                    continue

                valid = (
                    width > 0
                    and height > 0
                    and 0 <= xmin < xmax <= width + 1
                    and 0 <= ymin < ymax <= height + 1
                )
                if not valid:
                    invalid_boxes.append(
                        {
                            "file": xml_path.name,
                            "image_size": [width, height],
                            "box": [xmin, ymin, xmax, ymax],
                        }
                    )
                    continue
                box_width = xmax - xmin
                box_height = ymax - ymin
                box_area = (box_width * box_height) / (width * height)
                box_areas.append(box_area)
                class_box_stats[class_name]["width_px"].append(box_width)
                class_box_stats[class_name]["height_px"].append(box_height)
                class_box_stats[class_name]["aspect_ratio"].append(
                    box_width / box_height
                )

            if object_count == 0:
                empty_annotations += 1

        unexpected_classes = sorted(set(class_counts) - set(CLASSES))
        images_without_xml = sorted(image_stems - xml_stems)
        xml_without_images = sorted(xml_stems - image_stems)
        split_report = {
            "images": len(images),
            "xml_files": len(xml_files),
            "paired_files": len(image_stems & xml_stems),
            "images_without_xml": len(images_without_xml),
            "xml_without_image": len(xml_without_images),
            "pairing_examples": {
                "images_without_xml": images_without_xml[:10],
                "xml_without_image": xml_without_images[:10],
            },
            "classes": dict(sorted(class_counts.items())),
            "boxes": sum(class_counts.values()),
            "empty_annotations": empty_annotations,
            "invalid_boxes": len(invalid_boxes),
            "invalid_box_examples": invalid_boxes[:10],
            "dimensions": [
                {"width": width, "height": height, "count": count}
                for (width, height), count in dimensions.most_common()
            ],
            "normalized_box_area": {
                "p01": percentile(box_areas, 0.01),
                "p50": percentile(box_areas, 0.50),
                "p95": percentile(box_areas, 0.95),
            },
            "class_box_geometry": {
                class_name: {
                    metric: {
                        "p01": percentile(values, 0.01),
                        "p50": percentile(values, 0.50),
                        "p95": percentile(values, 0.95),
                    }
                    for metric, values in metrics.items()
                }
                for class_name, metrics in sorted(class_box_stats.items())
            },
        }
        report["splits"][split] = split_report

        if images_without_xml or xml_without_images:
            report["errors"].append(f"{split}: image/XML pairing is incomplete")
        if invalid_boxes:
            report["errors"].append(f"{split}: {len(invalid_boxes)} invalid boxes")
        if unexpected_classes:
            report["errors"].append(
                f"{split}: unexpected classes {unexpected_classes}; expected {CLASSES}"
            )
        if dimensions and set(dimensions) != {(expected_size, expected_size)}:
            report["warnings"].append(
                f"{split}: expected only {expected_size}x{expected_size} images, "
                f"got {list(dimensions)}"
            )
        if empty_annotations:
            report["warnings"].append(
                f"{split}: {empty_annotations} images contain no objects"
            )

    cross_split_sources = {
        source: sorted(splits)
        for source, splits in source_splits.items()
        if len(splits) > 1
    }
    report["source_overlap"] = {
        "unique_source_ids": len(source_splits),
        "cross_split_source_ids": len(cross_split_sources),
        "examples": dict(list(sorted(cross_split_sources.items()))[:30]),
    }
    if cross_split_sources:
        report["warnings"].append(
            f"{len(cross_split_sources)} Roboflow source ids occur in multiple splits; "
            "reported validation/test metrics may be optimistic."
        )
    return report


def patch_generated_coco_json(data_loader: Any) -> None:
    path = Path(data_loader.annotations_json_file)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.setdefault("info", {})
    payload.setdefault("licenses", [])
    path.write_text(json.dumps(payload), encoding="utf-8")


def validate_args(args: argparse.Namespace) -> list[str]:
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if args.evaluate_only:
        if not gpus or not set(gpus).issubset(ALLOWED_GPUS) or len(gpus) != len(set(gpus)):
            raise ValueError("Evaluation GPUs must be a unique subset of physical GPUs 2, 3 and 6.")
    elif set(gpus) != ALLOWED_GPUS or len(gpus) != len(ALLOWED_GPUS):
        raise ValueError("Training must use physical GPUs 2, 3 and 6 exactly.")
    if args.global_batch_size <= 0 or args.global_batch_size % len(gpus):
        raise ValueError("Global batch size must be positive and divisible by GPU count.")
    if args.eval_batch_size <= 0 or args.eval_batch_size % len(gpus):
        raise ValueError("Evaluation batch size must be positive and divisible by GPU count.")
    if args.epochs <= 0:
        raise ValueError("Epochs must be positive.")
    if args.checkpoint_every_epochs <= 0:
        raise ValueError("Checkpoint interval must be positive.")
    if args.keep_checkpoints <= 0:
        raise ValueError("At least one checkpoint must be retained.")
    if args.evaluate_only and not args.resume_weights:
        raise ValueError("--evaluate-only requires --resume-weights.")
    return gpus


def prune_checkpoints(tf: Any, directory: Path, keep: int) -> list[str]:
    prefixes: list[tuple[int, Path]] = []
    for index_path in directory.glob("ckpt-*.index"):
        try:
            epoch = int(index_path.stem.split("-", 1)[1])
        except (IndexError, ValueError):
            continue
        prefixes.append((epoch, index_path.with_suffix("")))
    prefixes.sort()
    retained = prefixes[-keep:]
    retained_set = {prefix for _, prefix in retained}
    for _, prefix in prefixes:
        if prefix in retained_set:
            continue
        index_path = prefix.with_suffix(".index")
        if index_path.exists():
            index_path.unlink()
        for data_path in prefix.parent.glob(f"{prefix.name}.data-*"):
            data_path.unlink()
    if retained:
        retained_names = [str(prefix) for _, prefix in retained]
        tf.compat.v1.train.update_checkpoint_state(
            str(directory),
            model_checkpoint_path=retained_names[-1],
            all_model_checkpoint_paths=retained_names,
        )
        return retained_names
    return []


def main() -> None:
    args = parse_args()
    gpus = validate_args(args)
    dataset = args.dataset.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    expected_size = 512 if args.model_name == "efficientdet_lite3" else 448
    audit = audit_dataset(dataset, expected_size=expected_size)
    save_json(output / "dataset_audit.json", audit)
    if audit["errors"]:
        raise RuntimeError(
            "Dataset audit failed:\n- " + "\n- ".join(audit["errors"])
        )
    if args.audit_only:
        print(json.dumps(audit, indent=2))
        return

    existing_artifacts = [
        path
        for path in output.iterdir()
        if path.name
        not in {
            "dataset_audit.json",
            "run_config.json",
            "train.log",
            "train.pid",
            "bootstrap.log",
            "bootstrap.pid",
        }
    ]
    if existing_artifacts and not args.resume_weights and not args.build_only:
        raise FileExistsError(
            f"Output directory is not empty: {output}. "
            "Choose a new output directory or pass --resume-weights."
        )

    # These must be set before importing TensorFlow.
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    os.environ.setdefault(
        "TFHUB_CACHE_DIR", "/home/ping/efficientdet_project/tfhub_cache"
    )
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

    import numpy as np
    import tensorflow as tf
    from tflite_model_maker import model_spec, object_detector
    from tflite_model_maker.config import ExportFormat, QuantizationConfig

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    visible_gpus = tf.config.list_physical_devices("GPU")
    for gpu in visible_gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if len(visible_gpus) != len(gpus):
        raise RuntimeError(
            f"Expected {len(gpus)} visible GPUs for physical ids {gpus}, got {visible_gpus}. "
            "Launch with the provided shell script so LD_LIBRARY_PATH is correct."
        )

    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    template = model_spec.get(args.model_name)
    spec = type(template)(
        model_name=template.model_name,
        uri=template.uri,
        model_dir=str(checkpoints),
        epochs=args.epochs,
        batch_size=args.global_batch_size,
        steps_per_execution=1,
        var_freeze_expr=None,
        tflite_max_detections=args.max_detections,
        strategy="gpus",
        tf_random_seed=args.seed,
        verbose=1,
    )
    if spec.ds_strategy.num_replicas_in_sync != len(gpus):
        raise RuntimeError(
            f"MirroredStrategy has {spec.ds_strategy.num_replicas_in_sync} replicas, "
            f"expected {len(gpus)}."
        )

    spec.config.anchor_scale = args.anchor_scale
    spec.config.aspect_ratios = list(args.aspect_ratios)
    spec.config.alpha = args.focal_alpha
    spec.config.gamma = args.focal_gamma
    spec.config.max_instances_per_image = args.max_instances
    spec.config.tflite_max_detections = args.max_detections
    spec.config.learning_rate = args.learning_rate
    spec.config.lr_warmup_init = args.learning_rate / 10.0
    # Keep evaluation threshold-free so AP is comparable across runs. The
    # speed_optimized deployment threshold is applied only when exporting.
    spec.config.nms_configs.score_thresh = 0.0
    spec.config.nms_configs.iou_thresh = args.iou_threshold
    spec.config.nms_configs.max_output_size = args.max_detections

    def load_split(split: str) -> Any:
        split_dir = dataset / split
        loader = object_detector.DataLoader.from_pascal_voc(
            images_dir=str(split_dir),
            annotations_dir=str(split_dir),
            label_map=list(CLASSES),
        )
        patch_generated_coco_json(loader)
        return loader

    train_data = load_split("train")
    valid_data = load_split("valid")
    test_data = load_split("test")
    train_steps = len(train_data) // args.global_batch_size
    spec.config.save_freq = train_steps * args.checkpoint_every_epochs
    run_config = {
        **vars(args),
        "dataset": str(dataset),
        "output_dir": str(output),
        "gpus": gpus,
        "visible_gpus": [device.name for device in visible_gpus],
        "replicas": spec.ds_strategy.num_replicas_in_sync,
        "per_gpu_batch_size": args.global_batch_size // len(gpus),
        "classes": list(CLASSES),
        "model_name": template.model_name,
        "image_size": list(spec.config.image_size),
        "evaluation_score_threshold": 0.0,
        "export_score_threshold": args.score_threshold,
        "train_steps_per_epoch": train_steps,
        "checkpoint_save_frequency_batches": spec.config.save_freq,
    }
    save_json(output / "run_config.json", run_config)

    if args.resume_weights:
        detector = object_detector.create(
            train_data,
            model_spec=spec,
            validation_data=valid_data,
            epochs=args.epochs,
            batch_size=args.global_batch_size,
            train_whole_model=True,
            do_train=False,
        )
        if not detector.model.built:
            detector.model(
                tf.zeros((1, *spec.config.image_size, 3), dtype=tf.float32),
                training=False,
            )
        status = detector.model.load_weights(
            str(args.resume_weights.expanduser().resolve())
        )
        status.assert_existing_objects_matched()
        if not args.evaluate_only:
            train_ds, train_steps, _ = detector._get_dataset_and_steps(
                train_data, args.global_batch_size, is_training=True
            )
            valid_ds, valid_steps, valid_json = detector._get_dataset_and_steps(
                valid_data, args.global_batch_size, is_training=False
            )
            detector.model_spec.train(
                detector.model,
                train_ds,
                train_steps,
                valid_ds,
                valid_steps,
                args.epochs,
                args.global_batch_size,
                valid_json,
            )
    else:
        detector = object_detector.create(
            train_data,
            model_spec=spec,
            validation_data=valid_data,
            epochs=args.epochs,
            batch_size=args.global_batch_size,
            train_whole_model=True,
            do_train=not args.build_only,
        )

    if not detector.model.built:
        detector.model(
            tf.zeros((1, *spec.config.image_size, 3), dtype=tf.float32),
            training=False,
        )
    if args.build_only:
        payload = {
            "status": "build_ok",
            "replicas": spec.ds_strategy.num_replicas_in_sync,
            "visible_gpus": [device.name for device in visible_gpus],
            "parameters": detector.model.count_params(),
            "train_images": len(train_data),
            "valid_images": len(valid_data),
            "test_images": len(test_data),
        }
        save_json(output / "build_check.json", payload)
        print(json.dumps(payload, indent=2))
        return

    if args.skip_keras_eval:
        keras_valid = {"status": "skipped", "reason": "--skip-keras-eval"}
        keras_test = {"status": "skipped", "reason": "--skip-keras-eval"}
    else:
        keras_valid = detector.evaluate(
            valid_data, batch_size=args.eval_batch_size
        )
        keras_test = detector.evaluate(test_data, batch_size=args.eval_batch_size)
    save_json(output / "valid_metrics.json", keras_valid)
    save_json(output / "test_metrics.json", keras_test)
    detector.model.save_weights(str(output / "final_weights"))

    spec.config.nms_configs.score_thresh = args.score_threshold
    tflite_name = args.tflite_filename or f"{args.model_name}_fp16.tflite"
    detector.export(
        export_dir=str(output),
        tflite_filename=tflite_name,
        quantization_config=QuantizationConfig.for_float16(),
        export_format=ExportFormat.TFLITE,
    )
    tflite_test = detector.evaluate_tflite(
        str(output / tflite_name), test_data
    )
    tflite_valid = detector.evaluate_tflite(
        str(output / tflite_name), valid_data
    )
    save_json(output / "test_tflite_metrics.json", tflite_test)
    save_json(output / "valid_tflite_metrics.json", tflite_valid)
    retained_checkpoints = prune_checkpoints(
        tf, checkpoints, args.keep_checkpoints
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "output_dir": str(output),
                "keras_test": jsonable(keras_test),
                "keras_valid": jsonable(keras_valid),
                "tflite_test": jsonable(tflite_test),
                "tflite_valid": jsonable(tflite_valid),
                "retained_checkpoints": retained_checkpoints,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
