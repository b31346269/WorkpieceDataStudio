#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
RF_SUFFIX = re.compile(r"\.rf\.[0-9a-f]+$")


def source_id(value: Path | str) -> str:
    stem = value.stem if isinstance(value, Path) else value
    return RF_SUFFIX.sub("", stem)


def files(folder: Path, suffixes: set[str]) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in suffixes)


def paired_stems(folder: Path) -> set[str]:
    images = {p.stem for p in files(folder, IMAGE_SUFFIXES)}
    xmls = {p.stem for p in files(folder, {".xml"})}
    if images != xmls:
        raise RuntimeError(
            f"Unpaired files in {folder}: "
            f"images-only={sorted(images - xmls)[:10]}, "
            f"xml-only={sorted(xmls - images)[:10]}"
        )
    return images


def clamp_boxes(dataset: Path) -> dict:
    changed_files = 0
    changed_boxes = 0
    removed_boxes = 0
    backup_root = dataset / ".xml_before_clamp"
    for split in ("train", "valid", "test"):
        folder = dataset / split
        if not folder.is_dir():
            continue
        for xml_path in files(folder, {".xml"}):
            tree = ET.parse(xml_path)
            root = tree.getroot()
            width = int(float(root.findtext("size/width", "0")))
            height = int(float(root.findtext("size/height", "0")))
            changed = False
            for obj in list(root.findall("object")):
                box = obj.find("bndbox")
                if box is None:
                    continue
                values = {}
                for key in ("xmin", "ymin", "xmax", "ymax"):
                    values[key] = int(round(float(box.findtext(key, "0"))))
                clipped = {
                    "xmin": max(0, min(values["xmin"], width - 1)),
                    "ymin": max(0, min(values["ymin"], height - 1)),
                    "xmax": max(1, min(values["xmax"], width)),
                    "ymax": max(1, min(values["ymax"], height)),
                }
                if clipped["xmin"] >= clipped["xmax"] or clipped["ymin"] >= clipped["ymax"]:
                    # A zero-area or fully out-of-frame box has no usable target.
                    # Preserve the original XML in the backup and remove only the
                    # invalid object instead of aborting the complete preparation.
                    root.remove(obj)
                    changed = True
                    removed_boxes += 1
                    continue
                if clipped != values:
                    for key, value in clipped.items():
                        box.find(key).text = str(value)
                    changed = True
                    changed_boxes += 1
            if changed:
                backup = backup_root / split / xml_path.name
                backup.parent.mkdir(parents=True, exist_ok=True)
                if not backup.exists():
                    shutil.copy2(xml_path, backup)
                tree.write(xml_path, encoding="utf-8", xml_declaration=False)
                changed_files += 1
    return {
        "changed_xml_files": changed_files,
        "changed_boxes": changed_boxes,
        "removed_boxes": removed_boxes,
    }


def remove_split_leakage(dataset: Path, extra_valid_sources: set[str] | None = None) -> dict:
    train = dataset / "train"
    valid = dataset / "valid"
    paired_stems(train)
    paired_stems(valid)
    valid_sources = {source_id(p) for p in files(valid, IMAGE_SUFFIXES)}
    valid_sources.update(extra_valid_sources or set())
    move_stems = {
        p.stem for p in files(train, IMAGE_SUFFIXES) if source_id(p) in valid_sources
    }
    moved = 0
    for stem in sorted(move_stems):
        for source in sorted(train.glob(f"{stem}.*")):
            if source.suffix.lower() not in IMAGE_SUFFIXES | {".xml"}:
                continue
            destination = valid / source.name
            if destination.exists():
                raise FileExistsError(destination)
            source.replace(destination)
            moved += 1
    return {"moved_pairs": len(move_stems), "moved_files": moved}


def link_pair(source_folder: Path, stem: str, destination: Path) -> None:
    matches = [
        p for p in source_folder.glob(f"{stem}.*")
        if p.suffix.lower() in IMAGE_SUFFIXES | {".xml"}
    ]
    if len(matches) != 2:
        raise RuntimeError(f"Expected image+xml for {source_folder / stem}, got {matches}")
    for source in matches:
        target = destination / source.name
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Dataset filename collision: {target}")
        target.symlink_to(source.resolve())


def build_mixed(new_root: Path, tool2_root: Path, output: Path, rehearsal_images: int, seed: int) -> dict:
    if output.exists():
        shutil.rmtree(output)
    for split in ("train", "valid", "test"):
        (output / split).mkdir(parents=True)

    new_train_stems = paired_stems(new_root / "train")
    new_valid_stems = paired_stems(new_root / "valid")
    tool_train_stems = paired_stems(tool2_root / "train")
    tool_test_stems = paired_stems(tool2_root / "test")

    for stem in sorted(new_train_stems):
        link_pair(new_root / "train", stem, output / "train")
    for stem in sorted(new_valid_stems):
        link_pair(new_root / "valid", stem, output / "valid")
    for stem in sorted(tool_test_stems):
        link_pair(tool2_root / "test", stem, output / "test")

    blocked_sources = {
        source_id(stem) for stem in new_valid_stems
    } | {
        source_id(stem) for stem in tool_test_stems
    }
    candidates: dict[str, list[str]] = {}
    for stem in tool_train_stems:
        sid = source_id(stem)
        if sid not in blocked_sources:
            candidates.setdefault(sid, []).append(stem)
    rng = random.Random(seed)
    source_ids = sorted(candidates)
    rng.shuffle(source_ids)
    selected: list[str] = []
    for sid in source_ids:
        selected.extend(sorted(candidates[sid]))
        if len(selected) >= rehearsal_images:
            break
    for stem in selected:
        link_pair(tool2_root / "train", stem, output / "train")

    counts = {
        "new_train_images": len(new_train_stems),
        "new_valid_images": len(new_valid_stems),
        "tool2_rehearsal_images": len(selected),
        "tool2_test_images": len(tool_test_stems),
        "rehearsal_source_groups": len({source_id(s) for s in selected}),
        "requested_rehearsal_images": rehearsal_images,
        "seed": seed,
    }
    (output / "preparation_report.json").write_text(
        json.dumps(counts, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-dataset", type=Path, required=True)
    parser.add_argument("--tool2-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rehearsal-images", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260721)
    args = parser.parse_args()

    new_root = args.new_dataset.resolve()
    tool2_root = args.tool2_dataset.resolve()
    output = args.output.resolve()
    tool2_test_sources = {
        source_id(p) for p in files(tool2_root / "test", IMAGE_SUFFIXES)
    }
    report = {
        "new_box_clamp": clamp_boxes(new_root),
        "tool2_box_clamp": clamp_boxes(tool2_root),
        "split_cleanup": remove_split_leakage(new_root, tool2_test_sources),
    }
    report["mixed_dataset"] = build_mixed(
        new_root, tool2_root, output, args.rehearsal_images, args.seed
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
