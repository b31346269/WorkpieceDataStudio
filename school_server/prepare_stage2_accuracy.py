#!/usr/bin/env python3
"""Build a source-disjoint stage-2 dataset without modifying source datasets."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
RF_SUFFIX = re.compile(r"\.rf\.[0-9a-f]+$")


def source_id(stem: str) -> str:
    return RF_SUFFIX.sub("", stem)


def paired_stems(folder: Path) -> set[str]:
    images = {
        path.stem for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    xmls = {path.stem for path in folder.glob("*.xml")}
    if images != xmls:
        raise RuntimeError(
            f"Unpaired files in {folder}: images-only={sorted(images-xmls)[:10]}, "
            f"xml-only={sorted(xmls-images)[:10]}"
        )
    return images


def link_pair(source: Path, stem: str, destination: Path) -> None:
    matches = [
        path for path in source.glob(f"{stem}.*")
        if path.suffix.lower() in IMAGE_SUFFIXES | {".xml"}
    ]
    if len(matches) != 2:
        raise RuntimeError(f"Expected image+xml for {source / stem}, got {matches}")
    for path in matches:
        target = destination / path.name
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Dataset filename collision: {target}")
        target.symlink_to(path.resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-dataset", type=Path, required=True)
    parser.add_argument("--tool2-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rehearsal-images", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260801)
    args = parser.parse_args()

    new_root = args.new_dataset.resolve()
    tool2_root = args.tool2_dataset.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing output: {output}")
    for split in ("train", "valid", "test"):
        (output / split).mkdir(parents=True)

    new_train = paired_stems(new_root / "train")
    new_valid = paired_stems(new_root / "valid")
    tool_train = paired_stems(tool2_root / "train")
    tool_test = paired_stems(tool2_root / "test")

    tool_test_sources = {source_id(stem) for stem in tool_test}
    clean_valid = {
        stem for stem in new_valid if source_id(stem) not in tool_test_sources
    }
    valid_sources = {source_id(stem) for stem in clean_valid}
    clean_train = {
        stem for stem in new_train
        if source_id(stem) not in valid_sources | tool_test_sources
    }

    rehearsal_groups: dict[str, list[str]] = {}
    blocked_sources = valid_sources | tool_test_sources
    for stem in tool_train:
        sid = source_id(stem)
        if sid not in blocked_sources:
            rehearsal_groups.setdefault(sid, []).append(stem)
    group_ids = sorted(rehearsal_groups)
    random.Random(args.seed).shuffle(group_ids)
    rehearsal: list[str] = []
    for sid in group_ids:
        rehearsal.extend(sorted(rehearsal_groups[sid]))
        if len(rehearsal) >= args.rehearsal_images:
            break

    for stem in sorted(clean_train):
        link_pair(new_root / "train", stem, output / "train")
    for stem in rehearsal:
        link_pair(tool2_root / "train", stem, output / "train")
    for stem in sorted(clean_valid):
        link_pair(new_root / "valid", stem, output / "valid")
    for stem in sorted(tool_test):
        link_pair(tool2_root / "test", stem, output / "test")

    report = {
        "new_train_images": len(clean_train),
        "new_train_excluded": len(new_train - clean_train),
        "new_valid_images": len(clean_valid),
        "new_valid_excluded_for_tool2_test_overlap": len(new_valid - clean_valid),
        "tool2_rehearsal_images": len(rehearsal),
        "tool2_rehearsal_source_groups": len({source_id(s) for s in rehearsal}),
        "tool2_test_images": len(tool_test),
        "requested_rehearsal_images": args.rehearsal_images,
        "seed": args.seed,
    }
    (output / "preparation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
