#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
RF_SUFFIX = re.compile(r"\.rf\.[0-9a-f]+$")


def source_id(stem: str) -> str:
    return RF_SUFFIX.sub("", stem)


def paired_stems(folder: Path) -> set[str]:
    images = {
        path.stem
        for path in folder.iterdir()
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
        path
        for path in source.glob(f"{stem}.*")
        if path.suffix.lower() in IMAGE_SUFFIXES | {".xml"}
    ]
    if len(matches) != 2:
        raise RuntimeError(f"Expected image+xml for {source / stem}, got {matches}")
    for path in matches:
        (destination / path.name).symlink_to(path.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create source-disjoint Pascal VOC splits with symlinks."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = args.input.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing output: {output}")

    stems = {split: paired_stems(source / split) for split in ("train", "valid", "test")}
    selected: dict[str, set[str]] = {}
    used_sources: set[str] = set()
    # Holdout priority prevents augmented versions of a test/valid source from
    # entering a lower-priority split.
    for split in ("test", "valid", "train"):
        selected[split] = {
            stem for stem in stems[split] if source_id(stem) not in used_sources
        }
        used_sources.update(source_id(stem) for stem in selected[split])

    for split in ("train", "valid", "test"):
        destination = output / split
        destination.mkdir(parents=True)
        # GNU cp performs the many link operations substantially faster than
        # issuing tens of thousands of individual Python filesystem calls on
        # the school server's storage.
        subprocess.run(
            ["cp", "-a", "-s", f"{source / split}/.", str(destination)],
            check=True,
        )
        excluded = stems[split] - selected[split]
        for path in destination.iterdir():
            if (
                path.stem in excluded
                and path.suffix.lower() in IMAGE_SUFFIXES | {".xml"}
            ):
                path.unlink()

    report = {
        "input": str(source),
        "output": str(output),
        "priority": ["test", "valid", "train"],
        "splits": {
            split: {
                "input_images": len(stems[split]),
                "selected_images": len(selected[split]),
                "removed_images": len(stems[split] - selected[split]),
                "selected_source_ids": len({source_id(s) for s in selected[split]}),
            }
            for split in ("train", "valid", "test")
        },
    }
    (output / "preparation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
