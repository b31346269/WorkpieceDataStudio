#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
RF_SUFFIX = re.compile(r"\.rf\.[0-9a-f]+$")


def source_id(stem):
    return RF_SUFFIX.sub("", stem)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(root):
    report = {"dataset": str(root.resolve()), "splits": {}, "cross_split": {}}
    source_sets = {}
    hashes = defaultdict(list)
    for split in ("train", "valid", "test"):
        folder = root / split
        if not folder.is_dir():
            continue
        images = {p.stem: p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES}
        xmls = {p.stem: p for p in folder.glob("*.xml")}
        classes = Counter()
        size_counts = Counter()
        invalid = []
        empty = []
        object_count = 0
        for stem, xml_path in xmls.items():
            try:
                tree = ET.parse(xml_path)
                xml_root = tree.getroot()
                width = int(float(xml_root.findtext("size/width", "0")))
                height = int(float(xml_root.findtext("size/height", "0")))
                size_counts[f"{width}x{height}"] += 1
                objects = xml_root.findall("object")
                if not objects:
                    empty.append(xml_path.name)
                for index, obj in enumerate(objects):
                    name = (obj.findtext("name") or "").strip()
                    classes[name] += 1
                    object_count += 1
                    box = obj.find("bndbox")
                    coords = {
                        key: float(box.findtext(key, "nan"))
                        for key in ("xmin", "ymin", "xmax", "ymax")
                    }
                    if not (
                        0 <= coords["xmin"] < coords["xmax"] <= width
                        and 0 <= coords["ymin"] < coords["ymax"] <= height
                    ):
                        invalid.append({"file": xml_path.name, "object": index, **coords})
            except Exception as exc:
                invalid.append({"file": xml_path.name, "parse_error": repr(exc)})
        for stem, image_path in images.items():
            hashes[sha256(image_path)].append(f"{split}/{image_path.name}")
        sources = {source_id(stem) for stem in images}
        source_sets[split] = sources
        source_multiplicity = Counter(source_id(stem) for stem in images)
        report["splits"][split] = {
            "images": len(images),
            "xmls": len(xmls),
            "objects": object_count,
            "classes": dict(sorted(classes.items())),
            "xml_sizes": dict(size_counts.most_common()),
            "images_without_xml": sorted(set(images) - set(xmls)),
            "xml_without_image": sorted(set(xmls) - set(images)),
            "empty_annotations": empty,
            "invalid_boxes": invalid,
            "unique_source_ids": len(sources),
            "augmentation_versions_per_source": dict(sorted(Counter(source_multiplicity.values()).items())),
        }
    split_names = sorted(source_sets)
    for i, left in enumerate(split_names):
        for right in split_names[i + 1 :]:
            overlap = sorted(source_sets[left] & source_sets[right])
            report["cross_split"][f"{left}_vs_{right}_source_ids"] = {
                "count": len(overlap),
                "examples": overlap[:30],
            }
    duplicate_groups = [paths for paths in hashes.values() if len(paths) > 1]
    report["exact_duplicate_image_groups"] = duplicate_groups[:100]
    report["exact_duplicate_image_group_count"] = len(duplicate_groups)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.dataset)
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
