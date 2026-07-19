from __future__ import annotations

import io
import tempfile
import time
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from workpiece_studio import storage
from workpiece_studio.exporter import export_yolov8
from workpiece_studio.jobs import GenerationJobs
from workpiece_studio.schemas import GenerationRequest
from workpiece_studio.storage import ProjectStore


def image_bytes() -> io.BytesIO:
    buffer = io.BytesIO()
    Image.new("RGB", (800, 600), (125, 132, 138)).save(buffer, "JPEG")
    buffer.seek(0)
    return buffer


def yolo_zip_bytes(unsafe: bool = False) -> io.BytesIO:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        if unsafe:
            archive.writestr("../escape.txt", "no")
        archive.writestr("train/images/sample.jpg", image_bytes().getvalue())
        archive.writestr(
            "train/labels/sample.txt",
            "0 0.25 0.25 0.1 0.1\n1 0.5 0.5 0.2 0.2\n2 0.7 0.7 0.2 0.3\n",
        )
        archive.writestr("data.yaml", "names: [hole, screw, tool]\n")
    buffer.seek(0)
    return buffer


class CoreWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        storage.PROJECTS = Path(self.temporary.name) / "projects"
        self.store = ProjectStore()
        self.project = self.store.create_project(
            "integration",
            ["hole", "screw", "tool"],
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_import_review_and_export(self) -> None:
        summary = self.store.import_yolo_zip(
            self.project["id"],
            "source.zip",
            yolo_zip_bytes(),
        )
        self.assertEqual(summary["image_count"], 1)
        self.assertEqual(summary["annotation_count"], 3)
        candidate = self.store.create_candidate(
            self.project["id"],
            Image.new("RGB", (640, 640), (80, 90, 100)),
            {"provider": "diffusers", "training_eligible": True, "seed": 7},
            [
                {
                    "id": "box1",
                    "class_id": 0,
                    "x1": 100,
                    "y1": 120,
                    "x2": 180,
                    "y2": 200,
                    "confidence": None,
                    "source": "manual",
                }
            ],
        )
        self.store.update_candidate(
            self.project["id"],
            candidate["id"],
            "approved",
            candidate["boxes"],
            {
                "workpiece_geometry": True,
                "holes_realistic": True,
                "screws_realistic": True,
                "tool_separate": True,
            },
        )
        output, report = export_yolov8(self.store, self.project["id"])
        self.assertEqual(report["candidate_count"], 1)
        with zipfile.ZipFile(output) as archive:
            names = set(archive.namelist())
            self.assertIn("data.yaml", names)
            self.assertIn("generation_manifest.json", names)
            label_name = next(name for name in names if name.endswith(".txt"))
            self.assertTrue(archive.read(label_name).decode().startswith("0 "))

    def test_mock_generation_is_visibly_marked_and_not_exportable(self) -> None:
        reference = self.store.save_reference(
            self.project["id"],
            "reference.jpg",
            image_bytes(),
        )
        jobs = GenerationJobs(self.store)
        job = jobs.submit(
            self.project["id"],
            GenerationRequest(
                reference_id=reference["id"],
                prompt="realistic industrial motor housing",
                count=2,
                provider="mock",
                prelabel=False,
            ),
        )
        deadline = time.time() + 10
        while time.time() < deadline:
            current = jobs.get(job["id"])
            if current["status"] in {"completed", "failed"}:
                break
            time.sleep(0.05)
        self.assertEqual(current["status"], "completed", current.get("error"))
        candidates = self.store.list_candidates(self.project["id"])
        self.assertEqual(len(candidates), 2)
        self.assertTrue(
            all(not item["generation"]["training_eligible"] for item in candidates)
        )
        for candidate in candidates:
            self.store.update_candidate(
                self.project["id"],
                candidate["id"],
                "approved",
                [],
                {
                    "workpiece_geometry": True,
                    "holes_realistic": True,
                    "screws_realistic": True,
                    "tool_separate": True,
                },
            )
        with self.assertRaisesRegex(ValueError, "No approved real-generation"):
            export_yolov8(self.store, self.project["id"])

    def test_zip_traversal_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsafe ZIP entry"):
            self.store.import_yolo_zip(
                self.project["id"],
                "unsafe.zip",
                yolo_zip_bytes(unsafe=True),
            )

    def test_duplicate_model_is_reused_and_can_be_deleted(self) -> None:
        payload = io.BytesIO(b"model" * 1024)
        first = self.store.save_yolo_model(self.project["id"], "best.pt", payload)
        second = self.store.save_yolo_model(
            self.project["id"],
            "best.pt",
            io.BytesIO(b"model" * 1024),
        )
        self.assertEqual(first["id"], second["id"])
        self.assertTrue(second["duplicate"])
        self.assertEqual(len(self.store.list_yolo_models(self.project["id"])), 1)
        self.store.delete_yolo_model(self.project["id"], first["id"])
        self.assertEqual(self.store.list_yolo_models(self.project["id"]), [])

    def test_project_can_be_deleted(self) -> None:
        deleted = self.store.delete_project(self.project["id"])
        self.assertEqual(deleted["id"], self.project["id"])
        with self.assertRaises(FileNotFoundError):
            self.store.get_project(self.project["id"])


if __name__ == "__main__":
    unittest.main()
