"""Lightweight pipeline checks; no model download or GPU needed."""

import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from segment_images import load_objects, mask_metrics, save_previews


class SegmentationTests(unittest.TestCase):
    def test_mask_metrics_detect_background_spill(self):
        mask = np.zeros((10, 20), dtype=bool)
        mask[2:8, 3:12] = True
        metrics = mask_metrics(mask, [3, 2, 9, 8], 0.9)
        self.assertEqual(metrics["foreground_pixels"], 54)
        self.assertAlmostEqual(metrics["outside_box_fraction"], 1 / 3)
        self.assertIn("mask extends beyond prompt box", metrics["flags"])

    def test_empty_mask_flagged(self):
        metrics = mask_metrics(np.zeros((10, 20), dtype=bool), [0, 0, 20, 10], 0.2)
        self.assertIn("empty mask", metrics["flags"])
        self.assertIn("low predicted quality", metrics["flags"])

    def test_foreground_alpha_preserves_mask_holes_and_original_rgb(self):
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary) / "object"
            rgb = np.full((20, 30, 3), [70, 120, 200], dtype=np.uint8)
            mask = np.zeros((20, 30), dtype=bool)
            mask[2:18, 3:27] = True
            mask[8:12, 10:20] = False
            save_previews(Image.fromarray(rgb), mask, [3, 2, 27, 18], prefix)
            with Image.open(f"{prefix}_foreground.png") as image:
                rgba = np.array(image)
            np.testing.assert_array_equal(rgba[:, :, 3] > 0, mask)
            np.testing.assert_array_equal(rgba[:, :, :3][mask], rgb[mask])

    def test_selection_filters_boxes_and_diversifies_categories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (root / "samples.csv").open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["image_id", "label_id", "category", "path"])
                writer.writerows([["one", "/a", "Apple", "one.jpg"],
                                  ["two", "/b", "Bird", "two.jpg"]])
            fields = ["ImageID", "LabelName", "Confidence", "XMin", "YMin", "XMax", "YMax",
                      "IsGroupOf", "IsDepiction", "IsInside", "IsOccluded", "IsTruncated"]
            with (root / "boxes.csv").open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(fields)
                writer.writerows([
                    ["one", "/a", 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                    ["one", "/a", 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                    ["two", "/b", 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                    ["two", "/b", 1, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                    ["two", "/b", 1, 0, 0, 1, 1, 0, 1, 0, 0, 0],
                    ["two", "/b", 1, 0, 0, 1, 1, 0, 0, 1, 0, 0],
                ])
            objects, skipped = load_objects(root, 42, set())
            self.assertEqual(skipped, 3)
            self.assertEqual(len(objects), 3)
            self.assertEqual({row["category"] for row in objects[:2]}, {"Apple", "Bird"})
            self.assertEqual(objects, load_objects(root, 42, set())[0])
            with self.assertRaises(ValueError):
                load_objects(root, 42, {"nonexistent"})


if __name__ == "__main__":
    unittest.main()
