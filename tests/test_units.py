"""Unit tests for colour naming, descriptions, tracking and the store.

Run with:  python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from objectlog import colors, describe, labels  # noqa: E402
from objectlog.store import Store  # noqa: E402
from objectlog.tracker import Detection, Tracker, iou  # noqa: E402


def solid(rgb, size=64):
    return np.full((size, size, 3), rgb, dtype=np.uint8)


class TestColors(unittest.TestCase):
    def test_primary_colours(self):
        cases = {
            (30, 90, 220): "blue",
            (210, 40, 40): "red",
            (40, 180, 70): "green",
            (240, 220, 40): "yellow",
        }
        for rgb, expected in cases.items():
            self.assertEqual(colors.dominant_color(solid(rgb)), expected, msg=str(rgb))

    def test_greyscale_named_by_brightness(self):
        self.assertEqual(colors.dominant_color(solid((10, 10, 12))), "black")
        self.assertEqual(colors.dominant_color(solid((128, 130, 129))), "grey")
        self.assertEqual(colors.dominant_color(solid((245, 245, 245))), "white")

    def test_dark_and_brown_special_cases(self):
        self.assertEqual(colors.dominant_color(solid((110, 60, 15))), "brown")
        self.assertTrue(colors.dominant_color(solid((10, 20, 70))).endswith("blue"))

    def test_orange_is_only_orange_when_vivid(self):
        # A traffic cone is orange; a beige coat sits in the same hue band but
        # is far too washed out to call orange.
        self.assertEqual(colors.dominant_color(solid((255, 140, 0))), "orange")
        self.assertEqual(colors.dominant_color(solid((143, 119, 97))), "brown")

    def test_analyse_returns_a_swatch_matching_the_name(self):
        # The dot shown in the UI must come from the pixels that won, not the
        # mean of the whole patch -- otherwise "orange" gets a grey dot.
        patch = np.full((80, 80, 3), (40, 40, 45), dtype=np.uint8)  # dark bg
        patch[10:70, 10:70] = (40, 90, 220)                          # blue object
        name, rgb = colors.analyse(patch)
        self.assertEqual(name, "blue")
        self.assertIsNotNone(rgb)
        # The swatch must actually be blue-dominant.
        self.assertGreater(rgb[2], rgb[0] + 60, msg=str(rgb))

    def test_analyse_returns_nothing_when_it_cannot_tell(self):
        rng = np.random.default_rng(11)
        noise = rng.integers(0, 256, size=(80, 80, 3), dtype=np.uint8)
        self.assertEqual(colors.analyse(noise), (None, None))
        self.assertEqual(colors.analyse(None), (None, None))

    def test_busy_patch_returns_none(self):
        rng = np.random.default_rng(7)
        noise = rng.integers(0, 256, size=(80, 80, 3), dtype=np.uint8)
        # Random colour confetti has no dominant hue -- better to say nothing.
        self.assertIsNone(colors.dominant_color(noise))

    def test_tiny_patch_returns_none(self):
        self.assertIsNone(colors.dominant_color(solid((0, 0, 255), size=3)))
        self.assertIsNone(colors.dominant_color(np.zeros((0, 0, 3), dtype=np.uint8)))

    def test_hair_tone(self):
        self.assertEqual(colors.hair_tone(solid((18, 18, 20))), "dark")
        self.assertEqual(colors.hair_tone(solid((225, 225, 228))), "light")


class TestDescribe(unittest.TestCase):
    def _person_frame(self):
        """A frame with dark hair on top and a blue torso beneath it."""
        frame = np.full((400, 300, 3), 200, dtype=np.uint8)
        box = (100.0, 40.0, 200.0, 360.0)
        # hair band: 2%-18% of the box height
        frame[int(40 + 320 * 0.02):int(40 + 320 * 0.18), 130:170] = (20, 18, 22)
        # torso band: 24%-58%
        frame[int(40 + 320 * 0.24):int(40 + 320 * 0.58), 122:178] = (35, 85, 210)
        return frame, box

    def test_person_gets_hair_and_clothing(self):
        frame, box = self._person_frame()
        result = describe.describe(frame, "person", box)
        self.assertEqual(result["attributes"].get("hair"), "dark")
        self.assertEqual(result["attributes"].get("wearing"), "blue")
        self.assertEqual(result["text"], "Person with dark hair wearing blue")
        self.assertTrue(result["swatch"].startswith("#"))
        self.assertEqual(len(result["swatch"]), 7)

    def test_object_gets_colour_prefix(self):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        frame[40:160, 40:160] = (200, 30, 30)
        result = describe.describe(frame, "car", (40.0, 40.0, 160.0, 160.0))
        self.assertEqual(result["text"], "Red car")

    def test_unknown_colour_falls_back_to_plain_label(self):
        rng = np.random.default_rng(3)
        frame = rng.integers(0, 256, size=(200, 200, 3), dtype=np.uint8)
        result = describe.describe(frame, "laptop", (20.0, 20.0, 180.0, 180.0))
        self.assertEqual(result["text"], "Laptop")

    def test_summarise_round_trips(self):
        self.assertEqual(
            describe.summarise("person", {"hair": "dark", "wearing": "blue"}),
            "Person with dark hair wearing blue")
        self.assertEqual(describe.summarise("cup", {"colour": "green"}),
                         "Green cup")


class TestLabels(unittest.TestCase):
    def test_known_and_unknown_categories(self):
        self.assertEqual(labels.category_for("person"), "People")
        self.assertEqual(labels.category_for("Cell Phone"), "Electronics")
        self.assertEqual(labels.category_for("flux capacitor"), "Other")

    def test_every_coco_class_has_a_category(self):
        # A COCO class landing in "Other" means the taxonomy has a hole.
        missing = [c for c in labels.COCO_CLASSES
                   if labels.category_for(c) == "Other"]
        self.assertEqual(missing, [])

    def test_unknown_categories_sort_last(self):
        ordered = sorted(["Zebras", "People", "Other"],
                         key=labels.category_sort_key)
        self.assertEqual(ordered[0], "People")
        self.assertEqual(ordered[-1], "Zebras")


class TestTracker(unittest.TestCase):
    def test_iou_basics(self):
        self.assertAlmostEqual(iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertEqual(iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)

    def test_same_object_keeps_one_track(self):
        tracker = Tracker(iou_threshold=0.3, max_missing=2, min_hits=2)
        ids = set()
        for offset in range(6):
            box = (10.0 + offset, 10.0, 60.0 + offset, 80.0)
            active, _ = tracker.update([Detection("person", 0.9, box)])
            ids.update(t.track_id for t in active)
        self.assertEqual(len(ids), 1, "a drifting object should stay one track")
        self.assertEqual(len(tracker.confirmed()), 1)

    def test_distinct_objects_get_distinct_tracks(self):
        tracker = Tracker()
        active, _ = tracker.update([
            Detection("person", 0.9, (0, 0, 50, 50)),
            Detection("person", 0.9, (300, 300, 350, 350)),
        ])
        self.assertEqual(len({t.track_id for t in active}), 2)

    def test_different_labels_never_merge(self):
        tracker = Tracker()
        tracker.update([Detection("person", 0.9, (0, 0, 50, 50))])
        active, _ = tracker.update([Detection("dog", 0.9, (0, 0, 50, 50))])
        self.assertEqual(len(tracker.tracks), 2)

    def test_track_closes_after_max_missing(self):
        tracker = Tracker(max_missing=2, min_hits=1)
        tracker.update([Detection("cup", 0.8, (0, 0, 20, 20))])
        closed_total = []
        for _ in range(4):
            _, closed = tracker.update([])
            closed_total.extend(closed)
        self.assertEqual(len(closed_total), 1)
        self.assertEqual(tracker.tracks, {})

    def test_min_hits_filters_flickers(self):
        tracker = Tracker(min_hits=3)
        tracker.update([Detection("cup", 0.8, (0, 0, 20, 20))])
        self.assertEqual(tracker.confirmed(), [])
        tracker.update([Detection("cup", 0.8, (0, 0, 20, 20))])
        tracker.update([Detection("cup", 0.8, (0, 0, 20, 20))])
        self.assertEqual(len(tracker.confirmed()), 1)


class TestCamera(unittest.TestCase):
    def test_synthetic_source_produces_moving_rgb_frames(self):
        from objectlog.camera import SyntheticCamera

        camera = SyntheticCamera(320, 240)
        first, second = camera.read(), camera.read()
        self.assertEqual(first.shape, (240, 320, 3))
        self.assertEqual(first.dtype, np.uint8)
        self.assertFalse(np.array_equal(first, second), "the scene should move")

    def test_rotation(self):
        from objectlog.camera import rotate

        frame = np.zeros((10, 20, 3), dtype=np.uint8)
        frame[0, 0] = (255, 0, 0)
        self.assertEqual(rotate(frame, 0).shape, (10, 20, 3))
        self.assertEqual(rotate(frame, 90).shape, (20, 10, 3))
        self.assertEqual(rotate(frame, 180).shape, (10, 20, 3))
        self.assertEqual(rotate(frame, 270).shape, (20, 10, 3))
        # 180 twice is the identity -- catches a flipped axis.
        self.assertTrue(np.array_equal(rotate(rotate(frame, 180), 180), frame))
        self.assertTrue(np.array_equal(rotate(rotate(frame, 90), 270), frame))
        with self.assertRaises(ValueError):
            rotate(frame, 45)

    def test_folder_source_loops_over_images(self):
        from PIL import Image

        from objectlog.camera import CameraError, FolderCamera

        directory = tempfile.mkdtemp(prefix="objectlog-folder-")
        try:
            with self.assertRaises(CameraError):
                FolderCamera(directory)  # empty folder
            for index, colour in enumerate([(255, 0, 0), (0, 255, 0)]):
                Image.new("RGB", (40, 30), colour).save(
                    os.path.join(directory, f"{index}.png"))

            camera = FolderCamera(directory)
            first, second, third = camera.read(), camera.read(), camera.read()
            self.assertEqual(first.shape, (30, 40, 3))
            self.assertFalse(np.array_equal(first, second))
            self.assertTrue(np.array_equal(first, third), "should wrap around")

            with self.assertRaises(CameraError):
                FolderCamera(os.path.join(directory, "nope"))
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_unknown_source_is_rejected(self):
        from objectlog import config as config_mod
        from objectlog.camera import CameraError, open_camera

        cfg = config_mod.load(None)
        cfg.set("camera.source", "telepathy")
        with self.assertRaises(CameraError):
            open_camera(cfg)


class TestStore(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="objectlog-test-")
        self.store = Store(db_path=os.path.join(self.dir, "log.db"),
                           snapshot_dir=os.path.join(self.dir, "snaps"),
                           max_entries=5)

    def tearDown(self):
        self.store.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def _add(self, label="person", **kwargs):
        now = time.time()
        return self.store.add_sighting(
            label=label, description=kwargs.pop("description", "Person"),
            confidence=kwargs.pop("confidence", 0.9),
            first_seen=now, last_seen=now, **kwargs)

    def test_add_and_read_back(self):
        entry_id = self._add(attributes={"wearing": "blue"}, swatch="#2255cc")
        rows = self.store.sightings()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], entry_id)
        self.assertEqual(rows[0]["category"], "People")
        self.assertEqual(rows[0]["attributes"], {"wearing": "blue"})

    def test_snapshot_written_and_served_path(self):
        frame = np.full((120, 160, 3), 90, dtype=np.uint8)
        self._add(box=[10, 10, 100, 100], frame=frame)
        row = self.store.sightings()[0]
        self.assertTrue(row["snapshot"].startswith("/snapshots/"))
        path = os.path.join(self.dir, "snaps", os.path.basename(row["snapshot"]))
        self.assertTrue(os.path.exists(path))
        self.assertGreater(os.path.getsize(path), 500)

    def test_touch_updates_description_and_last_seen(self):
        entry_id = self._add()
        later = time.time() + 10
        self.store.touch(entry_id, last_seen=later, confidence=0.99,
                         description="Person wearing red",
                         attributes={"wearing": "red"})
        row = self.store.sightings()[0]
        self.assertEqual(row["description"], "Person wearing red")
        self.assertGreaterEqual(row["duration"], 9)

    def test_prune_drops_oldest_and_their_snapshots(self):
        frame = np.full((60, 60, 3), 120, dtype=np.uint8)
        for _ in range(9):
            self._add(box=[5, 5, 50, 50], frame=frame)
        self.assertEqual(self.store.stats()["total"], 5)
        snaps = [f for f in os.listdir(os.path.join(self.dir, "snaps"))
                 if f.endswith(".jpg")]
        self.assertEqual(len(snaps), 5, "pruned rows must take their images too")

    def test_categories_and_label_counts(self):
        self._add("person")
        self._add("person")
        self._add("dog", description="Dog")
        categories = {c["name"]: c["count"] for c in self.store.categories()}
        self.assertEqual(categories, {"People": 2, "Animals": 1})
        self.assertEqual(self.store.label_counts()[0]["label"], "person")

    def test_pagination_and_filters(self):
        ids = [self._add() for _ in range(5)]
        newest = self.store.sightings(limit=2)
        self.assertEqual([r["id"] for r in newest], [ids[-1], ids[-2]])
        older = self.store.sightings(before_id=ids[-2], limit=2)
        self.assertEqual([r["id"] for r in older], [ids[-3], ids[-4]])
        self.assertEqual(self.store.sightings(category="Animals"), [])

    def test_clear_removes_rows_and_images(self):
        frame = np.full((60, 60, 3), 10, dtype=np.uint8)
        self._add(box=[1, 1, 50, 50], frame=frame)
        self.assertEqual(self.store.clear(), 1)
        self.assertEqual(self.store.stats()["total"], 0)
        self.assertEqual([f for f in os.listdir(os.path.join(self.dir, "snaps"))
                          if f.endswith(".jpg")], [])


if __name__ == "__main__":
    unittest.main()
