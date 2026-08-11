"""Tests for the Hailo output decoder.

No accelerator is needed: the decoder is a pure function, and these feed it the
shapes HailoRT is known to hand back. The device I/O around it cannot be tested
without hardware, which is exactly why the decoding is kept separate from it.

Hailo's on-chip NMS emits, per class, rows of
    (y_min, x_min, y_max, x_max, score)
normalised 0-1 -- note y before x, the reverse of everything else in this
project. Getting that backwards produces boxes that look plausible and are
wrong, so it is pinned down here.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from objectlog.backends.hailo_backend import (  # noqa: E402
    _iter_class_arrays, decode_nms_output)
from objectlog.labels import COCO_CLASSES  # noqa: E402


def one_person_at(y0, x0, y1, x1, score=0.9, num_classes=80):
    """Per-class lists with a single 'person' (class 0) detection."""
    classes = [np.zeros((0, 5), dtype=np.float32) for _ in range(num_classes)]
    classes[0] = np.array([[y0, x0, y1, x1, score]], dtype=np.float32)
    return classes


class TestOutputShapeNormalising(unittest.TestCase):
    """HailoRT returns this in several different wrappings by version."""

    def test_plain_list_per_class(self):
        raw = one_person_at(0.1, 0.2, 0.5, 0.6)
        self.assertEqual(len(_iter_class_arrays(raw)), 80)

    def test_batched_ndarray(self):
        raw = np.zeros((1, 80, 3, 5), dtype=np.float32)
        self.assertEqual(len(_iter_class_arrays(raw)), 80)

    def test_unbatched_ndarray(self):
        raw = np.zeros((80, 3, 5), dtype=np.float32)
        self.assertEqual(len(_iter_class_arrays(raw)), 80)

    def test_object_array_wrapping_a_batch(self):
        inner = one_person_at(0.1, 0.2, 0.5, 0.6)
        raw = np.empty(1, dtype=object)
        raw[0] = inner
        self.assertEqual(len(_iter_class_arrays(raw)), 80)

    def test_single_block_without_class_dimension(self):
        raw = np.array([[0.1, 0.2, 0.5, 0.6, 0.9]], dtype=np.float32)
        self.assertEqual(len(_iter_class_arrays(raw)), 1)

    def test_unreadable_shapes_yield_nothing(self):
        self.assertEqual(list(_iter_class_arrays(None)), [])
        self.assertEqual(list(_iter_class_arrays(42)), [])
        self.assertEqual(list(_iter_class_arrays("nonsense")), [])


class TestDecode(unittest.TestCase):
    def test_coordinates_are_y_first(self):
        # A box across the top-left: y 10-50%, x 20-60% of a 1000x500 frame.
        raw = one_person_at(0.1, 0.2, 0.5, 0.6)
        found = decode_nms_output(raw, COCO_CLASSES, 0.4, 1000, 500)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].label, "person")
        self.assertAlmostEqual(found[0].confidence, 0.9, places=5)
        # x0,y0,x1,y1 = 0.2*1000, 0.1*500, 0.6*1000, 0.5*500
        for actual, expected in zip(found[0].box, (200.0, 50.0, 600.0, 250.0)):
            self.assertAlmostEqual(actual, expected, delta=0.5)

    def test_confidence_threshold(self):
        raw = one_person_at(0.1, 0.2, 0.5, 0.6, score=0.35)
        self.assertEqual(decode_nms_output(raw, COCO_CLASSES, 0.4, 640, 640), [])
        self.assertEqual(
            len(decode_nms_output(raw, COCO_CLASSES, 0.3, 640, 640)), 1)

    def test_class_index_maps_to_the_right_name(self):
        classes = [np.zeros((0, 5), dtype=np.float32) for _ in range(80)]
        classes[15] = np.array([[0.1, 0.1, 0.4, 0.4, 0.8]], dtype=np.float32)
        found = decode_nms_output(classes, COCO_CLASSES, 0.4, 640, 640)
        self.assertEqual(found[0].label, COCO_CLASSES[15])
        self.assertEqual(found[0].label, "cat")

    def test_letterbox_is_undone(self):
        # A 1280x720 frame letterboxed into 640x640: scale 0.5, 140px of grey
        # top and bottom. A box filling the letterboxed image vertically from
        # 140..500px maps back to the full height of the original frame.
        raw = one_person_at(140 / 640, 0.0, 500 / 640, 1.0)
        found = decode_nms_output(
            raw, COCO_CLASSES, 0.4, 1280, 720,
            scale=0.5, pad_x=0.0, pad_y=140.0,
            input_width=640, input_height=640)
        self.assertEqual(len(found), 1)
        x0, y0, x1, y1 = found[0].box
        self.assertAlmostEqual(y0, 0.0, delta=1.0)
        self.assertAlmostEqual(y1, 720.0, delta=1.0)
        self.assertAlmostEqual(x0, 0.0, delta=1.0)
        self.assertAlmostEqual(x1, 1280.0, delta=1.0)

    def test_stretch_to_fit_uses_a_separate_scale_per_axis(self):
        # With hailo_letterbox off, a 1280x720 frame is squashed into a 640x640
        # input. Undoing that needs a different divisor per axis; using one for
        # both yields boxes that look plausible and are wrong.
        raw = one_person_at(0.25, 0.25, 0.75, 0.75)
        found = decode_nms_output(
            raw, COCO_CLASSES, 0.4, 1280, 720,
            scale=640 / 1280, scale_y=640 / 720,
            input_width=640, input_height=640)
        x0, y0, x1, y1 = found[0].box
        self.assertAlmostEqual(x0, 0.25 * 1280, delta=1.0)
        self.assertAlmostEqual(x1, 0.75 * 1280, delta=1.0)
        self.assertAlmostEqual(y0, 0.25 * 720, delta=1.0)
        self.assertAlmostEqual(y1, 0.75 * 720, delta=1.0)

    def test_bbox_first_layout_is_transposed(self):
        # yolov11m_h10 declares its output as (80, 5, 100): classes, then
        # [y0,x0,y1,x1,score], then up to 100 detections. Each class therefore
        # arrives as (5, N), not (N, 5). Read literally that is five bogus
        # boxes built from coordinate rows, so it must be transposed.
        classes = [np.zeros((5, 100), dtype=np.float32) for _ in range(80)]
        block = np.zeros((5, 100), dtype=np.float32)
        block[:, 0] = [0.1, 0.2, 0.5, 0.6, 0.9]     # first detection
        block[:, 1] = [0.3, 0.3, 0.4, 0.4, 0.7]     # second detection
        classes[0] = block
        found = decode_nms_output(classes, COCO_CLASSES, 0.4, 1000, 500)

        self.assertEqual(len(found), 2, "should read columns, not rows")
        self.assertEqual({d.label for d in found}, {"person"})
        best = max(found, key=lambda d: d.confidence)
        self.assertAlmostEqual(best.confidence, 0.9, places=5)
        for actual, expected in zip(best.box, (200.0, 50.0, 600.0, 250.0)):
            self.assertAlmostEqual(actual, expected, delta=0.5)

    def test_padding_in_a_dense_block_is_dropped(self):
        # Unused detection slots are zeros; a zero score must not become a box.
        classes = [np.zeros((5, 100), dtype=np.float32) for _ in range(80)]
        classes[0][:, 0] = [0.1, 0.1, 0.4, 0.4, 0.95]
        found = decode_nms_output(classes, COCO_CLASSES, 0.4, 640, 640)
        self.assertEqual(len(found), 1)

    def test_boxes_are_clamped_to_the_frame(self):
        # Models sometimes emit slightly out-of-range normalised coordinates.
        raw = one_person_at(-0.05, -0.10, 1.20, 1.05)
        found = decode_nms_output(raw, COCO_CLASSES, 0.4, 800, 600)
        x0, y0, x1, y1 = found[0].box
        self.assertGreaterEqual(x0, 0.0)
        self.assertGreaterEqual(y0, 0.0)
        self.assertLessEqual(x1, 800.0)
        self.assertLessEqual(y1, 600.0)

    def test_degenerate_boxes_are_dropped(self):
        classes = [np.zeros((0, 5), dtype=np.float32) for _ in range(80)]
        classes[0] = np.array([[0.5, 0.5, 0.5, 0.5, 0.9]], dtype=np.float32)
        self.assertEqual(decode_nms_output(classes, COCO_CLASSES, 0.4, 640, 640), [])

    def test_multiple_detections_across_classes(self):
        classes = [np.zeros((0, 5), dtype=np.float32) for _ in range(80)]
        classes[0] = np.array([[0.1, 0.1, 0.3, 0.3, 0.9],
                               [0.5, 0.5, 0.7, 0.7, 0.8]], dtype=np.float32)
        classes[2] = np.array([[0.2, 0.2, 0.6, 0.9, 0.7]], dtype=np.float32)
        found = decode_nms_output(classes, COCO_CLASSES, 0.4, 640, 640)
        self.assertEqual(len(found), 3)
        self.assertEqual(sorted(d.label for d in found),
                         ["car", "person", "person"])

    def test_empty_and_ragged_entries_are_survivable(self):
        classes = [np.zeros((0, 5), dtype=np.float32) for _ in range(80)]
        classes[1] = None                                   # missing
        classes[2] = np.zeros((0, 5), dtype=np.float32)     # empty
        classes[3] = np.array([0.1, 0.1, 0.4, 0.4, 0.9])    # 1-D, not 2-D
        classes[4] = np.array([[0.1, 0.2]])                 # too few columns
        found = decode_nms_output(classes, COCO_CLASSES, 0.4, 640, 640)
        self.assertEqual(len(found), 1, "the 1-D row should still decode")
        self.assertEqual(found[0].label, COCO_CLASSES[3])

    def test_custom_labels(self):
        raw = one_person_at(0.1, 0.2, 0.5, 0.6, num_classes=3)
        found = decode_nms_output(raw, ["widget", "gadget", "doohickey"],
                                  0.4, 640, 640)
        self.assertEqual(found[0].label, "widget")

    def test_unknown_class_index_does_not_crash(self):
        classes = [np.zeros((0, 5), dtype=np.float32) for _ in range(5)]
        classes[4] = np.array([[0.1, 0.1, 0.4, 0.4, 0.9]], dtype=np.float32)
        found = decode_nms_output(classes, ["a", "b"], 0.4, 640, 640)
        self.assertEqual(found[0].label, "class_4")


# The exact set of models `apt install hailo-all` puts on a Pi, taken from a
# real machine. Selection is tested against this rather than invented names.
INSTALLED_ON_A_REAL_PI = [
    "/usr/share/hailo-models/" + name + ".hef" for name in [
        "resnet_v1_50_h10", "resnet_v1_50_h8l", "scrfd_2.5g_h8l",
        "yolov11m_h10", "yolov5n_seg_h10", "yolov5n_seg_h8",
        "yolov5n_seg_h8l_mz", "yolov5s_personface_h8l", "yolov6n_h8",
        "yolov6n_h8l", "yolov8m_h10", "yolov8m_pose_h10", "yolov8s_h8",
        "yolov8s_h8l", "yolov8s_pose_h10", "yolov8s_pose_h8",
        "yolov8s_pose_h8l_pi", "yolox_s_leaky_h8l_rpi",
    ]
]


class TestModelSelection(unittest.TestCase):
    """Picking a detector out of what ships on the Pi.

    The failure this guards against is real: the first version sorted
    alphabetically and chose resnet_v1_50 -- an image classifier -- then
    advised configuring it as the object detector.
    """

    def setUp(self):
        from objectlog.backends.hailo_backend import score_hef, select_hef

        self.score_hef = score_hef
        self.select_hef = select_hef

    def test_classifiers_are_not_detectors(self):
        self.assertLess(self.score_hef("/x/resnet_v1_50_h10.hef"), 0)
        self.assertLess(self.score_hef("/x/mobilenet_v1.hef"), 0)

    def test_pose_and_segmentation_are_not_detectors(self):
        self.assertLess(self.score_hef("/x/yolov8m_pose_h10.hef"), 0)
        self.assertLess(self.score_hef("/x/yolov5n_seg_h10.hef"), 0)

    def test_face_and_single_purpose_models_are_excluded(self):
        # These detect boxes, but their classes are not COCO, so COCO labels
        # would mislabel everything they find.
        self.assertLess(self.score_hef("/x/scrfd_2.5g_h8l.hef"), 0)
        self.assertLess(self.score_hef("/x/yolov5s_personface_h8l.hef"), 0)

    def test_real_detectors_score_positively(self):
        for name in ("yolov11m_h10", "yolov8m_h10", "yolov8s_h8",
                     "yolov6n_h8", "yolox_s_leaky_h8l_rpi"):
            self.assertGreaterEqual(self.score_hef(f"/x/{name}.hef"), 0, name)

    def test_newer_and_larger_models_outrank_older_and_smaller(self):
        self.assertGreater(self.score_hef("/x/yolov11m_h10.hef"),
                           self.score_hef("/x/yolov8m_h10.hef"))
        self.assertGreater(self.score_hef("/x/yolov8m_h10.hef"),
                           self.score_hef("/x/yolov8s_h8.hef"))
        self.assertGreater(self.score_hef("/x/yolov8s_h8.hef"),
                           self.score_hef("/x/yolov6n_h8.hef"))

    def test_picks_the_best_model_for_a_hailo10h(self):
        chosen = self.select_hef(INSTALLED_ON_A_REAL_PI, "HAILO10H")
        self.assertTrue(chosen.endswith("yolov11m_h10.hef"), chosen)

    def test_picks_the_best_model_for_a_hailo8(self):
        chosen = self.select_hef(INSTALLED_ON_A_REAL_PI, "HAILO8")
        self.assertTrue(chosen.endswith("yolov8s_h8.hef"), chosen)

    def test_h8_does_not_match_an_h8l_build(self):
        # '_h8' is a prefix of '_h8l'; a naive substring match would load a
        # model the chip cannot run.
        chosen = self.select_hef(INSTALLED_ON_A_REAL_PI, "HAILO8L")
        self.assertTrue(chosen.endswith("_h8l.hef"), chosen)
        self.assertFalse(self.select_hef(
            ["/x/yolov8s_h8l.hef"], "HAILO8"), "an h8l build is not h8-loadable")

    def test_no_suitable_model_returns_nothing(self):
        # Better to say "none" than to hand back something that cannot load.
        self.assertIsNone(self.select_hef(
            ["/x/resnet_v1_50_h10.hef", "/x/scrfd_2.5g_h8l.hef"], "HAILO10H"))
        self.assertIsNone(self.select_hef([], "HAILO10H"))

    def test_unknown_architecture_still_picks_a_detector(self):
        chosen = self.select_hef(INSTALLED_ON_A_REAL_PI, None)
        self.assertIsNotNone(chosen)
        self.assertGreaterEqual(self.score_hef(chosen), 0)


class TestHefDiscovery(unittest.TestCase):
    def test_explicit_path_must_exist(self):
        from objectlog.backends.hailo_backend import find_hef

        self.assertIsNone(find_hef("/definitely/not/here.hef"))

    def test_explicit_path_is_returned_when_present(self):
        import tempfile

        from objectlog.backends.hailo_backend import find_hef

        with tempfile.NamedTemporaryFile(suffix=".hef") as handle:
            self.assertEqual(find_hef(handle.name), handle.name)


class TestBackendSelection(unittest.TestCase):
    def test_hailo_is_preferred_but_falls_back_cleanly(self):
        # No accelerator here, so auto must degrade to a CPU backend rather
        # than failing outright.
        from objectlog import config as config_mod
        from objectlog.backends import build

        cfg = config_mod.load(None)
        cfg.set("detector.backend", "auto")
        detector = build(cfg)
        self.assertIn(detector.name, {"hailo", "onnx", "ultralytics", "mock"})

    def test_asking_for_hailo_explicitly_raises_a_useful_error(self):
        from objectlog import config as config_mod
        from objectlog.backends import build

        cfg = config_mod.load(None)
        cfg.set("detector.backend", "hailo")
        try:
            build(cfg)
        except Exception as exc:
            message = str(exc)
            self.assertTrue(
                "hailo" in message.lower() or "hef" in message.lower(),
                f"unhelpful error: {message}")
        # If it did not raise, a real accelerator is present -- also fine.


if __name__ == "__main__":
    unittest.main()
