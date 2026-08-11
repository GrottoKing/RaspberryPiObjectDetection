"""End-to-end tests: ONNX decoding, the detection loop, and the web API.

These run with no camera and no downloaded model. The ONNX test builds a tiny
stand-in network whose output has the exact shape a real YOLO export produces,
which is enough to prove the letterbox maths and NMS are right.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from objectlog import config as config_mod  # noqa: E402
from objectlog.pipeline import Pipeline  # noqa: E402
from objectlog.store import Store  # noqa: E402
from objectlog.web.app import create_app  # noqa: E402


def temp_config(directory, **overrides):
    cfg = config_mod.load(None)
    cfg.set("storage.db_path", os.path.join(directory, "log.db"))
    cfg.set("storage.snapshot_dir", os.path.join(directory, "snaps"))
    cfg.set("camera.source", "synthetic")
    cfg.set("camera.width", 320)
    cfg.set("camera.height", 240)
    cfg.set("camera.fps_limit", 0)
    cfg.set("detector.backend", "mock")
    cfg.set("tracker.min_hits", 2)
    # These tests step frames in a tight loop, so no wall-clock time passes.
    # The time-based gates would never open; they have their own tests.
    cfg.set("tracker.min_seconds", 0.0)
    for key, value in overrides.items():
        cfg.set(key.replace("__", "."), value)
    return cfg


class TestOnnxDecoding(unittest.TestCase):
    """The ONNX backend against a synthetic model with a known output."""

    @classmethod
    def setUpClass(cls):
        try:
            import onnx  # noqa: F401
            import onnxruntime  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("onnx/onnxruntime not installed")

        cls.dir = tempfile.mkdtemp(prefix="objectlog-onnx-")
        cls.model_path = os.path.join(cls.dir, "fake.onnx")
        cls.dynamic_path = os.path.join(cls.dir, "fake_dynamic.onnx")
        cls.anchors = 300
        cls._build_model(cls.model_path, cls.anchors)
        cls._build_model(cls.dynamic_path, cls.anchors, dynamic=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    @staticmethod
    def _build_model(path, anchors, dynamic=False):
        """A model shaped like a YOLOv8/11 export: output (1, 84, anchors).

        Two planted detections:
          * a high-scoring 'person' at letterbox (320, 320, 100, 100)
          * a duplicate of it, to prove NMS collapses them
          * a low-scoring 'car' below the confidence threshold
        """
        import onnx
        from onnx import TensorProto, helper, numpy_helper

        data = np.zeros((1, 84, anchors), dtype=np.float32)
        # anchor 0: person, conf 0.90
        data[0, :4, 0] = [320.0, 320.0, 100.0, 100.0]
        data[0, 4, 0] = 0.90
        # anchor 1: near-identical person, conf 0.80 -> suppressed by NMS
        data[0, :4, 1] = [322.0, 321.0, 100.0, 100.0]
        data[0, 4, 1] = 0.80
        # anchor 2: car, conf 0.10 -> below threshold
        data[0, :4, 2] = [100.0, 100.0, 40.0, 40.0]
        data[0, 4 + 2, 2] = 0.10

        constant = helper.make_node(
            "Constant", inputs=[], outputs=["output0"],
            value=numpy_helper.from_array(data, name="preds"))
        # A dynamic export reports its spatial dims as names, not ints -- that
        # is what tells the detector it may send a rectangular frame.
        input_shape = [1, 3, "height", "width"] if dynamic else [1, 3, 640, 640]
        graph = helper.make_graph(
            [constant], "fake_yolo",
            inputs=[helper.make_tensor_value_info(
                "images", TensorProto.FLOAT, input_shape)],
            outputs=[helper.make_tensor_value_info(
                "output0", TensorProto.FLOAT, [1, 84, anchors])],
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 12)])
        model.ir_version = 8
        onnx.save(model, path)

    def _detector(self, **kwargs):
        from objectlog.backends.base import DetectionFilter
        from objectlog.backends.onnx_backend import OnnxDetector

        filter_keys = ("allowed", "excluded", "min_box_area", "max_box_area",
                       "ignore_regions")
        filter_kwargs = {k: kwargs.pop(k) for k in filter_keys if k in kwargs}
        params = dict(confidence=0.4, iou_threshold=0.45, input_size=640)
        params.update(kwargs)
        return OnnxDetector(self.model_path,
                            detection_filter=DetectionFilter(**filter_kwargs),
                            **params)

    def test_boxes_map_back_to_original_pixels(self):
        detector = self._detector()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        detections = detector.detect(frame)

        self.assertEqual(len(detections), 1, "NMS should collapse the duplicate")
        found = detections[0]
        self.assertEqual(found.label, "person")
        self.assertAlmostEqual(found.confidence, 0.90, places=5)

        # 1280x720 letterboxed into 640x640: scale 0.5, vertical pad 140px.
        # cxcywh (320,320,100,100) -> letterbox xyxy (270,270,370,370)
        # -> unpad y by 140 -> (270,130,370,230) -> /0.5 -> (540,260,740,460).
        for actual, expected in zip(found.box, (540.0, 260.0, 740.0, 460.0)):
            self.assertAlmostEqual(actual, expected, delta=1.5)

    def test_confidence_threshold_is_respected(self):
        self.assertEqual(len(self._detector(confidence=0.95).detect(
            np.zeros((720, 1280, 3), dtype=np.uint8))), 0)
        # Drop the threshold and the low-scoring car appears.
        labels = {d.label for d in self._detector(confidence=0.05).detect(
            np.zeros((720, 1280, 3), dtype=np.uint8))}
        self.assertEqual(labels, {"person", "car"})

    def test_class_filter(self):
        detector = self._detector(confidence=0.05, allowed=["car"])
        labels = {d.label for d in detector.detect(
            np.zeros((720, 1280, 3), dtype=np.uint8))}
        self.assertEqual(labels, {"car"})

    def test_min_box_area_drops_specks(self):
        # The person box is 200x300 of a 1280x720 frame -> 6.5% of the area.
        detector = self._detector(min_box_area=0.10)
        self.assertEqual(detector.detect(np.zeros((720, 1280, 3), dtype=np.uint8)), [])

    def test_square_frame_needs_no_padding(self):
        detector = self._detector()
        found = detector.detect(np.zeros((640, 640, 3), dtype=np.uint8))[0]
        for actual, expected in zip(found.box, (270.0, 270.0, 370.0, 370.0)):
            self.assertAlmostEqual(actual, expected, delta=1.5)

    def test_static_model_is_detected_as_such(self):
        self.assertFalse(self._detector().dynamic)
        self.assertEqual(self._detector(input_size=320).input_size, 640,
                         "a fixed-shape model must ignore the configured size")

    def test_dynamic_model_uses_a_rectangular_letterbox(self):
        from objectlog.backends.onnx_backend import OnnxDetector

        detector = OnnxDetector(self.dynamic_path, confidence=0.4)
        self.assertTrue(detector.dynamic)

        # 1280x720 at scale 0.5 -> 640x360, padded up to 640x384 (stride 32),
        # so the vertical pad is 12px rather than the square case's 140px.
        found = detector.detect(np.zeros((720, 1280, 3), dtype=np.uint8))[0]
        for actual, expected in zip(found.box, (540.0, 516.0, 740.0, 716.0)):
            self.assertAlmostEqual(actual, expected, delta=1.5)

    def test_letterbox_shapes(self):
        from objectlog.backends.onnx_backend import _letterbox

        wide = np.zeros((720, 1280, 3), dtype=np.uint8)
        square_canvas, _, _, _ = _letterbox(wide, 640, rectangular=False)
        self.assertEqual(square_canvas.shape[:2], (640, 640))

        rect_canvas, scale, pad_x, pad_y = _letterbox(wide, 640, rectangular=True)
        self.assertEqual(rect_canvas.shape[:2], (384, 640))
        self.assertEqual((scale, pad_x, pad_y), (0.5, 0, 12))
        # Padding is grey and centred, image content sits between the bars.
        self.assertTrue((rect_canvas[0] == 114).all())
        self.assertTrue((rect_canvas[-1] == 114).all())

        # Every rectangular canvas dimension must be a multiple of the stride,
        # or the network's downsampling will not line up.
        for shape in [(720, 1280), (1080, 1920), (480, 640), (240, 320), (500, 500)]:
            canvas, _, _, _ = _letterbox(np.zeros((*shape, 3), dtype=np.uint8),
                                         640, rectangular=True)
            self.assertEqual(canvas.shape[0] % 32, 0, msg=str(shape))
            self.assertEqual(canvas.shape[1] % 32, 0, msg=str(shape))

    def test_missing_model_gives_a_useful_error(self):
        from objectlog.backends.onnx_backend import OnnxDetector

        with self.assertRaises(FileNotFoundError) as ctx:
            OnnxDetector(os.path.join(self.dir, "nope.onnx"))
        self.assertIn("fetch_model.py", str(ctx.exception))


class TestPipelineAndWeb(unittest.TestCase):
    """Synthetic camera + mock detector, driven through the real pipeline."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="objectlog-pipe-")
        self.cfg = temp_config(self.dir)
        self.store = Store(
            db_path=self.cfg.get("storage.db_path"),
            snapshot_dir=self.cfg.get("storage.snapshot_dir"),
            max_entries=100)
        self.pipeline = Pipeline(self.cfg, self.store)

    def tearDown(self):
        self.pipeline.stop()
        self.store.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def _run_frames(self, count=6):
        for _ in range(count):
            self.pipeline._step()

    def test_sightings_are_logged_with_snapshots(self):
        self._run_frames(6)
        rows = self.store.sightings()
        self.assertGreater(len(rows), 0, "the synthetic scene should be logged")
        for row in rows:
            self.assertIsNotNone(row["snapshot"])
            path = os.path.join(self.dir, "snaps",
                                os.path.basename(row["snapshot"]))
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 500)

    def test_a_persistent_object_makes_one_entry_not_one_per_frame(self):
        self._run_frames(12)
        rows = self.store.sightings()
        # Three blocks in the synthetic scene; twelve frames. Without the
        # tracker this would be dozens of rows.
        self.assertLessEqual(len(rows), 6,
                             f"expected a handful of entries, got {len(rows)}")

    def test_colours_reach_the_description(self):
        self._run_frames(6)
        described = " ".join(r["description"] for r in self.store.sightings())
        self.assertTrue(any(word in described.lower()
                            for word in ("blue", "red", "green")),
                        f"no colour named in: {described}")

    def test_status_and_live_objects(self):
        self._run_frames(4)
        status = self.pipeline.status()
        self.assertEqual(status["camera"], "synthetic")
        self.assertEqual(status["backend"], "mock")
        self.assertEqual(status["frames"], 4)
        self.assertIsNone(status["error"])
        self.assertGreater(len(self.pipeline.live_objects()), 0)

    def test_stream_frame_is_a_jpeg(self):
        self._run_frames(3)
        jpeg = self.pipeline.snapshot_frame(max_width=320)
        self.assertTrue(jpeg.startswith(b"\xff\xd8"), "not a JPEG")

    # ------------------------------------------------------------ web layer

    def test_web_api(self):
        self._run_frames(6)
        app = create_app(self.cfg, self.store, self.pipeline)
        client = app.test_client()

        page = client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn(b"Camera Sightings", page.data)

        state = json.loads(client.get("/api/state").data)
        self.assertGreater(len(state["entries"]), 0)
        self.assertGreater(len(state["categories"]), 0)
        self.assertEqual(state["status"]["backend"], "mock")
        entry = state["entries"][0]
        for key in ("id", "description", "category", "snapshot", "attributes",
                    "confidence", "first_seen", "last_seen", "duration"):
            self.assertIn(key, entry)

        image = client.get(entry["snapshot"])
        self.assertEqual(image.status_code, 200)
        self.assertEqual(image.mimetype, "image/jpeg")
        image.close()

        health = client.get("/healthz")
        self.assertEqual(health.status_code, 200)

    def test_incremental_polling_returns_only_new_rows(self):
        self._run_frames(6)
        app = create_app(self.cfg, self.store, self.pipeline)
        client = app.test_client()

        first = json.loads(client.get("/api/state").data)
        newest_id = max(e["id"] for e in first["entries"])
        later = json.loads(client.get(
            f"/api/state?since_id={newest_id}&since_ts={time.time() + 60}").data)
        self.assertEqual(later["entries"], [],
                         "nothing new should come back on a repeat poll")

    def test_clear_endpoint_empties_the_log(self):
        self._run_frames(6)
        app = create_app(self.cfg, self.store, self.pipeline)
        client = app.test_client()
        self.assertGreater(json.loads(client.post("/api/clear").data)["cleared"], 0)
        self.assertEqual(json.loads(client.get("/api/state").data)["entries"], [])

    def test_stream_disabled_returns_404(self):
        self.cfg.set("web.stream", False)
        client = create_app(self.cfg, self.store, self.pipeline).test_client()
        self.assertEqual(client.get("/stream.mjpg").status_code, 404)

    def test_detector_failure_does_not_kill_the_loop(self):
        def explode(_frame):
            # Ask the loop to finish, then fail: this exercises the error
            # handler and still terminates in one pass.
            self.pipeline._stop.set()
            raise RuntimeError("camera unplugged")

        original = self.pipeline.detector.detect
        self.pipeline.detector.detect = explode
        try:
            self.pipeline._run()  # must return, not propagate
        finally:
            self.pipeline.detector.detect = original
        self.assertIn("camera unplugged", self.pipeline.status()["error"] or "")


class TestConfig(unittest.TestCase):
    def test_defaults_load_without_a_file(self):
        cfg = config_mod.load(None)
        self.assertEqual(cfg.get("web.port"), 8010)
        self.assertIsNone(cfg.get("nothing.here"))
        self.assertEqual(cfg.get("nothing.here", "fallback"), "fallback")

    def test_yaml_overrides_merge_key_by_key(self):
        directory = tempfile.mkdtemp(prefix="objectlog-cfg-")
        try:
            path = os.path.join(directory, "config.yaml")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("web:\n  port: 9999\ndetector:\n  confidence: 0.7\n")
            cfg = config_mod.load(path)
            self.assertEqual(cfg.get("web.port"), 9999)
            self.assertEqual(cfg.get("detector.confidence"), 0.7)
            # Untouched keys keep their defaults.
            self.assertEqual(cfg.get("web.host"), "0.0.0.0")
            self.assertEqual(cfg.get("detector.iou"), 0.45)
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_missing_explicit_config_raises(self):
        with self.assertRaises(FileNotFoundError):
            config_mod.load("/definitely/not/here.yaml")


if __name__ == "__main__":
    unittest.main()
