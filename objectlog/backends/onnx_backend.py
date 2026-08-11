"""YOLO (v5/v8/v11) inference through onnxruntime.

This is the recommended backend on a Raspberry Pi: onnxruntime is a small
wheel, needs no PyTorch, and runs YOLO11n at a few frames per second on a Pi 5
which is plenty for a sightings log.

Pre/post-processing is done here rather than by ultralytics so the Pi does not
have to carry a 2GB dependency tree.
"""

from __future__ import annotations

import ast
import os
from typing import List, Optional

import numpy as np
from PIL import Image

from ..labels import COCO_CLASSES
from ..tracker import Detection
from .base import DetectorBackend, nms


def _letterbox(frame: np.ndarray, size: int, rectangular: bool = False,
               stride: int = 32):
    """Resize preserving aspect ratio, padding the remainder with grey.

    With `rectangular`, the canvas is only padded up to the next multiple of
    the model stride instead of a full square. For a 16:9 camera frame that is
    640x384 rather than 640x640 -- 40% fewer pixels to push through the
    network, and it keeps more detail because nothing is wasted on grey bars.
    Only usable when the model was exported with dynamic input dimensions.

    Returns (padded_image, scale, pad_x, pad_y) so boxes can be mapped back.
    """
    height, width = frame.shape[:2]
    scale = min(size / width, size / height)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = np.asarray(
        Image.fromarray(frame).resize((new_w, new_h), Image.BILINEAR))

    if rectangular:
        canvas_w = int(np.ceil(new_w / stride) * stride)
        canvas_h = int(np.ceil(new_h / stride) * stride)
    else:
        canvas_w = canvas_h = size

    canvas = np.full((canvas_h, canvas_w, 3), 114, dtype=np.uint8)
    pad_x, pad_y = (canvas_w - new_w) // 2, (canvas_h - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y


class OnnxDetector(DetectorBackend):
    name = "onnx"

    def __init__(self, model_path: str, confidence: float = 0.4,
                 iou_threshold: float = 0.45, input_size: int = 640,
                 classes: Optional[List[str]] = None,
                 min_box_area: float = 0.0):
        try:
            import onnxruntime as ort  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is not installed (pip install onnxruntime)"
            ) from exc

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"model not found: {model_path}\n"
                "Fetch one with:  python3 scripts/fetch_model.py"
            )

        options = ort.SessionOptions()
        # A Pi has 4 cores; leaving one free keeps the web server responsive.
        options.intra_op_num_threads = max(1, (os.cpu_count() or 2) - 1)
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            model_path, options, providers=["CPUExecutionProvider"])

        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        # A model exported with dynamic axes reports its spatial dims as
        # strings ('height'/'width') rather than ints. Those can take a
        # rectangular input; a fixed-shape export must be fed a square.
        spatial = list(shape[2:4]) if len(shape) >= 4 else []
        fixed = [d for d in spatial if isinstance(d, int) and d > 0]
        self.dynamic = len(fixed) < 2
        self.input_size = int(input_size) if self.dynamic else fixed[0]

        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.min_box_area = min_box_area
        self.class_names = self._read_class_names()
        self.allowed = {c.lower() for c in (classes or [])}
        self.description = (
            f"onnxruntime · {os.path.basename(model_path)} · "
            f"{self.input_size}px{' dynamic' if self.dynamic else ''} · "
            f"{len(self.class_names)} classes"
        )

    def _read_class_names(self) -> List[str]:
        """Ultralytics stamps the class map into the ONNX metadata."""
        try:
            meta = self.session.get_modelmeta().custom_metadata_map or {}
            raw = meta.get("names")
            if raw:
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, dict):
                    return [parsed[key] for key in sorted(parsed, key=int)]
                if isinstance(parsed, list):
                    return list(parsed)
        except Exception:
            pass
        return list(COCO_CLASSES)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        height, width = frame.shape[:2]
        padded, scale, pad_x, pad_y = _letterbox(
            frame, self.input_size, rectangular=self.dynamic)
        tensor = padded.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None]  # NCHW

        outputs = self.session.run(None, {self.input_name: tensor})
        predictions = outputs[0]
        if predictions.ndim == 3:
            predictions = predictions[0]
        # v8/v11 export as (84, 8400) -- features first. v5 exports as
        # (25200, 85) -- anchors first. Decide by matching the class count,
        # and fall back to "anchors always outnumber features".
        widths = (len(self.class_names) + 4, len(self.class_names) + 5)
        if predictions.shape[1] in widths and predictions.shape[0] not in widths:
            pass  # already (anchors, features)
        elif predictions.shape[0] in widths and predictions.shape[1] not in widths:
            predictions = predictions.T
        elif predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T

        if predictions.shape[1] == len(self.class_names) + 5:
            # YOLOv5 layout: cx, cy, w, h, objectness, class scores
            objectness = predictions[:, 4:5]
            scores_all = predictions[:, 5:] * objectness
        else:
            # YOLOv8/v11 layout: cx, cy, w, h, class scores
            scores_all = predictions[:, 4:]

        class_ids = scores_all.argmax(axis=1)
        scores = scores_all[np.arange(scores_all.shape[0]), class_ids]
        keep = scores >= self.confidence
        if not np.any(keep):
            return []

        boxes_xywh = predictions[keep, :4]
        scores = scores[keep]
        class_ids = class_ids[keep]

        # cxcywh (letterboxed pixels) -> xyxy (original frame pixels)
        cx, cy, bw, bh = (boxes_xywh[:, 0], boxes_xywh[:, 1],
                          boxes_xywh[:, 2], boxes_xywh[:, 3])
        x0 = (cx - bw / 2 - pad_x) / scale
        y0 = (cy - bh / 2 - pad_y) / scale
        x1 = (cx + bw / 2 - pad_x) / scale
        y1 = (cy + bh / 2 - pad_y) / scale
        boxes = np.stack([
            np.clip(x0, 0, width - 1), np.clip(y0, 0, height - 1),
            np.clip(x1, 1, width), np.clip(y1, 1, height),
        ], axis=1)

        detections: List[Detection] = []
        frame_area = float(width * height)
        for class_id in np.unique(class_ids):
            mask = class_ids == class_id
            label = (self.class_names[int(class_id)]
                     if int(class_id) < len(self.class_names)
                     else f"class_{int(class_id)}")
            if self.allowed and label.lower() not in self.allowed:
                continue
            class_boxes, class_scores = boxes[mask], scores[mask]
            for index in nms(class_boxes, class_scores, self.iou_threshold):
                box = class_boxes[index]
                area = (box[2] - box[0]) * (box[3] - box[1])
                if frame_area > 0 and area / frame_area < self.min_box_area:
                    continue
                detections.append(Detection(
                    label=label,
                    confidence=float(class_scores[index]),
                    box=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                ))
        return detections
