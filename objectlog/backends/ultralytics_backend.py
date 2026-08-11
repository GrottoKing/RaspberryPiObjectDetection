"""Ultralytics YOLO backend.

Used when the `ultralytics` package is installed and pointed at a .pt model.
It downloads weights on first use and generally gives the same results as the
ONNX path, at the cost of a much larger install (it pulls in PyTorch).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..tracker import Detection
from .base import DetectorBackend


class UltralyticsDetector(DetectorBackend):
    name = "ultralytics"

    def __init__(self, model_path: str = "yolo11n.pt", confidence: float = 0.4,
                 iou_threshold: float = 0.45, input_size: int = 640,
                 classes: Optional[List[str]] = None,
                 min_box_area: float = 0.0):
        try:
            from ultralytics import YOLO  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError("ultralytics is not installed") from exc

        self.model = YOLO(model_path)
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.input_size = int(input_size)
        self.min_box_area = min_box_area
        self.allowed = {c.lower() for c in (classes or [])}
        self.description = f"ultralytics · {model_path} · {self.input_size}px"

    def detect(self, frame: np.ndarray) -> List[Detection]:
        results = self.model.predict(
            frame, imgsz=self.input_size, conf=self.confidence,
            iou=self.iou_threshold, verbose=False,
        )
        if not results:
            return []
        result = results[0]
        names = result.names
        frame_area = float(frame.shape[0] * frame.shape[1])

        detections: List[Detection] = []
        for box in result.boxes:
            label = str(names[int(box.cls)])
            if self.allowed and label.lower() not in self.allowed:
                continue
            x0, y0, x1, y1 = [float(v) for v in box.xyxy[0].tolist()]
            area = (x1 - x0) * (y1 - y0)
            if frame_area > 0 and area / frame_area < self.min_box_area:
                continue
            detections.append(Detection(
                label=label, confidence=float(box.conf), box=(x0, y0, x1, y1)))
        return detections
