"""Detector backend interface."""

from __future__ import annotations

from typing import List

import numpy as np

from ..tracker import Detection


class DetectorBackend:
    """A thing that turns an RGB frame into a list of Detections."""

    name = "base"
    #: Human-readable description shown in the web UI status bar.
    description = ""

    def detect(self, frame: np.ndarray) -> List[Detection]:
        raise NotImplementedError

    def close(self) -> None:
        pass


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> List[int]:
    """Plain numpy non-maximum suppression. Returns indices to keep."""
    if boxes.size == 0:
        return []
    x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x1 - x0) * np.maximum(0.0, y1 - y0)
    order = scores.argsort()[::-1]

    keep: List[int] = []
    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        ix0 = np.maximum(x0[current], x0[rest])
        iy0 = np.maximum(y0[current], y0[rest])
        ix1 = np.minimum(x1[current], x1[rest])
        iy1 = np.minimum(y1[current], y1[rest])
        inter = np.maximum(0.0, ix1 - ix0) * np.maximum(0.0, iy1 - iy0)
        union = areas[current] + areas[rest] - inter
        overlap = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
        order = rest[overlap <= threshold]
    return keep
