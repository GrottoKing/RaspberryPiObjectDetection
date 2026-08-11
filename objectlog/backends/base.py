"""Detector backend interface, plus the filtering every backend shares."""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from ..tracker import Detection


class DetectionFilter:
    """Drops detections you have decided you do not want.

    Every backend runs its output through one of these, so the rules behave
    identically whichever detector is in use. This is the main defence against
    false positives: a COCO model in an office will occasionally see a "car" in
    a filing cabinet, and no confidence threshold alone fixes that -- but
    telling it there are no cars indoors does.
    """

    def __init__(self, allowed: Optional[Sequence[str]] = None,
                 excluded: Optional[Sequence[str]] = None,
                 min_box_area: float = 0.0,
                 max_box_area: float = 1.0,
                 ignore_regions: Optional[Sequence[Sequence[float]]] = None):
        self.allowed = {c.lower().strip() for c in (allowed or [])}
        self.excluded = {c.lower().strip() for c in (excluded or [])}
        self.min_box_area = float(min_box_area)
        self.max_box_area = float(max_box_area)
        self.ignore_regions = [
            tuple(float(v) for v in region) for region in (ignore_regions or [])
        ]

    def allows_label(self, label: str) -> bool:
        """Cheap pre-check, so a backend can skip work for unwanted classes."""
        key = label.lower().strip()
        if self.excluded and key in self.excluded:
            return False
        if self.allowed and key not in self.allowed:
            return False
        return True

    def _in_ignored_region(self, box, width: int, height: int) -> bool:
        if not self.ignore_regions or width <= 0 or height <= 0:
            return False
        # Judge by the box centre: a detection whose middle sits in a masked
        # area is the thing being masked, even if its edges spill outside.
        cx = ((box[0] + box[2]) / 2) / width
        cy = ((box[1] + box[3]) / 2) / height
        for x0, y0, x1, y1 in self.ignore_regions:
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                return True
        return False

    def apply(self, detections: Sequence[Detection],
              frame_shape) -> List[Detection]:
        height, width = frame_shape[0], frame_shape[1]
        frame_area = float(width * height)
        kept: List[Detection] = []
        for det in detections:
            if not self.allows_label(det.label):
                continue
            if frame_area > 0:
                box = det.box
                share = ((box[2] - box[0]) * (box[3] - box[1])) / frame_area
                # Too small is usually noise; too large is usually the detector
                # locking onto a wall or a desk that fills the frame.
                if share < self.min_box_area or share > self.max_box_area:
                    continue
            if self._in_ignored_region(det.box, width, height):
                continue
            kept.append(det)
        return kept

    def describe(self) -> str:
        bits = []
        if self.allowed:
            bits.append(f"only {len(self.allowed)} classes")
        if self.excluded:
            bits.append(f"{len(self.excluded)} excluded")
        if self.ignore_regions:
            bits.append(f"{len(self.ignore_regions)} ignored region(s)")
        return " · ".join(bits)


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
