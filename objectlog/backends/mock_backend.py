"""A detector that invents plausible sightings.

Exists so the web UI, tracker and store can be exercised end to end with no
model file and no camera -- `--camera synthetic --backend mock`. It finds the
solid colour blocks the synthetic camera draws, so its boxes line up with real
pixels and the colour descriptions come out correctly.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..tracker import Detection
from .base import DetectionFilter, DetectorBackend

# Blocks the synthetic camera draws, in the order it draws them.
_PRETEND_LABELS = ["person", "car", "potted plant", "cup", "dog"]


class MockDetector(DetectorBackend):
    name = "mock"
    description = "mock detector (no model) · demo mode"

    def __init__(self, confidence: float = 0.4,
                 detection_filter: Optional[DetectionFilter] = None, **_ignored):
        self.confidence = confidence
        self.filter = detection_filter or DetectionFilter()
        self._frame_index = 0

    def detect(self, frame: np.ndarray) -> List[Detection]:
        self._frame_index += 1
        height, width = frame.shape[:2]

        # Find contiguous saturated regions by scanning a coarse grid -- crude,
        # but it locates the synthetic camera's blocks without needing OpenCV.
        step = max(4, min(height, width) // 60)
        small = frame[::step, ::step].astype(np.int16)
        spread = small.max(axis=2) - small.min(axis=2)
        mask = spread > 60

        detections: List[Detection] = []
        visited = np.zeros_like(mask)
        for row in range(mask.shape[0]):
            for col in range(mask.shape[1]):
                if not mask[row, col] or visited[row, col]:
                    continue
                # Flood fill this blob with an explicit stack (no recursion).
                stack = [(row, col)]
                pixels = []
                visited[row, col] = True
                while stack:
                    r, c = stack.pop()
                    pixels.append((r, c))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1]
                                and mask[nr, nc] and not visited[nr, nc]):
                            visited[nr, nc] = True
                            stack.append((nr, nc))
                if len(pixels) < 4:
                    continue
                rows = [p[0] for p in pixels]
                cols = [p[1] for p in pixels]
                box = (float(min(cols) * step), float(min(rows) * step),
                       float((max(cols) + 1) * step), float((max(rows) + 1) * step))
                # Stable label per blob position, so tracks stay coherent.
                label = _PRETEND_LABELS[len(detections) % len(_PRETEND_LABELS)]
                detections.append(Detection(
                    label=label, confidence=0.72 + 0.2 * ((self._frame_index % 5) / 5),
                    box=box))
        return self.filter.apply(detections, frame.shape)
