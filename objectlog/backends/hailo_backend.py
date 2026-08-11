"""Detection on a Hailo accelerator (AI HAT+ / AI Kit) via HailoRT.

The Hailo-8 runs models several tiers larger than anything the Pi's CPU can
manage, in a fraction of the time. Models are compiled ahead of time into .hef
files and are architecture-specific: a hailo8 build will not run on a hailo8l.

Most YOLO .hef builds do non-maximum suppression on-chip, so what comes back
is already decoded boxes -- one list per class, coordinates normalised 0-1 in
(y_min, x_min, y_max, x_max, score) order. That is the format this backend
reads. Run `scripts/hailo_probe.py` to confirm what yours actually produces.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence

import numpy as np

from ..labels import COCO_CLASSES
from ..tracker import Detection
from .base import DetectionFilter, DetectorBackend

# Where Raspberry Pi OS and the Hailo examples install compiled models.
HEF_SEARCH_PATHS = [
    "/usr/share/hailo-models",
    "/usr/local/hailo/resources",
    "/opt/hailo/resources",
    os.path.expanduser("~/hailo-rpi5-examples/resources"),
    "models",
]


def find_hef(preferred: Optional[str] = None) -> Optional[str]:
    """Locate a compiled model, preferring an explicitly configured one."""
    if preferred:
        return preferred if os.path.exists(preferred) else None
    import glob

    for directory in HEF_SEARCH_PATHS:
        matches = sorted(glob.glob(os.path.join(directory, "**", "*.hef"),
                                   recursive=True))
        if matches:
            return matches[0]
    return None


def _iter_class_arrays(raw) -> Sequence:
    """Normalise the many shapes a Hailo NMS output can arrive in.

    Returns a sequence indexed by class id, each entry an (N, 5) array. The
    runtime hands this back as a batched ndarray, an object array, or a plain
    list depending on version and how the model was compiled.
    """
    if isinstance(raw, np.ndarray):
        # A batched result: (1, classes, detections, 5) -> drop the batch.
        if raw.ndim == 4:
            return list(raw[0])
        if raw.ndim == 3:
            return list(raw)
        if raw.dtype == object:
            flat = raw.flatten()
            # A batch of one, wrapping the real per-class list.
            if flat.size == 1 and isinstance(flat[0], (list, tuple, np.ndarray)):
                return _iter_class_arrays(flat[0])
            return list(flat)
        # A single (N, 5) block with no class dimension at all.
        if raw.ndim == 2:
            return [raw]
        return []
    if isinstance(raw, (list, tuple)):
        # A batch of one, wrapping the real per-class list.
        if len(raw) == 1 and isinstance(raw[0], (list, tuple)) and len(raw[0]) > 5:
            return list(raw[0])
        return list(raw)
    return []


def decode_nms_output(raw, class_names: Sequence[str], confidence: float,
                      frame_width: int, frame_height: int,
                      scale: float = 1.0, pad_x: float = 0.0,
                      pad_y: float = 0.0, input_width: int = 0,
                      input_height: int = 0,
                      scale_y: Optional[float] = None) -> List[Detection]:
    """Turn an on-chip-NMS output into Detections in original frame pixels.

    Hailo gives normalised (y_min, x_min, y_max, x_max, score) per class --
    note y before x, which is the opposite of everything else here.

    `scale`/`pad_x`/`pad_y` undo the resize applied before inference. A
    letterbox uses one scale for both axes; a plain stretch-to-fit does not,
    hence `scale_y` (defaulting to `scale`).
    """
    if scale_y is None:
        scale_y = scale
    detections: List[Detection] = []
    input_width = input_width or frame_width
    input_height = input_height or frame_height

    for class_id, entries in enumerate(_iter_class_arrays(raw)):
        if entries is None:
            continue
        boxes = np.asarray(entries, dtype=np.float32)
        if boxes.size == 0:
            continue
        if boxes.ndim == 1:
            boxes = boxes.reshape(1, -1)
        if boxes.ndim != 2 or boxes.shape[1] < 5:
            continue

        label = (class_names[class_id] if class_id < len(class_names)
                 else f"class_{class_id}")

        for row in boxes:
            y_min, x_min, y_max, x_max, score = (float(row[0]), float(row[1]),
                                                 float(row[2]), float(row[3]),
                                                 float(row[4]))
            if score < confidence:
                continue
            # Normalised (0-1) against the network input, so scale up to input
            # pixels, undo the letterbox padding, then back to frame pixels.
            px0 = x_min * input_width
            px1 = x_max * input_width
            py0 = y_min * input_height
            py1 = y_max * input_height
            x0 = (px0 - pad_x) / scale
            x1 = (px1 - pad_x) / scale
            y0 = (py0 - pad_y) / scale_y
            y1 = (py1 - pad_y) / scale_y

            x0 = max(0.0, min(x0, frame_width - 1.0))
            y0 = max(0.0, min(y0, frame_height - 1.0))
            x1 = max(1.0, min(x1, float(frame_width)))
            y1 = max(1.0, min(y1, float(frame_height)))
            if x1 <= x0 or y1 <= y0:
                continue

            detections.append(Detection(label=label, confidence=score,
                                        box=(x0, y0, x1, y1)))
    return detections


class HailoDetector(DetectorBackend):
    name = "hailo"

    def __init__(self, hef_path: Optional[str] = None, confidence: float = 0.4,
                 detection_filter: Optional[DetectionFilter] = None,
                 labels: Optional[Sequence[str]] = None,
                 letterbox: bool = True, **_ignored):
        try:
            from hailo_platform import (ConfigureParams, FormatType, HEF,
                                        HailoStreamInterface, InferVStreams,
                                        InputVStreamParams,
                                        OutputVStreamParams, VDevice)
        except ImportError as exc:
            raise RuntimeError(
                "hailo_platform is not importable. It comes from apt, not "
                "pip:\n"
                "    sudo apt install -y hailo-all\n"
                "and the virtualenv must be able to see system packages "
                "(python3 -m venv --system-site-packages .venv).\n"
                "Run scripts/hailo_probe.py to diagnose."
            ) from exc

        resolved = find_hef(hef_path)
        if not resolved:
            raise FileNotFoundError(
                f"no Hailo model (.hef) found"
                f"{f' at {hef_path}' if hef_path else ''}.\n"
                "Run scripts/hailo_probe.py to see what is installed."
            )
        self.hef_path = resolved

        self._InferVStreams = InferVStreams
        self.confidence = confidence
        self.filter = detection_filter or DetectionFilter()
        self.class_names = list(labels) if labels else list(COCO_CLASSES)
        self.letterbox = letterbox

        self.hef = HEF(resolved)
        self._device = VDevice()
        params = ConfigureParams.create_from_hef(
            self.hef, interface=HailoStreamInterface.PCIe)
        self._network_group = self._device.configure(self.hef, params)[0]
        self._network_params = self._network_group.create_params()

        self._input_info = self.hef.get_input_vstream_infos()[0]
        self.input_height, self.input_width = self._input_info.shape[:2]

        self._in_params = InputVStreamParams.make(self._network_group,
                                                  format_type=FormatType.UINT8)
        self._out_params = OutputVStreamParams.make(self._network_group,
                                                    format_type=FormatType.FLOAT32)

        self.description = (
            f"hailo · {os.path.basename(resolved)} · "
            f"{self.input_width}x{self.input_height} · conf {confidence:g} · "
            f"{len(self.class_names)} classes"
        )
        extra = self.filter.describe()
        if extra:
            self.description += f" · {extra}"

    def _prepare(self, frame: np.ndarray):
        """Resize the frame to what the model wants, keeping the mapping back."""
        from PIL import Image

        height, width = frame.shape[:2]
        if not self.letterbox:
            # Stretch to fit: each axis gets its own scale and there is no pad.
            resized = np.asarray(Image.fromarray(frame).resize(
                (self.input_width, self.input_height), Image.BILINEAR))
            return (resized, self.input_width / width,
                    self.input_height / height, 0.0, 0.0)

        scale = min(self.input_width / width, self.input_height / height)
        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))
        resized = np.asarray(
            Image.fromarray(frame).resize((new_w, new_h), Image.BILINEAR))
        canvas = np.full((self.input_height, self.input_width, 3), 114,
                         dtype=np.uint8)
        pad_x = (self.input_width - new_w) // 2
        pad_y = (self.input_height - new_h) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        return canvas, scale, scale, float(pad_x), float(pad_y)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        height, width = frame.shape[:2]
        prepared, scale_x, scale_y, pad_x, pad_y = self._prepare(frame)
        batch = prepared[None].astype(np.uint8)

        with self._InferVStreams(self._network_group, self._in_params,
                                 self._out_params) as pipeline:
            with self._network_group.activate(self._network_params):
                results = pipeline.infer({self._input_info.name: batch})

        detections: List[Detection] = []
        for raw in results.values():
            detections.extend(decode_nms_output(
                raw, self.class_names, self.confidence, width, height,
                scale=scale_x, scale_y=scale_y, pad_x=pad_x, pad_y=pad_y,
                input_width=self.input_width, input_height=self.input_height))

        if not detections and results:
            # An empty result is normal. A result we could not read at all is
            # not -- say so rather than quietly logging nothing forever.
            self._warn_if_unreadable(results)

        return self.filter.apply(detections, frame.shape)

    _warned = False

    def _warn_if_unreadable(self, results) -> None:
        if HailoDetector._warned:
            return
        for raw in results.values():
            if len(_iter_class_arrays(raw)) == 0:
                HailoDetector._warned = True
                print("[hailo] the model's output could not be interpreted as "
                      "on-chip NMS results.")
                print("[hailo] run `python3 scripts/hailo_probe.py --hef "
                      f"{self.hef_path}` and check the layout.")
                return

    def close(self) -> None:
        try:
            self._device.release()
        except Exception:
            pass
