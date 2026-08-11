"""Frame sources.

`picamera2` is the real one (Camera Module 3 on a Pi). `opencv` covers USB
webcams and development on a laptop. `synthetic` generates moving shapes so the
whole pipeline can be exercised with no hardware at all.

All sources yield (H, W, 3) uint8 arrays in RGB order.
"""

from __future__ import annotations

import math
import os
import time
from typing import Optional

import numpy as np


class CameraError(RuntimeError):
    pass


class BaseCamera:
    name = "base"

    def read(self) -> Optional[np.ndarray]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class PiCamera(BaseCamera):
    """Raspberry Pi Camera Module via libcamera/picamera2."""

    name = "picamera2"

    def __init__(self, width: int, height: int, hflip: bool = False,
                 vflip: bool = False):
        try:
            from picamera2 import Picamera2  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - Pi only
            raise CameraError(
                "picamera2 is not installed. On Raspberry Pi OS: "
                "sudo apt install -y python3-picamera2"
            ) from exc

        self._picam = Picamera2()
        # RGB888 out of picamera2 is actually delivered BGR-ordered, so we
        # request it and flip the channels below -- this is a known quirk.
        config = self._picam.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"},
        )
        transform_applied = False
        if hflip or vflip:
            try:
                from libcamera import Transform  # noqa: PLC0415

                config["transform"] = Transform(hflip=int(hflip),
                                                vflip=int(vflip))
                transform_applied = True
            except Exception:  # pragma: no cover - depends on libcamera build
                pass
        self._flip_h = hflip and not transform_applied
        self._flip_v = vflip and not transform_applied

        self._picam.configure(config)
        self._picam.start()
        # Let auto-exposure and auto-white-balance settle before the first
        # detection, otherwise early colour readings are garbage.
        time.sleep(2.0)

    def read(self) -> Optional[np.ndarray]:
        frame = self._picam.capture_array()
        if frame is None:
            return None
        if frame.shape[2] == 4:
            frame = frame[:, :, :3]
        frame = frame[:, :, ::-1]  # BGR -> RGB
        if self._flip_h:
            frame = frame[:, ::-1]
        if self._flip_v:
            frame = frame[::-1, :]
        return np.ascontiguousarray(frame)

    def close(self) -> None:  # pragma: no cover - Pi only
        try:
            self._picam.stop()
            self._picam.close()
        except Exception:
            pass


class OpenCVCamera(BaseCamera):
    """USB webcam / V4L2 device via OpenCV."""

    name = "opencv"

    def __init__(self, device: int, width: int, height: int,
                 hflip: bool = False, vflip: bool = False):
        try:
            import cv2  # noqa: PLC0415
        except ImportError as exc:
            raise CameraError(
                "opencv-python is not installed (pip install opencv-python)"
            ) from exc
        self._cv2 = cv2
        self._cap = cv2.VideoCapture(device)
        if not self._cap.isOpened():
            raise CameraError(f"could not open video device {device}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._flip_h, self._flip_v = hflip, vflip

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        frame = frame[:, :, ::-1]  # BGR -> RGB
        if self._flip_h:
            frame = frame[:, ::-1]
        if self._flip_v:
            frame = frame[::-1, :]
        return np.ascontiguousarray(frame)

    def close(self) -> None:
        try:
            self._cap.release()
        except Exception:
            pass


class SyntheticCamera(BaseCamera):
    """A fake scene: coloured shapes drifting over a gradient.

    Useful for checking the web UI, the tracker and the store without a
    camera attached. Pair it with the `mock` detector backend.
    """

    name = "synthetic"

    def __init__(self, width: int = 640, height: int = 480):
        self.width, self.height = width, height
        self._t = 0.0
        yy, xx = np.mgrid[0:height, 0:width]
        base = np.zeros((height, width, 3), dtype=np.uint8)
        base[:, :, 0] = (30 + 40 * xx / max(width, 1)).astype(np.uint8)
        base[:, :, 1] = (35 + 35 * yy / max(height, 1)).astype(np.uint8)
        base[:, :, 2] = 55
        self._base = base

    def read(self) -> Optional[np.ndarray]:
        self._t += 0.35
        frame = self._base.copy()
        shapes = [
            ((60, 110, 220), 0.0, 0.22, 0.26),    # blue block
            ((215, 70, 60), 2.1, 0.55, 0.20),     # red block
            ((80, 190, 90), 4.2, 0.34, 0.16),     # green block
        ]
        for rgb, phase, y_frac, size_frac in shapes:
            cx = 0.5 + 0.32 * math.sin(self._t * 0.35 + phase)
            box_w = int(self.width * size_frac)
            box_h = int(self.height * size_frac * 1.4)
            x0 = int(cx * self.width) - box_w // 2
            y0 = int(y_frac * self.height)
            x0 = max(0, min(x0, self.width - box_w))
            y0 = max(0, min(y0, self.height - box_h))
            frame[y0:y0 + box_h, x0:x0 + box_w] = rgb
        return frame


class FolderCamera(BaseCamera):
    """Replays a folder of images, one per frame, looping forever.

    Handy for checking the detector and the web page against real photos
    before the camera is mounted where you actually want it.
    """

    name = "folder"

    def __init__(self, folder: str, width: int = 0, height: int = 0):
        try:
            from PIL import Image  # noqa: PLC0415
        except ImportError as exc:
            raise CameraError("Pillow is required for the folder source") from exc
        self._Image = Image

        if not os.path.isdir(folder):
            raise CameraError(f"not a folder: {folder}")
        extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        self.paths = sorted(
            os.path.join(folder, name) for name in os.listdir(folder)
            if name.lower().endswith(extensions))
        if not self.paths:
            raise CameraError(f"no images found in {folder}")
        self._index = 0
        self._max_width = width

    def read(self) -> Optional[np.ndarray]:
        path = self.paths[self._index % len(self.paths)]
        self._index += 1
        image = self._Image.open(path).convert("RGB")
        if self._max_width and image.width > self._max_width:
            ratio = self._max_width / image.width
            image = image.resize((self._max_width, max(1, int(image.height * ratio))))
        return np.asarray(image)


def open_camera(cfg) -> BaseCamera:
    """Open the configured source, falling back sensibly when auto."""
    source = str(cfg.get("camera.source", "auto")).lower()
    width = int(cfg.get("camera.width", 1280))
    height = int(cfg.get("camera.height", 720))
    hflip = bool(cfg.get("camera.hflip", False))
    vflip = bool(cfg.get("camera.vflip", False))
    device = int(cfg.get("camera.device", 0))

    order = {
        "auto": ["picamera2", "opencv", "synthetic"],
        "picamera2": ["picamera2"],
        "opencv": ["opencv"],
        "folder": ["folder"],
        "synthetic": ["synthetic"],
    }.get(source)
    if order is None:
        raise CameraError(f"unknown camera.source: {source!r}")

    errors = []
    for candidate in order:
        try:
            if candidate == "picamera2":
                return PiCamera(width, height, hflip, vflip)
            if candidate == "opencv":
                return OpenCVCamera(device, width, height, hflip, vflip)
            if candidate == "folder":
                return FolderCamera(str(cfg.get("camera.folder", "samples")),
                                    width)
            return SyntheticCamera(width, height)
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")
            if source != "auto":
                raise
    raise CameraError("no camera available -- " + "; ".join(errors))


def rotate(frame: np.ndarray, degrees: int) -> np.ndarray:
    """Rotate a frame by a multiple of 90 degrees."""
    degrees = int(degrees) % 360
    if degrees == 0:
        return frame
    if degrees == 90:
        return np.ascontiguousarray(np.rot90(frame, k=3))
    if degrees == 180:
        return np.ascontiguousarray(np.rot90(frame, k=2))
    if degrees == 270:
        return np.ascontiguousarray(np.rot90(frame, k=1))
    raise ValueError("camera.rotation must be 0, 90, 180 or 270")
