"""Detector backend selection.

`auto` prefers the ONNX model (light, fast, no PyTorch), falls back to
ultralytics if that is what is installed, and finally to the mock detector so
the app always starts and always shows something.
"""

from __future__ import annotations

import os

from .base import DetectorBackend, nms  # noqa: F401  (re-exported)

__all__ = ["DetectorBackend", "nms", "build"]


def build(cfg) -> DetectorBackend:
    requested = str(cfg.get("detector.backend", "auto")).lower()
    kwargs = {
        "confidence": float(cfg.get("detector.confidence", 0.4)),
        "iou_threshold": float(cfg.get("detector.iou", 0.45)),
        "input_size": int(cfg.get("detector.input_size", 640)),
        "classes": list(cfg.get("detector.classes", []) or []),
        "min_box_area": float(cfg.get("detector.min_box_area", 0.0)),
    }
    model = str(cfg.get("detector.model", "models/yolo11n.onnx"))

    order = {
        "auto": ["onnx", "ultralytics", "mock"],
        "onnx": ["onnx"],
        "ultralytics": ["ultralytics"],
        "mock": ["mock"],
    }.get(requested)
    if order is None:
        raise ValueError(f"unknown detector.backend: {requested!r}")

    errors = []
    for candidate in order:
        try:
            if candidate == "onnx":
                from .onnx_backend import OnnxDetector

                path = model if model.endswith(".onnx") else "models/yolo11n.onnx"
                return OnnxDetector(path, **kwargs)
            if candidate == "ultralytics":
                from .ultralytics_backend import UltralyticsDetector

                path = model if model.endswith(".pt") else "yolo11n.pt"
                return UltralyticsDetector(path, **kwargs)
            from .mock_backend import MockDetector

            return MockDetector(confidence=kwargs["confidence"],
                                classes=kwargs["classes"],
                                min_box_area=kwargs["min_box_area"])
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")
            if requested != "auto":
                raise
            print(f"[detector] {candidate} unavailable -- {exc}")

    raise RuntimeError("no detector backend available: " + "; ".join(errors))


def describe_model_hint(cfg) -> str:
    """A friendly nudge when the configured model file is missing."""
    model = str(cfg.get("detector.model", ""))
    if model.endswith(".onnx") and not os.path.exists(model):
        return (f"Model {model} not found -- run "
                f"`python3 scripts/fetch_model.py` to download it.")
    return ""
