"""Detector backend selection.

`auto` prefers the ONNX model (light, fast, no PyTorch), falls back to
ultralytics if that is what is installed, and finally to the mock detector so
the app always starts and always shows something.
"""

from __future__ import annotations

import os

from .base import DetectionFilter, DetectorBackend, nms  # noqa: F401

__all__ = ["DetectionFilter", "DetectorBackend", "nms", "build",
           "build_filter"]


def build_filter(cfg) -> DetectionFilter:
    """Assemble the shared detection filter from config."""
    return DetectionFilter(
        allowed=list(cfg.get("detector.classes", []) or []),
        excluded=list(cfg.get("detector.exclude_classes", []) or []),
        min_box_area=float(cfg.get("detector.min_box_area", 0.0)),
        max_box_area=float(cfg.get("detector.max_box_area", 1.0)),
        ignore_regions=list(cfg.get("detector.ignore_regions", []) or []),
    )


def build(cfg) -> DetectorBackend:
    requested = str(cfg.get("detector.backend", "auto")).lower()
    kwargs = {
        "confidence": float(cfg.get("detector.confidence", 0.4)),
        "iou_threshold": float(cfg.get("detector.iou", 0.45)),
        "input_size": int(cfg.get("detector.input_size", 640)),
        "detection_filter": build_filter(cfg),
    }
    model = str(cfg.get("detector.model", "models/yolo11n.onnx"))

    order = {
        # Hailo first: if the accelerator is there it is far faster and more
        # accurate than anything the CPU can do, so prefer it silently.
        "auto": ["hailo", "onnx", "ultralytics", "mock"],
        "hailo": ["hailo"],
        "onnx": ["onnx"],
        "ultralytics": ["ultralytics"],
        "mock": ["mock"],
    }.get(requested)
    if order is None:
        raise ValueError(f"unknown detector.backend: {requested!r}")

    errors = []
    for candidate in order:
        try:
            if candidate == "hailo":
                from .hailo_backend import HailoDetector

                return HailoDetector(
                    hef_path=cfg.get("detector.hef"),
                    confidence=kwargs["confidence"],
                    detection_filter=kwargs["detection_filter"],
                    labels=cfg.get("detector.labels") or None,
                    letterbox=bool(cfg.get("detector.hailo_letterbox", True)),
                )
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
                                detection_filter=kwargs["detection_filter"])
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")
            if requested != "auto":
                raise
            if candidate == "hailo":
                # Most Pis have no accelerator; that is not worth a paragraph.
                print("[detector] no Hailo accelerator in use, falling back")
            else:
                print(f"[detector] {candidate} unavailable -- {exc}")

    raise RuntimeError("no detector backend available: " + "; ".join(errors))


def describe_model_hint(cfg) -> str:
    """A friendly nudge when the configured model file is missing."""
    model = str(cfg.get("detector.model", ""))
    if model.endswith(".onnx") and not os.path.exists(model):
        return (f"Model {model} not found -- run "
                f"`python3 scripts/fetch_model.py` to download it.")
    return ""
