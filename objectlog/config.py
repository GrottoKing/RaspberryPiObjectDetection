"""Configuration loading.

Everything has a working default, so the app runs with no config file at all.
`config.yaml` (if present) overrides defaults key-by-key.
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict

DEFAULTS: Dict[str, Any] = {
    "camera": {
        # auto | picamera2 | opencv | synthetic
        "source": "auto",
        "width": 1280,
        "height": 720,
        # Upper bound on how often we run detection. The Pi cannot do 30fps
        # inference, and we do not need it to -- objects live for seconds.
        "fps_limit": 4.0,
        # 0 | 90 | 180 | 270
        "rotation": 0,
        "hflip": False,
        "vflip": False,
        # OpenCV device index, only used by the opencv source
        "device": 0,
        # Image folder, only used by the folder source
        "folder": "samples",
    },
    "detector": {
        # auto | hailo | onnx | ultralytics | mock
        "backend": "auto",
        "model": "models/yolo11n.onnx",
        # Hailo accelerator (AI HAT+ / AI Kit) settings. Leave hef empty to
        # auto-discover a compiled model in the usual system locations.
        "hef": "",
        # Class names for the .hef, in training order. Empty = COCO's 80.
        "labels": [],
        # Letterbox before inference (preserve aspect). Set false if boxes come
        # out consistently offset or stretched.
        "hailo_letterbox": True,
        "confidence": 0.40,
        "iou": 0.45,
        "input_size": 640,
        # Empty list = keep every class the model knows about.
        "classes": [],
        # Classes to drop. Easier than an allow-list when you only want to
        # silence a few persistent false positives.
        "exclude_classes": [],
        # Ignore boxes smaller than this fraction of the frame area. Kills a
        # lot of far-away noise.
        "min_box_area": 0.004,
        # ...and larger than this. A detector that has locked onto a wall or a
        # desk usually reports something covering most of the frame.
        "max_box_area": 1.0,
        # Rectangles of the frame to ignore, as fractions [x0, y0, x1, y1].
        # A detection whose centre falls inside one is discarded.
        "ignore_regions": [],
    },
    "tracker": {
        "iou_threshold": 0.30,
        # Seconds an object can go unseen before its entry is closed. In
        # seconds rather than frames so that changing fps_limit does not
        # silently change how readily an object is re-logged as a new one.
        "max_missing_seconds": 2.0,
        # An object must be seen this many frames AND for this long before it
        # earns a log entry. Raise either if you get flickery one-off entries.
        "min_hits": 3,
        "min_seconds": 0.4,
    },
    "storage": {
        "db_path": "data/objectlog.db",
        "snapshot_dir": "data/snapshots",
        # Oldest entries (and their snapshots) are pruned past this.
        "max_entries": 4000,
        "snapshot_max_width": 720,
        "snapshot_quality": 82,
    },
    "web": {
        "host": "0.0.0.0",
        "port": 8010,
        # Serve the live MJPEG preview at /stream.mjpg
        "stream": True,
        "stream_max_width": 640,
        "stream_fps": 4,
        # How often the browser asks for new log entries.
        "poll_ms": 1500,
    },
}


class Config:
    """Dict-backed config with dotted-path access."""

    def __init__(self, data: Dict[str, Any], path: str | None = None):
        self.data = data
        self.path = path

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node = self.data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def __getitem__(self, key: str) -> Any:
        return self.data[key]


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load(path: str | None = None) -> Config:
    """Load config.yaml if it exists, merged over the defaults."""
    candidates = [path] if path else ["config.yaml", "config.yml"]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            try:
                import yaml  # noqa: PLC0415  (optional dependency)
            except ImportError:
                print(f"[config] PyYAML not installed, ignoring {candidate}")
                break
            with open(candidate, "r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"{candidate} must contain a YAML mapping")
            return Config(_deep_merge(DEFAULTS, loaded), candidate)
    if path:
        raise FileNotFoundError(f"config file not found: {path}")
    return Config(copy.deepcopy(DEFAULTS), None)
