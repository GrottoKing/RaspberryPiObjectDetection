"""The detection loop.

Runs on its own thread: grab a frame, detect, track, describe, log. The web
server reads from the same Store and asks this object for the latest frame
when someone is watching the live preview.
"""

from __future__ import annotations

import threading
import time
import traceback
from typing import Dict, List, Optional

import numpy as np

from . import camera as camera_mod
from . import describe as describe_mod
from .backends import build as build_backend
from .labels import category_for
from .store import Store
from .tracker import Tracker

# A track's snapshot is refreshed while it is young and the detector gets more
# confident -- so the saved image tends to be a good look at the object rather
# than the blurry frame it first appeared in.
SNAPSHOT_IMPROVE_MARGIN = 0.04
MAX_SNAPSHOT_UPDATES = 4


class Pipeline:
    def __init__(self, cfg, store: Store):
        self.cfg = cfg
        self.store = store
        self.rotation = int(cfg.get("camera.rotation", 0))
        self.fps_limit = float(cfg.get("camera.fps_limit", 4.0))

        self.camera = camera_mod.open_camera(cfg)
        self.detector = build_backend(cfg)
        self.tracker = Tracker(
            iou_threshold=float(cfg.get("tracker.iou_threshold", 0.3)),
            max_missing_seconds=float(cfg.get("tracker.max_missing_seconds", 2.0)),
            min_hits=int(cfg.get("tracker.min_hits", 3)),
            min_seconds=float(cfg.get("tracker.min_seconds", 0.4)),
            rejoin_seconds=float(cfg.get("tracker.rejoin_seconds", 900.0)),
            rejoin_iou=float(cfg.get("tracker.rejoin_iou", 0.4)),
        )
        self._seed_tracker_memory()

        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_boxes: List[dict] = []
        self._frame_count = 0
        self._detect_ms = 0.0
        self._loop_ms = 0.0
        self._started_at = time.time()
        self._last_error: Optional[str] = None
        self._consecutive_failures = 0

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _seed_tracker_memory(self) -> None:
        """Carry the room's known fixtures across a restart.

        Without this, stopping and starting the service logs every piece of
        furniture again as though it had never been seen.
        """
        window = self.tracker.rejoin_seconds
        if window <= 0:
            return
        try:
            rows = self.store.recent_boxes(time.time() - window)
        except Exception:
            return
        for row in rows:
            self.tracker.remember(row["label"], tuple(row["box"]), row["id"],
                                  row["first_seen"], row["last_seen"])
        if rows:
            print(f"[tracker] remembering {len(rows)} object(s) already logged")

    # ------------------------------------------------------------- lifecycle

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="detector",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        try:
            self.camera.close()
        except Exception:
            pass
        try:
            self.detector.close()
        except Exception:
            pass

    # ------------------------------------------------------------- main loop

    def _run(self) -> None:
        min_interval = 1.0 / self.fps_limit if self.fps_limit > 0 else 0.0
        while not self._stop.is_set():
            started = time.time()
            try:
                self._step()
                self._consecutive_failures = 0
            except Exception as exc:  # keep the service alive through hiccups
                self._consecutive_failures += 1
                self._last_error = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
                # Back off so a hard failure does not spin the CPU.
                self._stop.wait(min(30.0, 1.0 * self._consecutive_failures))
                continue

            elapsed = time.time() - started
            self._loop_ms = elapsed * 1000.0
            if min_interval > elapsed:
                self._stop.wait(min_interval - elapsed)

        # Nothing special to do for open tracks on shutdown: each was written
        # to the store when it was confirmed and refreshed on every frame.
        self.tracker.flush()

    def _step(self) -> None:
        frame = self.camera.read()
        if frame is None:
            raise RuntimeError("camera returned no frame")
        if self.rotation:
            frame = camera_mod.rotate(frame, self.rotation)

        detect_started = time.time()
        detections = self.detector.detect(frame)
        self._detect_ms = (time.time() - detect_started) * 1000.0
        self._frame_count += 1

        active, _closed = self.tracker.update(detections)

        confirmed = {t.track_id for t in self.tracker.confirmed()}
        live: List[dict] = []
        for track in active:
            if track.track_id not in confirmed:
                continue
            self._record(track, frame)
            live.append({
                "id": track.entry_id,
                "track": track.track_id,
                "label": track.label,
                "category": category_for(track.label),
                "description": (track.description or {}).get("text", track.label),
                "confidence": round(track.confidence, 3),
                "box": [round(v, 1) for v in track.box],
                "age": round(track.age, 1),
            })

        with self._lock:
            self._latest_frame = frame
            self._latest_boxes = live

    def _record(self, track, frame: np.ndarray) -> None:
        """Create or refresh this track's row in the log."""
        if track.entry_id is None:
            description = describe_mod.describe(frame, track.label, track.box)
            track.description = description
            track.best_confidence = track.confidence
            track.entry_id = self.store.add_sighting(
                label=track.label,
                description=description["text"],
                confidence=track.confidence,
                first_seen=track.first_seen,
                last_seen=track.last_seen,
                attributes=description["attributes"],
                swatch=description["swatch"],
                box=list(track.box),
                frame=frame,
            )
            return

        improved = track.confidence > track.best_confidence + SNAPSHOT_IMPROVE_MARGIN
        if improved and track.snapshot_updates < MAX_SNAPSHOT_UPDATES:
            # Better look at the object: re-describe and replace the snapshot.
            track.best_confidence = track.confidence
            track.snapshot_updates += 1
            description = describe_mod.describe(frame, track.label, track.box)
            track.description = description
            self.store.write_snapshot(track.entry_id, frame, list(track.box),
                                      description["text"])
            self.store.touch(
                track.entry_id, last_seen=track.last_seen,
                confidence=track.confidence, description=description["text"],
                attributes=description["attributes"], swatch=description["swatch"],
                box=list(track.box))
        else:
            self.store.touch(track.entry_id, last_seen=track.last_seen,
                             confidence=track.confidence, box=list(track.box))

    # --------------------------------------------------------------- readers

    def live_objects(self) -> List[dict]:
        with self._lock:
            return list(self._latest_boxes)

    def snapshot_frame(self, max_width: int = 640, draw_boxes: bool = True):
        """A JPEG of the current view, for the live preview stream."""
        with self._lock:
            frame = self._latest_frame
            boxes = list(self._latest_boxes)
        if frame is None:
            return None

        from PIL import Image, ImageDraw  # noqa: PLC0415

        import io

        image = Image.fromarray(frame)
        if draw_boxes and boxes:
            draw = ImageDraw.Draw(image)
            width = max(2, int(min(image.size) * 0.005))
            for item in boxes:
                x0, y0, x1, y1 = item["box"]
                draw.rectangle([x0, y0, x1, y1], outline=(255, 214, 10),
                               width=width)
                draw.text((x0 + 4, max(0, y0 - 12)), item["description"][:40],
                          fill=(255, 214, 10))
        if image.width > max_width:
            ratio = max_width / image.width
            image = image.resize((max_width, max(1, int(image.height * ratio))),
                                 Image.BILINEAR)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=70)
        return buffer.getvalue()

    def status(self) -> Dict[str, object]:
        uptime = time.time() - self._started_at
        return {
            "camera": self.camera.name,
            "backend": self.detector.name,
            "backend_detail": getattr(self.detector, "description", ""),
            "frames": self._frame_count,
            "detect_ms": round(self._detect_ms, 1),
            "loop_ms": round(self._loop_ms, 1),
            "fps": round(self._frame_count / uptime, 2) if uptime > 0 else 0.0,
            "uptime": round(uptime, 1),
            "live_count": len(self.live_objects()),
            "tracks_open": len(self.tracker.tracks),
            "error": self._last_error,
        }
