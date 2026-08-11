"""A small IoU tracker.

Without this, a person standing in front of the camera for ten seconds would
produce forty identical log lines. With it, they produce one entry with a
first-seen time, a last-seen time and a single best snapshot.

Deliberately simple: greedy IoU matching, no motion model. That is enough at
the few-frames-per-second the Pi actually runs at.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

Box = Tuple[float, float, float, float]


def iou(a: Box, b: Box) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    track_id: int
    label: str
    box: Box
    confidence: float
    hits: int = 1
    missing: int = 0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    # Set once the track has been written to the store.
    entry_id: Optional[int] = None
    # Best confidence seen so far -- decides whether to refresh the snapshot.
    best_confidence: float = 0.0
    snapshot_updates: int = 0
    description: Optional[dict] = None

    @property
    def age(self) -> float:
        return self.last_seen - self.first_seen


@dataclass
class Detection:
    label: str
    confidence: float
    box: Box


class Tracker:
    def __init__(self, iou_threshold: float = 0.3, max_missing: int = 12,
                 min_hits: int = 3):
        self.iou_threshold = iou_threshold
        self.max_missing = max_missing
        self.min_hits = min_hits
        self.tracks: Dict[int, Track] = {}
        self._ids = itertools.count(1)

    def update(self, detections: Sequence[Detection]) -> Tuple[List[Track], List[Track]]:
        """Advance the tracker one frame.

        Returns (active_tracks, closed_tracks). A track is closed once it has
        been missing for `max_missing` consecutive frames.
        """
        now = time.time()
        unmatched = list(range(len(detections)))

        # Greedy match: strongest IoU pair first, one detection per track.
        pairs = []
        for track_id, track in self.tracks.items():
            for det_index in unmatched:
                det = detections[det_index]
                if det.label != track.label:
                    continue
                score = iou(track.box, det.box)
                if score >= self.iou_threshold:
                    pairs.append((score, track_id, det_index))
        pairs.sort(reverse=True)

        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        for _score, track_id, det_index in pairs:
            if track_id in used_tracks or det_index in used_dets:
                continue
            used_tracks.add(track_id)
            used_dets.add(det_index)
            det = detections[det_index]
            track = self.tracks[track_id]
            track.box = det.box
            track.confidence = det.confidence
            track.hits += 1
            track.missing = 0
            track.last_seen = now

        fresh: set[int] = set()
        for det_index, det in enumerate(detections):
            if det_index in used_dets:
                continue
            track_id = next(self._ids)
            fresh.add(track_id)
            self.tracks[track_id] = Track(
                track_id=track_id,
                label=det.label,
                box=det.box,
                confidence=det.confidence,
                first_seen=now,
                last_seen=now,
            )

        closed: List[Track] = []
        for track_id, track in list(self.tracks.items()):
            if track_id in used_tracks or track_id in fresh:
                continue  # matched this frame, or created this frame
            track.missing += 1
            if track.missing > self.max_missing:
                closed.append(self.tracks.pop(track_id))

        active = [t for t in self.tracks.values() if t.missing == 0]
        return active, closed

    def confirmed(self) -> List[Track]:
        """Tracks seen often enough to be believed."""
        return [t for t in self.tracks.values() if t.hits >= self.min_hits]

    def flush(self) -> List[Track]:
        """Close every track (used at shutdown)."""
        closed = list(self.tracks.values())
        self.tracks.clear()
        return closed
