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
    # Where it was first seen, so we can tell furniture from passers-by.
    first_box: Optional[Box] = None
    # Set once the track has been written to the store.
    entry_id: Optional[int] = None
    # Best confidence seen so far -- decides whether to refresh the snapshot.
    best_confidence: float = 0.0
    snapshot_updates: int = 0
    description: Optional[dict] = None
    # True when this track picked up an existing log entry rather than
    # starting a new one.
    rejoined: bool = False

    @property
    def age(self) -> float:
        return self.last_seen - self.first_seen

    @property
    def stationary(self) -> bool:
        """Has this object stayed put for its whole life?

        The distinction that matters for re-logging: a shelf that vanishes for
        a moment and comes back is the same shelf, but two people standing in
        the same doorway an hour apart are two sightings.
        """
        if self.first_box is None:
            return True
        return iou(self.first_box, self.box) >= 0.5


@dataclass
class Detection:
    label: str
    confidence: float
    box: Box


class Tracker:
    """Groups detections across frames into one entry per real object.

    Thresholds are in *seconds*, not frames. They were frame counts once, which
    quietly changed meaning whenever the frame rate did -- moving from 4fps to
    15fps shrank the tolerance for a missed detection from 3 seconds to 0.8,
    and a borderline detection that flickers would then be logged over and over
    as a "new" object. Wall-clock is the unit that actually matters here.
    """

    def __init__(self, iou_threshold: float = 0.3,
                 max_missing_seconds: float = 2.0, min_hits: int = 3,
                 min_seconds: float = 0.4, rejoin_seconds: float = 900.0,
                 rejoin_iou: float = 0.4):
        self.iou_threshold = iou_threshold
        self.max_missing_seconds = float(max_missing_seconds)
        self.min_hits = min_hits
        self.min_seconds = float(min_seconds)
        # How long a stationary object is remembered after it disappears, and
        # how well a new detection must overlap it to count as the same thing.
        self.rejoin_seconds = float(rejoin_seconds)
        self.rejoin_iou = float(rejoin_iou)
        self.tracks: Dict[int, Track] = {}
        self._ids = itertools.count(1)
        # Closed stationary tracks, kept so the fixtures of a room are not
        # logged afresh every time the detector blinks: (label, box, entry_id,
        # first_seen, closed_at).
        self._remembered: List[dict] = []

    def remember(self, label: str, box: Box, entry_id: Optional[int],
                 first_seen: float, closed_at: Optional[float] = None) -> None:
        """Note a stationary object so a later sighting resumes its entry."""
        if entry_id is None:
            return
        self._remembered = [r for r in self._remembered
                            if r["entry_id"] != entry_id]
        self._remembered.append({
            "label": label, "box": tuple(box), "entry_id": entry_id,
            "first_seen": first_seen,
            "closed_at": closed_at if closed_at is not None else time.time(),
        })

    def _rejoin(self, det: Detection, now: float) -> Optional[dict]:
        """Find a remembered stationary object matching this detection."""
        if self.rejoin_seconds <= 0:
            return None
        best, best_score = None, 0.0
        for entry in self._remembered:
            if entry["label"] != det.label:
                continue
            if now - entry["closed_at"] > self.rejoin_seconds:
                continue
            score = iou(entry["box"], det.box)
            if score >= self.rejoin_iou and score > best_score:
                best, best_score = entry, score
        return best

    def _forget_stale(self, now: float) -> None:
        self._remembered = [
            r for r in self._remembered
            if now - r["closed_at"] <= self.rejoin_seconds]

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
            track = Track(
                track_id=track_id,
                label=det.label,
                box=det.box,
                confidence=det.confidence,
                first_seen=now,
                last_seen=now,
                first_box=det.box,
            )
            # Something stationary we have logged before: reopen that entry
            # rather than reporting the room's furniture as a new sighting.
            remembered = self._rejoin(det, now)
            if remembered is not None:
                track.entry_id = remembered["entry_id"]
                track.first_seen = remembered["first_seen"]
                track.rejoined = True
                self._remembered.remove(remembered)
            self.tracks[track_id] = track

        closed: List[Track] = []
        for track_id, track in list(self.tracks.items()):
            if track_id in used_tracks or track_id in fresh:
                continue  # matched this frame, or created this frame
            track.missing += 1
            if now - track.last_seen > self.max_missing_seconds:
                closed.append(self.tracks.pop(track_id))
                if track.stationary and track.entry_id is not None:
                    self.remember(track.label, track.box, track.entry_id,
                                  track.first_seen, now)

        self._forget_stale(now)

        active = [t for t in self.tracks.values() if t.missing == 0]
        return active, closed

    def confirmed(self) -> List[Track]:
        """Tracks seen often enough, and for long enough, to be believed.

        The frame count alone is not enough: at 15fps three frames is a fifth
        of a second, which a flicker easily clears.
        """
        return [t for t in self.tracks.values()
                if t.hits >= self.min_hits and t.age >= self.min_seconds]

    def flush(self) -> List[Track]:
        """Close every track (used at shutdown)."""
        closed = list(self.tracks.values())
        self.tracks.clear()
        return closed
