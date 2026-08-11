"""SQLite-backed sighting log plus snapshot files on disk.

One row per object the camera saw, one JPEG per row. The log survives
restarts, and old entries are pruned along with their images so a Pi's SD card
does not fill up over a long run.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np

from .labels import category_for

SCHEMA = """
CREATE TABLE IF NOT EXISTS sightings (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    label         TEXT    NOT NULL,
    description   TEXT    NOT NULL,
    category      TEXT    NOT NULL,
    confidence    REAL    NOT NULL,
    first_seen    REAL    NOT NULL,
    last_seen     REAL    NOT NULL,
    snapshot      TEXT,
    attributes    TEXT    NOT NULL DEFAULT '{}',
    swatch        TEXT,
    box           TEXT
);
CREATE INDEX IF NOT EXISTS idx_sightings_first_seen ON sightings(first_seen DESC);
CREATE INDEX IF NOT EXISTS idx_sightings_category   ON sightings(category);
"""


def _to_jpeg(frame: np.ndarray, quality: int) -> bytes:
    from PIL import Image  # noqa: PLC0415

    import io

    buffer = io.BytesIO()
    Image.fromarray(frame).save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def annotate(frame: np.ndarray, box, label: str, max_width: int = 720) -> np.ndarray:
    """Draw the object's box on a copy of the frame and downscale it.

    The snapshot is what the user sees when hovering, so it wants to be small
    (fast to load) but still show where in the scene the thing was.
    """
    from PIL import Image, ImageDraw  # noqa: PLC0415

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = [float(v) for v in box]
    width = max(2, int(min(image.size) * 0.006))
    draw.rectangle([x0, y0, x1, y1], outline=(255, 214, 10), width=width)

    # A filled caption strip, placed inside the frame near the box.
    text = label[:48]
    pad = 4
    try:
        bbox = draw.textbbox((0, 0), text)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:  # pragma: no cover - very old Pillow
        tw, th = 6 * len(text), 11
    ty = max(0, y0 - th - 2 * pad)
    draw.rectangle([x0, ty, x0 + tw + 2 * pad, ty + th + 2 * pad],
                   fill=(255, 214, 10))
    draw.text((x0 + pad, ty + pad), text, fill=(20, 20, 20))

    if image.width > max_width:
        ratio = max_width / image.width
        image = image.resize((max_width, max(1, int(image.height * ratio))),
                             Image.BILINEAR)
    return np.asarray(image)


class Store:
    def __init__(self, db_path: str, snapshot_dir: str, max_entries: int = 4000,
                 snapshot_max_width: int = 720, snapshot_quality: int = 82):
        self.db_path = db_path
        self.snapshot_dir = snapshot_dir
        self.max_entries = max_entries
        self.snapshot_max_width = snapshot_max_width
        self.snapshot_quality = snapshot_quality
        self._lock = threading.Lock()

        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        os.makedirs(snapshot_dir, exist_ok=True)

        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        # WAL keeps the web thread's reads from blocking the detector's writes.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.commit()

    # ---------------------------------------------------------------- writes

    def add_sighting(self, *, label: str, description: str, confidence: float,
                     first_seen: float, last_seen: float,
                     attributes: Optional[dict] = None,
                     swatch: Optional[str] = None,
                     box: Optional[list] = None,
                     frame: Optional[np.ndarray] = None) -> int:
        """Insert a sighting, writing its snapshot if a frame is supplied."""
        category = category_for(label)
        with self._lock:
            cursor = self._conn.execute(
                """INSERT INTO sightings
                   (label, description, category, confidence, first_seen,
                    last_seen, snapshot, attributes, swatch, box)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (label, description, category, float(confidence),
                 float(first_seen), float(last_seen), None,
                 json.dumps(attributes or {}), swatch,
                 json.dumps([float(v) for v in box]) if box else None),
            )
            entry_id = int(cursor.lastrowid)
            self._conn.commit()

        if frame is not None:
            self.write_snapshot(entry_id, frame, box, description)
        self._prune()
        return entry_id

    def write_snapshot(self, entry_id: int, frame: np.ndarray, box,
                       label: str) -> str:
        """Render and store the snapshot for an entry (overwrites in place)."""
        filename = f"{entry_id:08d}.jpg"
        path = os.path.join(self.snapshot_dir, filename)
        image = annotate(frame, box, label, self.snapshot_max_width) if box is not None else frame
        data = _to_jpeg(image, self.snapshot_quality)
        tmp = path + ".tmp"
        with open(tmp, "wb") as handle:
            handle.write(data)
        os.replace(tmp, path)  # atomic, so the browser never sees half a file
        with self._lock:
            self._conn.execute("UPDATE sightings SET snapshot=? WHERE id=?",
                               (filename, entry_id))
            self._conn.commit()
        return filename

    def touch(self, entry_id: int, *, last_seen: float, confidence: float,
              description: Optional[str] = None,
              attributes: Optional[dict] = None,
              swatch: Optional[str] = None, box: Optional[list] = None) -> None:
        """Update an open sighting as its track keeps being seen."""
        fields = ["last_seen=?", "confidence=?"]
        values: List[Any] = [float(last_seen), float(confidence)]
        if description is not None:
            fields.append("description=?")
            values.append(description)
        if attributes is not None:
            fields.append("attributes=?")
            values.append(json.dumps(attributes))
        if swatch is not None:
            fields.append("swatch=?")
            values.append(swatch)
        if box is not None:
            fields.append("box=?")
            values.append(json.dumps([float(v) for v in box]))
        values.append(entry_id)
        with self._lock:
            self._conn.execute(
                f"UPDATE sightings SET {', '.join(fields)} WHERE id=?", values)
            self._conn.commit()

    def clear(self) -> int:
        """Wipe the log and every snapshot."""
        with self._lock:
            count = self._conn.execute(
                "SELECT COUNT(*) FROM sightings").fetchone()[0]
            self._conn.execute("DELETE FROM sightings")
            self._conn.commit()
        for name in os.listdir(self.snapshot_dir):
            if name.endswith(".jpg"):
                try:
                    os.remove(os.path.join(self.snapshot_dir, name))
                except OSError:
                    pass
        return int(count)

    def _prune(self) -> None:
        if self.max_entries <= 0:
            return
        with self._lock:
            rows = self._conn.execute(
                """SELECT id, snapshot FROM sightings
                   ORDER BY id DESC LIMIT -1 OFFSET ?""",
                (self.max_entries,),
            ).fetchall()
            if not rows:
                return
            self._conn.executemany("DELETE FROM sightings WHERE id=?",
                                   [(row["id"],) for row in rows])
            self._conn.commit()
        for row in rows:
            if row["snapshot"]:
                try:
                    os.remove(os.path.join(self.snapshot_dir, row["snapshot"]))
                except OSError:
                    pass

    # ----------------------------------------------------------------- reads

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": row["id"],
            "label": row["label"],
            "description": row["description"],
            "category": row["category"],
            "confidence": round(row["confidence"], 3),
            "first_seen": row["first_seen"],
            "last_seen": row["last_seen"],
            "duration": round(max(0.0, row["last_seen"] - row["first_seen"]), 1),
            "snapshot": f"/snapshots/{row['snapshot']}" if row["snapshot"] else None,
            "attributes": json.loads(row["attributes"] or "{}"),
            "swatch": row["swatch"],
        }

    def sightings(self, *, since_id: int = 0, before_id: Optional[int] = None,
                  limit: int = 300,
                  category: Optional[str] = None) -> List[Dict[str, Any]]:
        query = "SELECT * FROM sightings WHERE id > ?"
        params: List[Any] = [since_id]
        if before_id:
            query += " AND id < ?"
            params.append(before_id)
        if category:
            query += " AND category = ?"
            params.append(category)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def updated_since(self, timestamp: float, limit: int = 300) -> List[Dict[str, Any]]:
        """Entries whose last_seen moved recently (open tracks being refreshed)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM sightings WHERE last_seen > ? ORDER BY id DESC LIMIT ?",
                (timestamp, limit),
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def categories(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT category, COUNT(*) AS count, MAX(last_seen) AS latest
                   FROM sightings GROUP BY category"""
            ).fetchall()
        return [
            {"name": row["category"], "count": row["count"],
             "latest": row["latest"]}
            for row in rows
        ]

    def label_counts(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT label, category, COUNT(*) AS count,
                          MAX(last_seen) AS latest
                   FROM sightings GROUP BY label, category
                   ORDER BY count DESC"""
            ).fetchall()
        return [dict(row) for row in rows]

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            row = self._conn.execute(
                """SELECT COUNT(*) AS total, MIN(first_seen) AS earliest,
                          MAX(last_seen) AS latest FROM sightings"""
            ).fetchone()
        return {
            "total": row["total"],
            "earliest": row["earliest"],
            "latest": row["latest"],
            "now": time.time(),
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()
