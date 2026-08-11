"""The LAN web page and its JSON API."""

from __future__ import annotations

import os
import time
from typing import Optional

from flask import (Flask, Response, jsonify, render_template, request,
                   send_from_directory)

from ..labels import category_sort_key


def create_app(cfg, store, pipeline=None) -> Flask:
    app = Flask(__name__)
    snapshot_dir = os.path.abspath(cfg.get("storage.snapshot_dir",
                                           "data/snapshots"))

    @app.route("/")
    def index():
        return render_template(
            "index.html",
            poll_ms=int(cfg.get("web.poll_ms", 1500)),
            stream_enabled=bool(cfg.get("web.stream", True)) and pipeline is not None,
        )

    @app.route("/snapshots/<path:filename>")
    def snapshot(filename: str):
        response = send_from_directory(snapshot_dir, filename)
        # Snapshots are immutable once a track closes, but an open track can
        # replace one, so keep the cache short rather than forever.
        response.headers["Cache-Control"] = "public, max-age=30"
        return response

    @app.route("/api/state")
    def api_state():
        since_id = request.args.get("since_id", type=int, default=0)
        since_ts = request.args.get("since_ts", type=float, default=0.0)
        limit = min(request.args.get("limit", type=int, default=250), 1000)

        if since_id <= 0 and since_ts <= 0:
            entries = store.sightings(limit=limit)
        else:
            merged = {}
            for row in store.sightings(since_id=since_id, limit=limit):
                merged[row["id"]] = row
            # Open tracks keep moving their last_seen, so pick those up too.
            for row in store.updated_since(since_ts, limit=120):
                merged[row["id"]] = row
            entries = sorted(merged.values(), key=lambda r: r["id"], reverse=True)

        categories = sorted(store.categories(),
                            key=lambda c: category_sort_key(c["name"]))
        return jsonify({
            "entries": entries,
            "categories": categories,
            "labels": store.label_counts(),
            "stats": store.stats(),
            "live": pipeline.live_objects() if pipeline else [],
            "status": pipeline.status() if pipeline else {"backend": "none"},
            "server_time": time.time(),
        })

    @app.route("/api/log")
    def api_log():
        return jsonify({
            "entries": store.sightings(
                before_id=request.args.get("before_id", type=int),
                category=request.args.get("category"),
                limit=min(request.args.get("limit", type=int, default=200), 1000),
            ),
        })

    @app.route("/api/clear", methods=["POST"])
    def api_clear():
        removed = store.clear()
        return jsonify({"cleared": removed})

    @app.route("/stream.mjpg")
    def stream():
        if pipeline is None or not cfg.get("web.stream", True):
            return Response("live preview disabled", status=404,
                            mimetype="text/plain")

        max_width = int(cfg.get("web.stream_max_width", 640))
        interval = 1.0 / max(1.0, float(cfg.get("web.stream_fps", 4)))

        def frames():
            while True:
                jpeg: Optional[bytes] = pipeline.snapshot_frame(max_width)
                if jpeg:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
                           b"Content-Length: " + str(len(jpeg)).encode() +
                           b"\r\n\r\n" + jpeg + b"\r\n")
                time.sleep(interval)

        return Response(frames(),
                        mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/healthz")
    def healthz():
        status = pipeline.status() if pipeline else {}
        healthy = pipeline is None or status.get("error") is None
        return jsonify({"ok": healthy, "status": status}), (200 if healthy else 503)

    return app
