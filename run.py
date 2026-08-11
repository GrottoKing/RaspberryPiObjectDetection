#!/usr/bin/env python3
"""Start the camera sightings log.

    python3 run.py                       # use config.yaml / built-in defaults
    python3 run.py --camera synthetic --backend mock   # no hardware needed
    python3 run.py --port 8080 --fps 2
"""

from __future__ import annotations

import argparse
import signal
import socket
import sys

from objectlog import __version__, config as config_mod
from objectlog.backends import describe_model_hint
from objectlog.pipeline import Pipeline
from objectlog.store import Store
from objectlog.web.app import create_app


def local_ip() -> str:
    """Best guess at this machine's LAN address, for the startup banner."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # No packets are actually sent; this just picks the outbound interface.
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        sock.close()


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Log everything the camera sees, and serve it on the LAN.")
    parser.add_argument("-c", "--config", help="path to a config.yaml")
    parser.add_argument("--camera",
                        choices=["auto", "picamera2", "opencv", "folder", "synthetic"],
                        help="frame source")
    parser.add_argument("--folder", help="image folder for --camera folder")
    parser.add_argument("--backend", choices=["auto", "hailo", "onnx", "ultralytics", "mock"],
                        help="detector backend")
    parser.add_argument("--model", help="path to the detection model")
    parser.add_argument("--hef", help="path to a Hailo .hef (accelerator)")
    parser.add_argument("--conf", type=float, help="detection confidence threshold")
    parser.add_argument("--fps", type=float, help="max detections per second")
    parser.add_argument("--rotation", type=int, choices=[0, 90, 180, 270],
                        help="rotate the camera image")
    parser.add_argument("--host", help="web bind address (default 0.0.0.0)")
    parser.add_argument("--port", type=int, help="web port (default 8010)")
    parser.add_argument("--no-stream", action="store_true",
                        help="disable the live MJPEG preview")
    parser.add_argument("--reset", action="store_true",
                        help="wipe the existing log before starting")
    parser.add_argument("--version", action="version", version=__version__)
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace):
    cfg = config_mod.load(args.config)
    overrides = {
        "camera.source": args.camera,
        "camera.rotation": args.rotation,
        "camera.fps_limit": args.fps,
        "camera.folder": args.folder,
        "detector.backend": args.backend,
        "detector.model": args.model,
        "detector.hef": args.hef,
        "detector.confidence": args.conf,
        "web.host": args.host,
        "web.port": args.port,
    }
    for key, value in overrides.items():
        if value is not None:
            cfg.set(key, value)
    if args.no_stream:
        cfg.set("web.stream", False)
    return cfg


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = build_config(args)

    hint = describe_model_hint(cfg)
    if hint:
        print(f"[detector] {hint}")

    store = Store(
        db_path=cfg.get("storage.db_path", "data/objectlog.db"),
        snapshot_dir=cfg.get("storage.snapshot_dir", "data/snapshots"),
        max_entries=int(cfg.get("storage.max_entries", 4000)),
        snapshot_max_width=int(cfg.get("storage.snapshot_max_width", 720)),
        snapshot_quality=int(cfg.get("storage.snapshot_quality", 82)),
    )
    if args.reset:
        print(f"[store] cleared {store.clear()} previous sightings")

    try:
        pipeline = Pipeline(cfg, store)
    except Exception as exc:
        print(f"[fatal] could not start the detector: {exc}", file=sys.stderr)
        print("        try:  python3 run.py --camera synthetic --backend mock",
              file=sys.stderr)
        store.close()
        return 1

    pipeline.start()

    host = cfg.get("web.host", "0.0.0.0")
    port = int(cfg.get("web.port", 8010))
    status = pipeline.status()
    print()
    print(f"  camera   : {status['camera']}")
    print(f"  detector : {status['backend_detail'] or status['backend']}")
    print(f"  log      : {cfg.get('storage.db_path')}")
    print()
    print("  Open from any device on your network:")
    print(f"    http://{local_ip()}:{port}/")
    print(f"    http://{socket.gethostname()}.local:{port}/")
    print()

    app = create_app(cfg, store, pipeline)

    def shutdown(_signum, _frame):
        print("\n[shutdown] stopping detector…")
        pipeline.stop()
        store.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        # threaded=True matters: the MJPEG stream holds a connection open, and
        # the page must stay responsive while it does.
        app.run(host=host, port=port, threaded=True, debug=False,
                use_reloader=False)
    finally:
        pipeline.stop()
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
