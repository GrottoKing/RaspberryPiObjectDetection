#!/usr/bin/env python3
"""Find out what your camera is actually detecting, and at what confidence.

Samples frames from the real camera, runs detection with the threshold turned
right down, and reports every class it saw with the confidence scores it gave
them. That tells you where to set `detector.confidence` and which classes to
exclude, based on your room rather than on guesswork.

    python3 scripts/tune.py                    # 20 frames over ~20 seconds
    python3 scripts/tune.py --frames 60 --interval 2
    python3 scripts/tune.py --save-frames out/ # also write annotated images

Point the camera at the empty scene -- no people, nothing you want detected --
and everything it reports is by definition a false positive.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from objectlog import camera as camera_mod  # noqa: E402
from objectlog import config as config_mod  # noqa: E402
from objectlog.backends import build as build_backend  # noqa: E402

# Thresholds the summary reports against.
LADDER = [0.25, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--config", help="path to a config.yaml")
    parser.add_argument("--frames", type=int, default=20,
                        help="how many frames to sample (default 20)")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="seconds between frames (default 1)")
    parser.add_argument("--floor", type=float, default=0.20,
                        help="lowest confidence to report (default 0.20)")
    parser.add_argument("--camera", help="override the camera source")
    parser.add_argument("--folder", help="image folder for --camera folder")
    parser.add_argument("--save-frames", metavar="DIR",
                        help="write annotated frames here for inspection")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = config_mod.load(args.config)

    # Run wide open: we want to see what the detector *would* report, including
    # the weak detections the normal threshold hides.
    cfg.set("detector.confidence", args.floor)
    cfg.set("detector.classes", [])
    cfg.set("detector.exclude_classes", [])
    cfg.set("detector.min_box_area", 0.0)
    cfg.set("detector.max_box_area", 1.0)
    cfg.set("detector.ignore_regions", [])
    if args.camera:
        cfg.set("camera.source", args.camera)
    if args.folder:
        cfg.set("camera.folder", args.folder)

    camera = camera_mod.open_camera(cfg)
    detector = build_backend(cfg)
    rotation = int(cfg.get("camera.rotation", 0))

    print(f"camera   : {camera.name}")
    print(f"detector : {getattr(detector, 'description', detector.name)}")
    print(f"sampling : {args.frames} frames, {args.interval}s apart "
          f"(~{args.frames * args.interval:.0f}s)")
    print()
    print("Point the camera at the scene as it normally sits, with nothing in")
    print("it you actually want logged. Anything reported is a false positive.")
    print()

    if args.save_frames:
        os.makedirs(args.save_frames, exist_ok=True)

    scores = defaultdict(list)      # label -> [confidence, ...]
    areas = defaultdict(list)       # label -> [area fraction, ...]
    frames_seen = 0

    try:
        for index in range(args.frames):
            frame = camera.read()
            if frame is None:
                print(f"  frame {index + 1}: camera returned nothing")
                continue
            if rotation:
                frame = camera_mod.rotate(frame, rotation)
            frames_seen += 1

            detections = detector.detect(frame)
            frame_area = float(frame.shape[0] * frame.shape[1])
            for det in detections:
                scores[det.label].append(det.confidence)
                box = det.box
                areas[det.label].append(
                    ((box[2] - box[0]) * (box[3] - box[1])) / frame_area)

            summary = ", ".join(
                f"{d.label} {d.confidence:.2f}" for d in
                sorted(detections, key=lambda d: -d.confidence)[:6]) or "nothing"
            print(f"  frame {index + 1:>3}/{args.frames}: {summary}")

            if args.save_frames and detections:
                from objectlog.store import annotate

                from PIL import Image

                best = max(detections, key=lambda d: d.confidence)
                image = annotate(frame, best.box,
                                 f"{best.label} {best.confidence:.2f}", 900)
                Image.fromarray(image).save(
                    os.path.join(args.save_frames, f"frame{index + 1:03d}.jpg"))

            if index < args.frames - 1:
                time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n(stopped early)")
    finally:
        camera.close()

    # ------------------------------------------------------------- reporting

    print()
    print("=" * 72)
    if not scores:
        print(f"Nothing detected at all in {frames_seen} frames, even at "
              f"confidence {args.floor}.")
        print("Your current settings are not producing false positives here.")
        return 0

    print(f"WHAT IT SAW  ({frames_seen} frames, confidence floor {args.floor})")
    print("=" * 72)
    print(f"{'class':<16}{'frames':>7}{'max':>7}{'median':>8}{'typical size':>14}")
    print("-" * 72)
    ranked = sorted(scores.items(), key=lambda kv: -max(kv[1]))
    for label, values in ranked:
        ordered = sorted(values)
        median = ordered[len(ordered) // 2]
        area = sorted(areas[label])[len(areas[label]) // 2]
        print(f"{label:<16}{len(values):>7}{max(values):>7.2f}"
              f"{median:>8.2f}{area * 100:>13.1f}%")

    print()
    print("=" * 72)
    print("EFFECT OF RAISING detector.confidence")
    print("=" * 72)
    for threshold in LADDER:
        surviving = {label: [v for v in values if v >= threshold]
                     for label, values in scores.items()}
        surviving = {k: v for k, v in surviving.items() if v}
        if not surviving:
            print(f"  {threshold:.2f}  ->  nothing survives")
            continue
        listed = ", ".join(f"{label} x{len(v)}" for label, v in
                           sorted(surviving.items(), key=lambda kv: -len(kv[1])))
        print(f"  {threshold:.2f}  ->  {listed}")

    # ---------------------------------------------------------- suggestions

    highest = max(max(v) for v in scores.values())
    suggested = None
    for threshold in LADDER:
        if all(max(v) < threshold for v in scores.values()):
            suggested = threshold
            break

    print()
    print("=" * 72)
    print("SUGGESTED config.yaml")
    print("=" * 72)
    print("detector:")
    if suggested is not None:
        print(f"  confidence: {suggested:.2f}"
              f"   # above the strongest false positive seen ({highest:.2f})")
    else:
        print(f"  confidence: 0.60   # note: a false positive reached "
              f"{highest:.2f}, so")
        print("                     # the threshold alone will not clear these")
    print("  exclude_classes:")
    for label, _values in ranked:
        print(f"    - {label}")
    print()
    print("Remove from exclude_classes anything you DO want logged. Excluding a")
    print("class is the reliable fix -- a threshold high enough to stop a")
    print("stubborn false positive will also miss real things.")
    print()
    print("Then: sudo systemctl restart objectlog")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
