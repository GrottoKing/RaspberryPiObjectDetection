#!/usr/bin/env python3
"""Fetch a YOLO detection model in ONNX form.

Three ways to get one, tried in order:

1. Direct download of a pre-exported .onnx (no PyTorch needed) -- best on a Pi.
2. Export it locally with ultralytics, if that package is installed.
3. Tell you what to do by hand.

    python3 scripts/fetch_model.py                 # yolo11n, 640px
    python3 scripts/fetch_model.py --model yolo11s # a bit slower, more accurate
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import urllib.request

# Pre-exported ONNX weights, newest first. Each entry is a list of mirrors.
SOURCES = {
    "yolo11n": [
        "https://huggingface.co/Ultralytics/YOLO11/resolve/main/yolo11n.onnx?download=true",
    ],
    "yolo11s": [
        "https://huggingface.co/Ultralytics/YOLO11/resolve/main/yolo11s.onnx?download=true",
    ],
    "yolov8n": [
        "https://huggingface.co/Ultralytics/YOLOv8/resolve/main/yolov8n.onnx?download=true",
    ],
}


def download(url: str, destination: str) -> bool:
    print(f"  trying {url.split('?')[0]}")
    temporary = destination + ".part"
    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": "objectlog/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response:
            if response.status != 200:
                print(f"    HTTP {response.status}")
                return False
            with open(temporary, "wb") as handle:
                shutil.copyfileobj(response, handle)
    except Exception as exc:
        print(f"    failed: {exc}")
        if os.path.exists(temporary):
            os.remove(temporary)
        return False

    # A truncated or HTML error page is not a model; ONNX files start with the
    # protobuf field for ir_version and are megabytes, not kilobytes.
    if os.path.getsize(temporary) < 1_000_000:
        print("    file is too small to be a model, discarding")
        os.remove(temporary)
        return False
    os.replace(temporary, destination)
    return True


def export_with_ultralytics(model: str, destination: str, imgsz: int,
                            dynamic: bool = True) -> bool:
    try:
        from ultralytics import YOLO
    except ImportError:
        return False
    print(f"  exporting {model}.pt with ultralytics (this downloads weights)")
    try:
        yolo = YOLO(f"{model}.pt")
        # dynamic=True lets the detector feed a rectangular frame (e.g.
        # 640x384 for 16:9) instead of padding out to a full square, which is
        # both quicker and slightly more accurate.
        produced = yolo.export(format="onnx", imgsz=imgsz, opset=12,
                               dynamic=dynamic, simplify=False)
    except Exception as exc:
        print(f"    export failed: {exc}")
        if dynamic:
            print("    retrying with a fixed input size")
            return export_with_ultralytics(model, destination, imgsz, False)
        return False
    shutil.move(str(produced), destination)
    return True


def verify(path: str) -> bool:
    try:
        import onnxruntime as ort
    except ImportError:
        print("  (onnxruntime not installed, skipping verification)")
        return True
    try:
        session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    except Exception as exc:
        print(f"  model will not load: {exc}")
        return False
    shape = session.get_inputs()[0].shape
    outputs = [o.shape for o in session.get_outputs()]
    print(f"  verified · input {shape} · output {outputs}")
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="yolo11n", choices=sorted(SOURCES),
                        help="which model to fetch (default: yolo11n)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="input size when exporting locally")
    parser.add_argument("--out", default=None, help="output path")
    parser.add_argument("--force", action="store_true",
                        help="re-download even if the file exists")
    parser.add_argument("--static", action="store_true",
                        help="export a fixed 640x640 input instead of dynamic")
    args = parser.parse_args(argv)

    destination = args.out or os.path.join("models", f"{args.model}.onnx")
    os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)

    if os.path.exists(destination) and not args.force:
        print(f"{destination} already exists (use --force to replace it)")
        return 0 if verify(destination) else 1

    print(f"Fetching {args.model} -> {destination}")
    for url in SOURCES[args.model]:
        if download(url, destination):
            break
    else:
        print("  direct download unavailable, trying a local export")
        if not export_with_ultralytics(args.model, destination, args.imgsz,
                                       dynamic=not args.static):
            print()
            print("Could not fetch a model automatically. Two options:")
            print("  1. pip install ultralytics && python3 scripts/fetch_model.py")
            print("  2. Export on another machine and copy the .onnx into models/")
            print("  3. Or just run the demo with no model at all:")
            print("       python3 run.py --camera synthetic --backend mock")
            return 1

    if not verify(destination):
        return 1
    size_mb = os.path.getsize(destination) / 1e6
    print(f"\nDone: {destination} ({size_mb:.1f} MB)")
    print("Start it with:  python3 run.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
