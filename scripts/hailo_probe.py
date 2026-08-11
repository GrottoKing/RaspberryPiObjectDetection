#!/usr/bin/env python3
"""Report what the Hailo accelerator on this Pi actually looks like.

Run this before using the Hailo backend. It checks the driver and runtime are
installed, finds the device, locates any compiled models (.hef) on the system,
and -- if you point it at one -- runs a single inference and describes the
exact shape and layout of what comes back.

    python3 scripts/hailo_probe.py                    # environment + find HEFs
    python3 scripts/hailo_probe.py --hef <path>       # ...and test that model

The output layout of a Hailo model is decided when it is compiled, not by us,
so this is the only reliable way to know how to read its results. If the Hailo
backend misbehaves, paste this output rather than guessing.
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys

# Where Raspberry Pi OS and the Hailo examples put compiled models.
HEF_SEARCH_PATHS = [
    "/usr/share/hailo-models",
    "/usr/local/hailo/resources",
    "/opt/hailo/resources",
    os.path.expanduser("~/hailo-rpi5-examples/resources"),
    os.path.expanduser("~/hailo-rpi5-examples/resources/models"),
    "models",
    "resources",
]


def rule(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def run(command: list) -> tuple:
    """Run a command, returning (ok, output)."""
    if not shutil.which(command[0]):
        return False, f"{command[0]} not found on PATH"
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                timeout=30)
        return result.returncode == 0, (result.stdout + result.stderr).strip()
    except Exception as exc:
        return False, str(exc)


def check_environment() -> bool:
    rule("1. RUNTIME AND DEVICE")

    ok, output = run(["hailortcli", "fw-control", "identify"])
    if ok:
        print(output)
    else:
        print(f"hailortcli: {output}")
        print()
        print("If this failed, the runtime is not installed or the device is")
        print("not visible. On Raspberry Pi OS:")
        print("    sudo apt update && sudo apt install -y hailo-all")
        print("    sudo reboot")

    print()
    ok_pci, pci = run(["lspci"])
    if ok_pci:
        hailo_lines = [line for line in pci.splitlines()
                       if "hailo" in line.lower()]
        print("PCIe:", hailo_lines[0] if hailo_lines
              else "no Hailo device found on the PCIe bus")
    else:
        print("PCIe: could not run lspci")

    print()
    try:
        import hailo_platform  # noqa: F401

        version = getattr(hailo_platform, "__version__", "unknown")
        print(f"python bindings: hailo_platform {version} — importable")
        importable = True
    except ImportError as exc:
        print(f"python bindings: NOT importable — {exc}")
        print()
        print("The bindings come from apt, not pip. If you are using the")
        print("project virtualenv it must be able to see system packages:")
        print("    python3 -m venv --system-site-packages .venv")
        print("(scripts/install.sh does this, but a venv made by hand may not)")
        importable = False

    # PCIe Gen 3 roughly doubles the throughput to the accelerator.
    try:
        with open("/boot/firmware/config.txt", "r", encoding="utf-8") as handle:
            config = handle.read()
        if "pciex1_gen=3" in config:
            print("PCIe speed: Gen 3 configured")
        else:
            print("PCIe speed: Gen 2 (default). For full throughput add")
            print("            dtparam=pciex1_gen=3   to /boot/firmware/config.txt")
    except OSError:
        pass

    return importable


def find_hefs() -> list:
    rule("2. COMPILED MODELS (.hef) ON THIS SYSTEM")
    found = []
    for directory in HEF_SEARCH_PATHS:
        for path in sorted(glob.glob(os.path.join(directory, "**", "*.hef"),
                                     recursive=True)):
            if path not in found:
                found.append(path)

    if not found:
        print("None found. Looked in:")
        for directory in HEF_SEARCH_PATHS:
            print(f"    {directory}")
        print()
        print("`sudo apt install hailo-all` normally installs some into")
        print("/usr/share/hailo-models/. Otherwise fetch them from the Hailo")
        print("model zoo, choosing the build that matches your chip:")
        print("    https://github.com/hailo-ai/hailo_model_zoo")
        print("A hailo8 model will NOT run on a hailo8l, or vice versa.")
        return found

    for path in found:
        size = os.path.getsize(path) / 1e6
        print(f"    {path}  ({size:.1f} MB)")
    return found


def inspect_hef(hef_path: str) -> None:
    rule(f"3. MODEL LAYOUT — {os.path.basename(hef_path)}")
    try:
        from hailo_platform import HEF
    except ImportError as exc:
        print(f"cannot import hailo_platform: {exc}")
        return

    try:
        hef = HEF(hef_path)
    except Exception as exc:
        print(f"could not open the model: {exc}")
        return

    for info in hef.get_input_vstream_infos():
        print(f"  INPUT   {info.name}")
        print(f"          shape  {info.shape}   (height, width, channels)")
        print(f"          format {getattr(info, 'format', '?')}")

    for info in hef.get_output_vstream_infos():
        print(f"  OUTPUT  {info.name}")
        print(f"          shape  {info.shape}")
        fmt = getattr(info, "format", None)
        order = getattr(fmt, "order", None)
        print(f"          format {fmt}")
        if order is not None and "NMS" in str(order).upper():
            print("          ^ this model does NMS on-chip: results come back")
            print("            as decoded boxes, one list per class.")


def test_inference(hef_path: str) -> None:
    rule(f"4. ONE INFERENCE — {os.path.basename(hef_path)}")
    try:
        import numpy as np
        from hailo_platform import (ConfigureParams, FormatType, HEF,
                                    HailoStreamInterface, InferVStreams,
                                    InputVStreamParams, OutputVStreamParams,
                                    VDevice)
    except ImportError as exc:
        print(f"cannot import what is needed: {exc}")
        return

    try:
        hef = HEF(hef_path)
        with VDevice() as target:
            params = ConfigureParams.create_from_hef(
                hef, interface=HailoStreamInterface.PCIe)
            network_group = target.configure(hef, params)[0]
            network_params = network_group.create_params()

            input_info = hef.get_input_vstream_infos()[0]
            height, width, channels = input_info.shape
            print(f"feeding a {width}x{height}x{channels} test frame")

            in_params = InputVStreamParams.make(network_group,
                                                format_type=FormatType.UINT8)
            out_params = OutputVStreamParams.make(network_group,
                                                  format_type=FormatType.FLOAT32)
            frame = np.zeros((1, height, width, channels), dtype=np.uint8)

            with InferVStreams(network_group, in_params,
                               out_params) as pipeline:
                with network_group.activate(network_params):
                    results = pipeline.infer({input_info.name: frame})

        print()
        print("RESULT STRUCTURE — this is what the backend has to read:")
        for name, value in results.items():
            print(f"  output '{name}': {type(value).__name__}")
            described = describe_value(value, indent=4, depth=0)
            if not described:
                print("    (empty)")
    except Exception as exc:
        print(f"inference failed: {type(exc).__name__}: {exc}")
        print()
        print("A 'device in use' error usually means the service is running:")
        print("    sudo systemctl stop objectlog")


def describe_value(value, indent: int = 4, depth: int = 0) -> bool:
    """Print the shape of a possibly-nested result, without dumping the data."""
    pad = " " * indent
    if depth > 3:
        return False
    try:
        import numpy as np
    except ImportError:
        return False

    if isinstance(value, np.ndarray):
        print(f"{pad}ndarray shape={value.shape} dtype={value.dtype}")
        if value.dtype == object and value.size:
            print(f"{pad}(object array — inspecting first element)")
            return describe_value(value.flat[0], indent + 2, depth + 1)
        if value.size and value.ndim <= 2 and value.size <= 20:
            print(f"{pad}values: {value.tolist()}")
        return True
    if isinstance(value, (list, tuple)):
        print(f"{pad}{type(value).__name__} of length {len(value)}")
        if value:
            non_empty = next(
                (item for item in value
                 if not (hasattr(item, "__len__") and len(item) == 0)), None)
            if non_empty is None:
                print(f"{pad}(all entries empty — expected for a blank frame)")
                return describe_value(value[0], indent + 2, depth + 1)
            print(f"{pad}first non-empty entry:")
            return describe_value(non_empty, indent + 2, depth + 1)
        return True
    print(f"{pad}{type(value).__name__}: {value!r}"[:200])
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hef", help="path to a .hef to inspect and test")
    parser.add_argument("--no-inference", action="store_true",
                        help="describe the model but do not run it")
    args = parser.parse_args(argv)

    print("Hailo probe — reporting what this Pi actually has")
    importable = check_environment()
    found = find_hefs()

    target = args.hef or (found[0] if found else None)
    if target and not os.path.exists(target):
        print(f"\nno such file: {target}")
        return 1

    if target and importable:
        inspect_hef(target)
        if not args.no_inference:
            test_inference(target)
    elif target:
        print(f"\nWould inspect {target}, but the Python bindings are missing.")

    rule("NEXT")
    if not importable:
        print("Install the runtime first:  sudo apt install -y hailo-all")
        print("then re-run this probe.")
    elif not found and not args.hef:
        print("Get a compiled model, then re-run with --hef <path>.")
    else:
        print("If sections 3 and 4 look sane, enable the backend:")
        print()
        print("    detector:")
        print("      backend: hailo")
        print(f"      hef: {target}")
        print()
        print("Then: sudo systemctl restart objectlog")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
