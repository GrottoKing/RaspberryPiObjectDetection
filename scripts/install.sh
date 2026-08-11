#!/usr/bin/env bash
# One-shot setup on Raspberry Pi OS (Bookworm or later).
#
#   bash scripts/install.sh
#
# Creates a virtualenv that can still see the system picamera2, installs the
# Python dependencies, and fetches a detection model.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

echo "==> Installing system packages (needs sudo)"
sudo apt update
# python3-picamera2 comes from apt -- the pip build does not work on the Pi.
sudo apt install -y python3-picamera2 python3-venv python3-pip libcap-dev

if [ ! -d .venv ]; then
  echo "==> Creating .venv (with access to the system picamera2)"
  # --system-site-packages is what lets the venv import the apt-installed
  # picamera2 and libcamera bindings.
  python3 -m venv --system-site-packages .venv
fi

echo "==> Installing Python dependencies"
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt

echo "==> Fetching a detection model"
./.venv/bin/python scripts/fetch_model.py || {
  echo "    Model download failed. The app will still run in demo mode:"
  echo "      ./.venv/bin/python run.py --camera synthetic --backend mock"
}

echo
echo "==> Checking for a Hailo accelerator"
if command -v hailortcli >/dev/null 2>&1 && hailortcli fw-control identify >/dev/null 2>&1; then
  echo "    Hailo device found. It will be used automatically."
  echo "    Verify with: python3 scripts/hailo_probe.py"
elif lspci 2>/dev/null | grep -qi hailo; then
  echo "    A Hailo device is on the PCIe bus but the runtime is missing:"
  echo "        sudo apt install -y hailo-all && sudo reboot"
else
  echo "    None detected -- using CPU inference."
fi

echo
echo "==> Checking the camera"
if ./.venv/bin/python -c "import picamera2" 2>/dev/null; then
  echo "    picamera2 OK"
else
  echo "    picamera2 NOT importable -- the app will fall back to a webcam or"
  echo "    the synthetic test pattern."
fi

cat <<EOF

Done. Start it with:

    cd $ROOT
    ./.venv/bin/python run.py

Then open the address it prints from any device on your network.

To start it automatically on every boot:

    sudo bash scripts/install-service.sh
EOF
