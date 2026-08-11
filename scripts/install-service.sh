#!/usr/bin/env bash
# Install the sightings log as a systemd service, so it starts on boot and
# restarts if it falls over.
#
#   sudo bash scripts/install-service.sh
#
# Works out the right user and paths itself rather than making you edit the
# unit file -- Raspberry Pi OS has not defaulted to the "pi" user for years,
# and a wrong User= is the usual reason this silently fails to start.
#
# Set DRY_RUN=1 to print the unit file and stop without touching the system.
set -euo pipefail

SERVICE_NAME=objectlog
UNIT_PATH="/etc/systemd/system/${SERVICE_NAME}.service"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${REPO_DIR}/.venv/bin/python"

# When run under sudo, SUDO_USER is the human who typed it -- that is who the
# service should run as, not root. The camera and the data directory belong to
# them, and running a network service as root for no reason is worth avoiding.
RUN_USER="${SUDO_USER:-$(id -un)}"

if [ "$RUN_USER" = "root" ]; then
  echo "Refusing to install a service that runs as root." >&2
  echo "Run it with sudo from your normal user account instead:" >&2
  echo "    sudo bash scripts/install-service.sh" >&2
  exit 1
fi

if [ ! -x "$PYTHON" ]; then
  echo "No virtualenv found at $PYTHON" >&2
  echo "Run 'bash scripts/install.sh' first." >&2
  exit 1
fi

# The camera device is owned by the 'video' group. Without membership the
# service starts and then fails on every frame, which looks like a camera fault.
if ! id -nG "$RUN_USER" 2>/dev/null | tr ' ' '\n' | grep -qx video; then
  echo "Note: user '$RUN_USER' is not in the 'video' group; adding it."
  echo "      (log out and back in for this to apply to your shell too)"
  usermod -aG video "$RUN_USER" 2>/dev/null || \
    echo "      could not add automatically -- run: sudo usermod -aG video $RUN_USER"
fi

UNIT_CONTENT="[Unit]
Description=Camera sightings log
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${RUN_USER}
WorkingDirectory=${REPO_DIR}
ExecStart=${PYTHON} run.py
Restart=on-failure
RestartSec=5
# The camera stack is happier with a moment to settle after a reboot.
ExecStartPre=/bin/sleep 5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target"

echo "==> Service configuration"
echo "    user      : ${RUN_USER}"
echo "    directory : ${REPO_DIR}"
echo "    python    : ${PYTHON}"
echo

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "==> DRY_RUN=1, would write ${UNIT_PATH}:"
  echo
  echo "$UNIT_CONTENT"
  exit 0
fi

if [ "$(id -u)" -ne 0 ]; then
  echo "This needs root to write ${UNIT_PATH}." >&2
  echo "Re-run as:  sudo bash scripts/install-service.sh" >&2
  exit 1
fi

echo "==> Writing ${UNIT_PATH}"
printf '%s\n' "$UNIT_CONTENT" > "$UNIT_PATH"

echo "==> Enabling and starting ${SERVICE_NAME}"
systemctl daemon-reload
systemctl enable "$SERVICE_NAME"
systemctl restart "$SERVICE_NAME"

# Give it a moment to either come up or fall over, then report honestly.
sleep 6
echo
if systemctl is-active --quiet "$SERVICE_NAME"; then
  PORT="$(grep -E '^\s*port:' "${REPO_DIR}/config.yaml" 2>/dev/null \
          | head -1 | tr -dc '0-9')"
  PORT="${PORT:-8010}"
  IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
  echo "Running. It will now start automatically on boot."
  echo
  echo "    http://${IP:-<pi-ip>}:${PORT}/"
  echo "    http://$(hostname).local:${PORT}/"
  echo
  echo "Useful commands:"
  echo "    journalctl -u ${SERVICE_NAME} -f      # watch the log"
  echo "    sudo systemctl restart ${SERVICE_NAME}"
  echo "    sudo systemctl stop ${SERVICE_NAME}"
  echo "    sudo systemctl disable ${SERVICE_NAME}   # stop starting on boot"
else
  echo "The service failed to start. Its own output:" >&2
  echo >&2
  journalctl -u "$SERVICE_NAME" -n 30 --no-pager >&2 || true
  exit 1
fi
