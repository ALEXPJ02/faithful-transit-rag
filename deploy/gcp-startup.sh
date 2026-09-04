#!/usr/bin/env bash
#
# GCE startup script for the TfNSW delay collector.
#
# Runs as root on every boot of the instance. The point of doing the whole
# setup here rather than over SSH is that the machine provisions itself: one
# `gcloud compute instances create` and the collector is running, with no
# interactive session and nothing to forget.
#
# It is also idempotent — GCE re-runs startup scripts on every boot, so this
# has to be safe to execute against an already-configured machine.
#
# The API key arrives as instance metadata (`tfnsw-api-key`). Metadata is
# readable by anyone with project access, which for a single-owner student
# project is an acceptable trade for not standing up Secret Manager. Move it
# there if the project ever gains collaborators.

set -euo pipefail

REPO_URL="https://github.com/ALEXPJ02/faithful-transit-rag.git"
APP_DIR="/opt/transit-rag"
APP_USER="collector"
LOG_TAG="transit-rag-startup"

log() { logger -t "$LOG_TAG" "$*"; echo "[$LOG_TAG] $*"; }

log "starting provisioning"

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip git ca-certificates

# pyproject requires >=3.12. Debian 12 ships 3.11 and fails deep inside pip
# with a message about package metadata, which reads like a packaging bug
# rather than a wrong base image. Check it here, where the fix is obvious.
PY_VERSION="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)'; then
  log "FATAL: python3 is ${PY_VERSION}, but this project requires 3.12 or newer."
  log "       Recreate the instance with an image that ships 3.12+:"
  log "       --image-family=ubuntu-2404-lts-amd64 --image-project=ubuntu-os-cloud"
  exit 1
fi
log "python3 ${PY_VERSION} — ok"

id -u "$APP_USER" >/dev/null 2>&1 || useradd --system --create-home --shell /usr/sbin/nologin "$APP_USER"

if [ -d "$APP_DIR/.git" ]; then
  log "repo present, updating"
  git -C "$APP_DIR" fetch --quiet origin main
  git -C "$APP_DIR" reset --hard --quiet origin/main
else
  log "cloning $REPO_URL"
  rm -rf "$APP_DIR"
  git clone --quiet --depth 1 "$REPO_URL" "$APP_DIR"
fi

# The venv is rebuilt only when missing; `pip install -e` is cheap enough to
# re-run every boot and keeps the install in step with the checkout.
if [ ! -x "$APP_DIR/.venv/bin/python" ]; then
  log "creating venv"
  python3 -m venv "$APP_DIR/.venv"
fi
"$APP_DIR/.venv/bin/pip" install --quiet --upgrade pip
"$APP_DIR/.venv/bin/pip" install --quiet -e "$APP_DIR[realtime]"

# Fetch the key from instance metadata rather than baking it into the image
# or the repo.
API_KEY="$(curl -sf -H 'Metadata-Flavor: Google' \
  'http://metadata.google.internal/computeMetadata/v1/instance/attributes/tfnsw-api-key' || true)"
if [ -z "$API_KEY" ]; then
  log "FATAL: instance metadata 'tfnsw-api-key' is empty — refusing to start"
  exit 1
fi

install -d -o "$APP_USER" -g "$APP_USER" -m 0750 "$APP_DIR/data"
umask 077
cat > "$APP_DIR/.env" <<ENV
TFNSW_API_KEY=${API_KEY}
COLLECTION_SINK=sqlite
COLLECTION_DB_PATH=${APP_DIR}/data/delay_observations.db
COLLECTION_ROUTES_LOOKUP=${APP_DIR}/data/routes_lookup.csv
POLLER_INTERVAL_SECONDS=120
POLLER_ROUTES=T1,T4
POLLER_MAX_UPCOMING_STOPS=3
ENV
chown "$APP_USER:$APP_USER" "$APP_DIR/.env"
chmod 0400 "$APP_DIR/.env"

# The route lookup is tracked in the repo, so the clone already has it. Copy
# it where .env points; --require-routes refuses to run without it, which is
# the behaviour we want on an unattended box.
if [ -f "$APP_DIR/data/routes_lookup.csv" ]; then
  log "route lookup present"
else
  log "FATAL: data/routes_lookup.csv missing from the checkout"
  exit 1
fi
chown -R "$APP_USER:$APP_USER" "$APP_DIR/data"

cat > /etc/systemd/system/transit-poller.service <<'UNIT'
[Unit]
Description=TfNSW GTFS-Realtime delay collector
After=network-online.target
Wants=network-online.target
# Belongs in [Unit], not [Service] — systemd ignores it in [Service] with only
# a warning in the journal. Zero disables the start rate limiter entirely, so
# a repeatedly failing collector keeps retrying instead of being given up on.
# On a collection window that cannot be re-run, a unit systemd has marked
# failed is worse than one that is still trying.
StartLimitIntervalSec=0

[Service]
Type=simple
User=collector
WorkingDirectory=/opt/transit-rag
EnvironmentFile=/opt/transit-rag/.env
ExecStart=/opt/transit-rag/.venv/bin/transit-poller --require-routes
Restart=always
RestartSec=30
StandardOutput=journal
StandardError=journal
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/transit-rag/data

[Install]
WantedBy=multi-user.target
UNIT

# A consistent, readable copy for download. Two reasons this exists rather
# than telling anyone to scp the database directly:
#
#   1. The data directory is owned by the collector user, so an SSH login
#      cannot read it — and scp reports that as "No such file or directory",
#      which reads like the file is missing rather than unreadable.
#   2. The database is in WAL mode and written to every two minutes. Copying
#      the file can capture a torn read; sqlite3's backup API takes a
#      consistent snapshot of a live database, which is the whole point of it.
cat > /usr/local/bin/transit-snapshot <<'SNAPSHOT_SCRIPT'
#!/usr/bin/env bash
set -euo pipefail
DEST="${1:-/tmp/delay_observations.db}"
/opt/transit-rag/.venv/bin/python - "$DEST" <<'SNAPSHOT_PY'
import sqlite3
import sys

source = sqlite3.connect("file:/opt/transit-rag/data/delay_observations.db?mode=ro", uri=True)
destination = sqlite3.connect(sys.argv[1])
with destination:
    source.backup(destination)
rows = destination.execute("SELECT COUNT(*) FROM stop_observations").fetchone()[0]
print(f"{rows:,} stop events")
destination.close()
source.close()
SNAPSHOT_PY
chmod 0644 "$DEST"
echo "snapshot ready: $DEST"
SNAPSHOT_SCRIPT
chmod 0755 /usr/local/bin/transit-snapshot
log "installed /usr/local/bin/transit-snapshot"

systemctl daemon-reload
systemctl enable --now transit-poller.service
log "provisioning complete; poller enabled"
