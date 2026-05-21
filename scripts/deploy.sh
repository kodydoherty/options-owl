#!/usr/bin/env bash
# deploy.sh — rsync the OptionsOwl project to a remote server.
#
# Usage:
#   bash scripts/deploy.sh owl@1.2.3.4
#   bash scripts/deploy.sh owl@my-server.example.com
#
# What this does:
#   - rsyncs the project to ~/options-owl on the remote
#   - excludes journal/, .venv/, __pycache__, .git, secrets
#   - preserves the remote .env if one already exists
#   - rebuilds and restarts docker compose

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 user@host"
    echo "Example: $0 owl@1.2.3.4"
    exit 1
fi

REMOTE="$1"
REMOTE_DIR="${REMOTE_DIR:-options-owl}"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo ">>> Syncing $LOCAL_DIR  →  $REMOTE:~/$REMOTE_DIR"

rsync -avz --delete \
    --exclude '.git/' \
    --exclude '.venv/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.pytest_cache/' \
    --exclude '.ruff_cache/' \
    --exclude 'journal/' \
    --exclude '.env' \
    --exclude '.DS_Store' \
    --exclude 'node_modules/' \
    "$LOCAL_DIR/" "$REMOTE:~/$REMOTE_DIR/"

echo ""
echo ">>> Project synced. Now copying .env if it doesn't exist on the remote..."

# Only copy .env if the remote doesn't already have one (don't overwrite remote secrets)
ssh "$REMOTE" "test -f ~/$REMOTE_DIR/.env" && {
    echo "Remote .env already exists — leaving it alone."
} || {
    if [ -f "$LOCAL_DIR/.env" ]; then
        echo "Copying local .env to remote (first time only)…"
        scp "$LOCAL_DIR/.env" "$REMOTE:~/$REMOTE_DIR/.env"
    else
        echo "WARNING: no local .env found and no remote .env. You'll need to create one before starting."
    fi
}

echo ""
echo ">>> Rebuilding and restarting docker compose on remote…"
ssh "$REMOTE" "cd ~/$REMOTE_DIR && docker compose up -d --build"

echo ""
echo ">>> Done. Tail logs with:"
echo "    ssh $REMOTE 'cd ~/$REMOTE_DIR && docker compose logs -f --tail 50'"
