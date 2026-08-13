#!/bin/bash
# OptionsOwl Babysitter — Runs Claude Code health check every 10 minutes
# Called by cron during market hours (Mon-Fri, market hours UTC)
# Runs as user 'kody' (has docker group access) since Claude Code
# blocks --dangerously-skip-permissions as root.
#
# Setup: ./scripts/setup-babysit.sh <ANTHROPIC_API_KEY>
# Logs: /root/options-owl/journal/babysit.log

set -euo pipefail

PROJECT_DIR="/root/options-owl"
LOG_DIR="$PROJECT_DIR/journal"
LOCK_FILE="$LOG_DIR/babysit.lock"
PROMPT_FILE="$PROJECT_DIR/scripts/monitor-prompt.md"

# Prevent overlapping runs
if [ -f "$LOCK_FILE" ]; then
    LOCK_AGE=$(($(date +%s) - $(stat -c %Y "$LOCK_FILE" 2>/dev/null || echo 0)))
    if [ "$LOCK_AGE" -lt 300 ]; then
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] SKIP — previous run still active (${LOCK_AGE}s old)"
        exit 0
    fi
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] WARN — stale lock (${LOCK_AGE}s), removing"
    rm -f "$LOCK_FILE"
fi

trap 'rm -f "$LOCK_FILE"' EXIT
touch "$LOCK_FILE"

echo ""
echo "=========================================="
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Babysit check starting"
echo "=========================================="

cd "$PROJECT_DIR"

# Run as 'kody' user (has docker group, avoids root restriction on --dangerously-skip-permissions)
# Pass ANTHROPIC_API_KEY through to the subprocess
sudo -u kody \
    ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY}" \
    HOME="/home/kody" \
    claude -p "$(cat "$PROMPT_FILE")" \
        --max-turns 15 \
        --dangerously-skip-permissions \
        --output-format text \
        2>&1

EXIT_CODE=$?
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Babysit check completed (exit=$EXIT_CODE)"
