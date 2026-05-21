#!/usr/bin/env bash
# rebuild.sh — Local DevOps pipeline: test → sync → build → staggered restart
#
# Usage:
#   ./scripts/rebuild.sh              # full pipeline: test + rebuild ALL agents
#   ./scripts/rebuild.sh owlet-kody   # full pipeline: test + rebuild just one agent
#   ./scripts/rebuild.sh --skip-tests # skip tests (emergency hotfix only)
#
# Pipeline steps:
#   1. Run full test suite locally (pytest + ruff lint)
#   2. rsync local code → droplet (excludes .env, journal data, .venv, .git)
#   3. docker compose build --no-cache on droplet
#   4. Staggered restart (15s between bots to avoid Webull 429)
#   5. Verify all containers healthy

set -euo pipefail

# ---------------------------------------------------------------------------
# Production droplet config
# ---------------------------------------------------------------------------
DROPLET_IP="129.212.138.145"
DROPLET_USER="root"
DROPLET="$DROPLET_USER@$DROPLET_IP"
SSH_KEY="$HOME/.ssh/id_ed25519_do"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_DIR="/root/options-owl"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
SKIP_TESTS=false
TARGET=""

for arg in "$@"; do
  case "$arg" in
    --skip-tests)
      SKIP_TESTS=true
      echo "WARNING: Skipping tests — use only for emergency hotfixes!"
      ;;
    *)
      TARGET="$arg"
      ;;
  esac
done

# ---------------------------------------------------------------------------
# Step 1: Local test suite (MANDATORY unless --skip-tests)
# ---------------------------------------------------------------------------
if [ "$SKIP_TESTS" = false ]; then
  echo "=== Step 1a: Running test suite ==="
  cd "$LOCAL_DIR"
  if ! python -m pytest tests/ -x -q --tb=short; then
    echo ""
    echo "DEPLOY BLOCKED: Tests failed. Fix failing tests before deploying."
    echo "Use --skip-tests only for emergency hotfixes."
    exit 1
  fi

  echo "=== Step 1b: Running lint ==="
  if ! ruff check options_owl/ --select E,F --quiet 2>/dev/null; then
    echo "WARNING: Lint issues found (non-blocking). Review before next deploy."
  fi

  echo ""
  echo "All tests passed. Proceeding to deploy..."
  echo ""
else
  echo "=== Step 1: SKIPPED (--skip-tests) ==="
fi

# ---------------------------------------------------------------------------
# Step 2: Sync code to droplet
# ---------------------------------------------------------------------------
echo "=== Step 2: Syncing code to droplet ($DROPLET_IP) ==="
rsync -avz \
  --exclude='.venv' --exclude='__pycache__' --exclude='.DS_Store' \
  --exclude='*.pyc' --exclude='/journal/' --exclude='.env' --exclude='*.log*' \
  --exclude='did.bin' --exclude='.git' --exclude='.dockerignore' \
  --exclude='webull_trade_sdk.log*' \
  -e "ssh -i $SSH_KEY" \
  "$LOCAL_DIR/" "$DROPLET:$REMOTE_DIR/"

# ---------------------------------------------------------------------------
# Step 3: Build Docker images (no cache)
# ---------------------------------------------------------------------------
if [ -n "$TARGET" ]; then
  echo "=== Step 3: Rebuilding $TARGET (no cache) ==="
  ssh -i "$SSH_KEY" "$DROPLET" "cd $REMOTE_DIR && docker compose build --no-cache $TARGET"
  echo "=== Step 4: Restarting $TARGET ==="
  ssh -i "$SSH_KEY" "$DROPLET" "cd $REMOTE_DIR && docker compose restart $TARGET"
else
  echo "=== Step 3: Rebuilding ALL images (no cache) ==="
  ssh -i "$SSH_KEY" "$DROPLET" "cd $REMOTE_DIR && docker compose build --no-cache"
  echo "=== Step 4: Staggered restart (15s between bots) ==="
  ssh -i "$SSH_KEY" "$DROPLET" "cd $REMOTE_DIR && bash scripts/restart-staggered.sh"
fi

# ---------------------------------------------------------------------------
# Step 5: Verify
# ---------------------------------------------------------------------------
echo "=== Step 5: Verifying ==="
ssh -i "$SSH_KEY" "$DROPLET" "cd $REMOTE_DIR && docker compose ps"

echo ""
echo "=== Deploy complete ==="
