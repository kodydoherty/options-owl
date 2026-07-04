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
# Step 1c: Safety check — PAPER_TRADE and KILL_SWITCH in docker-compose.yml
# Prevents accidental deploys that switch live bots to paper mode.
# ---------------------------------------------------------------------------
echo "=== Step 1c: Verifying docker-compose.yml safety flags ==="
KODY_PAPER=$(grep -A15 'owlet-kody:' "$LOCAL_DIR/docker-compose.yml" | grep 'PAPER_TRADE=' | head -1 | sed 's/.*PAPER_TRADE=//')
KODY_KILL=$(grep -A15 'owlet-kody:' "$LOCAL_DIR/docker-compose.yml" | grep 'WEBULL_KILL_SWITCH=' | head -1 | sed 's/.*WEBULL_KILL_SWITCH=//')
if [ "$KODY_PAPER" = "true" ] || [ "$KODY_KILL" = "true" ]; then
  echo ""
  echo "DEPLOY BLOCKED: owlet-kody has PAPER_TRADE=$KODY_PAPER WEBULL_KILL_SWITCH=$KODY_KILL"
  echo "This would switch LIVE trading to paper mode on deploy."
  echo "If intentional, edit docker-compose.yml first, then re-run."
  exit 1
fi
echo "owlet-kody: PAPER_TRADE=$KODY_PAPER WEBULL_KILL_SWITCH=$KODY_KILL — OK"

# adam is now a LIVE bot (margin) too — guard it the same way (wider -A: adam's PAPER_TRADE sits deeper).
ADAM_PAPER=$(grep -A40 'owlet-adam:' "$LOCAL_DIR/docker-compose.yml" | grep 'PAPER_TRADE=' | head -1 | sed 's/.*PAPER_TRADE=//')
ADAM_KILL=$(grep -A40 'owlet-adam:' "$LOCAL_DIR/docker-compose.yml" | grep 'WEBULL_KILL_SWITCH=' | head -1 | sed 's/.*WEBULL_KILL_SWITCH=//')
if [ "$ADAM_PAPER" = "true" ] || [ "$ADAM_KILL" = "true" ]; then
  echo ""
  echo "DEPLOY BLOCKED: owlet-adam has PAPER_TRADE=$ADAM_PAPER WEBULL_KILL_SWITCH=$ADAM_KILL"
  echo "adam is a LIVE bot (margin). This would switch it to paper. If intentional, edit then re-run."
  exit 1
fi
echo "owlet-adam: PAPER_TRADE=$ADAM_PAPER WEBULL_KILL_SWITCH=$ADAM_KILL — OK (live, margin)"

# dennis → LIVE 2026-06-30 (Kody's call — 3rd live bot after kody+adam). Guard the
# live direction — block if dennis is accidentally flipped back to paper.
DENNIS_PAPER=$(grep -A15 'owlet-dennis:' "$LOCAL_DIR/docker-compose.yml" | grep 'PAPER_TRADE=' | head -1 | sed 's/.*PAPER_TRADE=//')
DENNIS_KILL=$(grep -A15 'owlet-dennis:' "$LOCAL_DIR/docker-compose.yml" | grep 'WEBULL_KILL_SWITCH=' | head -1 | sed 's/.*WEBULL_KILL_SWITCH=//')
if [ "$DENNIS_PAPER" = "true" ] || [ "$DENNIS_KILL" = "true" ]; then
  echo ""
  echo "DEPLOY BLOCKED: owlet-dennis has PAPER_TRADE=$DENNIS_PAPER WEBULL_KILL_SWITCH=$DENNIS_KILL"
  echo "dennis is a LIVE bot. This would switch it to paper. If intentional, edit then re-run."
  exit 1
fi
echo "owlet-dennis: PAPER_TRADE=$DENNIS_PAPER WEBULL_KILL_SWITCH=$DENNIS_KILL — OK (live)"

# vinny → LIVE on MARGIN 2026-07-01 (Kody's call — 4th live bot). Guard the live
# direction — block if vinny is accidentally flipped back to paper. (-A40: vinny's
# PAPER_TRADE sits deeper in its block, like adam's.)
VINNY_PAPER=$(grep -A40 'owlet-vinny:' "$LOCAL_DIR/docker-compose.yml" | grep 'PAPER_TRADE=' | head -1 | sed 's/.*PAPER_TRADE=//')
VINNY_KILL=$(grep -A40 'owlet-vinny:' "$LOCAL_DIR/docker-compose.yml" | grep 'WEBULL_KILL_SWITCH=' | head -1 | sed 's/.*WEBULL_KILL_SWITCH=//')
if [ "$VINNY_PAPER" = "true" ] || [ "$VINNY_KILL" = "true" ]; then
  echo ""
  echo "DEPLOY BLOCKED: owlet-vinny has PAPER_TRADE=$VINNY_PAPER WEBULL_KILL_SWITCH=$VINNY_KILL"
  echo "vinny is a LIVE bot (margin). This would switch it to paper. If intentional, edit then re-run."
  exit 1
fi
echo "owlet-vinny: PAPER_TRADE=$VINNY_PAPER WEBULL_KILL_SWITCH=$VINNY_KILL — OK (live, margin)"

# alan → LIVE on MARGIN 2026-07-04 (Kody's call — 5th live bot, $10k). Guard the live
# direction — block if alan is accidentally flipped back to paper. (-A20: alan's
# PAPER_TRADE sits near the top of its block.)
ALAN_PAPER=$(grep -A20 'owlet-alan:' "$LOCAL_DIR/docker-compose.yml" | grep 'PAPER_TRADE=' | head -1 | sed 's/.*PAPER_TRADE=//')
ALAN_KILL=$(grep -A20 'owlet-alan:' "$LOCAL_DIR/docker-compose.yml" | grep 'WEBULL_KILL_SWITCH=' | head -1 | sed 's/.*WEBULL_KILL_SWITCH=//')
if [ "$ALAN_PAPER" = "true" ] || [ "$ALAN_KILL" = "true" ]; then
  echo ""
  echo "DEPLOY BLOCKED: owlet-alan has PAPER_TRADE=$ALAN_PAPER WEBULL_KILL_SWITCH=$ALAN_KILL"
  echo "alan is a LIVE bot (margin). This would switch it to paper. If intentional, edit then re-run."
  exit 1
fi
echo "owlet-alan: PAPER_TRADE=$ALAN_PAPER WEBULL_KILL_SWITCH=$ALAN_KILL — OK (live, margin)"

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
  ssh -i "$SSH_KEY" "$DROPLET" "cd $REMOTE_DIR && docker compose up -d $TARGET"
else
  echo "=== Step 3: Rebuilding ALL images (no cache) ==="
  ssh -i "$SSH_KEY" "$DROPLET" "cd $REMOTE_DIR && docker compose build --no-cache"
  echo "=== Step 4: Staggered restart (15s between bots) ==="
  ssh -i "$SSH_KEY" "$DROPLET" "cd $REMOTE_DIR && bash scripts/restart-staggered.sh"
fi

# ---------------------------------------------------------------------------
# Step 4b: Prune Docker build cache (prevents 100GB+ buildup)
# ---------------------------------------------------------------------------
echo "=== Step 4b: Pruning Docker build cache ==="
ssh -i "$SSH_KEY" "$DROPLET" "docker builder prune --all -f 2>/dev/null || true"

# ---------------------------------------------------------------------------
# Step 5: Verify
# ---------------------------------------------------------------------------
echo "=== Step 5: Verifying ==="
ssh -i "$SSH_KEY" "$DROPLET" "cd $REMOTE_DIR && docker compose ps"

# Post-deploy: verify owlet-kody is LIVE (not accidentally paper)
echo ""
echo "=== Step 6: Verifying owlet-kody is LIVE ==="
LIVE_PAPER=$(ssh -i "$SSH_KEY" "$DROPLET" "docker exec owlet-kody env 2>/dev/null | grep PAPER_TRADE= | head -1" || echo "PAPER_TRADE=UNKNOWN")
LIVE_KILL=$(ssh -i "$SSH_KEY" "$DROPLET" "docker exec owlet-kody env 2>/dev/null | grep WEBULL_KILL_SWITCH= | head -1" || echo "WEBULL_KILL_SWITCH=UNKNOWN")
echo "  $LIVE_PAPER"
echo "  $LIVE_KILL"
if echo "$LIVE_PAPER" | grep -q "true"; then
  echo ""
  echo "WARNING: owlet-kody is running in PAPER mode! Check docker-compose.yml."
fi

echo ""
echo "=== Deploy complete ==="
