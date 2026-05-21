#!/usr/bin/env bash
# kill-trades.sh — emergency: flip Webull kill switch ON and restart owlet-kody.
#
# Run from the project root (or use the 'kill-trades' alias from anywhere).
# Safe to run from a phone over SSH in 5 seconds.

set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f .env ]; then
    echo "ERROR: .env not found"
    exit 1
fi

echo ">>> Flipping WEBULL_KILL_SWITCH=true in .env"
if grep -q '^WEBULL_KILL_SWITCH=' .env; then
    sed -i.bak 's/^WEBULL_KILL_SWITCH=.*/WEBULL_KILL_SWITCH=true/' .env
else
    echo 'WEBULL_KILL_SWITCH=true' >> .env
fi
rm -f .env.bak

echo ">>> Restarting owlet-kody (LIVE bot)"
docker compose restart owlet-kody

echo ""
echo "================================================================"
echo "KILL SWITCH ACTIVE — owlet-kody will reject all new orders."
echo "Existing open positions are NOT auto-closed."
echo ""
echo "To re-enable: bash scripts/start-trades.sh"
echo "================================================================"
