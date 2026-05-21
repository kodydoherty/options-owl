#!/usr/bin/env bash
# start-trades.sh — re-enable live trading: flip Webull kill switch OFF
# and restart owlet-kody.

set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f .env ]; then
    echo "ERROR: .env not found"
    exit 1
fi

echo ">>> Flipping WEBULL_KILL_SWITCH=false in .env"
if grep -q '^WEBULL_KILL_SWITCH=' .env; then
    sed -i.bak 's/^WEBULL_KILL_SWITCH=.*/WEBULL_KILL_SWITCH=false/' .env
else
    echo 'WEBULL_KILL_SWITCH=false' >> .env
fi
rm -f .env.bak

echo ">>> Restarting owlet-kody (LIVE bot)"
docker compose restart owlet-kody

echo ""
echo "================================================================"
echo "Kill switch DISABLED — owlet-kody will resume placing orders."
echo "================================================================"
