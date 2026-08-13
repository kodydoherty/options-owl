#!/usr/bin/env bash
# Out-of-process safety net for the LIVE bots. Cron every 5 min during market hours.
#
# Runs scripts/safety_check.py per live bot in a throwaway container with the SOURCE
# MOUNTED, so the check can be updated by rsync alone -- no image rebuild, no restart of a
# live trading bot just to change a watchdog.
#
# Why this is separate from the bot's own monitors: on 2026-08-13 a phantom quantity blocked
# an exit for 35 minutes (+$1,313 -> -$1,280) while three layers of monitoring stayed silent.
# Two of them lived inside the bot process, so they could never have caught a wedged loop,
# and the third (babysit.sh) had been failing on a missing exec bit for 87% of its runs.
#
# This script therefore assumes nothing works: it verifies its own preconditions, and any
# failure to RUN is reported as loudly as a failure it finds. Silence is never health.
set -uo pipefail

ROOT=/root/options-owl
TODAY=$(date -u +%F)
BOTS="${SAFETY_BOTS:-kody dennis}"
FAILED=0

cd "$ROOT" || { echo "SAFETY BLIND: cannot cd to $ROOT"; exit 2; }

for b in $BOTS; do
  db="$ROOT/journal/owlet-$b/raw_messages.db"
  if [ ! -f "$db" ]; then
    echo "SAFETY BLIND [$b]: no trade DB at $db"
    FAILED=1
    continue
  fi

  out=$(docker run --rm --network options-owl_default \
    -v "$ROOT":/src -w /src --env-file "$ROOT/.env" \
    "options-owl-owlet-$b" \
    python scripts/safety_check.py \
      --bot "$b" \
      --db "/src/journal/owlet-$b/raw_messages.db" \
      --log "/src/journal/owlet-$b/logs/options_owl_${TODAY}.log" \
      --status "/src/journal/safety_$b.status" 2>&1)
  rc=$?

  # rc: 0 ok, 1 breach found, 2 could not run. 2 is NOT better than 1 -- being blind is
  # its own emergency, and was precisely the state during the 08-13 incident.
  case $rc in
    0) echo "[$(date -u +%H:%M:%S)] SAFETY OK [$b]" ;;
    1) echo "[$(date -u +%H:%M:%S)] *** SAFETY BREACH [$b] ***"; echo "$out"; FAILED=1 ;;
    *) echo "[$(date -u +%H:%M:%S)] *** SAFETY BLIND [$b] (rc=$rc) ***"; echo "$out"; FAILED=1 ;;
  esac
done

exit $FAILED
