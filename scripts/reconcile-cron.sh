#!/usr/bin/env bash
# Periodic PG-mirror reconcile (dashboard data durability).
#
# Keeps the dashboard's Postgres tables (trades, agent_state) matching each bot's SQLite
# source-of-truth. The bot's own PG sync is fire-and-forget + UPDATE-only-on-close, so it drifts
# (phantom opens, missing trades, no balance) — this backstops it every few minutes.
#
# SAFETY: reads a temp COPY of each small raw_messages.db (never touches the source) and writes
# ONLY the dashboard mirror. Verified NO bot reads PG trades/agent_state — so this has ZERO
# trading-path impact. Runs from cron on the droplet.
set -uo pipefail
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
cd /root/options-owl || exit 1
exec docker compose exec -T owlet-dashboard \
  python scripts/reconcile_pg_from_sqlite.py
