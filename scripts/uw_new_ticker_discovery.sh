#!/bin/bash
# TRACK A1: after the new-ticker thetadata download finishes, run the flow discovery on the 7
# new names (V7 exits, per-month consistency) so we learn which are PROFITABLE — not just high-flow.
# CHAINED on the download PID. Output -> /tmp/uw_new_discovery.log (+ overwrites the discovery CSV).
set -u
cd "$(dirname "$0")/.."
if [ -f /tmp/dl_new_tickers.pid ]; then
  P=$(cat /tmp/dl_new_tickers.pid)
  echo "[$(date +%H:%M)] waiting for new-ticker download PID $P to finish..."
  while kill -0 "$P" 2>/dev/null; do sleep 60; done
  echo "[$(date +%H:%M)] download done — running flow discovery on new tickers"
fi
# only include names that actually landed (>0 option rows)
HAVE=$(sqlite3 journal/thetadata_options.db "SELECT GROUP_CONCAT(ticker) FROM (SELECT DISTINCT ticker FROM option_ohlc WHERE ticker IN ('MU','SMH','MRVL','TSM','INTC','ORCL','QCOM'))" 2>/dev/null)
echo "[$(date +%H:%M)] new tickers with data: ${HAVE:-NONE}"
[ -z "$HAVE" ] && { echo "no new-ticker data landed — aborting"; exit 1; }
UW_DISCOVERY_TICKERS="$HAVE" python scripts/uw_ticker_discovery.py
echo "[$(date +%H:%M)] A1 done — new-ticker flow discovery complete (compare PF/total/mo+% to baseline)"
