#!/bin/bash
# Expand flow-ticker coverage: download the NEXT tier of high-flow names from the universe ranking
# that we're still blind to. Full pipeline: option OHLC -> Polygon stock backfill (fixes the
# --ohlc-only/no-greeks/no-stock_ohlc gotcha) -> flow discovery. Append to thetadata_options.db.
set -u
cd "$(dirname "$0")/.."
START=2026-03-01
END=2026-06-12
# next tier (not in current 21): semis/tech/ETFs with recurring whale flow
TICKERS=(ASML ARM IBM DELL CRWD GLD SLV SOXX LRCX GOOG)
LOG=/tmp/thetadata_more_tickers.log
: > "$LOG"

echo "[$(date +%H:%M)] STEP 1/3: option OHLC download" | tee -a "$LOG"
for tk in "${TICKERS[@]}"; do
  echo "[$(date +%H:%M)] downloading $tk..." | tee -a "$LOG"
  python scripts/download_thetadata.py --ticker "$tk" --start "$START" --end "$END" \
      --ohlc-only --otm 6 --otm-below 8 >> "$LOG" 2>&1
  echo "[$(date +%H:%M)] $tk option rows: $(sqlite3 journal/thetadata_options.db "SELECT COUNT(*) FROM option_ohlc WHERE ticker='$tk'" 2>/dev/null)" | tee -a "$LOG"
done

echo "[$(date +%H:%M)] STEP 2/3: Polygon stock_ohlc backfill" | tee -a "$LOG"
CSV=$(IFS=,; echo "${TICKERS[*]}")
python scripts/backfill_stock_ohlc_polygon.py --tickers "$CSV" --start "$START" --end "$END" >> "$LOG" 2>&1

echo "[$(date +%H:%M)] STEP 3/3: flow discovery on the new tier" | tee -a "$LOG"
# only names that actually landed with BOTH option + stock data
HAVE=$(sqlite3 journal/thetadata_options.db "SELECT GROUP_CONCAT(t) FROM (SELECT DISTINCT o.ticker t FROM option_ohlc o WHERE o.ticker IN ('ASML','ARM','IBM','DELL','CRWD','GLD','SLV','SOXX','LRCX','GOOG') AND EXISTS(SELECT 1 FROM stock_ohlc s WHERE s.ticker=o.ticker))" 2>/dev/null)
echo "[$(date +%H:%M)] tier names with full data: ${HAVE:-NONE}" | tee -a "$LOG"
[ -z "$HAVE" ] && { echo "no data landed — abort"; exit 1; }
UW_DISCOVERY_TICKERS="$HAVE" python scripts/uw_ticker_discovery.py 2>&1 | tee -a "$LOG"
echo "[$(date +%H:%M)] DONE — review per-ticker PF/total/mo+% to pick whitelist adds" | tee -a "$LOG"
