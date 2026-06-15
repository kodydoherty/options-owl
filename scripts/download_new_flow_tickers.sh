#!/bin/bash
# Download the top high-flow tickers we're currently blind to (from uw_flow_universe ranking),
# OHLC-only (flow discovery needs option+stock close, not greeks/quotes), over the UW-flow
# overlap window + buffer. Appends to the main thetadata_options.db so uw_ticker_discovery can
# immediately backtest them. ATM±strikes wide enough for both call rallies and put crashes.
set -u
cd "$(dirname "$0")/.."
START=2026-03-01
END=2026-06-12
TICKERS=(MU SMH MRVL TSM INTC ORCL QCOM)   # MU = top-3 both sides; rest = semis/large-cap tech
LOG=/tmp/thetadata_new_tickers.log
: > "$LOG"

for tk in "${TICKERS[@]}"; do
  echo "[$(date +%H:%M)] ===== downloading $tk ($START..$END, OHLC-only) =====" | tee -a "$LOG"
  python scripts/download_thetadata.py --ticker "$tk" --start "$START" --end "$END" \
      --ohlc-only --otm 6 --otm-below 8 >> "$LOG" 2>&1
  echo "[$(date +%H:%M)] done $tk (rows: $(sqlite3 journal/thetadata_options.db "SELECT COUNT(*) FROM option_ohlc WHERE ticker='$tk'" 2>/dev/null))" | tee -a "$LOG"
done

echo "[$(date +%H:%M)] ALL NEW-TICKER DOWNLOADS COMPLETE" | tee -a "$LOG"
echo "Coverage now:" | tee -a "$LOG"
sqlite3 journal/thetadata_options.db "SELECT ticker, COUNT(*) FROM option_ohlc WHERE ticker IN ('MU','SMH','MRVL','TSM','INTC','ORCL','QCOM') GROUP BY ticker" 2>&1 | tee -a "$LOG"
