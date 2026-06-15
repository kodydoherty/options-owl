#!/bin/bash
# Per-ticker ML for the NEW flow tickers (overnight pipeline). Per-ticker signal/runner models
# (train_option_signals_v2.py) need GREEKS (iv/delta/theta/vega), but the new tickers were
# downloaded --ohlc-only. So: (1) wait for the OHLC expansion to finish, (2) download greeks for
# every new ticker (per-data-type gating fetches ONLY greeks, skips existing OHLC), (3) train
# per-ticker models. Robust: continues on per-ticker failure; logs everything.
set -u
cd "$(dirname "$0")/.."
START=2026-03-01
END=2026-06-12
# all new tickers (first batch + expansion). MU partial greeks already started.
NEW=(MU SMH MRVL TSM INTC ORCL QCOM ASML ARM IBM DELL CRWD GLD SLV SOXX LRCX GOOG)
LOG=/tmp/per_ticker_ml.log
: > "$LOG"

# 1) wait for the OHLC expansion download to finish (avoid ThetaData contention)
if [ -f /tmp/more_tickers.pid ]; then
  P=$(cat /tmp/more_tickers.pid)
  echo "[$(date +%H:%M)] waiting for OHLC expansion PID $P..." | tee -a "$LOG"
  while kill -0 "$P" 2>/dev/null; do sleep 120; done
fi

# 2) greeks download per ticker (skips already-downloaded OHLC, fetches greeks + extracts stock)
echo "[$(date +%H:%M)] STEP 1: greeks download (the slow part)" | tee -a "$LOG"
for tk in "${NEW[@]}"; do
  echo "[$(date +%H:%M)] greeks $tk..." | tee -a "$LOG"
  python scripts/download_thetadata.py --ticker "$tk" --start "$START" --end "$END" >> "$LOG" 2>&1 \
    && echo "[$(date +%H:%M)] $tk greeks rows: $(sqlite3 journal/thetadata_options.db "SELECT COUNT(*) FROM option_greeks WHERE ticker='$tk'" 2>/dev/null)" | tee -a "$LOG" \
    || echo "[$(date +%H:%M)] $tk greeks FAILED (continuing)" | tee -a "$LOG"
done

# 3) train per-ticker signal/runner models
echo "[$(date +%H:%M)] STEP 2: per-ticker model training" | tee -a "$LOG"
for tk in "${NEW[@]}"; do
  echo "[$(date +%H:%M)] training $tk..." | tee -a "$LOG"
  python scripts/train_option_signals_v2.py --ticker "$tk" >> "$LOG" 2>&1 \
    && echo "[$(date +%H:%M)] $tk trained OK" | tee -a "$LOG" \
    || echo "[$(date +%H:%M)] $tk training FAILED (continuing)" | tee -a "$LOG"
done

echo "[$(date +%H:%M)] PER-TICKER ML PIPELINE COMPLETE" | tee -a "$LOG"
echo "new per-ticker models:" | tee -a "$LOG"
ls -1 journal/models/signal_ml_v2/ 2>/dev/null | grep -iE '^(runner|pattern|entry)_(MU|SMH|MRVL|TSM|INTC|ORCL|QCOM|ASML|ARM|IBM|DELL|CRWD|GLD|SLV|SOXX|LRCX|GOOG)' | tee -a "$LOG"
