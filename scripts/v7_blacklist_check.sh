#!/bin/bash
# "100% sure" blacklist validation: do the PUT_EXCLUDED_TICKERS (PLTR,AMD,MSTR,AVGO,AMZN,GOOGL)
# STILL lose as GENERAL-ML puts under V7 wide-trail exits? Clears the PUT exclusion
# (PUT_INCLUDE_BLACKLIST=1) so those names trade puts, runs --puts-only --v7-exits, and dumps the
# per-ticker PUT P&L. Negative per-ticker P&L => blacklist still justified for the general pipeline.
# (The flow path bypasses this gate by design; see uw_ticker_discovery for the flow side.)
# CHAINED: waits for any running rebuild.sh to finish first.
set -u
cd /Users/kody/dev/options-owl

while pgrep -f 'rebuild.sh' >/dev/null 2>&1; do
  echo "[$(date +%H:%M)] waiting for rebuild.sh to finish before blacklist backtest..."; sleep 30
done
echo "[$(date +%H:%M)] running general-ML V7 PUT backtest WITH blacklisted names..."

LOG=/tmp/v7_blacklist.log
PUT_INCLUDE_BLACKLIST=1 PORTFOLIO_START=20000 \
  python scripts/backtest_gold_standard.py --puts-only --v7-exits \
    --start 2026-03-16 --end 2026-06-09 > "$LOG" 2>&1

echo "=== PER-TICKER PUT P&L (V7 exits, general-ML entries) — blacklisted names in [] ==="
# the per-ticker table prints as:  TICKER  trades  wr  $pnl  $avg
awk '/^  [A-Z]+ +[0-9]/ {print}' "$LOG" | while read -r tk tr wr pnl avg; do
  case " PLTR AMD MSTR AVGO AMZN GOOGL " in
    *" $tk "*) flag="[BLACKLISTED]";;
    *) flag="";;
  esac
  echo "  $tk  trades=$tr  wr=$wr  pnl=$pnl  avg=$avg  $flag"
done
echo ""
echo "=== VERDICT (blacklisted names with NEGATIVE pnl => keep blacklisted) ==="
awk '/^  [A-Z]+ +[0-9]/ {print}' "$LOG" | while read -r tk tr wr pnl avg; do
  case " PLTR AMD MSTR AVGO AMZN GOOGL " in
    *" $tk "*)
      n=$(echo "$pnl" | tr -d '$+,')
      if awk "BEGIN{exit !($n < 0)}"; then echo "  $tk: $pnl  -> KEEP blacklisted (loses under V7)";
      else echo "  $tk: $pnl  -> REVIEW (profitable under V7 general-ML)"; fi;;
  esac
done
echo ""
echo "Full log -> $LOG"
