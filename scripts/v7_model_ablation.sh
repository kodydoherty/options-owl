#!/bin/bash
# Model-ablation: at the chosen config (pattern 0.62 / entry 0.80), toggle the directional-
# regime gate ON/OFF across 3 windows to see if it pulls its weight (prune dead weight).
# NOTE: signal_quality + stop_calibration need their own toggle flags (TODO) to ablate;
# this covers the gates that are toggleable today (pattern, entry-timing already swept).
set -u
cd /Users/kody/dev/options-owl
OUT=journal/v3_eval_results/v7_ablation_results.csv
echo "window,regime,pnl,pf,wr,dd,trades" > "$OUT"
WINDOWS=("2026-03-16:2026-04-09" "2026-04-13:2026-05-08" "2026-05-11:2026-06-09")

run() {
  local s=$1 e=$2 reg=$3
  V7_PATTERN_THRESH=0.62 V7_ENTRY_THRESH=0.80 V7_DIRECTIONAL_REGIME=$reg PORTFOLIO_START=20000 \
    python scripts/backtest_gold_standard_v7.py --window oos2 --start "$s" --end "$e" > /tmp/abl_run.log 2>&1
  local row; row=$(grep '^oos2,V7 convex' journal/v3_eval_results/v7_convex_comparison.csv | tail -1)
  echo "$s..$e,$reg,$(echo "$row"|cut -d, -f3),$(echo "$row"|cut -d, -f4),$(echo "$row"|cut -d, -f5),$(echo "$row"|cut -d, -f6),$(echo "$row"|cut -d, -f7)" >> "$OUT"
  echo "[$(date +%H:%M)] w=$s regime=$reg -> $(echo "$row"|cut -d, -f3,5,7)"
}

for w in "${WINDOWS[@]}"; do
  for reg in 1 0; do run "${w%:*}" "${w#*:}" "$reg"; done
done

echo "=== ABLATION: directional-regime ON vs OFF (avg across windows) ==="
python3 - "$OUT" <<'PY'
import csv,sys,statistics as st
from collections import defaultdict
g=defaultdict(list)
for r in csv.DictReader(open(sys.argv[1])): g[r['regime']].append(r)
print(f"{'regime':>8}{'avg_ret%':>10}{'avg_pf':>8}{'avg_wr':>8}{'avg_dd':>8}{'avg_trades':>11}")
for reg,rs in sorted(g.items()):
    ret=st.mean(float(r['pnl'])/20000*100 for r in rs)
    print(f"{('ON' if reg=='1' else 'OFF'):>8}{ret:>10.0f}{st.mean(float(r['pf']) for r in rs):>8.2f}{st.mean(float(r['wr']) for r in rs):>8.1f}{st.mean(float(r['dd']) for r in rs):>8.1f}{st.mean(int(r['trades']) for r in rs):>11.0f}")
print("\nIf OFF >= ON on risk-adjusted return + similar DD -> directional-regime is dead weight; prune it.")
PY
echo "Results -> $OUT"
