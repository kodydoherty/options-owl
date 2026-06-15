#!/bin/bash
# V7 gate sweep: vary the two ML thresholds (pattern, entry-timing) that gate 95%+ of
# candidates, across 3 non-overlapping time windows, to find the config that lifts trade
# volume + captures more runners WITHOUT wrecking risk-adjusted return — and is CONSISTENT
# across periods. The convex thesis: looser entry + V7 exits (cut losers, ride runners).
set -u
cd /Users/kody/dev/options-owl
OUT=journal/v3_eval_results/v7_sweep_results.csv
echo "window,pattern,entry,pnl,pf,wr,dd,trades,scanned,pass_pct" > "$OUT"

WINDOWS=("2026-03-16:2026-04-09" "2026-04-13:2026-05-08" "2026-05-11:2026-06-09")
PATTERNS=(0.74 0.62 0.50)
ENTRIES=(0.80 0.00)

run() {
  local start=$1 end=$2 p=$3 e=$4
  V7_PATTERN_THRESH=$p V7_ENTRY_THRESH=$e PORTFOLIO_START=20000 \
    python scripts/backtest_gold_standard_v7.py --window oos2 --start "$start" --end "$end" \
    > /tmp/sweep_run.log 2>&1
  # V7 funnel
  local fn scanned pass
  fn=$(grep 'V7 convex.*FUNNEL' /tmp/sweep_run.log | tail -1)
  scanned=$(echo "$fn" | grep -oE 'scanned=[0-9,]+' | tr -d 'scanned=,')
  pass=$(echo "$fn" | grep -oE '\([0-9.]+% pass' | grep -oE '[0-9.]+')
  # V7 comparison row (oos2)
  local row
  row=$(grep '^oos2,V7 convex' journal/v3_eval_results/v7_convex_comparison.csv | tail -1)
  local pnl pf wr dd tr
  pnl=$(echo "$row" | cut -d, -f3); pf=$(echo "$row" | cut -d, -f4)
  wr=$(echo "$row" | cut -d, -f5); dd=$(echo "$row" | cut -d, -f6); tr=$(echo "$row" | cut -d, -f7)
  echo "$start..$end,$p,$e,$pnl,$pf,$wr,$dd,$tr,${scanned:-0},${pass:-0}" >> "$OUT"
  echo "[$(date +%H:%M)] w=$start p=$p e=$e -> trades=$tr pnl=$pnl wr=$wr pass=${pass}%"
}

for w in "${WINDOWS[@]}"; do
  for p in "${PATTERNS[@]}"; do
    for e in "${ENTRIES[@]}"; do
      run "${w%:*}" "${w#*:}" "$p" "$e"
    done
  done
done

echo "=== SWEEP DONE — consistency summary (avg trades/day & return % by config, across windows) ==="
python3 - "$OUT" <<'PY'
import csv, sys, statistics as st
from collections import defaultdict
rows=list(csv.DictReader(open(sys.argv[1])))
days={'2026-03-16..2026-04-09':18,'2026-04-13..2026-05-08':18,'2026-05-11..2026-06-09':21}
g=defaultdict(list)
for r in rows:
    g[(r['pattern'],r['entry'])].append(r)
print(f"{'pattern':>8}{'entry':>6}{'avg_tr/day':>11}{'avg_ret%':>10}{'avg_pf':>8}{'avg_wr':>8}{'avg_dd':>8}{'ret_std':>9}")
for (p,e),rs in sorted(g.items()):
    tpd=st.mean(int(r['trades'])/days.get(r['window'],20) for r in rs)
    ret=[float(r['pnl'])/20000*100 for r in rs]
    pf=st.mean(float(r['pf']) for r in rs); wr=st.mean(float(r['wr']) for r in rs); dd=st.mean(float(r['dd']) for r in rs)
    print(f"{p:>8}{e:>6}{tpd:>11.1f}{st.mean(ret):>10.1f}{pf:>8.2f}{wr:>8.1f}{dd:>8.1f}{st.pstdev(ret):>9.1f}")
PY
echo "Results -> $OUT"
