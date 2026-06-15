#!/bin/bash
# Multi-VALUE gate sweep: tune the actual thresholds (not on/off) of the two value-tunable
# gates — pattern model + entry-timing — jointly across 3 non-overlapping windows, to find
# the config that is CONSISTENTLY best risk-adjusted (low return-std), not a local optimum.
# (regime/signal_quality/stop_cal threshold-tuning needs further parameterization — TODO.)
set -u
cd /Users/kody/dev/options-owl
OUT=journal/v3_eval_results/v7_multivalue_sweep.csv
echo "window,pattern,entry,pnl,pf,wr,dd,trades" > "$OUT"
WINDOWS=("2026-03-16:2026-04-09" "2026-04-13:2026-05-08" "2026-05-11:2026-06-09")
PATTERNS=(0.68 0.62 0.56)
ENTRIES=(0.90 0.80 0.70)   # actual entry-timing VALUES (not on/off)

run() {
  V7_PATTERN_THRESH=$3 V7_ENTRY_THRESH=$4 PORTFOLIO_START=20000 \
    python scripts/backtest_gold_standard_v7.py --window oos2 --start "$1" --end "$2" > /tmp/mv_run.log 2>&1
  local r; r=$(grep '^oos2,V7 convex' journal/v3_eval_results/v7_convex_comparison.csv | tail -1)
  echo "$1..$2,$3,$4,$(echo "$r"|cut -d, -f3),$(echo "$r"|cut -d, -f4),$(echo "$r"|cut -d, -f5),$(echo "$r"|cut -d, -f6),$(echo "$r"|cut -d, -f7)" >> "$OUT"
  echo "[$(date +%H:%M)] p=$3 e=$4 w=$1 -> $(echo "$r"|cut -d, -f3,4,6,7)"
}

for w in "${WINDOWS[@]}"; do
  for p in "${PATTERNS[@]}"; do for e in "${ENTRIES[@]}"; do run "${w%:*}" "${w#*:}" "$p" "$e"; done; done
done

echo "=== MULTI-VALUE SWEEP — ranked by consistency (low return-std) + risk-adjusted ==="
python3 - "$OUT" <<'PY'
import csv,sys,statistics as st
from collections import defaultdict
days={'2026-03-16..2026-04-09':18,'2026-04-13..2026-05-08':18,'2026-05-11..2026-06-09':21}
g=defaultdict(list)
for r in csv.DictReader(open(sys.argv[1])): g[(r['pattern'],r['entry'])].append(r)
rows=[]
for (p,e),rs in g.items():
    ret=[float(r['pnl'])/20000*100 for r in rs]
    rows.append((p,e,st.mean(ret),st.pstdev(ret),st.mean(float(r['pf']) for r in rs),
                 st.mean(float(r['dd']) for r in rs),st.mean(int(r['trades'])/days.get(r['window'],20) for r in rs)))
print(f"{'pattern':>8}{'entry':>7}{'avg_ret%':>10}{'ret_std':>9}{'avg_pf':>8}{'avg_dd':>8}{'tr/day':>8}")
# rank: prefer high mean return per unit of std (consistency-adjusted), decent PF
for p,e,m,sd,pf,dd,tpd in sorted(rows, key=lambda x:-(x[2]/(x[3]+1))):
    print(f"{p:>8}{e:>7}{m:>10.0f}{sd:>9.0f}{pf:>8.2f}{dd:>8.1f}{tpd:>8.1f}")
print("\nTop row = best return-per-consistency. Cross-check PF + DD before adopting.")
PY
echo "Results -> $OUT"
