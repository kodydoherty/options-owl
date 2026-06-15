#!/bin/bash
# Step 2 of the prune plan: does the ML stop_calibration model BEAT V7's wide-trail exits?
# V7's thesis is "ride runners / cut losers via wide adaptive trails" — a per-trade ML tight-stop
# could conflict with that. This runs ONE clean comparison at the documented sweet-spot gate config
# (pattern 0.62 / entry 0.80) across the 3 non-overlapping windows:
#   A) V7 wide-trail exits (V7_USE_ML_STOP=0, current/default)
#   B) V7 + ML dynamic stop override (V7_USE_ML_STOP=1)
# CHAINED: waits for the multi-value sweep (PID in /tmp/v7_mv.pid) to finish first — no CPU contention.
set -u
cd /Users/kody/dev/options-owl
OUT=journal/v3_eval_results/v7_stopcal_test.csv

# ── wait for the multi-value sweep to finish (strict ordering, no contention) ──
if [ -f /tmp/v7_mv.pid ]; then
  P=$(cat /tmp/v7_mv.pid)
  echo "[$(date +%H:%M)] waiting for multi-value sweep PID $P to finish before stop_cal test..."
  while kill -0 "$P" 2>/dev/null; do sleep 60; done
  echo "[$(date +%H:%M)] sweep done — starting stop_cal test"
fi

echo "window,variant,pnl,pf,wr,dd,trades" > "$OUT"
WINDOWS=("2026-03-16:2026-04-09" "2026-04-13:2026-05-08" "2026-05-11:2026-06-09")
PAT=0.62; ENT=0.80

run() {  # $1=start $2=end $3=variant_label $4=use_ml_stop
  V7_PATTERN_THRESH=$PAT V7_ENTRY_THRESH=$ENT V7_USE_ML_STOP=$4 PORTFOLIO_START=20000 \
    python scripts/backtest_gold_standard_v7.py --window oos2 --start "$1" --end "$2" > /tmp/sc_run.log 2>&1
  local r; r=$(grep '^oos2,V7 convex' journal/v3_eval_results/v7_convex_comparison.csv | tail -1)
  echo "$1..$2,$3,$(echo "$r"|cut -d, -f3),$(echo "$r"|cut -d, -f4),$(echo "$r"|cut -d, -f5),$(echo "$r"|cut -d, -f6),$(echo "$r"|cut -d, -f7)" >> "$OUT"
  echo "[$(date +%H:%M)] $3 w=$1 -> pnl/pf/dd/tr $(echo "$r"|cut -d, -f3,4,6,7)"
}

for w in "${WINDOWS[@]}"; do
  run "${w%:*}" "${w#*:}" "V7_widetrail" 0
  run "${w%:*}" "${w#*:}" "V7_mlstop"    1
done

echo "=== STOP_CAL TEST — does ML dynamic stop beat V7 wide-trail? ==="
python3 - "$OUT" <<'PY'
import csv,sys,statistics as st
from collections import defaultdict
g=defaultdict(list)
for r in csv.DictReader(open(sys.argv[1])): g[r['variant']].append(r)
print(f"{'variant':>14}{'avg_pnl':>10}{'avg_pf':>8}{'avg_wr':>8}{'avg_dd':>8}{'avg_tr':>8}")
res={}
for v,rs in g.items():
    m=lambda k,f=float:st.mean(f(r[k]) for r in rs)
    res[v]=(m('pnl'),m('pf'),m('wr'),m('dd'),m('trades',lambda x:int(x)))
    print(f"{v:>14}{res[v][0]:>10.0f}{res[v][1]:>8.2f}{res[v][2]:>8.1f}{res[v][3]:>8.1f}{res[v][4]:>8.0f}")
if 'V7_widetrail' in res and 'V7_mlstop' in res:
    w,m=res['V7_widetrail'],res['V7_mlstop']
    print(f"\nΔ ml_stop vs widetrail: pnl {m[0]-w[0]:+.0f}, pf {m[1]-w[1]:+.2f}, dd {m[3]-w[3]:+.1f}")
    print("VERDICT:", "KEEP stop_calibration (ml_stop wins)" if (m[1]>w[1] and m[0]>=w[0]*0.95)
          else "DROP stop_calibration (V7 wide-trail wins — it's dead weight in V7)")
PY
echo "Results -> $OUT"
