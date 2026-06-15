#!/bin/bash
# Exit-only ablation: does swapping ONLY the V7 exits into V6 beat plain V6?
# Entry + sizing held identical (prod relaxed_C: anti_chase OFF, momentum ON, current sizing).
# 4 runs: {OOS, OOS2} x {V6 exits (baseline), V7 exits}. Same retrained models, same windows
# as the v7 backtest, so results are directly comparable.
set -u
cd /Users/kody/dev/options-owl
STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="journal/v3_eval_results/exit_ablation_${STAMP}"
mkdir -p "$LOGDIR"
SUM="$LOGDIR/SUMMARY.txt"
PY=python
GATES="--gate-anti-chase off --gate-momentum on"   # match production/relaxed_C entry config

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$SUM"; }

run() {
    # run <label> <start> <end> <extra-flags...>
    local label="$1" start="$2" end="$3"; shift 3
    log "START  $label  ($start..$end)"
    $PY scripts/backtest_gold_standard.py --start "$start" --end "$end" $GATES "$@" \
        > "$LOGDIR/${label}.log" 2>&1
    # extract the key line items
    local pnl wr pf
    pnl=$(grep -E 'Total P&L:' "$LOGDIR/${label}.log" | tail -1 | grep -oE '[-+$,0-9.]+' | tail -1)
    wr=$(grep -E 'Win Rate:' "$LOGDIR/${label}.log" | tail -1 | grep -oE '[0-9.]+%' | tail -1)
    pf=$(grep -E 'Profit Factor:' "$LOGDIR/${label}.log" | tail -1 | grep -oE '[0-9.inf]+' | tail -1)
    log "DONE   $label  P&L=$pnl  WR=$wr  PF=$pf"
}

log "=== EXIT-ONLY ABLATION (entry+sizing held at prod relaxed_C) ==="
run oos_v6_exits   2025-09-08 2025-12-07
run oos_v7_exits   2025-09-08 2025-12-07 --v7-exits
run oos2_v6_exits  2025-03-15 2025-08-31
run oos2_v7_exits  2025-03-15 2025-08-31 --v7-exits

log "=== EXIT-REASON CHECK (v7 arm must show NO scaleout / profit_target) ==="
for f in oos_v7_exits oos2_v7_exits; do
    log "-- $f exit reasons --"
    grep -iE 'scaleout|profit_target|adaptive_trail|soft_trail|scalp_trail|theta' "$LOGDIR/${f}.log" \
        | grep -iE 'reason|:' | head -8 | tee -a "$SUM"
done
log "ABLATION COMPLETE. Logs in $LOGDIR"
