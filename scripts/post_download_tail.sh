#!/bin/bash
# Critical-path tail of the retrain pipeline (afternoon/midday skipped — v7 doesn't load them).
# put_pattern -> put_entry_timing -> runner_prediction -> v7 (IS + OOS + OOS2).
set -u
cd /Users/kody/dev/options-owl
LOGDIR="journal/v3_eval_results/retrain_20260611_124224"
SUMMARY="$LOGDIR/SUMMARY.txt"
PY=python

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$SUMMARY"; }
stage() {
    local name="$1"; shift
    log "START  $name"
    local t0=$(date +%s)
    if "$@" > "$LOGDIR/${name}.log" 2>&1; then
        log "PASS   $name ($(($(date +%s)-t0))s)"
    else
        log "FAILED $name (exit $?) — see $LOGDIR/${name}.log"
    fi
}

log "--- TAIL pipeline (afternoon/midday skipped; not used by v7) ---"
stage put_pattern        $PY scripts/train_put_pattern.py
stage put_entry_timing   $PY scripts/train_put_entry_timing.py
stage runner_prediction  $PY scripts/runner_prediction.py
stage v7_backtest        $PY scripts/backtest_gold_standard_v7.py

log "TAIL COMPLETE. Comparison:"
cat journal/v3_eval_results/v7_convex_comparison.csv | tee -a "$SUMMARY"
