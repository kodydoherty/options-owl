#!/bin/bash
# Post-download full ML retrain + v7 backtest pipeline.
#
# Waits for the Jan–Aug 2025 ThetaData download to finish, then:
#   1. Backs up current ml_v3 models
#   2. Retrains the full model suite on the extended data (auto-uses full DB range)
#   3. Regenerates leak-free P(runner) predictions (DATE_LO now 2025-01)
#   4. Re-runs v7 across IS + OOS + OOS2 (2025-H1) windows
#
# Stages run SEQUENTIALLY (one heavy job at a time → no OOM). Each stage logs
# PASS/FAIL but the pipeline continues so one failed trainer doesn't lose the rest.
set -u
cd /Users/kody/dev/options-owl

DL_PID=44016
STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="journal/v3_eval_results/retrain_${STAMP}"
mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/SUMMARY.txt"
PY=python

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$SUMMARY"; }

stage() {
    # stage "name" command args...
    local name="$1"; shift
    log "START  $name"
    local t0=$(date +%s)
    if "$@" > "$LOGDIR/${name}.log" 2>&1; then
        log "PASS   $name ($(($(date +%s)-t0))s)"
    else
        log "FAILED $name (exit $?) — see $LOGDIR/${name}.log"
    fi
}

log "Waiting for download (PID $DL_PID) to finish..."
while kill -0 "$DL_PID" 2>/dev/null; do sleep 60; done
log "Download finished. Verifying 2025 coverage..."
sqlite3 journal/thetadata_options.db \
    "SELECT substr(date,1,7) ym, COUNT(DISTINCT date) FROM download_log WHERE date LIKE '2025-0%' GROUP BY ym ORDER BY ym;" \
    | tee -a "$SUMMARY"

# 1. Back up current models (rollback safety)
log "Backing up ml_v3 models -> journal/models/ml_v3_backup_${STAMP}"
cp -r journal/models/ml_v3 "journal/models/ml_v3_backup_${STAMP}"

# 2. Full model retrain (each auto-reads full DB range incl. new Jan–Aug 2025)
stage pattern_entry      $PY scripts/train_pattern_entry.py
stage ml_v3_suite        $PY scripts/train_ml_models_v3.py
stage pattern_afternoon  $PY scripts/train_pattern_afternoon.py
stage pattern_midday     $PY scripts/train_pattern_midday.py
stage put_pattern        $PY scripts/train_put_pattern.py
stage put_entry_timing   $PY scripts/train_put_entry_timing.py

# 3. Regenerate leak-free runner predictions on extended data
stage runner_prediction  $PY scripts/runner_prediction.py

# 4. Re-run v7 across IS + OOS + OOS2
stage v7_backtest        $PY scripts/backtest_gold_standard_v7.py

log "PIPELINE COMPLETE. Comparison:"
cat journal/v3_eval_results/v7_convex_comparison.csv | tee -a "$SUMMARY"
log "All stage logs in $LOGDIR"
