#!/bin/bash
# Watch for V3 model training completion, then auto-run evaluation.
# Usage: ./scripts/watch_and_evaluate_v3.sh

MODEL_DIR="journal/models/ml_v3"
EVAL_SCRIPT="scripts/evaluate_v3_models.py"

# Models we expect from training
EXPECTED_MODELS=(
    "entry_timing.txt"
    "exit_timing.txt"
    "regime_classifier.txt"
    "ticker_selection.txt"
    "stop_calibration.txt"
    "signal_quality.txt"
)

echo "Watching for V3 model training completion..."
echo "Expected models: ${EXPECTED_MODELS[*]}"
echo ""

while true; do
    found=0
    missing=""
    for model in "${EXPECTED_MODELS[@]}"; do
        if [ -f "$MODEL_DIR/$model" ]; then
            found=$((found + 1))
        else
            missing="$missing $model"
        fi
    done

    echo "[$(date '+%H:%M:%S')] Models found: $found/${#EXPECTED_MODELS[@]}  Missing:$missing"

    if [ $found -eq ${#EXPECTED_MODELS[@]} ]; then
        echo ""
        echo "All models ready! Starting evaluation..."
        echo "============================================"
        python $EVAL_SCRIPT 2>&1 | tee journal/v3_eval_results/evaluation_log.txt
        echo ""
        echo "Evaluation complete. Report at: journal/v3_eval_results/v3_evaluation_report.md"
        exit 0
    fi

    # Also run partial evaluation if at least 2 models exist
    if [ $found -ge 2 ] && [ ! -f "journal/v3_eval_results/.partial_done" ]; then
        echo ""
        echo "Running PARTIAL evaluation with $found models..."
        python $EVAL_SCRIPT 2>&1 | tee journal/v3_eval_results/partial_evaluation_log.txt
        touch journal/v3_eval_results/.partial_done
        echo "Partial evaluation saved. Continuing to wait for remaining models..."
        echo ""
    fi

    sleep 120  # check every 2 minutes
done
