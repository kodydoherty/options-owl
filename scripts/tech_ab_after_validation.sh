#!/usr/bin/env bash
# Tech-book A/B (2026-08-05): does the retrained expansion model preserve the CURRENT tech book's P&L,
# or should we route (old model for tech, new model only for new tickers)?
# Runs the full current universe on the OLD model (pattern_entry) then the NEW model
# (pattern_entry_expansion), same window/flags, and prints the comparison. Waits for the new-ticker
# validation to finish first so it doesn't thrash the 192GB DB.
set -u
cd "$(dirname "$0")/.." || exit 1
LOG=journal/thetadata_expansion_logs
mkdir -p "$LOG"
COMMON="--days 126 --pattern-threshold 0.62 --no-entry-filter --model-fill-miss on"

echo "[tech-ab] waiting for the new-ticker validation to finish..."
while pgrep -f 'backtest_expansion_tickers.py' >/dev/null 2>&1; do sleep 60; done
# small grace so the last child backtest fully exits + DB cache settles
sleep 30
echo "[tech-ab] validation done — starting A/B $(date '+%F %H:%M:%S')"

echo "[tech-ab] RUN 1/2: OLD model (pattern_entry) — baseline tech book"
python scripts/backtest_gold_standard.py $COMMON > "$LOG/tech_ab_OLD.txt" 2>&1
echo "[tech-ab] RUN 2/2: NEW model (pattern_entry_expansion)"
PATTERN_MODEL_STEM=pattern_entry_expansion python scripts/backtest_gold_standard.py $COMMON > "$LOG/tech_ab_NEW.txt" 2>&1

echo ""
echo "================= TECH A/B RESULT ================="
echo "--- OLD model (pattern_entry) ---"
grep -E 'Headline \(excl|Include-losers:' "$LOG/tech_ab_OLD.txt" | head -2
echo "--- NEW model (pattern_entry_expansion) ---"
grep -E 'Headline \(excl|Include-losers:' "$LOG/tech_ab_NEW.txt" | head -2
echo "=================================================="
echo "[tech-ab] VERDICT: if NEW >= OLD → combined model is safe on tech (use one model)."
echo "[tech-ab]          if NEW <  OLD → route (old model for tech, new model for new tickers)."
echo "[tech-ab] done $(date '+%F %H:%M:%S')"
