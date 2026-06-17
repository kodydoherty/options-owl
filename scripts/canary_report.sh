#!/usr/bin/env bash
# Daily canary report — runs ON the droplet (cron), writes journal/canary_report_<date>.md.
# Observe-first validation of the regime-budget + runner_v1 paper canaries (adam/vinny vs control
# kody/dennis). Deployed 2026-06-17. No SSH / no cloud — reads droplet logs + DBs directly.
set -u
cd /root/options-owl || exit 1
DATE=$(date -u +%F)              # bot logs + sqlite date('now') are UTC
OUT="journal/canary_report_${DATE}.md"

{
  echo "# Canary Report — ${DATE}"
  echo "_Generated $(date -u '+%Y-%m-%d %H:%M UTC') on droplet. Canaries: owlet-adam, owlet-vinny (flags ON) vs control kody/dennis (flags OFF)._"
  echo

  echo "## 1. runner_v1 P(runner) fidelity (THE key check)"
  for b in adam vinny; do
    LOG="journal/owlet-${b}/logs/options_owl_${DATE}.log"
    echo "### owlet-${b}"
    if [ ! -f "$LOG" ]; then echo "- log missing ($LOG)"; echo; continue; fi
    vals=$(grep -oE 'RUNNER_V1: .*p_runner=[0-9.]+' "$LOG" | grep -oE 'p_runner=[0-9.]+' | cut -d= -f2)
    n=$(printf '%s\n' "$vals" | grep -c .)
    if [ "$n" -eq 0 ]; then
      echo "- no CALL entries scored today (0 RUNNER_V1 lines) — nothing to judge"
    else
      printf '%s\n' "$vals" | awk '
        BEGIN{mn=9;mx=-9;s=0}
        {if($1<mn)mn=$1; if($1>mx)mx=$1; s+=$1}
        END{mean=s/NR; sp=mx-mn;
          printf "- n=%d  min=%.3f  max=%.3f  mean=%.3f  spread=%.3f\n", NR, mn, mx, mean, sp;
          if(sp<0.10) printf "- FLAG: degenerate (spread %.3f < 0.10) — live features likely skewed vs training; recalibrate cut points before trusting sizing\n", sp;
          else printf "- PASS: healthy spread across the range\n"}'
      echo "- quartile sizing tiers hit:"
      grep -oE 'RUNNER_V1_SIZING: .*Q[1-4] ×[0-9.]+' "$LOG" | grep -oE 'Q[1-4] ×[0-9.]+' | sort | uniq -c | sed 's/^/    /'
    fi
    echo "- scorer failures: $(grep -c 'RUNNER_V1: failed' "$LOG")"
    echo
  done

  echo "## 2. Regime call/put budget"
  for b in adam vinny; do
    LOG="journal/owlet-${b}/logs/options_owl_${DATE}.log"
    [ -f "$LOG" ] || continue
    echo "### owlet-${b}"
    if [ "$(grep -c 'REGIME_BUDGET:' "$LOG")" = "0" ]; then echo "    (none fired today)"; else
      grep -oE 'REGIME_BUDGET: .*(DOWN|UP|FLAT) regime, (call|put)×[0-9.]+' "$LOG" \
        | grep -oE '(DOWN|UP|FLAT) regime, (call|put)×[0-9.]+' | sort | uniq -c | sed 's/^/    /'
    fi
    echo
  done

  echo "## 3. Day P&L (observational — tiny n, do not over-read one day)"
  for b in adam vinny kody dennis; do
    DB="journal/owlet-${b}/raw_messages.db"
    [ -f "$DB" ] || continue
    tag="control"; case "$b" in adam|vinny) tag="CANARY";; esac
    echo "- **owlet-${b}** (${tag}):"
    sqlite3 "$DB" "SELECT '    '||option_type||': '||COUNT(*)||' trades, '||SUM(CASE WHEN pnl_dollars>0 THEN 1 ELSE 0 END)||'W, '||printf('\$%.0f',COALESCE(SUM(pnl_dollars),0)) FROM paper_trades WHERE status='closed' AND exit_source='ai' AND date(opened_at)=date('now') GROUP BY option_type" 2>/dev/null
    [ "$(sqlite3 "$DB" "SELECT COUNT(*) FROM paper_trades WHERE status='closed' AND exit_source='ai' AND date(opened_at)=date('now')" 2>/dev/null)" = "0" ] && echo "    (no closed trades today)"
  done
  echo
  echo "_Compare call-side adam/vinny (flags ON) vs kody/dennis (OFF) over SEVERAL days — one day is noise._"
  echo

  echo "## 4. Health"
  for b in adam vinny; do
    LOG="journal/owlet-${b}/logs/options_owl_${DATE}.log"
    st=$(docker inspect -f '{{.State.Status}} {{if .State.Health}}{{.State.Health.Status}}{{end}}' "owlet-${b}" 2>/dev/null)
    crit=0; [ -f "$LOG" ] && crit=$(grep -cE 'CRITICAL|Traceback' "$LOG")
    echo "- owlet-${b}: ${st:-unknown} | CRITICAL/Traceback lines: ${crit}"
  done
} > "$OUT" 2>&1

echo "wrote $OUT"
