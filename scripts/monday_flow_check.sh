#!/bin/bash
# Monday-morning UW-flow check — run ON THE DROPLET (cd /root/options-owl) after ~10:30 ET.
# Confirms whale-flow signals fire at the expected rate, at sane prices, and that LIVE bots
# (kody/dennis) produce the same flow trades as the PAPER bots (adam/vinny/yank) on identical signals.
#   ssh -i ~/.ssh/id_ed25519_do root@129.212.138.145 'cd /root/options-owl && bash scripts/monday_flow_check.sh'
set -u
cd "$(dirname "$0")/.." 2>/dev/null || cd /root/options-owl
DAY=${1:-$(date -u +%Y-%m-%d)}
LIVE="owlet-kody owlet-dennis"
PAPER="owlet-adam owlet-vinny owlet-yank"

echo "==================== UW FLOW CHECK — $DAY ===================="

flow_signals() {  # count UW_FLOW signals seen in today's log for a bot
  local b=$1 f="journal/$b/logs/options_owl_${DAY}.log"
  [ -f "$f" ] && grep -c 'UW_FLOW.*dispatch\|UW_FLOW.*signal\|on_flow_signal' "$f" 2>/dev/null || echo 0
}

flow_trades() {  # per-bot flow trades today from the DB
  local b=$1 db="journal/$b/raw_messages.db"
  [ -f "$db" ] || { echo "  (no db)"; return; }
  sqlite3 "$db" "
    SELECT printf('  %-5s %-4s %3dc @ \$%.2f  %-8s %-14s \$%+.0f  %s',
      ticker, direction, contracts, premium_per_contract, status,
      COALESCE(exit_reason,'-'), COALESCE(pnl_dollars,0),
      CASE WHEN webull_order_id IS NOT NULL THEN 'WEBULL' ELSE 'PAPER' END)
    FROM paper_trades
    WHERE bot_source='uw_flow' AND date(opened_at)='$DAY'
    ORDER BY opened_at" 2>/dev/null
}

for grp in "LIVE:$LIVE" "PAPER:$PAPER"; do
  label=${grp%%:*}; bots=${grp#*:}
  echo ""; echo "########## $label ##########"
  for b in $bots; do
    sig=$(flow_signals "$b")
    db="journal/$b/raw_messages.db"
    n=$([ -f "$db" ] && sqlite3 "$db" "SELECT COUNT(*) FROM paper_trades WHERE bot_source='uw_flow' AND date(opened_at)='$DAY'" 2>/dev/null || echo 0)
    echo "── $b: $sig flow signals seen, $n flow trades ──"
    flow_trades "$b"
  done
done

echo ""
echo "==================== SANITY ===================="
# 1) signal rate ~3/day (sum over a representative bot)
S=$(flow_signals owlet-adam)
echo "Signal rate (adam): $S today  — expect ~3/day; 0 = collector not firing (check WS/logs)"
# 2) insane entry premiums (>\$9 non-index or <=\$0) across all bots
for b in $LIVE $PAPER; do
  db="journal/$b/raw_messages.db"; [ -f "$db" ] || continue
  bad=$(sqlite3 "$db" "SELECT COUNT(*) FROM paper_trades WHERE bot_source='uw_flow' AND date(opened_at)='$DAY' AND (premium_per_contract<=0 OR premium_per_contract>9)" 2>/dev/null)
  [ "${bad:-0}" -gt 0 ] && echo "  ⚠️  $b: $bad flow trades at suspicious premium (<=0 or >\$9)"
done
# 3) live-vs-paper divergence: same tickers traded?
echo ""
echo "Live-vs-paper ticker overlap (same signals should hit both):"
for db in journal/owlet-kody/raw_messages.db journal/owlet-adam/raw_messages.db; do
  [ -f "$db" ] && echo "  $(basename $(dirname $db)): $(sqlite3 "$db" "SELECT GROUP_CONCAT(DISTINCT ticker) FROM paper_trades WHERE bot_source='uw_flow' AND date(opened_at)='$DAY'" 2>/dev/null)"
done
echo ""
echo "If LIVE traded tickers PAPER didn't (or vice-versa) → investigate (premium/spread/cap gate divergence or Webull reject)."
echo "Deep-dive a trade:  ./scripts/trade-pnl.py --droplet --id <N>   |   gate trace: grep 'uw_flow\\|UW_FLOW' journal/owlet-kody/logs/options_owl_${DAY}.log"
