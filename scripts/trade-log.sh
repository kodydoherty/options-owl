#!/usr/bin/env bash
# trade-log.sh — Query trade events and logs from the droplet
#
# Usage:
#   ./scripts/trade-log.sh                    # today's trade events
#   ./scripts/trade-log.sh 2026-04-23         # specific date
#   ./scripts/trade-log.sh events             # all trade events (last 50)
#   ./scripts/trade-log.sh trades             # all trades with Webull status
#   ./scripts/trade-log.sh logs [date]        # grep trade-relevant log lines
#   ./scripts/trade-log.sh webull [date]      # grep Webull order lines

set -euo pipefail

DROPLET="root@129.212.138.145"
SSH_KEY="$HOME/.ssh/id_ed25519_do"
DB="journal/owlet-kody/raw_messages.db"
LOG_DIR="journal/owlet-kody/logs"

CMD="${1:-today}"
DATE="${2:-$(TZ=America/New_York date +%Y-%m-%d)}"

case "$CMD" in
  today|events)
    if [ "$CMD" = "today" ]; then
      WHERE="WHERE date(created_at) = '$DATE'"
    else
      WHERE=""
    fi
    echo "=== Trade Events ${DATE} ==="
    ssh -i "$SSH_KEY" "$DROPLET" "cd /root/options-owl && sqlite3 $DB \"
      SELECT id, ticker, event_type, detail, substr(created_at, 12, 8) as time
      FROM trade_events $WHERE
      ORDER BY id DESC LIMIT 50
    \" -column -header" 2>/dev/null || echo "(no trade_events table yet — will appear after next market day)"
    ;;

  trades)
    BOT="${3:-kody}"
    DB="journal/owlet-$BOT/raw_messages.db"
    echo "=== LIVE Trades for owlet-$BOT (parents + partials) ==="
    ssh -i "$SSH_KEY" "$DROPLET" "cd /root/options-owl && sqlite3 \$DB \"
      SELECT t.id,
        CASE WHEN t.parent_trade_id IS NOT NULL THEN '  ^' || t.parent_trade_id ELSE '' END as parent,
        t.ticker, t.option_type as type, t.strike, t.contracts as qty,
        printf('\\\$%.2f', t.premium_per_contract) as entry,
        printf('\\\$%.2f', COALESCE(NULLIF(t.webull_exit_fill_price,0), t.exit_premium)) as exit_p,
        printf('\\\$%.2f', t.pnl_dollars) as pnl,
        CASE WHEN t.pnl_dollars > 0 THEN 'WIN' ELSE 'LOSS' END as result,
        t.exit_reason, date(t.opened_at) as date
      FROM paper_trades t
      WHERE t.status = 'closed'
        AND (length(t.webull_order_id) > 0
             OR t.parent_trade_id IN (SELECT id FROM paper_trades WHERE length(webull_order_id) > 0))
      ORDER BY t.opened_at, t.id
    \" -column -header" DB="$DB" 2>/dev/null
    echo ""
    echo "=== Summary ==="
    ssh -i "$SSH_KEY" "$DROPLET" "cd /root/options-owl && sqlite3 $DB \"
      SELECT COUNT(*) as trades,
        SUM(CASE WHEN pnl_dollars > 0 THEN 1 ELSE 0 END) as wins,
        SUM(CASE WHEN pnl_dollars <= 0 THEN 1 ELSE 0 END) as losses,
        printf('\\\$%.2f', SUM(pnl_dollars)) as total_pnl,
        printf('\\\$%.2f', SUM(CASE WHEN date(opened_at) = '$(TZ=America/New_York date +%Y-%m-%d)' THEN pnl_dollars ELSE 0 END)) as today_pnl
      FROM paper_trades t
      WHERE t.status = 'closed'
        AND (length(t.webull_order_id) > 0
             OR t.parent_trade_id IN (SELECT id FROM paper_trades WHERE length(webull_order_id) > 0))
    \" -column -header" 2>/dev/null
    ;;

  logs)
    echo "=== Trade logs for $DATE ==="
    ssh -i "$SSH_KEY" "$DROPLET" "grep -E 'TradeLifecycle|WEBULL ENTRY|PAPER TRADE|Pipeline|SIZING|SmartEntry' /root/options-owl/$LOG_DIR/options_owl_${DATE}.log 2>/dev/null" || echo "(no log file for $DATE)"
    ;;

  webull)
    echo "=== Webull orders for $DATE ==="
    ssh -i "$SSH_KEY" "$DROPLET" "grep -iE 'WEBULL ORDER|WEBULL ENTRY|WEBULL.*FILL|WEBULL.*ERROR|WEBULL.*FAIL|PARAM_ERR' /root/options-owl/$LOG_DIR/options_owl_${DATE}.log 2>/dev/null" || echo "(no log file for $DATE)"
    ;;

  *)
    # Treat first arg as a date
    DATE="$CMD"
    echo "=== Trade Events for $DATE ==="
    ssh -i "$SSH_KEY" "$DROPLET" "cd /root/options-owl && sqlite3 $DB \"
      SELECT id, ticker, event_type, detail, substr(created_at, 12, 8) as time
      FROM trade_events WHERE date(created_at) = '$DATE'
      ORDER BY id
    \" -column -header" 2>/dev/null || echo "(no trade_events table yet)"
    ;;
esac
