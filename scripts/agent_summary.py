"""Show today's trade summary for the current agent."""
import sqlite3
import os

db = os.environ.get("DB_PATH", "journal/raw_messages.db")
if not os.path.exists(db):
    print("No DB found")
    exit()

conn = sqlite3.connect(db)
row = conn.execute(
    "SELECT COUNT(*), "
    "SUM(CASE WHEN pnl_dollars > 0 THEN 1 ELSE 0 END), "
    "SUM(CASE WHEN pnl_dollars <= 0 THEN 1 ELSE 0 END), "
    "SUM(pnl_dollars), "
    "SUM(CASE WHEN webull_order_id IS NOT NULL THEN 1 ELSE 0 END) "
    "FROM paper_trades "
    "WHERE date(opened_at) >= '2026-05-12' AND status = 'closed'"
).fetchone()

cnt, wins, losses, total, wb = row
wins = wins or 0
losses = losses or 0
total = total or 0
wb = wb or 0
print("Summary: {} trades, {}W/{}L, PnL: ${:+.2f}, Webull: {}".format(
    cnt, wins, losses, total, wb))
print()

trades = conn.execute(
    "SELECT id, ticker, option_type, strike, contracts, "
    "premium_per_contract, exit_premium, pnl_dollars, "
    "webull_order_id, exit_source, exit_reason "
    "FROM paper_trades "
    "WHERE date(opened_at) >= '2026-05-12' AND status = 'closed' "
    "ORDER BY id"
).fetchall()

for r in trades:
    tid, ticker, otype, strike, contracts = r[0], r[1], r[2], r[3], r[4]
    entry, exit_p, pnl = r[5], r[6] or 0, r[7] or 0
    woid = "WB" if r[8] else "PP"
    src = r[9] or "ai"
    reason = r[10] or ""
    print("  #{:3d} {:5s} {:4s} ${:7.1f} x{:2d} "
          "entry=${:.2f} exit=${:.2f} pnl=${:+9.2f} [{}] {}/{}".format(
              tid, ticker, otype, strike, contracts,
              entry, exit_p, pnl, woid, src, reason))

conn.close()
