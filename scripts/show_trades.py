import sqlite3
conn = sqlite3.connect('journal/raw_messages.db')
conn.row_factory = sqlite3.Row
rows = conn.execute(
    "SELECT id, ticker, option_type, strike, contracts, "
    "premium_per_contract, exit_premium, pnl_dollars, "
    "webull_order_id, webull_exit_order_id, "
    "exit_source, exit_reason, status "
    "FROM paper_trades WHERE date(opened_at) >= '2026-05-11' ORDER BY id"
).fetchall()
for r in rows:
    woid = (r['webull_order_id'] or '')[:12]
    exoid = (r['webull_exit_order_id'] or '')[:12]
    entry = r['premium_per_contract'] or 0
    exit_p = r['exit_premium'] or 0
    pnl = r['pnl_dollars'] or 0
    print(
        f"#{r['id']:3d} {r['ticker']:5s} {r['option_type']:4s} "
        f"${r['strike']:7.1f} x{r['contracts']:2d} "
        f"entry=${entry:.2f} exit=${exit_p:.2f} pnl=${pnl:+.2f} "
        f"[{r['status']:6s}] src={r['exit_source'] or ''} "
        f"reason={r['exit_reason'] or ''} woid={woid} exoid={exoid}"
    )
conn.close()
