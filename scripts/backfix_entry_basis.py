"""Back-fix historical trades whose recorded entry basis (premium_per_contract) does NOT
match the ACTUAL Webull fill (webull_entry_fill_price). The fill is ground truth; the
recorded premium can be a stale/pre-fill quote (see the 2026-06-26 FSM entry-basis bug).

Corrects, for CLOSED non-DCA trades only (and skips scaleout parents/children to avoid the
known partial-close P&L complications — those are left for manual review and logged):
  - premium_per_contract  -> webull_entry_fill_price (true entry basis)
  - total_cost            -> fill * contracts * 100
  - pnl_dollars / pnl_pct -> recomputed from the actual exit basis (webull_exit_fill_price
                             when present, else exit_premium)
  - mfe_pnl_pct / mae_pnl_pct -> recomputed off the corrected entry (premiums unchanged)

DRY-RUN by default. Pass --apply to write. Run per-bot DB on the droplet.
Usage:  python scripts/backfix_entry_basis.py <db_path> [--apply]
"""
from __future__ import annotations

import sqlite3
import sys

TOL = 0.02  # ignore sub-2% / 2-cent rounding noise


def _has_scaleout_children(conn, trade_id):
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM paper_trades WHERE parent_trade_id = ?", (trade_id,)
        ).fetchone()
        return (row[0] or 0) > 0
    except sqlite3.OperationalError:
        return False  # no parent_trade_id column → no scaleout model


def main():
    if len(sys.argv) < 2:
        print("usage: backfix_entry_basis.py <db_path> [--apply]")
        sys.exit(1)
    db_path = sys.argv[1]
    apply = "--apply" in sys.argv

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT * FROM paper_trades
           WHERE webull_entry_fill_price IS NOT NULL AND webull_entry_fill_price > 0
             AND premium_per_contract IS NOT NULL
             AND ABS(premium_per_contract - webull_entry_fill_price) > MAX(?, ?*webull_entry_fill_price)
             AND COALESCE(dca_total_contracts, 0) = 0
             AND (dca_last_add_at IS NULL)""",
        (TOL, TOL),
    ).fetchall()

    print(f"\n=== {db_path} — {len(rows)} candidate(s) ===")
    fixed = skipped = 0
    for r in rows:
        tid = r["id"]
        fill = float(r["webull_entry_fill_price"])
        old_entry = float(r["premium_per_contract"])
        contracts = int(r["contracts"] or 1)
        status = r["status"]

        if status != "closed":
            print(f"  #{tid} {r['ticker']}: status={status} (open) — SKIP")
            skipped += 1
            continue
        if _has_scaleout_children(conn, tid):
            print(f"  #{tid} {r['ticker']}: has scaleout children — SKIP (manual review)")
            skipped += 1
            continue
        # NEVER fabricate P&L: only correct trades with a VERIFIABLE two-sided Webull fill.
        # Manual/orphan closes have an APPROXIMATE exit (webull_exit_fill_price=0) — their
        # recorded P&L is intentionally conservative; recomputing it from the approx exit
        # price would invent realized P&L that never happened. Leave them untouched.
        exit_fill = float(r["webull_exit_fill_price"] or 0.0)
        reason = (r["exit_reason"] or "")
        if r["exit_source"] == "manual" or reason.startswith("orphan") or exit_fill <= 0:
            print(f"  #{tid} {r['ticker']}: no verifiable exit fill "
                  f"(source={r['exit_source']}, reason={reason[:24]}) — SKIP (entry stale but P&L not recomputable)")
            skipped += 1
            continue

        exit_basis = exit_fill  # real Webull exit fill only
        new_cost = fill * contracts * 100
        new_pnl = (exit_basis - fill) * contracts * 100 if exit_basis > 0 else r["pnl_dollars"]
        new_pnl_pct = ((exit_basis - fill) / fill * 100) if (exit_basis > 0 and fill > 0) else r["pnl_pct"]

        # Recompute peak/trough % off the corrected entry (premiums unchanged).
        new_mfe_pct = r["mfe_pnl_pct"]
        new_mae_pct = r["mae_pnl_pct"]
        if r["mfe_premium"] and fill > 0:
            new_mfe_pct = (float(r["mfe_premium"]) - fill) / fill * 100
        if r["mae_premium"] and fill > 0:
            new_mae_pct = (float(r["mae_premium"]) - fill) / fill * 100

        print(
            f"  #{tid} {r['ticker']} {r['option_type']}: "
            f"entry ${old_entry:.2f}→${fill:.2f}  "
            f"pnl ${float(r['pnl_dollars'] or 0):+.0f}→${new_pnl:+.0f}  "
            f"pnl% {float(r['pnl_pct'] or 0):+.1f}→{new_pnl_pct:+.1f}  "
            f"(exit_basis=${exit_basis:.2f}, x{contracts})"
        )

        if apply:
            conn.execute(
                """UPDATE paper_trades SET
                     premium_per_contract = ?, total_cost = ?,
                     pnl_dollars = ?, pnl_pct = ?, mfe_pnl_pct = ?, mae_pnl_pct = ?
                   WHERE id = ?""",
                (fill, new_cost, new_pnl, new_pnl_pct, new_mfe_pct, new_mae_pct, tid),
            )
        fixed += 1

    if apply:
        conn.commit()
        print(f"  APPLIED: {fixed} corrected, {skipped} skipped.")
    else:
        print(f"  DRY-RUN: {fixed} would be corrected, {skipped} skipped. Pass --apply to write.")
    conn.close()


if __name__ == "__main__":
    main()
