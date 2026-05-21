"""Replay only REAL Webull trades (trades with webull_order_id) with different sizing.

This filters out paper-only trades that never actually executed on Webull,
giving an accurate picture of what different sizing would have produced
on trades that actually filled.

Usage:
    python scripts/backtest_sizing_real_only.py
"""

import json
import subprocess
import sys
from collections import defaultdict

PORTFOLIO_SIZE = 5000.0
MAX_PORTFOLIO_RISK_PCT = 75.0
MAX_CONCURRENT = 5
LIQUIDITY_CAP = 20

SSH_KEY = "~/.ssh/id_ed25519_do"
DROPLET = "root@129.212.138.145"
DB_PATH = "journal/owlet-kody/raw_messages.db"

QUERY = (
    "SELECT id, ticker, direction, score, strike, option_type, contracts, "
    "premium_per_contract, total_cost, entry_price, pnl_dollars, pnl_pct, "
    "exit_reason, duration_minutes, opened_at, closed_at, status, "
    "webull_order_id "
    "FROM paper_trades WHERE status='closed' ORDER BY id"
)


def score_to_contracts(
    score: int,
    cost_per_contract: float,
    balance: float,
    max_position_pct: float,
    max_concurrent: int,
    max_portfolio_risk_pct: float,
) -> int:
    if score < 70 or cost_per_contract <= 0:
        return 0
    total_deployable = balance * (max_portfolio_risk_pct / 100)
    target_per_trade = total_deployable / max(1, max_concurrent)
    raw_contracts = int(target_per_trade / cost_per_contract)
    max_spend = balance * (max_position_pct / 100)
    max_by_position = int(max_spend / cost_per_contract)
    capped = min(raw_contracts, max_by_position, LIQUIDITY_CAP)
    return max(1, capped)


def run_backtest(trades: list[dict], max_position_pct: float) -> dict:
    daily_pnl: dict[str, float] = defaultdict(float)
    daily_trades: dict[str, list] = defaultdict(list)
    total_pnl = 0.0
    wins = losses = 0

    for t in trades:
        premium = t["premium_per_contract"]
        cost_per_contract = premium * 100
        date_str = t["opened_at"][:10]

        new_contracts = score_to_contracts(
            score=t["score"], cost_per_contract=cost_per_contract,
            balance=PORTFOLIO_SIZE, max_position_pct=max_position_pct,
            max_concurrent=MAX_CONCURRENT, max_portfolio_risk_pct=MAX_PORTFOLIO_RISK_PCT,
        )

        old_contracts = t["contracts"]
        if old_contracts <= 0:
            continue
        scale = new_contracts / old_contracts
        new_pnl = t["pnl_dollars"] * scale
        new_cost = new_contracts * cost_per_contract

        daily_pnl[date_str] += new_pnl
        total_pnl += new_pnl
        if new_pnl >= 0:
            wins += 1
        else:
            losses += 1

        daily_trades[date_str].append({
            "ticker": t["ticker"],
            "old_contracts": old_contracts,
            "new_contracts": new_contracts,
            "old_pnl": t["pnl_dollars"],
            "new_pnl": new_pnl,
            "new_cost": new_cost,
            "exit_reason": t["exit_reason"],
            "pnl_pct": t["pnl_pct"],
        })

    return {
        "max_position_pct": max_position_pct,
        "daily_pnl": dict(daily_pnl),
        "daily_trades": dict(daily_trades),
        "total_pnl": total_pnl,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / (wins + losses) * 100 if (wins + losses) > 0 else 0,
    }


def main():
    cmd = [
        "ssh", "-i", SSH_KEY, DROPLET,
        f"cd /root/options-owl && sqlite3 {DB_PATH} \"{QUERY}\" -json",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        print(f"Error: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    all_trades = json.loads(result.stdout)

    # Split into real (Webull) vs paper-only
    real_trades = [t for t in all_trades if t["webull_order_id"] and t["pnl_pct"] != 0.0]
    paper_only = [t for t in all_trades if not t["webull_order_id"] and t["pnl_pct"] != 0.0]

    print(f"Total trades: {len(all_trades)}")
    print(f"  Real Webull trades: {len(real_trades)}")
    print(f"  Paper-only (no Webull order): {len(paper_only)}")
    print(f"\nPortfolio: ${PORTFOLIO_SIZE:,.0f} | Risk Cap: {MAX_PORTFOLIO_RISK_PCT}% | Max Concurrent: {MAX_CONCURRENT}")
    print(f"Target per slot: ${PORTFOLIO_SIZE * MAX_PORTFOLIO_RISK_PCT / 100 / MAX_CONCURRENT:,.0f}")

    # Show real Webull P&L summary first
    print(f"\n{'='*70}")
    print("  ACTUAL WEBULL TRADES (what really happened)")
    print(f"{'='*70}")
    real_daily: dict[str, list] = defaultdict(list)
    for t in real_trades:
        real_daily[t["opened_at"][:10]].append(t)

    actual_total = 0.0
    for date in sorted(real_daily.keys()):
        day_trades = real_daily[date]
        day_pnl = sum(t["pnl_dollars"] for t in day_trades)
        actual_total += day_pnl
        print(f"\n  {date}: ${day_pnl:+8,.2f}  ({len(day_trades)} trades)")
        for t in day_trades:
            print(
                f"    {t['ticker']:6s} x{t['contracts']}  "
                f"${t['total_cost']:>7,.0f} cost  →  ${t['pnl_dollars']:>+8,.2f}  "
                f"({t['pnl_pct']:+.1f}%)  [{t['exit_reason']}]"
            )
    print(f"\n  ACTUAL TOTAL: ${actual_total:+,.2f}")

    # Now backtest with different sizing on REAL trades only
    sizing_levels = [10.0, 15.0, 20.0]

    for pct in sizing_levels:
        r = run_backtest(real_trades, pct)
        print(f"\n{'='*70}")
        print(f"  REAL TRADES @ {pct}% sizing  (max ${PORTFOLIO_SIZE * pct / 100:,.0f}/trade)")
        print(f"{'='*70}")
        print(f"  Total P&L: ${r['total_pnl']:+,.2f}")
        print(f"  Win Rate:  {r['win_rate']:.0f}% ({r['wins']}W / {r['losses']}L)")

        for date in sorted(r["daily_pnl"].keys()):
            dpnl = r["daily_pnl"][date]
            trades_today = r["daily_trades"][date]
            print(f"\n  {date}: ${dpnl:+8,.2f}  ({len(trades_today)} trades)")
            for tt in trades_today:
                resize = ""
                if tt["new_contracts"] != tt["old_contracts"]:
                    resize = f" (was x{tt['old_contracts']})"
                print(
                    f"    {tt['ticker']:6s} x{tt['new_contracts']}{resize:12s} "
                    f"${tt['new_cost']:>7,.0f} cost  →  ${tt['new_pnl']:>+8,.2f}  "
                    f"({tt['pnl_pct']:+.1f}%)  [{tt['exit_reason']}]"
                )

    # Also show paper-only trades that were MISSED
    print(f"\n{'='*70}")
    print("  PAPER-ONLY TRADES (never hit Webull — missed opportunities)")
    print(f"{'='*70}")
    paper_daily: dict[str, list] = defaultdict(list)
    for t in paper_only:
        paper_daily[t["opened_at"][:10]].append(t)

    missed_total = 0.0
    for date in sorted(paper_daily.keys()):
        day_trades = paper_daily[date]
        day_pnl = sum(t["pnl_dollars"] for t in day_trades)
        missed_total += day_pnl
        print(f"\n  {date}: ${day_pnl:+8,.2f}  ({len(day_trades)} trades, paper-only)")
        for t in day_trades:
            print(
                f"    {t['ticker']:6s} x{t['contracts']}  "
                f"${t['total_cost']:>7,.0f} cost  →  ${t['pnl_dollars']:>+8,.2f}  "
                f"({t['pnl_pct']:+.1f}%)  [{t['exit_reason']}]"
            )
    print(f"\n  PAPER-ONLY TOTAL: ${missed_total:+,.2f}")
    print(f"  (Would have been extra P&L if all had executed on Webull)")

    # Final comparison
    print(f"\n{'='*70}")
    print("  COMPARISON: Real trades with different sizing")
    print(f"{'='*70}")
    print(f"  {'Sizing':<12} {'Total P&L':>12} {'Win Rate':>10} {'Avg/Trade':>12}")
    print(f"  {'-'*48}")
    print(f"  {'ACTUAL':<12} ${actual_total:>+10,.2f}   {'':>5}    {'':>10}")
    for pct in sizing_levels:
        r = run_backtest(real_trades, pct)
        n = r["wins"] + r["losses"]
        avg = r["total_pnl"] / n if n > 0 else 0
        print(f"  {pct:>5.0f}%       ${r['total_pnl']:>+10,.2f}   {r['win_rate']:>5.0f}%    ${avg:>+10,.2f}")


if __name__ == "__main__":
    main()
