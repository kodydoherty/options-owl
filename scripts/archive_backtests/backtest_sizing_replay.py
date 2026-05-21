"""Replay all historical live trades with different MAX_POSITION_PCT values.

Uses actual trade outcomes (entry premium, exit premium, pnl_pct) and re-sizes
contracts based on the dollar-target sizing formula to show what daily P&L
would have been at 10% vs 15% vs 20% per trade.

Usage:
    python scripts/backtest_sizing_replay.py
"""

import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime

# Current production settings
PORTFOLIO_SIZE = 5000.0
MAX_PORTFOLIO_RISK_PCT = 75.0
MAX_CONCURRENT = 5
LIQUIDITY_CAP = 20

# Pull trades from droplet
SSH_KEY = "~/.ssh/id_ed25519_do"
DROPLET = "root@129.212.138.145"
DB_PATH = "journal/owlet-kody/raw_messages.db"

QUERY = (
    "SELECT id, ticker, direction, score, strike, option_type, contracts, "
    "premium_per_contract, total_cost, entry_price, pnl_dollars, pnl_pct, "
    "exit_reason, duration_minutes, opened_at, closed_at, status "
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
    if score < 70:
        return 0
    if cost_per_contract <= 0:
        return 0

    total_deployable = balance * (max_portfolio_risk_pct / 100)
    target_per_trade = total_deployable / max(1, max_concurrent)
    raw_contracts = int(target_per_trade / cost_per_contract)

    max_spend = balance * (max_position_pct / 100)
    max_by_position = int(max_spend / cost_per_contract)
    capped = min(raw_contracts, max_by_position)
    capped = min(capped, LIQUIDITY_CAP)
    return max(1, capped)


def run_backtest(trades: list[dict], max_position_pct: float) -> dict:
    """Replay trades with a given MAX_POSITION_PCT, return daily P&L summary."""
    daily_pnl: dict[str, float] = defaultdict(float)
    daily_trades: dict[str, list] = defaultdict(list)
    total_pnl = 0.0
    wins = 0
    losses = 0

    for t in trades:
        premium = t["premium_per_contract"]
        cost_per_contract = premium * 100
        date_str = t["opened_at"][:10]

        new_contracts = score_to_contracts(
            score=t["score"],
            cost_per_contract=cost_per_contract,
            balance=PORTFOLIO_SIZE,
            max_position_pct=max_position_pct,
            max_concurrent=MAX_CONCURRENT,
            max_portfolio_risk_pct=MAX_PORTFOLIO_RISK_PCT,
        )

        # Scale P&L proportionally: new_pnl = original_pnl * (new_contracts / old_contracts)
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
    # Fetch trades from droplet
    cmd = [
        "ssh", "-i", SSH_KEY, DROPLET,
        f"cd /root/options-owl && sqlite3 {DB_PATH} \"{QUERY}\" -json",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        print(f"Error fetching trades: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    trades = json.loads(result.stdout)
    # Filter out phantom/zero-pnl trades
    trades = [t for t in trades if t["pnl_pct"] != 0.0 and t["exit_reason"] != "phantom_cleanup"]

    print(f"Replaying {len(trades)} trades from {trades[0]['opened_at'][:10]} to {trades[-1]['opened_at'][:10]}")
    print(f"Portfolio: ${PORTFOLIO_SIZE:,.0f} | Risk Cap: {MAX_PORTFOLIO_RISK_PCT}% | Max Concurrent: {MAX_CONCURRENT}")
    print(f"Target per slot: ${PORTFOLIO_SIZE * MAX_PORTFOLIO_RISK_PCT / 100 / MAX_CONCURRENT:,.0f}")
    print()

    sizing_levels = [10.0, 15.0, 20.0, 25.0]

    for pct in sizing_levels:
        result = run_backtest(trades, pct)
        print(f"{'='*70}")
        print(f"  MAX_POSITION_PCT = {pct}%  (max ${PORTFOLIO_SIZE * pct / 100:,.0f} per trade)")
        print(f"{'='*70}")
        print(f"  Total P&L: ${result['total_pnl']:+,.2f}")
        print(f"  Win Rate:  {result['win_rate']:.0f}% ({result['wins']}W / {result['losses']}L)")
        print()

        # Daily breakdown
        for date in sorted(result["daily_pnl"].keys()):
            dpnl = result["daily_pnl"][date]
            trades_today = result["daily_trades"][date]
            print(f"  {date}: ${dpnl:+8,.2f}  ({len(trades_today)} trades)")
            for tt in trades_today:
                arrow = "+" if tt["new_pnl"] >= 0 else ""
                resize = ""
                if tt["new_contracts"] != tt["old_contracts"]:
                    resize = f" (was x{tt['old_contracts']})"
                print(
                    f"    {tt['ticker']:6s} x{tt['new_contracts']}{resize:12s} "
                    f"${tt['new_cost']:>7,.0f} cost  →  {arrow}${tt['new_pnl']:>8,.2f}  "
                    f"({tt['pnl_pct']:+.1f}%)  [{tt['exit_reason']}]"
                )
        print()

    # Comparison table
    print(f"\n{'='*70}")
    print("  COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Sizing':<12} {'Total P&L':>12} {'Win Rate':>10} {'Avg/Trade':>12}")
    print(f"  {'-'*48}")
    for pct in sizing_levels:
        r = run_backtest(trades, pct)
        n = r["wins"] + r["losses"]
        avg = r["total_pnl"] / n if n > 0 else 0
        print(f"  {pct:>5.0f}%       ${r['total_pnl']:>+10,.2f}   {r['win_rate']:>5.0f}%    ${avg:>+10,.2f}")


if __name__ == "__main__":
    main()
