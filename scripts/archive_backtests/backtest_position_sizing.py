#!/usr/bin/env python3
"""
Backtest different MAX_POSITION_PCT values on historical trades.

Replays all Kody's trades with different position sizing to find
the optimal allocation that maximizes P&L while keeping risk reasonable.
"""

from dataclasses import dataclass

PORTFOLIO = 5000.0
MAX_CONCURRENT = 3
MAX_CONTRACTS = 20
MIN_CONTRACTS = 1

# All Kody trades: (ticker, date, premium, exit_price, pnl_pct)
TRADES = [
    # 4/23 - Today
    ("AMZN call", "4/23", 1.618, 1.978, 22.3),
    ("SPY call", "4/23", 1.266, 1.512, 19.4),
    ("QQQ call", "4/23", 1.638, 1.861, 13.6),
    ("AMD call", "4/23", 4.492, 4.773, 6.3),
    ("AVGO call", "4/23", 4.191, 3.777, -9.9),
    ("AAPL call", "4/23", 1.397, 0.987, -29.3),
    ("PLTR put", "4/23", 1.548, 1.694, 9.5),
    # 4/22
    ("GOOGL put", "4/22", 2.101, 3.276, 55.9),
    ("NVDA call", "4/22", 1.598, 1.963, 22.8),
    ("QQQ call", "4/22", 1.538, 1.964, 27.7),
    ("AMZN put", "4/22", 1.859, 2.485, 33.6),
    ("META put", "4/22", 4.191, 5.175, 23.5),
    ("NVDA call2", "4/22", 0.291, 0.119, -58.6),
    ("SPY put", "4/22", 2.060, 0.408, -80.2),
    ("AVGO call", "4/22", 2.000, 0.726, -61.0),
    # 4/21
    ("SPY call", "4/21", 1.236, 1.507, 21.9),
    ("NVDA call", "4/21", 1.005, 0.391, -61.1),
    ("IWM call", "4/21", 0.623, 0.239, -58.6),
    ("MSFT call", "4/21", 3.236, 2.327, -28.1),
    ("NVDA call2", "4/21", 1.377, 1.002, -27.2),
    ("AMZN call", "4/21", 2.221, 1.648, -25.8),
    # 4/17
    ("SPY call", "4/17", 2.221, 3.327, 49.8),
    ("IWM call", "4/17", 0.824, 0.478, -37.7),
    ("QQQ call", "4/17", 1.538, 1.493, -31.3),
    ("AMZN call", "4/17", 1.075, 0.786, -20.2),
    # 4/15
    ("QQQ call", "4/15", 1.175, 1.573, 33.8),
    ("AMZN call", "4/15", 0.804, 0.993, 23.5),
    ("GOOGL call", "4/15", 0.864, 0.338, -60.9),
    ("SPY call", "4/15", 0.432, 0.259, -12.5),
]


@dataclass
class TradeResult:
    ticker: str
    date: str
    premium: float
    exit_price: float
    pnl_pct: float
    contracts: int
    cost: float
    pnl_dollar: float


def size_trade(portfolio: float, max_pos_pct: float, premium: float) -> int:
    """Calculate contracts using Vinny sizing: target = portfolio * pct / max_concurrent."""
    target_per_trade = portfolio * (max_pos_pct / 100.0) / MAX_CONCURRENT
    cost_per_contract = premium * 100  # options are 100 shares
    contracts = int(target_per_trade / cost_per_contract)
    return max(MIN_CONTRACTS, min(contracts, MAX_CONTRACTS))


def run_backtest(max_pos_pct: float) -> list[TradeResult]:
    results = []
    for ticker, date, premium, exit_price, pnl_pct, in TRADES:
        contracts = size_trade(PORTFOLIO, max_pos_pct, premium)
        cost = contracts * premium * 100
        pnl_dollar = contracts * (exit_price - premium) * 100
        results.append(TradeResult(
            ticker=ticker, date=date, premium=premium,
            exit_price=exit_price, pnl_pct=pnl_pct,
            contracts=contracts, cost=cost, pnl_dollar=pnl_dollar,
        ))
    return results


def calc_metrics(results: list[TradeResult]) -> dict:
    pnls = [r.pnl_dollar for r in results]
    winners = [r for r in results if r.pnl_dollar > 0]
    losers = [r for r in results if r.pnl_dollar <= 0]

    total_pnl = sum(pnls)
    win_rate = len(winners) / len(results) * 100
    avg_winner = sum(r.pnl_dollar for r in winners) / len(winners) if winners else 0
    avg_loser = sum(r.pnl_dollar for r in losers) / len(losers) if losers else 0
    gross_profit = sum(r.pnl_dollar for r in winners)
    gross_loss = abs(sum(r.pnl_dollar for r in losers))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Max drawdown: running cumulative P&L
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for pnl in pnls:
        cumulative += pnl
        peak = max(peak, cumulative)
        dd = peak - cumulative
        max_dd = max(max_dd, dd)

    # Avg cost per trade
    avg_cost = sum(r.cost for r in results) / len(results)
    max_cost = max(r.cost for r in results)

    # Avg contracts
    avg_contracts = sum(r.contracts for r in results) / len(results)

    return {
        "total_pnl": total_pnl,
        "win_rate": win_rate,
        "avg_winner": avg_winner,
        "avg_loser": avg_loser,
        "profit_factor": profit_factor,
        "max_drawdown": max_dd,
        "avg_cost": avg_cost,
        "max_cost": max_cost,
        "avg_contracts": avg_contracts,
        "total_trades": len(results),
        "winners": len(winners),
        "losers": len(losers),
        "return_pct": total_pnl / PORTFOLIO * 100,
    }


def main():
    pct_values = [10, 15, 20, 25, 30, 33, 40, 50]

    print("=" * 120)
    print(f"POSITION SIZING BACKTEST — {len(TRADES)} trades, ${PORTFOLIO:.0f} portfolio, MAX_CONCURRENT={MAX_CONCURRENT}")
    print(f"Sizing formula: target_per_trade = ${PORTFOLIO:.0f} * MAX_POSITION_PCT / {MAX_CONCURRENT}")
    print("=" * 120)

    all_metrics = {}
    for pct in pct_values:
        results = run_backtest(pct)
        metrics = calc_metrics(results)
        all_metrics[pct] = metrics

    # Summary table
    print()
    print(f"{'PCT':>5} | {'Target/Trade':>12} | {'AvgContr':>8} | {'TotalP&L':>10} | {'Return%':>8} | "
          f"{'WinRate':>7} | {'AvgWin':>8} | {'AvgLoss':>8} | {'PF':>5} | {'MaxDD':>8} | {'AvgCost':>8}")
    print("-" * 120)

    for pct in pct_values:
        m = all_metrics[pct]
        target = PORTFOLIO * pct / 100 / MAX_CONCURRENT
        print(f"{pct:>4}% | ${target:>10.0f} | {m['avg_contracts']:>8.1f} | "
              f"${m['total_pnl']:>+9.2f} | {m['return_pct']:>+7.1f}% | "
              f"{m['win_rate']:>6.1f}% | ${m['avg_winner']:>7.2f} | ${m['avg_loser']:>7.2f} | "
              f"{m['profit_factor']:>5.2f} | ${m['max_drawdown']:>7.2f} | ${m['avg_cost']:>7.0f}")

    # Find optimal
    best_pct = max(pct_values, key=lambda p: all_metrics[p]["total_pnl"])
    best = all_metrics[best_pct]

    # Also find best risk-adjusted (P&L / max_drawdown)
    best_risk_adj_pct = max(pct_values, key=lambda p: (
        all_metrics[p]["total_pnl"] / all_metrics[p]["max_drawdown"]
        if all_metrics[p]["max_drawdown"] > 0 else float('inf')
    ))

    print()
    print("=" * 120)
    print(f"BEST RAW P&L:           {best_pct}% -> ${best['total_pnl']:+.2f} ({best['return_pct']:+.1f}% return)")
    ra = all_metrics[best_risk_adj_pct]
    ratio = ra["total_pnl"] / ra["max_drawdown"] if ra["max_drawdown"] > 0 else 0
    print(f"BEST RISK-ADJUSTED:     {best_risk_adj_pct}% -> ${ra['total_pnl']:+.2f} "
          f"(P&L/DD ratio: {ratio:.2f})")
    print("=" * 120)

    # Per-day breakdown for current (20%) and best
    print()
    print("PER-DAY P&L BREAKDOWN (current 20% vs best)")
    print("-" * 80)

    for pct_show in sorted(set([20, best_pct])):
        results = run_backtest(pct_show)
        days = {}
        for r in results:
            days.setdefault(r.date, []).append(r)

        print(f"\n  MAX_POSITION_PCT = {pct_show}%:")
        running = 0.0
        for date in ["4/15", "4/17", "4/21", "4/22", "4/23"]:
            if date not in days:
                continue
            day_pnl = sum(r.pnl_dollar for r in days[date])
            day_wins = sum(1 for r in days[date] if r.pnl_dollar > 0)
            day_total = len(days[date])
            running += day_pnl
            print(f"    {date}: ${day_pnl:>+8.2f}  ({day_wins}/{day_total} wins)  "
                  f"running: ${running:>+9.2f}")

    # Detailed trade list for current 20%
    print()
    print("=" * 120)
    print("DETAILED TRADES @ 20% (current setting)")
    print(f"{'Ticker':<16} {'Date':<6} {'Prem':>6} {'Exit':>6} {'P&L%':>7} {'Contr':>5} {'Cost':>8} {'P&L$':>9}")
    print("-" * 80)
    results_20 = run_backtest(20)
    for r in sorted(results_20, key=lambda x: x.pnl_dollar, reverse=True):
        print(f"{r.ticker:<16} {r.date:<6} ${r.premium:>5.3f} ${r.exit_price:>5.3f} "
              f"{r.pnl_pct:>+6.1f}% {r.contracts:>5} ${r.cost:>7.2f} ${r.pnl_dollar:>+8.2f}")
    total_20 = sum(r.pnl_dollar for r in results_20)
    print(f"{'':>16} {'':>6} {'':>6} {'':>6} {'':>7} {'':>5} {'':>8} ${total_20:>+8.2f}")

    # Show what changes at each level
    print()
    print("=" * 120)
    print("CONTRACT COUNT COMPARISON BY PREMIUM LEVEL")
    print(f"{'Premium':>8}", end="")
    for pct in pct_values:
        print(f" | {pct}%", end="")
    print()
    print("-" * 80)
    sample_premiums = [0.30, 0.50, 0.80, 1.00, 1.50, 2.00, 3.00, 4.00, 5.00]
    for prem in sample_premiums:
        print(f"${prem:>6.2f} ", end="")
        for pct in pct_values:
            c = size_trade(PORTFOLIO, pct, prem)
            print(f" | {c:>3}", end="")
        print()


if __name__ == "__main__":
    main()
