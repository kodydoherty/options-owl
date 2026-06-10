"""Simulate profit-based retracement on real live trades.

For each closed trade, replay the premium history and compare:
  A) What actually happened (real exit reason + real PnL)
  B) What would happen with 35% profit retrace (below +40% gain)
  C) What would happen with adaptive trail only (current production after fix)

Uses real MFE and premium data from the trade database.
"""

import json
import sys


def simulate_profit_retrace(trade, retrace_pct=35.0, min_gain_pct=10.0, adaptive_dormant=40.0):
    """Simulate what profit retrace would do on a real trade.

    Returns what the exit would look like if profit retrace was active
    in the zone below adaptive_dormant (where adaptive trail is dormant).
    """
    entry = trade["premium_per_contract"]
    exit_prem = trade["exit_premium"]
    mfe = trade["mfe_premium"]

    if not entry or entry <= 0 or not mfe:
        return None

    peak_gain_pct = (mfe - entry) / entry * 100

    # Only applies in the zone where adaptive trail is dormant
    if peak_gain_pct >= adaptive_dormant:
        return {
            "applies": False,
            "reason": f"Peak gain +{peak_gain_pct:.1f}% >= {adaptive_dormant}% → adaptive trail handles this",
            "would_change": False,
        }

    # Check if min gain threshold met
    if peak_gain_pct < min_gain_pct:
        return {
            "applies": False,
            "reason": f"Peak gain +{peak_gain_pct:.1f}% < min {min_gain_pct}% → retrace not active",
            "would_change": False,
        }

    # Profit retrace applies here. Calculate the exit point.
    profit_at_peak = mfe - entry  # e.g., $1.50 - $1.00 = $0.50
    retrace_amount = profit_at_peak * (retrace_pct / 100)  # 35% of $0.50 = $0.175
    retrace_exit_price = mfe - retrace_amount  # $1.50 - $0.175 = $1.325

    # What actually happened
    actual_pnl_per = (exit_prem - entry)
    actual_pnl_pct = actual_pnl_per / entry * 100

    # What profit retrace would give
    # The exit happens at retrace_exit_price IF the real exit was below that
    # (meaning the trade DID retrace past this point)
    if exit_prem < retrace_exit_price:
        # Profit retrace would have exited earlier at a better price
        retrace_pnl_per = (retrace_exit_price - entry)
        retrace_pnl_pct = retrace_pnl_per / entry * 100
        contracts = trade["contracts"]
        actual_pnl_dollars = actual_pnl_per * contracts * 100
        retrace_pnl_dollars = retrace_pnl_per * contracts * 100
        improvement = retrace_pnl_dollars - actual_pnl_dollars

        return {
            "applies": True,
            "would_change": True,
            "actual_exit": exit_prem,
            "retrace_exit": retrace_exit_price,
            "actual_pnl": actual_pnl_dollars,
            "retrace_pnl": retrace_pnl_dollars,
            "improvement": improvement,
            "actual_pnl_pct": actual_pnl_pct,
            "retrace_pnl_pct": retrace_pnl_pct,
            "peak_gain_pct": peak_gain_pct,
            "reason": (f"Retrace would exit at ${retrace_exit_price:.3f} "
                      f"(kept {100-retrace_pct:.0f}% of ${profit_at_peak:.3f} profit) "
                      f"vs actual ${exit_prem:.3f}"),
        }
    else:
        # Trade exited above the retrace line — retrace wouldn't have helped
        # But it also wouldn't have hurt (trade exited for another reason first)
        return {
            "applies": True,
            "would_change": False,
            "reason": (f"Actual exit ${exit_prem:.3f} >= retrace line ${retrace_exit_price:.3f} "
                      f"→ other exit fired first ({trade['exit_reason']})"),
        }


def main():
    trades_json = """TRADES_PLACEHOLDER"""

    trades = [
        {"id":1,"ticker":"IWM","option_type":"call","strike":263.0,"contracts":16,"premium_per_contract":0.3015,"exit_premium":0.34825,"mfe_premium":0.41,"pnl_dollars":74.8,"exit_reason":"velocity_exit","opened_at":"2026-04-13T14:36:12"},
        {"id":2,"ticker":"SPY","option_type":"call","strike":680.0,"contracts":6,"premium_per_contract":0.804,"exit_premium":1.30345,"mfe_premium":1.5,"pnl_dollars":299.67,"exit_reason":"velocity_exit","opened_at":"2026-04-13T14:51:13"},
        {"id":3,"ticker":"QQQ","option_type":"call","strike":613.0,"contracts":8,"premium_per_contract":0.6432,"exit_premium":0.68655,"mfe_premium":0.81,"pnl_dollars":34.68,"exit_reason":"velocity_exit","opened_at":"2026-04-13T15:00:31"},
        {"id":4,"ticker":"AMZN","option_type":"call","strike":237.5,"contracts":7,"premium_per_contract":0.75375,"exit_premium":0.94525,"mfe_premium":1.13,"pnl_dollars":134.05,"exit_reason":"velocity_exit","opened_at":"2026-04-13T15:24:19"},
        {"id":5,"ticker":"META","option_type":"call","strike":627.5,"contracts":3,"premium_per_contract":1.7085,"exit_premium":2.5074,"mfe_premium":3.03,"pnl_dollars":239.67,"exit_reason":"velocity_exit","opened_at":"2026-04-13T15:39:19"},
        {"id":6,"ticker":"MSTR","option_type":"call","strike":139.0,"contracts":1,"premium_per_contract":3.66825,"exit_premium":3.82652,"mfe_premium":3.84575,"pnl_dollars":15.83,"exit_reason":"t1_hit","opened_at":"2026-04-15T14:51:54"},
        {"id":7,"ticker":"GOOGL","option_type":"call","strike":335.0,"contracts":4,"premium_per_contract":0.8643,"exit_premium":0.3383,"mfe_premium":0.8643,"pnl_dollars":-210.4,"exit_reason":"stop_hit","opened_at":"2026-04-15T14:54:53"},
        {"id":8,"ticker":"QQQ","option_type":"call","strike":632.0,"contracts":2,"premium_per_contract":1.2663,"exit_premium":1.38305,"mfe_premium":1.65,"pnl_dollars":23.35,"exit_reason":"velocity_exit","opened_at":"2026-04-15T15:01:10"},
        {"id":10,"ticker":"AAPL","option_type":"call","strike":262.5,"contracts":8,"premium_per_contract":0.4221,"exit_premium":1.2338,"mfe_premium":1.43,"pnl_dollars":649.36,"exit_reason":"velocity_exit","opened_at":"2026-04-15T15:15:55"},
        {"id":11,"ticker":"SPY","option_type":"call","strike":698.0,"contracts":2,"premium_per_contract":1.1256,"exit_premium":1.1343,"mfe_premium":1.33,"pnl_dollars":14.0,"exit_reason":"velocity_exit","opened_at":"2026-04-15T18:33:48"},
        {"id":12,"ticker":"SPY","option_type":"call","strike":700.0,"contracts":7,"premium_per_contract":0.43215,"exit_premium":0.2587,"mfe_premium":0.52,"pnl_dollars":-35.0,"exit_reason":"setup_failed","opened_at":"2026-04-15T19:24:48"},
        {"id":13,"ticker":"SPY","option_type":"call","strike":709.0,"contracts":1,"premium_per_contract":1.61805,"exit_premium":2.57705,"mfe_premium":2.59,"pnl_dollars":93.0,"exit_reason":"t1_hit","opened_at":"2026-04-17T14:25:12"},
        {"id":14,"ticker":"AMZN","option_type":"call","strike":255.0,"contracts":2,"premium_per_contract":1.07535,"exit_premium":0.78605,"mfe_premium":1.58,"pnl_dollars":-40.0,"exit_reason":"trailing_stop","opened_at":"2026-04-17T14:34:11"},
        {"id":15,"ticker":"IWM","option_type":"call","strike":277.0,"contracts":4,"premium_per_contract":0.8241,"exit_premium":0.4776,"mfe_premium":0.92,"pnl_dollars":-116.0,"exit_reason":"setup_failed","opened_at":"2026-04-17T14:43:00"},
        {"id":17,"ticker":"QQQ","option_type":"call","strike":649.0,"contracts":2,"premium_per_contract":1.53765,"exit_premium":1.4925,"mfe_premium":1.73,"pnl_dollars":-92.0,"exit_reason":"setup_failed","opened_at":"2026-04-17T14:46:02"},
        {"id":18,"ticker":"IWM","option_type":"call","strike":280.0,"contracts":5,"premium_per_contract":0.6231,"exit_premium":0.2388,"mfe_premium":0.6231,"pnl_dollars":-170.0,"exit_reason":"stop_hit","opened_at":"2026-04-21T14:31:08"},
        {"id":19,"ticker":"MSFT","option_type":"call","strike":427.5,"contracts":1,"premium_per_contract":3.2361,"exit_premium":2.3269,"mfe_premium":3.3586,"pnl_dollars":-90.92,"exit_reason":"no_momentum","opened_at":"2026-04-21T14:42:59"},
        {"id":20,"ticker":"NVDA","option_type":"call","strike":202.5,"contracts":2,"premium_per_contract":1.37685,"exit_premium":1.00181,"mfe_premium":1.37685,"pnl_dollars":-75.01,"exit_reason":"no_momentum","opened_at":"2026-04-21T14:46:01"},
        {"id":21,"ticker":"QQQ","option_type":"call","strike":647.0,"contracts":2,"premium_per_contract":1.51755,"exit_premium":1.78105,"mfe_premium":1.83,"pnl_dollars":70.0,"exit_reason":"dollar_trail","opened_at":"2026-04-21T15:10:01"},
        {"id":22,"ticker":"NVDA","option_type":"call","strike":202.5,"contracts":3,"premium_per_contract":1.005,"exit_premium":0.39053,"mfe_premium":1.005,"pnl_dollars":-184.34,"exit_reason":"stop_hit","opened_at":"2026-04-21T15:57:48"},
        {"id":23,"ticker":"AMZN","option_type":"call","strike":252.5,"contracts":1,"premium_per_contract":2.22105,"exit_premium":1.64777,"mfe_premium":2.39105,"pnl_dollars":-57.33,"exit_reason":"no_momentum","opened_at":"2026-04-21T16:06:21"},
        {"id":24,"ticker":"SPY","option_type":"put","strike":705.0,"contracts":1,"premium_per_contract":2.06025,"exit_premium":0.40795,"mfe_premium":2.06025,"pnl_dollars":-165.23,"exit_reason":"stop_hit","opened_at":"2026-04-22T13:27:41"},
        {"id":25,"ticker":"AVGO","option_type":"call","strike":412.5,"contracts":1,"premium_per_contract":2.86425,"exit_premium":3.184,"mfe_premium":3.2,"pnl_dollars":51.0,"exit_reason":"t1_hit","opened_at":"2026-04-22T15:00:42"},
        {"id":26,"ticker":"MSTR","option_type":"call","strike":180.0,"contracts":1,"premium_per_contract":5.67825,"exit_premium":5.93343,"mfe_premium":5.96325,"pnl_dollars":25.52,"exit_reason":"t1_hit","opened_at":"2026-04-22T15:03:41"},
        {"id":27,"ticker":"NVDA","option_type":"call","strike":202.5,"contracts":10,"premium_per_contract":0.29145,"exit_premium":0.1194,"mfe_premium":0.35,"pnl_dollars":-170.0,"exit_reason":"no_momentum","opened_at":"2026-04-22T15:18:39"},
        {"id":28,"ticker":"AMZN","option_type":"call","strike":252.5,"contracts":5,"premium_per_contract":0.6231,"exit_premium":0.92535,"mfe_premium":0.94,"pnl_dollars":185.0,"exit_reason":"dollar_trail","opened_at":"2026-04-22T15:51:40"},
        {"id":29,"ticker":"PLTR","option_type":"call","strike":150.0,"contracts":1,"premium_per_contract":3.3969,"exit_premium":3.53166,"mfe_premium":3.5494,"pnl_dollars":13.48,"exit_reason":"t1_hit","opened_at":"2026-04-22T16:15:42"},
        {"id":30,"ticker":"AVGO","option_type":"call","strike":420.0,"contracts":1,"premium_per_contract":1.99995,"exit_premium":0.72635,"mfe_premium":2.0,"pnl_dollars":-114.0,"exit_reason":"stop_hit","opened_at":"2026-04-22T16:24:40"},
    ]

    print("=" * 110)
    print("LIVE SIGNAL REPLAY: 35% Profit Retrace (below +40% gain) + Adaptive Trail (above +40%)")
    print("=" * 110)
    print()
    print("Rule: If peak gain is 10-40%, exit when 35% of profit is given back.")
    print("      If peak gain >= 40%, adaptive trail handles it (unchanged).")
    print()

    total_actual = 0
    total_retrace = 0
    changed_trades = []
    unchanged_trades = []

    for trade in trades:
        result = simulate_profit_retrace(trade)
        actual_pnl = trade["pnl_dollars"]
        total_actual += actual_pnl

        if result is None:
            total_retrace += actual_pnl
            continue

        if result["would_change"]:
            total_retrace += result["retrace_pnl"]
            changed_trades.append((trade, result))
        else:
            total_retrace += actual_pnl
            unchanged_trades.append((trade, result))

    # Print changed trades
    print(f"{'#':<4} {'Ticker':<6} {'Type':<5} {'Entry$':>8} {'MFE$':>8} {'Peak%':>7} "
          f"{'ActExit$':>9} {'RetExit$':>9} {'ActPnL$':>9} {'RetPnL$':>9} {'Diff$':>8} {'ActReason':<16}")
    print("-" * 110)

    for trade, result in changed_trades:
        entry = trade["premium_per_contract"]
        mfe = trade["mfe_premium"]
        peak_pct = (mfe - entry) / entry * 100

        print(f"#{trade['id']:<3} {trade['ticker']:<6} {trade['option_type']:<5} "
              f"${entry:>7.3f} ${mfe:>7.3f} +{peak_pct:>5.1f}% "
              f"${result['actual_exit']:>8.3f} ${result['retrace_exit']:>8.3f} "
              f"${result['actual_pnl']:>+8.0f} ${result['retrace_pnl']:>+8.0f} "
              f"${result['improvement']:>+7.0f} {trade['exit_reason']:<16}")

    print()
    print(f"--- Trades where retrace would NOT change outcome ---")
    for trade, result in unchanged_trades:
        entry = trade["premium_per_contract"]
        mfe = trade["mfe_premium"] or entry
        peak_pct = (mfe - entry) / entry * 100
        applies_str = "In range" if result["applies"] else "Out of range"
        print(f"  #{trade['id']:<3} {trade['ticker']:<6} Peak +{peak_pct:>5.1f}%  "
              f"PnL ${trade['pnl_dollars']:>+8.0f}  {trade['exit_reason']:<16}  "
              f"({result['reason'][:70]})")

    # Summary
    improvement = total_retrace - total_actual
    print()
    print("=" * 110)
    print(f"SUMMARY")
    print(f"  Total trades: {len(trades)}")
    print(f"  Trades changed by profit retrace: {len(changed_trades)}")
    print(f"  Trades unchanged: {len(unchanged_trades) + len(trades) - len(changed_trades) - len(unchanged_trades)}")
    print()
    print(f"  Actual total PnL:           ${total_actual:>+,.2f}")
    print(f"  With profit retrace PnL:    ${total_retrace:>+,.2f}")
    print(f"  Improvement:                ${improvement:>+,.2f}")
    print()

    if changed_trades:
        improvements = [r["improvement"] for _, r in changed_trades]
        print(f"  Avg improvement per changed trade: ${sum(improvements)/len(improvements):>+,.2f}")
        print(f"  Best improvement:  ${max(improvements):>+,.2f}")
        print(f"  Worst improvement: ${min(improvements):>+,.2f}")

    # Win rate comparison
    actual_wins = sum(1 for t in trades if t["pnl_dollars"] >= 0)
    retrace_wins = 0
    for trade in trades:
        result = simulate_profit_retrace(trade)
        if result and result["would_change"]:
            if result["retrace_pnl"] >= 0:
                retrace_wins += 1
        else:
            if trade["pnl_dollars"] >= 0:
                retrace_wins += 1

    print()
    print(f"  Actual win rate:  {actual_wins}/{len(trades)} = {actual_wins/len(trades)*100:.1f}%")
    print(f"  Retrace win rate: {retrace_wins}/{len(trades)} = {retrace_wins/len(trades)*100:.1f}%")


if __name__ == "__main__":
    main()
