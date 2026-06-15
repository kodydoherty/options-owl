#!/usr/bin/env python3
"""
Combinatorial filter optimization on historical trade data.

Inspired by the "v10 surgical filter" approach: find rule combinations that
remove the most losers with zero (or near-zero) winner casualties.

Usage:
    python scripts/optimize_filters.py                           # default: kody's DB
    python scripts/optimize_filters.py --db journal/owlet-adam/raw_messages.db
    python scripts/optimize_filters.py --webull-only             # only real trades
    python scripts/optimize_filters.py --max-combo 3             # limit combo size
    python scripts/optimize_filters.py --min-trades 20           # min trades after filter
    python scripts/optimize_filters.py --all-agents              # merge all agent DBs
"""

import argparse
import glob
import itertools
import json
import os
import random
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# ── Market open in ET (9:30 AM) ──────────────────────────────────────────────
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30


def minutes_since_open(hour: int, minute: int) -> int:
    """Minutes elapsed since 9:30 AM ET."""
    return (hour - MARKET_OPEN_HOUR) * 60 + (minute - MARKET_OPEN_MINUTE)


# ── Candidate filter rules ───────────────────────────────────────────────────
# Each rule: (name, function(trade_dict) -> bool).  True = keep, False = remove.

CANDIDATE_RULES: list[tuple[str, Any]] = [
    # ── Time-based ──
    ("skip_before_935", lambda t: t["minutes_since_open"] >= 5),
    ("skip_before_940", lambda t: t["minutes_since_open"] >= 10),
    ("skip_before_945", lambda t: t["minutes_since_open"] >= 15),
    ("skip_before_950", lambda t: t["minutes_since_open"] >= 20),
    ("skip_before_1000", lambda t: t["minutes_since_open"] >= 30),
    ("skip_after_1500", lambda t: t["minutes_since_open"] <= 330),
    ("skip_after_1530", lambda t: t["minutes_since_open"] <= 360),
    ("skip_midday_1200_1330", lambda t: not (150 <= t["minutes_since_open"] <= 240)),
    ("skip_early_afternoon_1300_1430", lambda t: not (210 <= t["minutes_since_open"] <= 300)),
    ("skip_last_hour", lambda t: t["minutes_since_open"] <= 330),

    # ── Score-based ──
    ("min_score_80", lambda t: t["score"] >= 80),
    ("min_score_85", lambda t: t["score"] >= 85),
    ("min_score_90", lambda t: t["score"] >= 90),
    ("min_score_95", lambda t: t["score"] >= 95),
    ("min_score_100", lambda t: t["score"] >= 100),
    ("max_score_140", lambda t: t["score"] <= 140),
    ("max_score_120", lambda t: t["score"] <= 120),

    # ── Premium-based ──
    ("max_premium_8", lambda t: t["entry_premium"] <= 8.0),
    ("max_premium_5", lambda t: t["entry_premium"] <= 5.0),
    ("max_premium_3", lambda t: t["entry_premium"] <= 3.0),
    ("max_premium_2", lambda t: t["entry_premium"] <= 2.0),
    ("min_premium_0.30", lambda t: t["entry_premium"] >= 0.30),
    ("min_premium_0.50", lambda t: t["entry_premium"] >= 0.50),
    ("min_premium_1.00", lambda t: t["entry_premium"] >= 1.00),

    # ── Ticker-based ──
    ("skip_index_puts", lambda t: not (t["ticker"] in ("SPY", "QQQ", "IWM") and t["direction"] == "put")),
    ("skip_index_calls", lambda t: not (t["ticker"] in ("SPY", "QQQ", "IWM") and t["direction"] == "call")),
    ("skip_mstr", lambda t: t["ticker"] != "MSTR"),
    ("skip_pltr", lambda t: t["ticker"] != "PLTR"),
    ("skip_amd", lambda t: t["ticker"] != "AMD"),
    ("only_high_vol", lambda t: t["ticker"] in ("MSTR", "AMD", "TSLA", "NVDA", "AVGO", "META", "COIN", "SMCI", "PLTR")),
    ("only_mega_cap", lambda t: t["ticker"] in ("AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AVGO")),

    # ── Day-of-week ──
    ("skip_monday", lambda t: t["day_of_week"] != 0),
    ("skip_tuesday", lambda t: t["day_of_week"] != 1),
    ("skip_wednesday", lambda t: t["day_of_week"] != 2),
    ("skip_thursday", lambda t: t["day_of_week"] != 3),
    ("skip_friday", lambda t: t["day_of_week"] != 4),

    # ── Direction-based ──
    ("only_calls", lambda t: t["direction"] == "call"),
    ("only_puts", lambda t: t["direction"] == "put"),

    # ── Contract count / sizing ──
    ("min_2_contracts", lambda t: t["contracts"] >= 2),
    ("max_20_contracts", lambda t: t["contracts"] <= 20),
    ("max_10_contracts", lambda t: t["contracts"] <= 10),

    # ── Duration-based (if available) ──
    ("min_duration_5m", lambda t: (t.get("duration_minutes") or 999) >= 5),
    ("max_duration_120m", lambda t: (t.get("duration_minutes") or 0) <= 120),

    # ── Premium relative to strike (cheap vs expensive) ──
    ("premium_under_1pct_strike", lambda t: t["entry_premium"] / max(t["strike"], 1) * 100 < 1.0),
    ("premium_over_0.1pct_strike", lambda t: t["entry_premium"] / max(t["strike"], 1) * 100 > 0.1),
]


def load_trades(db_path: str, webull_only: bool = False) -> list[dict]:
    """Load closed trades from SQLite and compute derived features."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = """
        SELECT id, ticker, direction, score, premium_per_contract,
               opened_at, closed_at, status, pnl_dollars, pnl_pct,
               exit_reason, exit_source, webull_order_id, contracts,
               total_cost, strike, parent_trade_id, duration_minutes,
               entry_price
        FROM paper_trades
        WHERE status = 'closed'
    """
    if webull_only:
        query += " AND webull_order_id IS NOT NULL"

    # Exclude scaleout children — their P&L is part of the parent
    query += " AND parent_trade_id IS NULL"
    query += " ORDER BY id"

    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()

    trades = []
    for row in rows:
        opened_at = row["opened_at"] or ""
        # Parse datetime — format: 2026-04-13T14:36:12.502813
        try:
            dt = datetime.fromisoformat(opened_at)
        except (ValueError, TypeError):
            continue

        entry_premium = row["premium_per_contract"] or 0.0
        total_cost = row["total_cost"] or (entry_premium * (row["contracts"] or 1) * 100)
        pnl_dollars = row["pnl_dollars"] or 0.0
        pnl_pct = row["pnl_pct"] if row["pnl_pct"] is not None else (
            (pnl_dollars / total_cost * 100) if total_cost > 0 else 0.0
        )

        hour = dt.hour
        minute = dt.minute
        mso = minutes_since_open(hour, minute)

        trade = {
            "id": row["id"],
            "ticker": row["ticker"],
            "direction": row["direction"],
            "score": row["score"] or 0,
            "entry_premium": entry_premium,
            "strike": row["strike"] or 0,
            "contracts": row["contracts"] or 1,
            "total_cost": total_cost,
            "pnl_dollars": pnl_dollars,
            "pnl_pct": pnl_pct,
            "exit_reason": row["exit_reason"] or "",
            "exit_source": row["exit_source"] or "ai",
            "webull_order_id": row["webull_order_id"],
            "is_winner": pnl_dollars > 0,
            "hour": hour,
            "minute": minute,
            "minutes_since_open": mso,
            "day_of_week": dt.weekday(),
            "date": dt.strftime("%Y-%m-%d"),
            "duration_minutes": row["duration_minutes"],
            "parent_trade_id": row["parent_trade_id"],
        }
        trades.append(trade)

    return trades


def load_trades_multi(db_paths: list[str], webull_only: bool = False) -> list[dict]:
    """Load and merge trades from multiple agent DBs."""
    all_trades = []
    for db_path in db_paths:
        if os.path.exists(db_path):
            trades = load_trades(db_path, webull_only)
            # Tag with agent name
            agent = Path(db_path).parent.name
            for t in trades:
                t["agent"] = agent
            all_trades.extend(trades)
    return all_trades


def compute_stats(trades: list[dict]) -> dict:
    """Compute aggregate stats for a set of trades."""
    if not trades:
        return {
            "count": 0, "winners": 0, "losers": 0, "win_rate": 0.0,
            "avg_pnl": 0.0, "total_pnl": 0.0, "avg_winner": 0.0,
            "avg_loser": 0.0,
        }
    winners = [t for t in trades if t["is_winner"]]
    losers = [t for t in trades if not t["is_winner"]]
    total_pnl = sum(t["pnl_dollars"] for t in trades)
    return {
        "count": len(trades),
        "winners": len(winners),
        "losers": len(losers),
        "win_rate": len(winners) / len(trades) * 100 if trades else 0,
        "avg_pnl": total_pnl / len(trades) if trades else 0,
        "total_pnl": total_pnl,
        "avg_winner": (sum(t["pnl_dollars"] for t in winners) / len(winners)) if winners else 0,
        "avg_loser": (sum(t["pnl_dollars"] for t in losers) / len(losers)) if losers else 0,
    }


def apply_rules(trades: list[dict], rules: list[tuple[str, Any]]) -> list[dict]:
    """Apply a set of rules — keep trades that pass ALL rules."""
    result = trades
    for _name, fn in rules:
        result = [t for t in result if fn(t)]
    return result


def evaluate_combination(
    trades: list[dict],
    baseline_winners: int,
    baseline_losers: int,
    rules: list[tuple[str, Any]],
    min_trades: int,
) -> dict | None:
    """Evaluate a rule combination. Returns stats dict or None if below min_trades."""
    filtered = apply_rules(trades, rules)
    if len(filtered) < min_trades:
        return None

    stats = compute_stats(filtered)
    rule_names = [name for name, _ in rules]

    removed_trades = [t for t in trades if t not in set_from_list(filtered)]
    winners_removed = sum(1 for t in removed_trades if t["is_winner"])
    losers_removed = sum(1 for t in removed_trades if not t["is_winner"])

    loser_removal_rate = losers_removed / baseline_losers * 100 if baseline_losers > 0 else 0
    winner_casualty_rate = winners_removed / baseline_winners * 100 if baseline_winners > 0 else 0

    return {
        "rules": rule_names,
        "num_rules": len(rules),
        "trades_remaining": stats["count"],
        "winners": stats["winners"],
        "losers": stats["losers"],
        "win_rate": stats["win_rate"],
        "avg_pnl": stats["avg_pnl"],
        "total_pnl": stats["total_pnl"],
        "losers_removed": losers_removed,
        "loser_removal_rate": loser_removal_rate,
        "winners_removed": winners_removed,
        "winner_casualty_rate": winner_casualty_rate,
        "avg_winner": stats["avg_winner"],
        "avg_loser": stats["avg_loser"],
    }


def set_from_list(lst: list[dict]) -> set:
    """Create a set of trade IDs for fast membership testing."""
    return {t["id"] for t in lst}


def apply_rules_ids(trades: list[dict], rules: list[tuple[str, Any]]) -> set:
    """Apply rules and return set of surviving trade IDs."""
    result = trades
    for _name, fn in rules:
        result = [t for t in result if fn(t)]
    return {t["id"] for t in result}


def evaluate_combination_fast(
    trades: list[dict],
    baseline_winners: int,
    baseline_losers: int,
    rules: list[tuple[str, Any]],
    min_trades: int,
    winner_ids: set,
    loser_ids: set,
) -> dict | None:
    """Fast evaluation using set operations."""
    surviving_ids = apply_rules_ids(trades, rules)
    count = len(surviving_ids)
    if count < min_trades:
        return None

    surviving_winners = len(surviving_ids & winner_ids)
    surviving_losers = len(surviving_ids & loser_ids)

    winners_removed = baseline_winners - surviving_winners
    losers_removed = baseline_losers - surviving_losers

    loser_removal_rate = losers_removed / baseline_losers * 100 if baseline_losers > 0 else 0
    winner_casualty_rate = winners_removed / baseline_winners * 100 if baseline_winners > 0 else 0

    # Compute P&L stats for surviving trades
    surviving_trades = [t for t in trades if t["id"] in surviving_ids]
    total_pnl = sum(t["pnl_dollars"] for t in surviving_trades)
    winner_trades = [t for t in surviving_trades if t["is_winner"]]
    loser_trades = [t for t in surviving_trades if not t["is_winner"]]

    return {
        "rules": [name for name, _ in rules],
        "num_rules": len(rules),
        "trades_remaining": count,
        "winners": surviving_winners,
        "losers": surviving_losers,
        "win_rate": surviving_winners / count * 100 if count > 0 else 0,
        "avg_pnl": total_pnl / count if count > 0 else 0,
        "total_pnl": total_pnl,
        "losers_removed": losers_removed,
        "loser_removal_rate": loser_removal_rate,
        "winners_removed": winners_removed,
        "winner_casualty_rate": winner_casualty_rate,
        "avg_winner": (sum(t["pnl_dollars"] for t in winner_trades) / len(winner_trades)) if winner_trades else 0,
        "avg_loser": (sum(t["pnl_dollars"] for t in loser_trades) / len(loser_trades)) if loser_trades else 0,
    }


def rank_results(results: list[dict]) -> list[dict]:
    """
    Rank results:
      1. Zero winner casualties first
      2. Then by loser removal rate (descending)
      3. Then by win rate (descending)
      4. Then by total P&L (descending)
    """
    return sorted(results, key=lambda r: (
        r["winners_removed"] == 0,   # True sorts after False, we want True first
        r["loser_removal_rate"],
        r["win_rate"],
        r["total_pnl"],
    ), reverse=True)


def print_report(
    baseline: dict,
    ranked: list[dict],
    top_n: int = 20,
) -> None:
    """Print the optimization report."""
    sep = "=" * 100
    print(f"\n{sep}")
    print("  COMBINATORIAL FILTER OPTIMIZATION REPORT")
    print(sep)

    print(f"\n  BASELINE (no filters):")
    print(f"    Trades:     {baseline['count']}")
    print(f"    Winners:    {baseline['winners']}  |  Losers: {baseline['losers']}")
    print(f"    Win Rate:   {baseline['win_rate']:.1f}%")
    print(f"    Avg P&L:    ${baseline['avg_pnl']:.2f}")
    print(f"    Total P&L:  ${baseline['total_pnl']:.2f}")
    print(f"    Avg Winner: ${baseline['avg_winner']:.2f}  |  Avg Loser: ${baseline['avg_loser']:.2f}")

    zero_casualty = [r for r in ranked if r["winners_removed"] == 0]
    near_zero = [r for r in ranked if 0 < r["winners_removed"] <= 2]

    print(f"\n  SEARCH RESULTS:")
    print(f"    Total combinations evaluated: {len(ranked) + 0}")
    print(f"    Zero-casualty combos:         {len(zero_casualty)}")
    print(f"    Near-zero casualty (<=2):     {len(near_zero)}")

    print(f"\n{sep}")
    print(f"  TOP {min(top_n, len(ranked))} FILTER COMBINATIONS")
    print(sep)

    for i, r in enumerate(ranked[:top_n], 1):
        casualty_tag = "ZERO CASUALTIES" if r["winners_removed"] == 0 else f"{r['winners_removed']} winner(s) lost"
        pnl_delta = r["total_pnl"] - baseline["total_pnl"]
        pnl_arrow = "+" if pnl_delta >= 0 else ""

        print(f"\n  #{i}  [{casualty_tag}]")
        print(f"    Rules ({r['num_rules']}): {', '.join(r['rules'])}")
        print(f"    Trades: {r['trades_remaining']}/{baseline['count']}  "
              f"| WR: {r['win_rate']:.1f}% (was {baseline['win_rate']:.1f}%)  "
              f"| Total P&L: ${r['total_pnl']:.2f} ({pnl_arrow}${pnl_delta:.2f})")
        print(f"    Winners: {r['winners']}/{baseline['winners']}  "
              f"| Losers: {r['losers']}/{baseline['losers']}  "
              f"| Losers removed: {r['losers_removed']} ({r['loser_removal_rate']:.1f}%)")
        print(f"    Avg P&L: ${r['avg_pnl']:.2f}  "
              f"| Avg winner: ${r['avg_winner']:.2f}  "
              f"| Avg loser: ${r['avg_loser']:.2f}")

    # Final recommendation
    if ranked:
        best = ranked[0]
        print(f"\n{sep}")
        print("  RECOMMENDED FILTER SET")
        print(sep)
        print(f"    Rules: {', '.join(best['rules'])}")
        print(f"    Effect: removes {best['losers_removed']}/{baseline['losers']} losers "
              f"({best['loser_removal_rate']:.1f}%) with {best['winners_removed']} winner casualties")
        print(f"    Result: {best['win_rate']:.1f}% WR, ${best['total_pnl']:.2f} total P&L "
              f"({best['trades_remaining']} trades)")
        print()


def run_optimization(
    trades: list[dict],
    max_combo: int = 5,
    min_trades: int = 20,
    max_random_samples: int = 50000,
    rules: list[tuple[str, Any]] | None = None,
) -> list[dict]:
    """
    Run the combinatorial search over all rule combinations.
    Returns ranked list of results.
    """
    if rules is None:
        rules = CANDIDATE_RULES

    baseline = compute_stats(trades)
    baseline_winners = baseline["winners"]
    baseline_losers = baseline["losers"]

    # Pre-compute winner/loser ID sets
    winner_ids = {t["id"] for t in trades if t["is_winner"]}
    loser_ids = {t["id"] for t in trades if not t["is_winner"]}

    # First, prune rules that have no effect (remove 0 trades)
    active_rules = []
    for name, fn in rules:
        removed = sum(1 for t in trades if not fn(t))
        if removed > 0:
            active_rules.append((name, fn))
        else:
            pass  # Skip no-op rules silently

    print(f"  Active rules (remove >= 1 trade): {len(active_rules)}/{len(rules)}")

    results = []
    total_combos = 0
    evaluated = 0
    start = time.time()

    for combo_size in range(1, max_combo + 1):
        n_combos = len(list(itertools.combinations(range(len(active_rules)), combo_size)))
        total_combos += n_combos

        if n_combos > max_random_samples:
            # Random sampling for large combo spaces
            print(f"  Combo size {combo_size}: {n_combos:,} combos -> sampling {max_random_samples:,} random")
            indices = list(range(len(active_rules)))
            sampled = set()
            attempts = 0
            while len(sampled) < max_random_samples and attempts < max_random_samples * 3:
                combo = tuple(sorted(random.sample(indices, combo_size)))
                sampled.add(combo)
                attempts += 1

            for combo_idx in sampled:
                combo_rules = [active_rules[i] for i in combo_idx]
                result = evaluate_combination_fast(
                    trades, baseline_winners, baseline_losers,
                    combo_rules, min_trades, winner_ids, loser_ids,
                )
                if result is not None:
                    results.append(result)
                evaluated += 1

                if evaluated % 10000 == 0:
                    elapsed = time.time() - start
                    print(f"    ... {evaluated:,} evaluated, {len(results):,} valid, {elapsed:.1f}s elapsed")
        else:
            print(f"  Combo size {combo_size}: {n_combos:,} combos (exhaustive)")
            for combo in itertools.combinations(range(len(active_rules)), combo_size):
                combo_rules = [active_rules[i] for i in combo]
                result = evaluate_combination_fast(
                    trades, baseline_winners, baseline_losers,
                    combo_rules, min_trades, winner_ids, loser_ids,
                )
                if result is not None:
                    results.append(result)
                evaluated += 1

                if evaluated % 10000 == 0:
                    elapsed = time.time() - start
                    print(f"    ... {evaluated:,} evaluated, {len(results):,} valid, {elapsed:.1f}s elapsed")

    elapsed = time.time() - start
    print(f"\n  Search complete: {evaluated:,} combos evaluated, "
          f"{len(results):,} valid results, {elapsed:.1f}s")

    ranked = rank_results(results)
    return ranked


def main():
    parser = argparse.ArgumentParser(
        description="Combinatorial filter optimization on historical trade data"
    )
    parser.add_argument(
        "--db", type=str, default=None,
        help="Path to raw_messages.db (default: journal/owlet-kody/raw_messages.db)"
    )
    parser.add_argument(
        "--all-agents", action="store_true",
        help="Merge trades from all agent DBs in journal/"
    )
    parser.add_argument(
        "--webull-only", action="store_true",
        help="Only include trades with a Webull order ID"
    )
    parser.add_argument(
        "--max-combo", type=int, default=5,
        help="Maximum number of rules in a combination (default: 5)"
    )
    parser.add_argument(
        "--min-trades", type=int, default=20,
        help="Minimum trades remaining after filter (default: 20)"
    )
    parser.add_argument(
        "--max-samples", type=int, default=50000,
        help="Max random samples per combo size when exhaustive is too large (default: 50000)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()
    random.seed(args.seed)

    # Resolve DB paths
    base_dir = Path(__file__).resolve().parent.parent
    if args.all_agents:
        db_paths = sorted(glob.glob(str(base_dir / "journal" / "owlet-*" / "raw_messages.db")))
        if not db_paths:
            print("ERROR: No agent DBs found in journal/owlet-*/")
            return
        print(f"Loading trades from {len(db_paths)} agent DBs...")
        trades = load_trades_multi(db_paths, webull_only=args.webull_only)
    else:
        db_path = args.db or str(base_dir / "journal" / "owlet-kody" / "raw_messages.db")
        if not os.path.exists(db_path):
            print(f"ERROR: Database not found: {db_path}")
            return
        print(f"Loading trades from {db_path}...")
        trades = load_trades(db_path, webull_only=args.webull_only)

    if not trades:
        print("ERROR: No closed trades found.")
        return

    baseline = compute_stats(trades)
    print(f"Loaded {baseline['count']} trades "
          f"({baseline['winners']}W / {baseline['losers']}L, "
          f"{baseline['win_rate']:.1f}% WR, "
          f"${baseline['total_pnl']:.2f} total P&L)")
    if args.webull_only:
        print("  (Webull-only filter active)")
    print()

    # Run optimization
    ranked = run_optimization(
        trades,
        max_combo=args.max_combo,
        min_trades=args.min_trades,
        max_random_samples=args.max_samples,
    )

    # Print report
    print_report(baseline, ranked, top_n=20)


if __name__ == "__main__":
    main()
