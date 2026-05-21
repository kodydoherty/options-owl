"""Backtest scoring mitigations against real Webull trade data.

Tests each adjustment INDEPENDENTLY against the baseline:
  A) Baseline: current production sizing (score tiers as-is)
  B) Flat multipliers: all scores 78-134 get same 75% mult
  C) Put penalty: reduce put sizing by 50%
  D) Put score floor: require 90+ for puts (block puts 78-89)
  E) Score 100 demotion: treat score=100 as score=90 (suspected default)
  F) Combined best: flat mults + put penalty

Uses REAL Webull trade outcomes — rescales P&L by contract ratio since
FSM exit gates are independent of position size (except scaleout).

Usage:
    python scripts/backtest_sizing_mitigations.py
"""

from __future__ import annotations

import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

BOT_DBS = {
    "kody": PROJECT_DIR / "journal" / "owlet-kody" / "raw_messages.db",
}

PORTFOLIO = 23000

# ---------------------------------------------------------------------------
# Current production sizing (must match vinny_strategy.py exactly)
# ---------------------------------------------------------------------------

PROD_SCORE_TIERS = [
    # (min_score, budget_mult, position_cap_pct)
    (135, 1.00, 0.15),
    (120, 0.85, 0.12),
    (100, 0.85, 0.08),
    (90,  0.50, 0.08),
    (78,  0.25, 0.08),
]

PROD_MAX_RISK_PCT = 0.75
PROD_MAX_CONCURRENT = 4
PROD_MAX_POSITION_PCT = 0.10  # current production: min(tier, 10%)


@dataclass
class Trade:
    trade_id: int
    bot: str
    ticker: str
    option_type: str
    score: int
    contracts: int
    entry_premium: float
    pnl_dollars: float
    pnl_pct: float
    exit_reason: str
    opened_at: str


def load_trades() -> list[Trade]:
    trades = []
    for bot, db_path in BOT_DBS.items():
        if not db_path.exists():
            continue
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT id, ticker, option_type, score, contracts,
                   webull_entry_fill_price, premium_per_contract,
                   pnl_dollars, pnl_pct, exit_reason, opened_at
            FROM paper_trades
            WHERE status='closed' AND webull_order_id IS NOT NULL
            ORDER BY id
        """).fetchall()
        conn.close()
        for r in rows:
            entry = r["webull_entry_fill_price"] or r["premium_per_contract"] or 1.0
            trades.append(Trade(
                trade_id=r["id"], bot=bot, ticker=r["ticker"],
                option_type=r["option_type"] or "call",
                score=r["score"] or 100,
                contracts=r["contracts"] or 1,
                entry_premium=entry,
                pnl_dollars=r["pnl_dollars"] or 0,
                pnl_pct=r["pnl_pct"] or 0,
                exit_reason=r["exit_reason"] or "unknown",
                opened_at=r["opened_at"] or "",
            ))
    return trades


def compute_contracts(
    score: int,
    cost_per_contract: float,
    score_tiers: list[tuple[int, float, float]],
    max_risk_pct: float = PROD_MAX_RISK_PCT,
    max_concurrent: int = PROD_MAX_CONCURRENT,
    max_position_pct: float = PROD_MAX_POSITION_PCT,
    portfolio: float = PORTFOLIO,
) -> int:
    """Compute contract count using given tier table."""
    if score < 78 or cost_per_contract <= 0:
        return 0

    deployable = portfolio * max_risk_pct
    per_slot = deployable / max_concurrent

    score_mult = 0.25
    tier_pos_pct = 0.08
    for threshold, mult, pos_pct in score_tiers:
        if score >= threshold:
            score_mult = mult
            tier_pos_pct = pos_pct
            break

    effective_pos_pct = min(tier_pos_pct, max_position_pct)
    position_cap = portfolio * effective_pos_pct

    scaled_target = per_slot * score_mult
    raw_contracts = int(scaled_target / cost_per_contract) if cost_per_contract > 0 else 1
    pos_cap_contracts = int(position_cap / cost_per_contract) if cost_per_contract > 0 else 1
    if pos_cap_contracts == 0:
        return 0
    return max(1, min(raw_contracts, pos_cap_contracts))


def simulate_config(
    trades: list[Trade],
    label: str,
    score_tiers: list[tuple[int, float, float]],
    put_mult: float = 1.0,
    put_min_score: int = 78,
    score_override: dict[int, int] | None = None,
) -> dict:
    """Run a sizing config across all trades, rescaling P&L by contract ratio."""
    total_pnl = 0.0
    entered = 0
    skipped = 0
    wins = 0
    skip_bad = 0.0
    skip_good = 0.0
    details = []

    for t in trades:
        effective_score = t.score
        if score_override and t.score in score_override:
            effective_score = score_override[t.score]

        # Check put floor
        if t.option_type == "put" and effective_score < put_min_score:
            skipped += 1
            if t.pnl_dollars < 0:
                skip_bad += t.pnl_dollars
            else:
                skip_good += t.pnl_dollars
            details.append((t, 0, 0, "put_below_floor"))
            continue

        cost_per = t.entry_premium * 100
        new_contracts = compute_contracts(effective_score, cost_per, score_tiers)

        # Apply put multiplier
        if t.option_type == "put" and put_mult != 1.0:
            new_contracts = max(1, int(new_contracts * put_mult))

        if new_contracts == 0:
            skipped += 1
            if t.pnl_dollars < 0:
                skip_bad += t.pnl_dollars
            else:
                skip_good += t.pnl_dollars
            details.append((t, 0, 0, "zero_contracts"))
            continue

        # Rescale P&L: pnl_per_contract × new_contracts
        if t.contracts > 0:
            pnl_per_contract = t.pnl_dollars / t.contracts
        else:
            pnl_per_contract = t.pnl_dollars

        adjusted_pnl = pnl_per_contract * new_contracts
        total_pnl += adjusted_pnl
        entered += 1
        if adjusted_pnl > 0:
            wins += 1
        details.append((t, new_contracts, adjusted_pnl, "entered"))

    wr = wins / entered * 100 if entered > 0 else 0
    return {
        "label": label,
        "entered": entered,
        "skipped": skipped,
        "pnl": total_pnl,
        "wr": wr,
        "wins": wins,
        "skip_bad": skip_bad,
        "skip_good": skip_good,
        "details": details,
    }


def print_result(r, baseline_pnl):
    delta = r["pnl"] - baseline_pnl
    net_skip = abs(r["skip_bad"]) - r["skip_good"]
    print(f"  {r['label']:<50} {r['entered']:>4} {r['skipped']:>4} "
          f"${r['pnl']:>+10,.2f} ${delta:>+10,.2f} {r['wr']:>5.1f}%")


def main():
    trades = load_trades()
    print(f"Loaded {len(trades)} closed Webull trades\n")

    calls = [t for t in trades if t.option_type == "call"]
    puts = [t for t in trades if t.option_type == "put"]
    print(f"  Calls: {len(calls)}  Puts: {len(puts)}")
    print(f"  Score=100 trades: {sum(1 for t in trades if t.score == 100)}")
    print()

    # ===================================================================
    # A) Baseline: current production
    # ===================================================================
    baseline = simulate_config(trades, "A) Baseline (current production)", PROD_SCORE_TIERS)
    baseline_pnl = baseline["pnl"]

    print(f"{'=' * 110}")
    print(f"SIZING MITIGATION BACKTEST — {len(trades)} trades, Portfolio ${PORTFOLIO:,}")
    print(f"{'=' * 110}\n")

    print(f"  {'Config':<50} {'In':>4} {'Out':>4} {'P&L':>12} {'Delta':>12} {'WR':>6}")
    print(f"  {'-' * 100}")

    print_result(baseline, baseline_pnl)

    # ===================================================================
    # B) Flat multipliers: 78-134 all get 75%, 135+ gets 100%
    # ===================================================================
    flat_tiers = [
        (135, 1.00, 0.15),
        (78,  0.75, 0.08),
    ]
    r = simulate_config(trades, "B) Flat 75% for 78-134, 100% for 135+", flat_tiers)
    print_result(r, baseline_pnl)

    # B2) Even flatter: everyone gets 75%
    flat_all = [
        (78, 0.75, 0.08),
    ]
    r = simulate_config(trades, "B2) Flat 75% for ALL scores", flat_all)
    print_result(r, baseline_pnl)

    # B3) Flat 60%
    flat_60 = [
        (135, 1.00, 0.15),
        (78,  0.60, 0.08),
    ]
    r = simulate_config(trades, "B3) Flat 60% for 78-134, 100% for 135+", flat_60)
    print_result(r, baseline_pnl)

    # ===================================================================
    # C) Put penalty: reduce put sizing by 50%
    # ===================================================================
    r = simulate_config(trades, "C) Put penalty 50% (half contracts)", PROD_SCORE_TIERS, put_mult=0.5)
    print_result(r, baseline_pnl)

    r = simulate_config(trades, "C2) Put penalty 75% (quarter fewer)", PROD_SCORE_TIERS, put_mult=0.75)
    print_result(r, baseline_pnl)

    r = simulate_config(trades, "C3) Put penalty 33% (third contracts)", PROD_SCORE_TIERS, put_mult=0.33)
    print_result(r, baseline_pnl)

    # ===================================================================
    # D) Put score floor: require 90+ for puts
    # ===================================================================
    r = simulate_config(trades, "D) Put floor 90 (block puts 78-89)", PROD_SCORE_TIERS, put_min_score=90)
    print_result(r, baseline_pnl)

    r = simulate_config(trades, "D2) Put floor 100 (block puts 78-99)", PROD_SCORE_TIERS, put_min_score=100)
    print_result(r, baseline_pnl)

    r = simulate_config(trades, "D3) Put floor 110 (block puts 78-109)", PROD_SCORE_TIERS, put_min_score=110)
    print_result(r, baseline_pnl)

    # ===================================================================
    # E) Score 100 demotion: treat 100 as 90 (suspected default)
    # ===================================================================
    r = simulate_config(trades, "E) Score 100 → treat as 90", PROD_SCORE_TIERS, score_override={100: 90})
    print_result(r, baseline_pnl)

    r = simulate_config(trades, "E2) Score 100 → treat as 85", PROD_SCORE_TIERS, score_override={100: 85})
    print_result(r, baseline_pnl)

    # ===================================================================
    # F) Combined: best of each
    # ===================================================================
    print(f"\n  {'--- COMBINED CONFIGS ---':<50}")

    r = simulate_config(trades, "F1) Flat 75% + put penalty 50%", flat_tiers, put_mult=0.5)
    print_result(r, baseline_pnl)

    r = simulate_config(trades, "F2) Flat 75% + put floor 90", flat_tiers, put_min_score=90)
    print_result(r, baseline_pnl)

    r = simulate_config(trades, "F3) Flat 75% + put floor 100", flat_tiers, put_min_score=100)
    print_result(r, baseline_pnl)

    r = simulate_config(trades, "F4) Flat 75% + score100→90 + put 50%", flat_tiers,
                        put_mult=0.5, score_override={100: 90})
    print_result(r, baseline_pnl)

    r = simulate_config(trades, "F5) Flat 60% + put floor 90", flat_60, put_min_score=90)
    print_result(r, baseline_pnl)

    r = simulate_config(trades, "F6) Flat 60% + put 50% + score100→90", flat_60,
                        put_mult=0.5, score_override={100: 90})
    print_result(r, baseline_pnl)

    # ===================================================================
    # Detailed breakdown of best config
    # ===================================================================

    # Find best
    all_configs = [
        ("A) Baseline", simulate_config(trades, "A", PROD_SCORE_TIERS)),
        ("B) Flat 75%", simulate_config(trades, "B", flat_tiers)),
        ("C) Put 50%", simulate_config(trades, "C", PROD_SCORE_TIERS, put_mult=0.5)),
        ("D) Put floor 90", simulate_config(trades, "D", PROD_SCORE_TIERS, put_min_score=90)),
        ("E) Score100→90", simulate_config(trades, "E", PROD_SCORE_TIERS, score_override={100: 90})),
        ("F1) Flat+Put50%", simulate_config(trades, "F1", flat_tiers, put_mult=0.5)),
        ("F2) Flat+PutFloor90", simulate_config(trades, "F2", flat_tiers, put_min_score=90)),
        ("F4) Flat+100→90+Put50%", simulate_config(trades, "F4", flat_tiers, put_mult=0.5, score_override={100: 90})),
    ]

    best_label, best_r = max(all_configs, key=lambda x: x[1]["pnl"])
    print(f"\n{'=' * 110}")
    print(f"BEST CONFIG: {best_label}")
    print(f"  P&L: ${best_r['pnl']:,.2f} (delta: ${best_r['pnl'] - baseline_pnl:+,.2f})")
    print(f"  Win Rate: {best_r['wr']:.1f}% ({best_r['wins']}/{best_r['entered']})")
    print(f"{'=' * 110}\n")

    # Per-ticker impact of best config
    print(f"Per-ticker impact (best config vs baseline):\n")
    print(f"  {'Ticker':<8} {'Type':<5} {'BasePnL':>10} {'NewPnL':>10} {'Delta':>10} {'BaseCt':>6} {'NewCt':>6}")
    print(f"  {'-' * 65}")

    baseline_by_ticker: dict[str, float] = {}
    best_by_ticker: dict[str, float] = {}
    baseline_ct: dict[str, int] = {}
    best_ct: dict[str, int] = {}

    for t, new_ct, adj_pnl, status in baseline["details"]:
        key = f"{t.ticker}_{t.option_type}"
        baseline_by_ticker[key] = baseline_by_ticker.get(key, 0) + (adj_pnl if status == "entered" else t.pnl_dollars)
        baseline_ct[key] = baseline_ct.get(key, 0) + 1

    for t, new_ct, adj_pnl, status in best_r["details"]:
        key = f"{t.ticker}_{t.option_type}"
        best_by_ticker[key] = best_by_ticker.get(key, 0) + adj_pnl
        best_ct[key] = best_ct.get(key, 0) + (1 if status == "entered" else 0)

    all_keys = sorted(set(baseline_by_ticker.keys()) | set(best_by_ticker.keys()))
    for key in all_keys:
        ticker, otype = key.rsplit("_", 1)
        base_pnl = baseline_by_ticker.get(key, 0)
        new_pnl = best_by_ticker.get(key, 0)
        delta = new_pnl - base_pnl
        b_ct = baseline_ct.get(key, 0)
        n_ct = best_ct.get(key, 0)
        if abs(delta) > 1:
            print(f"  {ticker:<8} {otype:<5} ${base_pnl:>+9,.2f} ${new_pnl:>+9,.2f} ${delta:>+9,.2f} {b_ct:>6} {n_ct:>6}")

    # Skipped trades detail
    if best_r["skipped"] > 0:
        print(f"\n  Skipped trades ({best_r['skipped']}):")
        print(f"  {'Ticker':<8} {'Type':<5} {'Score':>5} {'OrigPnL':>10} {'Reason':<20}")
        print(f"  {'-' * 55}")
        for t, new_ct, adj_pnl, status in best_r["details"]:
            if status != "entered":
                print(f"  {t.ticker:<8} {t.option_type:<5} {t.score:>5} ${t.pnl_dollars:>+9,.2f} {status:<20}")

    # Contract count comparison for a few example trades
    print(f"\n\n{'=' * 110}")
    print(f"SAMPLE CONTRACT CHANGES (baseline → best)")
    print(f"{'=' * 110}\n")
    print(f"  {'Ticker':<8} {'Type':<5} {'Score':>5} {'Entry':>7} {'OldCt':>5} {'NewCt':>5} {'OldPnL':>10} {'NewPnL':>10}")
    print(f"  {'-' * 70}")

    base_details = {(t.trade_id, t.bot): (new_ct, adj_pnl) for t, new_ct, adj_pnl, s in baseline["details"]}
    for t, new_ct, adj_pnl, status in best_r["details"][:30]:
        old_ct, old_pnl = base_details.get((t.trade_id, t.bot), (t.contracts, t.pnl_dollars))
        if old_ct != new_ct or abs(old_pnl - adj_pnl) > 1:
            print(f"  {t.ticker:<8} {t.option_type:<5} {t.score:>5} ${t.entry_premium:>5.2f} {old_ct:>5} {new_ct:>5} "
                  f"${old_pnl:>+9,.2f} ${adj_pnl:>+9,.2f}")


if __name__ == "__main__":
    main()
