"""Starting-portfolio-size sweep (2026-07-14) — what account size should a user start with?

Runs the KODY-FAITHFUL gold-standard ML book (calls+puts, conf_linear 0.4-1.8, delta gate, V7 exits +
the live -25% hardstops/stall-cut/early-lock/profit-lock) at several STARTING balances over the same
window, and reports for each: final equity, return %, PF, WR, max DD, #trades — PLUS the affordability
breakdown that answers Kody's hypothesis (small accounts can't afford expensive premiums so they SKIP
those trades: score_to_contracts returns 0 when one contract > 15% of balance).

Models load ONCE; each size re-runs the full compounding loop (trade selection IS balance-dependent via
circuit-breakers/GFV/concurrency, so we can't decouple). Sizing is prod score_to_contracts with kody's
conf_linear — the integer-contract flooring + 15% position cap are what make small accounts skip trades.

Caveats (stated, not hidden): ML book ONLY (no UW flow — flow is ~4x the ML edge and its per-trade sleeve
is roughly size-linear, so it lifts every account but doesn't change the RANKING). runner_v1/VVIX/flow-
conviction sizing layers are not modeled here (roughly balance-independent scalars). Read the $ levels as a
conservative ML-only estimate; read the RELATIVE comparison across sizes as the real signal.

Usage: python scripts/portfolio_size_sweep.py --days 126 --sizes 5000,10000,15000,20000
"""
import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest_gold_standard as gs  # noqa: E402


def _apply_kody_config():
    """Replicate the prod-faithful global setup that main() does for the kody invocation:
    --pattern-threshold 0.62 --no-entry-filter --puts (delta gate on, conf_linear 0.4-1.8, V7 exits)."""
    gs.ENABLE_PUTS = True
    gs.PUTS_ONLY = False
    gs.ENABLE_DIP_CONFIRM = True
    # prod gates (module defaults are already True, set explicitly for clarity)
    gs.ENABLE_ANTI_CHASE = True
    gs.ENABLE_MOMENTUM_CONFIRM = True
    gs.ENABLE_CONSECUTIVE_LOSER = True
    gs.ENABLE_CORRELATION_CAP = True
    gs.ENABLE_DIRECTIONAL_REGIME = True
    gs.ENABLE_PUT_BEARISH_CONFIRM = True
    # delta gate ON, old static price gates OFF (matches prod)
    gs.ENABLE_DELTA_GATE = True
    gs.ENABLE_PRICE_GATES = False
    gs.DELTA_MIN = 0.15
    gs.DELTA_MAX = 0.70
    # kody sizing: real score_to_contracts + conf_linear 0.4-1.8
    gs.SIZING_MODE = "current"
    gs.CONF_LINEAR_CURRENT = True
    gs.CONF_BUDGET_MIN = 0.4
    gs.CONF_BUDGET_MAX = 1.8
    gs.MAX_SIZING_DOLLARS = 0.0   # no-op below $50k
    gs.MULTI_DAY_CAP = 2          # prod paper_trader multi-day cap
    gs.V7_EXITS_OVERRIDE = True
    gs.ALLOW_REENTRIES = True


def _reset_size_stats():
    for k in gs._SIZE_STATS:
        gs._SIZE_STATS[k] = 0 if not k.endswith("_sum") else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=126, help="Trading days (126 ≈ 6 months)")
    ap.add_argument("--sizes", type=str, default="5000,10000,15000,20000")
    ap.add_argument("--pattern-threshold", type=float, default=0.62)
    ap.add_argument("--regime-threshold", type=float, default=0.02)
    args = ap.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]

    _apply_kody_config()

    # date range = last N trading days available in thetadata
    conn = sqlite3.connect(gs.THETADATA_DB)
    all_dates = [r[0] for r in conn.execute(
        "SELECT DISTINCT substr(timestamp,1,10) FROM option_ohlc WHERE ticker='SPY' ORDER BY 1 DESC"
    ).fetchall()]
    conn.close()
    end_date = all_dates[0]
    start_date = all_dates[min(args.days - 1, len(all_dates) - 1)]
    tickers = [t for t in gs.TICKERS if t not in gs.EXCLUDED_TICKERS]

    print("=" * 78)
    print("PORTFOLIO-SIZE SWEEP — kody-faithful ML book (calls+puts)")
    print(f"Window: {start_date} → {end_date}  ({args.days} trading days)  |  {len(tickers)} tickers")
    print("Sizing: conf_linear[0.4,1.8], 15% pos cap, 75% risk, 8 concurrent, multiday_cap=2")
    print("=" * 78)

    print("\nLoading models (once)...")
    models = gs.load_models(use_entry_filter=False, use_regime=True)
    (pattern_model, pattern_meta, entry_model, entry_features, stop_model, regime_model,
     signal_model, put_pattern_model, put_pattern_meta, put_entry_model, put_entry_features,
     put_entry_threshold) = models

    rows = []
    for size in sizes:
        gs.PORTFOLIO_START = size
        _reset_size_stats()
        print(f"\n{'#' * 60}\n### RUN: starting portfolio = ${size:,}\n{'#' * 60}", flush=True)
        r = gs.run_backtest(
            pattern_model, pattern_meta, entry_model, entry_features,
            args.pattern_threshold, 0.80, tickers, start_date, end_date, stop_model,
            regime_model, args.regime_threshold, signal_model,
            put_pattern_model, put_pattern_meta,
            put_entry_model, put_entry_features, put_entry_threshold, "none",
        )
        stats = dict(gs._SIZE_STATS)
        rows.append((size, r, stats))

    # ── Summary tables ─────────────────────────────────────────────────────
    print("\n\n" + "=" * 92)
    print("RESULTS — performance by starting portfolio")
    print("=" * 92)
    hdr = f"{'Start':>8} {'FinalEq':>11} {'Return%':>9} {'CAGR-ish':>9} {'PF':>6} {'WR%':>6} {'Trades':>7} {'MaxDD%':>8} {'AvgContr':>9}"
    print(hdr)
    print("-" * 92)
    for size, r, stats in rows:
        final_eq = size + r["total_pnl"]
        avg_contr = stats["contracts_sum"] / max(1, stats["taken"])
        # simple annualization off the window length (days/252)
        ann = ((final_eq / size) ** (252.0 / max(args.days, 1)) - 1) * 100 if final_eq > 0 else -100
        print(f"${size:>7,} ${final_eq:>10,.0f} {r['return_pct']:>8.1f}% {ann:>8.0f}% "
              f"{r['profit_factor']:>6.2f} {r['win_rate']:>6.1f} {r['trades']:>7} "
              f"{r['max_drawdown_pct']:>7.1f}% {avg_contr:>9.2f}")

    print("\n" + "=" * 92)
    print("AFFORDABILITY — does a small account get priced out of expensive trades? (Kody's hypothesis)")
    print("=" * 92)
    print(f"{'Start':>8} {'Taken':>7} {'SkipCantAfford':>15} {'Afford%':>8} {'CapClipped':>11} "
          f"{'AvgPremTaken':>13} {'AvgPremSkip':>12}")
    print("-" * 92)
    for size, r, stats in rows:
        taken = stats["taken"]
        skip = stats["skip_unafford"]
        elig = taken + skip  # size-eligible universe (excludes score/conf-floor rejects)
        afford_pct = 100.0 * taken / max(1, elig)
        cap = stats["cap_bound"]
        avg_prem_taken = stats["taken_prem_sum"] / max(1, taken)
        avg_prem_skip = stats["unafford_prem_sum"] / max(1, skip)
        print(f"${size:>7,} {taken:>7} {skip:>15} {afford_pct:>7.1f}% {cap:>11} "
              f"${avg_prem_taken:>11.2f} ${avg_prem_skip:>10.2f}")
    print("\nSkipCantAfford = passed entry gates but 1 contract > 15% of balance → trade skipped.")
    print("Afford% = of size-eligible signals, how many the account could actually take.")
    print("CapClipped = trades where the 15% position cap reduced the contract count (wanted more).")


if __name__ == "__main__":
    main()
