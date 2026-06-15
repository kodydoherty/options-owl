"""Retrain all per-ticker ML models when new data arrives.

Automates the full retraining pipeline:
1. Check data freshness per ticker in thetadata_options.db
2. Train per-ticker signal + runner models for tickers with sufficient data
3. Train a GENERIC model using all tickers' data combined
4. Compare per-ticker AUC vs GENERIC AUC to show which tickers benefit
5. Print a summary report

Usage:
    python scripts/retrain_all_models.py                          # retrain all tickers
    python scripts/retrain_all_models.py --tickers SPY,TSLA,NVDA  # specific tickers
    python scripts/retrain_all_models.py --dry-run                 # check data only
    python scripts/retrain_all_models.py --min-days 30             # require 30 days of data
    python scripts/retrain_all_models.py --no-uw                   # skip UW adjustments
    python scripts/retrain_all_models.py --skip-backtest           # train only, no FSM backtest
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

from loguru import logger

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

# Import training functions from the existing v2 script
from scripts.train_option_signals_v2 import (
    FEATURE_COLS,
    MODEL_DIR as DEFAULT_MODEL_DIR,
    THETADATA_DB as DEFAULT_THETADATA_DB,
    TICKERS as ALL_TICKERS,
    UW_DB as DEFAULT_UW_DB,
    build_dataset_for_ticker,
    run_backtest,
    train_runner_model,
    train_ticker_model,
    UWScoreAdjuster,
)

import pandas as pd


# ---------------------------------------------------------------------------
# Data freshness check
# ---------------------------------------------------------------------------

def check_data_freshness(
    conn: sqlite3.Connection,
    tickers: list[str],
) -> dict[str, dict]:
    """Query thetadata_options.db for data availability per ticker.

    Returns {ticker: {days: int, first_date: str, last_date: str, row_count: int}}.
    """
    freshness = {}
    for ticker in tickers:
        row = conn.execute(
            """
            SELECT
                COUNT(DISTINCT substr(timestamp, 1, 10)) as days,
                MIN(substr(timestamp, 1, 10)) as first_date,
                MAX(substr(timestamp, 1, 10)) as last_date,
                COUNT(*) as row_count
            FROM option_ohlc
            WHERE ticker = ?
            """,
            (ticker,),
        ).fetchone()

        days, first_date, last_date, row_count = row
        freshness[ticker] = {
            "days": days or 0,
            "first_date": first_date or "N/A",
            "last_date": last_date or "N/A",
            "row_count": row_count or 0,
        }
    return freshness


# ---------------------------------------------------------------------------
# Train a single ticker (with timing and error handling)
# ---------------------------------------------------------------------------

def train_single_ticker(
    conn: sqlite3.Connection,
    ticker: str,
    model_dir: Path,
) -> dict | None:
    """Train signal + runner models for one ticker. Returns combined metadata or None."""
    start = time.time()
    logger.info("Training {} ...", ticker)

    try:
        df = build_dataset_for_ticker(conn, ticker)
        if df.empty:
            logger.warning("{}: no dataset could be built (no profitable moves found)", ticker)
            return None

        # Temporarily override MODEL_DIR in the v2 module so models save to the right place
        import scripts.train_option_signals_v2 as v2_mod
        original_dir = v2_mod.MODEL_DIR
        v2_mod.MODEL_DIR = model_dir
        model_dir.mkdir(parents=True, exist_ok=True)

        try:
            signal_meta = train_ticker_model(df, ticker)
            runner_meta = train_runner_model(df, ticker)
        finally:
            v2_mod.MODEL_DIR = original_dir

        elapsed = time.time() - start
        logger.info("{} done in {:.1f}s", ticker, elapsed)

        return {
            "ticker": ticker,
            "dataset_size": len(df),
            "signal": signal_meta,
            "runner": runner_meta,
            "elapsed_seconds": elapsed,
            "df": df,  # keep for GENERIC training
        }

    except Exception:
        elapsed = time.time() - start
        logger.exception("FAILED training {} after {:.1f}s", ticker, elapsed)
        return None


# ---------------------------------------------------------------------------
# Train GENERIC model from combined data
# ---------------------------------------------------------------------------

def train_generic_model(
    all_dfs: dict[str, pd.DataFrame],
    model_dir: Path,
) -> dict | None:
    """Train a GENERIC model from all tickers' combined data."""
    if not all_dfs:
        logger.warning("No ticker datasets available for GENERIC model")
        return None

    combined = pd.concat(list(all_dfs.values()), ignore_index=True)
    if len(combined) < 100:
        logger.warning("Combined dataset too small ({} samples) for GENERIC model", len(combined))
        return None

    logger.info("Training GENERIC model ({} samples from {} tickers) ...", len(combined), len(all_dfs))
    start = time.time()

    import scripts.train_option_signals_v2 as v2_mod
    original_dir = v2_mod.MODEL_DIR
    v2_mod.MODEL_DIR = model_dir
    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        signal_meta = train_ticker_model(combined, "GENERIC")
        runner_meta = train_runner_model(combined, "GENERIC")
    finally:
        v2_mod.MODEL_DIR = original_dir

    elapsed = time.time() - start
    logger.info("GENERIC done in {:.1f}s", elapsed)

    return {
        "ticker": "GENERIC",
        "dataset_size": len(combined),
        "signal": signal_meta,
        "runner": runner_meta,
        "elapsed_seconds": elapsed,
    }


# ---------------------------------------------------------------------------
# Compare per-ticker vs GENERIC
# ---------------------------------------------------------------------------

def compare_models(
    ticker_results: list[dict],
    generic_result: dict | None,
) -> list[dict]:
    """Compare each per-ticker model's test AUC against the GENERIC model.

    Returns list of comparison dicts for the summary table.
    """
    generic_auc = 0.0
    if generic_result and generic_result.get("signal"):
        generic_auc = generic_result["signal"]["metrics"].get("auc", 0.0)

    comparisons = []
    for result in ticker_results:
        ticker = result["ticker"]
        signal = result.get("signal")
        if not signal:
            comparisons.append({
                "ticker": ticker,
                "days": 0,
                "train_auc": 0.0,
                "test_auc": 0.0,
                "threshold": 0.0,
                "per_ticker_better": None,
                "delta": 0.0,
            })
            continue

        metrics = signal["metrics"]
        # Approximate train AUC from the model metadata
        # The v2 script only stores test metrics, so we use test AUC for both display cols
        # and note that "Train AUC" is not separately tracked — we show test AUC
        test_auc = metrics.get("auc", 0.0)
        train_samples = metrics.get("train_samples", 0)
        test_samples = metrics.get("test_samples", 0)

        delta = test_auc - generic_auc if generic_auc > 0 else 0.0
        per_ticker_better = delta > 0 if generic_auc > 0 else None

        comparisons.append({
            "ticker": ticker,
            "train_samples": train_samples,
            "test_samples": test_samples,
            "test_auc": test_auc,
            "threshold": signal.get("optimal_threshold", 0.0),
            "precision": metrics.get("precision", 0.0),
            "recall": metrics.get("recall", 0.0),
            "f1": metrics.get("f1", 0.0),
            "per_ticker_better": per_ticker_better,
            "delta": delta,
            "generic_auc": generic_auc,
        })

    return comparisons


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def print_freshness_report(freshness: dict[str, dict], min_days: int) -> None:
    """Print data freshness table."""
    print(f"\n{'='*75}")
    print("DATA FRESHNESS REPORT")
    print(f"{'='*75}")
    print(f"{'Ticker':<8} {'Days':>6} {'Rows':>10} {'First Date':>12} {'Last Date':>12} {'Status':>10}")
    print("-" * 75)
    for ticker, info in sorted(freshness.items()):
        status = "OK" if info["days"] >= min_days else f"SKIP (<{min_days}d)"
        print(
            f"{ticker:<8} {info['days']:>6} {info['row_count']:>10,} "
            f"{info['first_date']:>12} {info['last_date']:>12} {status:>10}"
        )
    eligible = sum(1 for info in freshness.values() if info["days"] >= min_days)
    print(f"\n{eligible}/{len(freshness)} tickers eligible for training (>= {min_days} days)")


def print_summary_report(
    comparisons: list[dict],
    generic_result: dict | None,
    freshness: dict[str, dict],
    total_elapsed: float,
) -> None:
    """Print the final summary table."""
    generic_auc = 0.0
    if generic_result and generic_result.get("signal"):
        generic_auc = generic_result["signal"]["metrics"].get("auc", 0.0)

    print(f"\n{'='*90}")
    print("RETRAIN SUMMARY")
    print(f"{'='*90}")
    print(
        f"{'Ticker':<8} {'Days':>5} {'Train':>7} {'Test':>6} "
        f"{'AUC':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Thresh':>7} {'vs GENERIC':>14}"
    )
    print("-" * 90)

    for c in comparisons:
        days = freshness.get(c["ticker"], {}).get("days", 0)

        if c.get("test_auc", 0) == 0 and c.get("per_ticker_better") is None:
            print(f"{c['ticker']:<8} {days:>5} {'':>7} {'':>6} {'FAILED':>6}")
            continue

        if c["per_ticker_better"] is True:
            verdict = f"Yes (+{c['delta']:.3f})"
        elif c["per_ticker_better"] is False:
            verdict = f"No (GEN: {c['generic_auc']:.3f})"
        else:
            verdict = "N/A"

        print(
            f"{c['ticker']:<8} {days:>5} {c.get('train_samples', 0):>7} {c.get('test_samples', 0):>6} "
            f"{c['test_auc']:>5.3f} {c.get('precision', 0)*100:>5.1f}% {c.get('recall', 0)*100:>5.1f}% "
            f"{c.get('f1', 0)*100:>5.1f}% {c['threshold']:>6.2f} {verdict:>14}"
        )

    # GENERIC row
    if generic_result and generic_result.get("signal"):
        gm = generic_result["signal"]["metrics"]
        print("-" * 90)
        print(
            f"{'GENERIC':<8} {'ALL':>5} {gm.get('train_samples', 0):>7} {gm.get('test_samples', 0):>6} "
            f"{generic_auc:>5.3f} {gm.get('precision', 0)*100:>5.1f}% {gm.get('recall', 0)*100:>5.1f}% "
            f"{gm.get('f1', 0)*100:>5.1f}% {generic_result['signal'].get('optimal_threshold', 0):>6.2f} {'(baseline)':>14}"
        )

    better_count = sum(1 for c in comparisons if c.get("per_ticker_better") is True)
    total_count = sum(1 for c in comparisons if c.get("per_ticker_better") is not None)
    print(f"\n{better_count}/{total_count} tickers benefit from per-ticker models over GENERIC")
    print(f"Total training time: {total_elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def retrain_all(
    tickers: list[str] | None = None,
    min_days: int = 20,
    db_path: str = DEFAULT_THETADATA_DB,
    uw_db_path: str = DEFAULT_UW_DB,
    models_dir: str | None = None,
    no_uw: bool = False,
    dry_run: bool = False,
    skip_backtest: bool = False,
) -> dict:
    """Run the full retrain pipeline.

    Returns a summary dict with results per ticker and GENERIC comparison.
    """
    model_dir = Path(models_dir) if models_dir else DEFAULT_MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)

    effective_tickers = tickers or list(ALL_TICKERS)
    logger.info("Checking data freshness for {} tickers in {}", len(effective_tickers), db_path)

    # Step 1: Check data freshness
    freshness = check_data_freshness(conn, effective_tickers)
    print_freshness_report(freshness, min_days)

    eligible = [t for t in effective_tickers if freshness[t]["days"] >= min_days]

    if not eligible:
        logger.warning("No tickers have >= {} days of data. Nothing to train.", min_days)
        conn.close()
        return {"freshness": freshness, "trained": [], "generic": None, "comparisons": []}

    if dry_run:
        logger.info("Dry run complete. {} tickers eligible for training.", len(eligible))
        conn.close()
        return {"freshness": freshness, "trained": [], "generic": None, "comparisons": []}

    # Step 2: Train per-ticker models
    total_start = time.time()
    ticker_results = []
    all_dfs = {}

    print(f"\n{'='*75}")
    print(f"TRAINING PER-TICKER MODELS ({len(eligible)} tickers)")
    print(f"{'='*75}\n")

    for i, ticker in enumerate(eligible, 1):
        logger.info("[{}/{}] {}", i, len(eligible), ticker)
        result = train_single_ticker(conn, ticker, model_dir)
        if result:
            ticker_results.append(result)
            if result.get("df") is not None:
                all_dfs[ticker] = result["df"]
                # Drop the dataframe reference from result to save memory in the summary
                # (we already have it in all_dfs)

    # Step 3: Train GENERIC model
    print(f"\n{'='*75}")
    print("TRAINING GENERIC MODEL")
    print(f"{'='*75}\n")

    generic_result = train_generic_model(all_dfs, model_dir)

    # Step 4: Compare per-ticker vs GENERIC
    comparisons = compare_models(ticker_results, generic_result)

    total_elapsed = time.time() - total_start

    # Step 5: Print summary
    print_summary_report(comparisons, generic_result, freshness, total_elapsed)

    # Step 6 (optional): Run backtest with new models
    if not skip_backtest:
        print(f"\n{'='*75}")
        print("BACKTESTING WITH NEW MODELS")
        print(f"{'='*75}\n")

        uw_adjuster = None
        if not no_uw:
            try:
                uw_adjuster = UWScoreAdjuster(uw_db_path)
            except Exception as e:
                logger.warning("Could not load UW adjuster: {}", e)

        backtest_results = []
        for ticker in eligible:
            try:
                r = run_backtest(conn, ticker, portfolio=20000, uw_adjuster=uw_adjuster)
                if r:
                    backtest_results.append(r)
            except Exception:
                logger.exception("Backtest failed for {}", ticker)

        if backtest_results:
            total_trades = sum(r["trades"] for r in backtest_results)
            total_wins = sum(r["wins"] for r in backtest_results)
            total_pnl = sum(r["total_pnl"] for r in backtest_results)
            wr = total_wins / total_trades * 100 if total_trades > 0 else 0
            print(f"\nBACKTEST COMBINED: {total_trades} trades | {total_wins} wins ({wr:.1f}%) | P&L: ${total_pnl:,.0f}")

        if uw_adjuster:
            uw_adjuster.close()

    conn.close()

    # Clean up dataframe refs from results before returning
    for r in ticker_results:
        r.pop("df", None)

    logger.info("Retrain complete. Models saved to {}", model_dir)

    return {
        "freshness": freshness,
        "trained": ticker_results,
        "generic": generic_result,
        "comparisons": comparisons,
        "total_elapsed": total_elapsed,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Retrain all per-ticker ML models when new data arrives"
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Comma-separated list of tickers (default: all 14)",
    )
    parser.add_argument(
        "--min-days",
        type=int,
        default=20,
        help="Minimum training days per ticker (default: 20)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=DEFAULT_THETADATA_DB,
        help=f"Path to thetadata DB (default: {DEFAULT_THETADATA_DB})",
    )
    parser.add_argument(
        "--uw-db",
        type=str,
        default=DEFAULT_UW_DB,
        help=f"Path to UW DB (default: {DEFAULT_UW_DB})",
    )
    parser.add_argument(
        "--no-uw",
        action="store_true",
        help="Skip UW score adjustments",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help=f"Where to save models (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check data freshness only, don't train",
    )
    parser.add_argument(
        "--skip-backtest",
        action="store_true",
        help="Train only, skip the full FSM backtest (faster)",
    )
    args = parser.parse_args()

    # Configure loguru
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}",
        level="INFO",
    )

    tickers = None
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    retrain_all(
        tickers=tickers,
        min_days=args.min_days,
        db_path=args.db,
        uw_db_path=args.uw_db,
        models_dir=args.models_dir,
        no_uw=args.no_uw,
        dry_run=args.dry_run,
        skip_backtest=args.skip_backtest,
    )


if __name__ == "__main__":
    main()
