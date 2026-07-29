"""6-month backtest of the NEW put filter (whippy-ticker exclusion + underlying-down -0.5% trigger)
vs unfiltered, on our thetadata universe. Puts-only, bearish-confirm OFF so the model actually
generates a put stream (the harness is near-0 with prod put gates). Tests whether the filter — which
flipped the LIVE 6wk book from -$3,642 to +$766 in-sample — holds over a longer window.

    SIZING_MODE=current CONF_LINEAR_CURRENT=1 CONF_BUDGET_MIN=0.4 CONF_BUDGET_MAX=1.8 \
    python scripts/put_filter_6mo.py --days 126
"""
import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest_gold_standard as gs  # noqa: E402

WHIPPY = {"TSLA", "NVDA", "NFLX", "MU", "BA", "AMZN", "PLTR"}  # live per-ticker put losers

CONFIGS = [
    ("UNFILTERED (all puts)",        dict(excl=set(),   trig=-0.15)),
    ("FILTER: exclude whippy",       dict(excl=WHIPPY,  trig=-0.15)),
    ("FILTER: +underlying<=-0.5%",   dict(excl=WHIPPY,  trig=-0.5)),
    ("FILTER: +underlying<=-1.0%",   dict(excl=WHIPPY,  trig=-1.0)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=126)
    ap.add_argument("--pattern-threshold", type=float, default=0.62)
    ap.add_argument("--regime-threshold", type=float, default=0.02)
    args = ap.parse_args()

    # generate a PUT stream: bearish-confirm OFF (else ~0 puts); keep other prod gates
    gs.ENABLE_ANTI_CHASE = True; gs.ENABLE_MOMENTUM_CONFIRM = True
    gs.ENABLE_DIRECTIONAL_REGIME = True; gs.ENABLE_CONSECUTIVE_LOSER = True
    gs.ENABLE_CORRELATION_CAP = True
    gs.ENABLE_PUT_BEARISH_CONFIRM = False
    gs.ENABLE_PUTS = True; gs.PUTS_ONLY = True
    gs.PORTFOLIO_START = 23000

    conn = sqlite3.connect(gs.THETADATA_DB)
    all_dates = [r[0] for r in conn.execute(
        "SELECT DISTINCT substr(timestamp,1,10) FROM option_ohlc WHERE ticker='SPY' ORDER BY 1 DESC").fetchall()]
    conn.close()
    end_date = all_dates[0]; start_date = all_dates[min(args.days - 1, len(all_dates) - 1)]
    base_tickers = [t for t in gs.TICKERS if t not in gs.EXCLUDED_TICKERS]

    print("=" * 84)
    print(f"PUT FILTER 6-MONTH TEST | {start_date} → {end_date} ({args.days}d) | bearish-confirm OFF (put stream)")
    print("=" * 84)
    print("Loading models once...")
    m = gs.load_models(use_entry_filter=False, use_regime=True)

    rows = []
    for name, cfg in CONFIGS:
        gs.PUT_EXCLUDED_TICKERS = set(cfg["excl"])
        gs.PUT_DIRECTION_TRIGGER_PCT = cfg["trig"]
        tickers = [t for t in base_tickers if t not in cfg["excl"]]
        print(f"\n{'#'*60}\n### {name}  (trig {cfg['trig']}, {len(tickers)} tickers)\n{'#'*60}", flush=True)
        r = gs.run_backtest(m[0], m[1], m[2], m[3], args.pattern_threshold, 0.80, tickers,
                            start_date, end_date, m[4], m[5], args.regime_threshold, m[6],
                            m[7], m[8], m[9], m[10], m[11], "none")
        rows.append((name, r))

    print("\n\n" + "=" * 84)
    print(f"PUT FILTER RESULTS — {args.days}d ({start_date}→{end_date})")
    print("=" * 84)
    print(f"{'config':<30}{'trades':>7}{'WR%':>6}{'P&L':>10}{'PF':>7}{'DD%':>7}")
    print("-" * 84)
    base = None
    for name, r in rows:
        if base is None:
            base = r["total_pnl"]
        d = f"  ({r['total_pnl']-base:+.0f} vs base)" if name != rows[0][0] else ""
        print(f"{name:<30}{r['trades']:>7}{r['win_rate']:>6.0f}{r['total_pnl']:>+10.0f}"
              f"{r['profit_factor']:>7.2f}{r['max_drawdown_pct']:>7.1f}{d}")


if __name__ == "__main__":
    main()
