#!/usr/bin/env python3
"""Backtest the 20 universe-expansion candidates on the HONEST-fill gold-standard harness,
one ticker at a time, in parallel, and print a ranked edge table.

Each ticker runs: backtest_gold_standard.py --tickers <T> --pattern-threshold 0.62
--no-entry-filter --model-fill-miss on  (honest fills + fill-miss modeling).

Metrics parsed from STDOUT (per-ticker capture — no shared-report-file race):
  * Headline (excl. losers): P&L / PF / WR / trades
  * Include-losers: P&L / PF / WR / trades   (less selection bias)
  * fill_miss count

Winners (positive P&L, PF > ~1.5, enough trades) → promote CALL-side, flag-gated. SLV/SMH are an
honest RE-TEST (they lost only on the old fantasy-fill harness).

Usage:
  python scripts/backtest_expansion_tickers.py                 # all 20, 126-day window
  python scripts/backtest_expansion_tickers.py --days 400      # longer / multi-regime
  python scripts/backtest_expansion_tickers.py --concurrency 6
  python scripts/backtest_expansion_tickers.py --only TQQQ,SOXL
"""
import argparse
import concurrent.futures
import re
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_DIR / "journal" / "thetadata_expansion_logs"

TICKERS = ["TQQQ", "SOXL", "SPXL", "TNA", "SMH", "QCOM", "CRM", "UBER", "XLF", "GLD",
           "TLT", "SQQQ", "IBIT", "HOOD", "XLE", "KRE", "XBI", "SLV", "UNG", "MARA"]

_HEAD = re.compile(r"Headline \(excl\. losers\):\s*P&L \$([+-]?[\d,]+)\s*\|\s*PF\s*([\d.]+|inf)\s*\|\s*WR\s*([\d.]+)%\s*\|\s*(\d+)\s*trades")
_INCL = re.compile(r"Include-losers:\s*P&L \$([+-]?[\d,]+)\s*\|\s*PF\s*([\d.]+|inf)\s*\|\s*WR\s*([\d.]+)%\s*\|\s*(\d+)\s*trades")
_MISS = re.compile(r"METRIC fill_miss (\d+)")


def _num(s):
    return float(s.replace(",", "")) if s not in (None, "inf") else float("inf")


def backtest_one(ticker: str, days: int) -> dict:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LOG_DIR / f"bt_{ticker}.log"
    cmd = [sys.executable, str(PROJECT_DIR / "scripts" / "backtest_gold_standard.py"),
           "--days", str(days), "--tickers", ticker,
           "--pattern-threshold", "0.62", "--no-entry-filter", "--model-fill-miss", "on"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = proc.stdout + "\n" + proc.stderr
    out_path.write_text(out)
    r = {"ticker": ticker, "trades": 0, "pnl": 0.0, "pf": 0.0, "wr": 0.0,
         "itrades": 0, "ipnl": 0.0, "miss": 0, "err": proc.returncode != 0}
    m = _HEAD.search(out)
    if m:
        r.update(pnl=_num(m.group(1)), pf=_num(m.group(2)), wr=float(m.group(3)), trades=int(m.group(4)))
    mi = _INCL.search(out)
    if mi:
        r.update(ipnl=_num(mi.group(1)), itrades=int(mi.group(4)))
    mm = _MISS.search(out)
    if mm:
        r["miss"] = int(mm.group(1))
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=126, help="backtest window in trading days (default 126 = ~6mo)")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--only", type=str, default=None)
    args = ap.parse_args()

    todo = TICKERS
    if args.only:
        want = {t.strip().upper() for t in args.only.split(",") if t.strip()}
        todo = [t for t in TICKERS if t in want]

    print(f"Backtesting {len(todo)} tickers, {args.days}d window, concurrency {args.concurrency}\n", flush=True)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(backtest_one, t, args.days): t for t in todo}
        for fut in concurrent.futures.as_completed(futs):
            r = fut.result()
            results.append(r)
            flag = "ERR" if r["err"] else ("0 trades" if r["trades"] == 0 else "ok")
            print(f"  done {r['ticker']:5s} pnl=${r['pnl']:>+9,.0f} PF={r['pf']:>4.2f} "
                  f"WR={r['wr']:>4.1f}% tr={r['trades']:>3d} miss={r['miss']:>3d}  [{flag}]", flush=True)

    # Rank by the HEADLINE per-ticker P&L (the real isolated number). NOTE: the harness's
    # "Include-losers" line is a stale GLOBAL aggregate, NOT this ticker — do not rank on it.
    results.sort(key=lambda x: (-x["pnl"], -x["pf"]))
    print("\n" + "=" * 78)
    print(f"{'RANK  TICKER':14s}{'P&L':>12s}{'PF':>6s}{'WR%':>7s}{'trades':>8s}{'fillmiss':>9s}  verdict")
    print("=" * 78)
    for i, r in enumerate(results, 1):
        if r["err"]:
            verdict = "ERROR"
        elif r["trades"] < 20:
            verdict = f"thin ({r['trades']} tr)"
        elif r["pnl"] > 0 and r["pf"] >= 1.5:
            verdict = "★ WINNER"
        elif r["pnl"] > 0:
            verdict = "marginal +"
        else:
            verdict = "loser"
        print(f"{i:<4d}  {r['ticker']:6s}{r['pnl']:>+12,.0f}{r['pf']:>6.2f}{r['wr']:>6.1f}%"
              f"{r['trades']:>8d}{r['miss']:>9d}  {verdict}", flush=True)
    print("=" * 78)
    winners = [r["ticker"] for r in results if not r["err"] and r["trades"] >= 20 and r["pnl"] > 0 and r["pf"] >= 1.5]
    print(f"\n★ WINNERS (promote CALL-side, flag-gated): {', '.join(winners) if winners else 'none'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
