#!/usr/bin/env python3
"""Per-ticker solo models for the new tickers (2026-08-05, Kody's architecture).

Steps 1-2 of the plan: for each new ticker with ENOUGH data, train a SOLO pattern model on ONLY that
ticker's 2.5yr, then validate it on the honest harness over the last 126 days — which lands on the
model's held-out 20% test period (train_pattern_entry does an 80/20 TIME split), so it's out-of-sample /
walk-forward. The generic model + tech book are NOT touched. Promote only the solo models that survive.

Only tickers with >=3000 training samples are included — thinner ones (SPXL 65, TNA 157, SMH 943,
UNG/SOXL/QCOM <2.3k) can't train a trustworthy solo model (their high AUCs are overfit artifacts).

Self-healing: per-ticker retry, atomic status, resumable. Run: python scripts/per_ticker_models.py
"""
import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
LOG = PROJECT / "journal" / "thetadata_expansion_logs"
STATUS = LOG / "_per_ticker_status.json"

# new tickers with >=3000 samples (data-sufficient for a solo model)
TICKERS = ["HOOD", "MARA", "SLV", "IBIT", "TQQQ", "CRM", "SQQQ", "UBER"]

_HEAD = re.compile(r"Headline \(excl\. losers\):\s*P&L \$([+-]?[\d,]+)\s*\|\s*PF\s*([\d.]+|inf)\s*\|\s*WR\s*([\d.]+)%\s*\|\s*(\d+)\s*trades")
_TESTAUC = re.compile(r"Test AUC[:=]\s*([\d.]+)")
_AUC = re.compile(r"AUC[=:]\s*([\d.]+)")


def _num(s):
    return float(s.replace(",", "")) if s not in (None, "inf") else float("inf")


def _load():
    try:
        return json.loads(STATUS.read_text())
    except Exception:
        return {}


def _save(d):
    try:
        LOG.mkdir(parents=True, exist_ok=True)
        tmp = STATUS.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(d, indent=2))
        tmp.replace(STATUS)
    except Exception:
        pass


def process(ticker: str, start: str) -> dict:
    LOG.mkdir(parents=True, exist_ok=True)
    stem = f"pattern_solo_{ticker}"
    tlog = LOG / f"solo_train_{ticker}.log"
    vlog = LOG / f"solo_val_{ticker}.log"
    r = {"ticker": ticker, "stem": stem, "train": "?", "auc": None,
         "pnl": None, "pf": None, "wr": None, "trades": None}

    # 1) TRAIN solo on only this ticker
    tcmd = [sys.executable, str(PROJECT / "scripts" / "train_pattern_entry.py"),
            "--ticker", ticker, "--out", stem, "--start", start]
    for attempt in range(1, 4):
        rc = subprocess.call(tcmd, stdout=open(tlog, "w"), stderr=subprocess.STDOUT)
        if rc == 0:
            break
        print(f"  [{ticker}] train attempt {attempt} rc={rc} — {'retry' if attempt < 3 else 'GIVE UP'}", flush=True)
        time.sleep(5)
    else:
        r["train"] = "FAIL"
        return r
    r["train"] = "OK"
    txt = tlog.read_text(errors="ignore")
    m = _TESTAUC.search(txt) or _AUC.search(txt)
    if m:
        r["auc"] = float(m.group(1))

    # 2) VALIDATE on the held-out last-126d window with the SOLO model
    vcmd = [sys.executable, str(PROJECT / "scripts" / "backtest_gold_standard.py"),
            "--days", "126", "--tickers", ticker, "--pattern-threshold", "0.62",
            "--no-entry-filter", "--model-fill-miss", "on"]
    env = dict(os.environ, PATTERN_MODEL_STEM=stem)
    proc = subprocess.run(vcmd, env=env, capture_output=True, text=True)
    vlog.write_text(proc.stdout + "\n" + proc.stderr)
    m = _HEAD.search(proc.stdout)
    if m:
        r.update(pnl=_num(m.group(1)), pf=_num(m.group(2)), wr=float(m.group(3)), trades=int(m.group(4)))
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=2, help="parallel (train+validate) pipelines; DB-heavy")
    ap.add_argument("--start", type=str, default="2024-01-01")
    ap.add_argument("--only", type=str, default=None)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    todo = TICKERS
    if args.only:
        want = {t.strip().upper() for t in args.only.split(",") if t.strip()}
        todo = [t for t in TICKERS if t in want]

    status = {} if args.fresh else _load()
    pending = [t for t in todo if status.get(t, {}).get("train") != "OK" or status.get(t, {}).get("pnl") is None]
    print(f"Per-ticker solo models: {len(pending)} pending of {len(todo)} — {', '.join(pending)}", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(process, t, args.start): t for t in pending}
        for fut in concurrent.futures.as_completed(futs):
            r = fut.result()
            status[r["ticker"]] = r
            _save(status)
            print(f"  done {r['ticker']:5s} train={r['train']} auc={r['auc']} "
                  f"pnl=${(r['pnl'] or 0):>+8,.0f} PF={r['pf']} tr={r['trades']}", flush=True)

    results = [status[t] for t in todo if t in status]
    results.sort(key=lambda x: -(x.get("pnl") or -1e9))
    print("\n" + "=" * 74)
    print(f"{'TICKER':8s}{'testAUC':>8s}{'P&L':>11s}{'PF':>6s}{'WR%':>7s}{'trades':>8s}  verdict")
    print("=" * 74)
    winners = []
    for r in results:
        pnl = r.get("pnl")
        if r.get("train") != "OK" or pnl is None:
            v = "train/val fail"
        elif (r.get("trades") or 0) < 20:
            v = f"thin ({r.get('trades')} tr)"
        elif pnl > 0 and (r.get("pf") or 0) >= 1.5:
            v = "★ PROMOTE"; winners.append(r["ticker"])
        elif pnl > 0:
            v = "marginal +"
        else:
            v = "loser"
        print(f"{r['ticker']:8s}{(r.get('auc') or 0):>8.3f}{(pnl or 0):>+11,.0f}"
              f"{(r.get('pf') or 0):>6.2f}{(r.get('wr') or 0):>6.1f}%{(r.get('trades') or 0):>8d}  {v}", flush=True)
    print("=" * 74)
    print(f"\n★ SOLO-MODEL WINNERS (route ticker→pattern_solo_<T>, flag-gated): "
          f"{', '.join(winners) if winners else 'none'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
