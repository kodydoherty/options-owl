#!/usr/bin/env python3
"""Class-based pattern models for the new tickers (2026-08-06, Kody's architecture).

Groups related tickers into ONE pooled model per asset class (more data + better calibration than
solo, isolated from the tech book so no dilution), trained with HONEST-FILL labels, then validates
each member ticker out-of-sample at the class model's own threshold.

Classes (edit as needed):
  leveraged  : TQQQ SOXL SPXL TNA SQQQ   (3x index ETFs — near-identical decay/vol dynamics)
  crypto     : IBIT MARA                 (bitcoin-driven)
  commodity  : SLV UNG                   (macro-driven; split later if metals vs energy diverge)
  singlenames: HOOD UBER CRM QCOM        (individual equities — same 'area' as tech, kept separate)

Route: each promoted member ticker → its class model (pattern_class_<name>) + that model's threshold;
tech book stays on the untouched generic. Runner models per class come AFTER (separate step).

Self-healing, resumable. Run: python scripts/class_models.py   (add --only crypto to target one class)
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
STATUS = LOG / "_class_status.json"

CLASSES = {
    "leveraged":  ["TQQQ", "SOXL", "SPXL", "TNA", "SQQQ"],
    "crypto":     ["IBIT", "MARA"],
    "commodity":  ["SLV", "UNG"],
    "singlenames": ["HOOD", "UBER", "CRM", "QCOM"],
}

_HEAD = re.compile(r"Headline \(excl\. losers\):\s*P&L \$([+-]?[\d,]+)\s*\|\s*PF\s*([\d.]+|inf)\s*\|\s*WR\s*([\d.]+)%\s*\|\s*(\d+)\s*trades")
_AUC = re.compile(r"AUC[=:]\s*([\d.]+)")


def _num(s):
    return float(s.replace(",", "")) if s not in (None, "inf") else float("inf")


def _load():
    try:
        return json.loads(STATUS.read_text())
    except Exception:
        return {}


def _save(d):
    LOG.mkdir(parents=True, exist_ok=True)
    tmp = STATUS.with_suffix(".json.tmp"); tmp.write_text(json.dumps(d, indent=2)); tmp.replace(STATUS)


def process(cls: str, members: list, start: str) -> dict:
    stem = f"pattern_class_{cls}"
    tlog = LOG / f"class_train_{cls}.log"
    r = {"class": cls, "stem": stem, "members": members, "train": "?", "auc": None, "val": {}}

    # 1) TRAIN one pooled model on all class members, HONEST labels
    tcmd = [sys.executable, str(PROJECT / "scripts" / "train_pattern_entry.py"),
            "--ticker", ",".join(members), "--out", stem, "--start", start]
    env = dict(os.environ, HONEST_FILL_LABELS="1")
    for attempt in range(1, 4):
        rc = subprocess.call(tcmd, stdout=open(tlog, "w"), stderr=subprocess.STDOUT, env=env)
        if rc == 0:
            break
        print(f"  [{cls}] train attempt {attempt} rc={rc} — {'retry' if attempt < 3 else 'GIVE UP'}", flush=True)
        time.sleep(5)
    else:
        r["train"] = "FAIL"
        return r
    r["train"] = "OK"
    m = _AUC.search(tlog.read_text(errors="ignore"))
    if m:
        r["auc"] = float(m.group(1))

    # 2) VALIDATE each member out-of-sample at the CLASS model's own threshold (--pattern-threshold 0.0)
    for tk in members:
        vlog = LOG / f"class_val_{cls}_{tk}.txt"
        vcmd = [sys.executable, str(PROJECT / "scripts" / "backtest_gold_standard.py"),
                "--days", "126", "--tickers", tk, "--pattern-threshold", "0.0",
                "--no-entry-filter", "--model-fill-miss", "on"]
        venv = dict(os.environ, PATTERN_MODEL_STEM=stem)
        proc = subprocess.run(vcmd, env=venv, capture_output=True, text=True)
        vlog.write_text(proc.stdout + "\n" + proc.stderr)
        mm = _HEAD.search(proc.stdout)
        r["val"][tk] = ({"pnl": _num(mm.group(1)), "pf": _num(mm.group(2)),
                         "wr": float(mm.group(3)), "trades": int(mm.group(4))} if mm
                        else {"pnl": None, "pf": None, "wr": None, "trades": None})
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--start", type=str, default="2024-01-01")
    ap.add_argument("--only", type=str, default=None, help="comma-separated class names")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    classes = CLASSES
    if args.only:
        want = {c.strip().lower() for c in args.only.split(",")}
        classes = {k: v for k, v in CLASSES.items() if k in want}

    status = {} if args.fresh else _load()
    pending = [c for c in classes if status.get(c, {}).get("train") != "OK" or not status.get(c, {}).get("val")]
    print(f"Class models: {len(pending)} pending — {', '.join(pending)}", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(process, c, classes[c], args.start): c for c in pending}
        for fut in concurrent.futures.as_completed(futs):
            r = fut.result()
            status[r["class"]] = r
            _save(status)
            print(f"  done class={r['class']:11s} train={r['train']} auc={r['auc']}", flush=True)

    # Report: per-member verdicts under each class model
    print("\n" + "=" * 78)
    print(f"{'CLASS':12s}{'TICKER':8s}{'P&L':>11s}{'PF':>6s}{'WR%':>7s}{'trades':>8s}  verdict")
    print("=" * 78)
    promote = []
    for c in classes:
        r = status.get(c, {})
        for tk, v in (r.get("val") or {}).items():
            pnl = v.get("pnl")
            if pnl is None:
                verdict = "val fail"
            elif (v.get("trades") or 0) < 20:
                verdict = f"thin ({v.get('trades')} tr)"
            elif pnl > 0 and (v.get("pf") or 0) >= 1.5:
                verdict = "★ PROMOTE"; promote.append(f"{tk}→{c}")
            elif pnl > 0:
                verdict = "marginal +"
            else:
                verdict = "loser"
            print(f"{c:12s}{tk:8s}{(pnl or 0):>+11,.0f}{(v.get('pf') or 0):>6.2f}"
                  f"{(v.get('wr') or 0):>6.1f}%{(v.get('trades') or 0):>8d}  {verdict}", flush=True)
    print("=" * 78)
    print(f"\n★ PROMOTE (route ticker→class model, flag-gated): {', '.join(promote) if promote else 'none'}", flush=True)
    print("NEXT: train per-class RUNNER models for the promoted classes (sizing).", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
