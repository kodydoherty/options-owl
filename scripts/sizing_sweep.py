#!/usr/bin/env python3
"""Position-sizing sweep driver.

Runs backtest_gold_standard.py under the validated relaxed_C gate baseline for a
set of sizing configs, captures the SWEEP_METRICS block + the per-trade raw JSON,
and writes a combined CSV. Terminates each backtest right after the raw JSON is
written (before the expensive include-losers bias-check pass) to halve runtime.

Usage:
    python scripts/sizing_sweep.py --window is    # in-sample
    python scripts/sizing_sweep.py --window oos    # out-of-sample
    python scripts/sizing_sweep.py --window is --configs baseline,flat
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BT = ROOT / "scripts" / "backtest_gold_standard.py"
RAW = ROOT / "journal" / "v3_eval_results" / "gold_standard_raw.json"
OUTDIR = ROOT / "journal" / "v3_eval_results" / "sizing_sweep"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Validated gate baseline ("relaxed_C")
BASE_GATES = [
    "--puts", "--no-regime",
    "--gate-anti-chase", "off",
    "--gate-momentum", "off",
    "--gate-consecutive-loser", "off",
    "--delta-floor", "0.12",
    "--tod-buffer-min", "5",
]

WINDOWS = {
    "is": ("2026-02-01", "2026-05-20"),
    "oos": ("2025-09-08", "2025-12-07"),
}

# name -> list of sizing CLI args appended to BASE_GATES
CONFIGS: dict[str, list[str]] = {
    # Controls
    "baseline":        ["--sizing-mode", "current"],
    "flat85":          ["--sizing-mode", "flat"],
    # Monotonic confidence-linear curves (budget% over conf 0.74->0.95)
    "conf_lin_60_120": ["--sizing-mode", "conf_linear", "--conf-budget-min", "0.60", "--conf-budget-max", "1.20"],
    "conf_lin_70_110": ["--sizing-mode", "conf_linear", "--conf-budget-min", "0.70", "--conf-budget-max", "1.10"],
    "conf_lin_85_115": ["--sizing-mode", "conf_linear", "--conf-budget-min", "0.85", "--conf-budget-max", "1.15"],
    # Multi-day cap variants on the flat control (flat isolates the cap effect)
    "flat_mdcap2":     ["--sizing-mode", "flat", "--multiday-cap", "2"],
    "flat_mdcap4":     ["--sizing-mode", "flat", "--multiday-cap", "4"],
    "flat_mdcap6":     ["--sizing-mode", "flat", "--multiday-cap", "6"],
    "flat_mdcap_off":  ["--sizing-mode", "flat", "--multiday-cap", "off"],
    # Multi-day cap on baseline (current sizing) for prod-parity reference
    "baseline_mdcap2": ["--sizing-mode", "current", "--multiday-cap", "2"],
}

METRIC_RE = re.compile(r"^\s*METRIC (\S+) (\S+)\s*$")


def run_one(name: str, sizing_args: list[str], start: str, end: str, window: str) -> dict:
    cmd = [sys.executable, str(BT), *BASE_GATES, "--start", start, "--end", end, *sizing_args]
    logpath = OUTDIR / f"{window}_{name}.log"
    rawpath = OUTDIR / f"{window}_{name}_raw.json"
    metrics: dict[str, str] = {}
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    print(f"  → {name}: {' '.join(sizing_args)}", flush=True)
    with open(logpath, "w") as lf:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, env=env, cwd=str(ROOT))
        try:
            for line in proc.stdout:
                lf.write(line)
                lf.flush()
                m = METRIC_RE.match(line)
                if m:
                    metrics[m.group(1)] = m.group(2)
                # Raw JSON is fully written by the time this prints; kill before
                # the include-losers bias pass to save ~50% runtime.
                if "Raw data:" in line:
                    # snapshot the raw JSON for this config immediately
                    if RAW.exists():
                        shutil.copy2(RAW, rawpath)
                    proc.send_signal(signal.SIGTERM)
                    break
            proc.wait(timeout=30)
        except Exception:
            proc.kill()
    metrics["_config"] = name
    metrics["_sizing_args"] = " ".join(sizing_args)
    metrics["_raw"] = str(rawpath)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", choices=["is", "oos"], required=True)
    ap.add_argument("--configs", type=str, default=None,
                    help="comma-separated subset of config names")
    args = ap.parse_args()
    start, end = WINDOWS[args.window]
    names = args.configs.split(",") if args.configs else list(CONFIGS)

    rows = []
    for name in names:
        if name not in CONFIGS:
            print(f"  ! unknown config {name}, skipping", flush=True)
            continue
        rows.append(run_one(name, CONFIGS[name], start, end, args.window))

    # Write combined CSV
    cols = ["_config", "total_pnl", "profit_factor", "sharpe", "max_drawdown_pct",
            "win_rate", "trades", "runners_100", "big_runners_200", "runner_pnl",
            "avg_win", "avg_loss", "_sizing_args", "_raw"]
    csvpath = OUTDIR / f"{args.window}_sweep.csv"
    with open(csvpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {csvpath}", flush=True)
    # Pretty print
    print(f"\n{'config':<18}{'P&L':>10}{'PF':>7}{'Shrp':>7}{'maxDD':>7}{'WR':>6}{'N':>5}{'R100':>6}")
    for r in rows:
        try:
            print(f"{r['_config']:<18}{float(r.get('total_pnl',0)):>10,.0f}"
                  f"{float(r.get('profit_factor',0)):>7.2f}{float(r.get('sharpe',0)):>7.2f}"
                  f"{float(r.get('max_drawdown_pct',0)):>6.1f}%{float(r.get('win_rate',0)):>5.0f}%"
                  f"{int(r.get('trades',0)):>5}{int(r.get('runners_100',0)):>6}")
        except (ValueError, KeyError):
            print(f"{r['_config']:<18} (no metrics — check log)")


if __name__ == "__main__":
    main()
