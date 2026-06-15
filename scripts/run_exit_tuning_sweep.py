#!/usr/bin/env python3
"""Disciplined one-param-at-a-time EXIT tuning sweep for the gold-standard backtest.

Offline experiment only — no live trading, no deploy. Runs the REAL ExitFSM via
backtest_gold_standard.py with runtime CLI overrides on top of the validated
relaxed_C gate config. Each run's full stdout is written to its own log under
journal/v3_eval_results/tuning/ and the Headline + Runners lines are parsed.

Usage:
    python scripts/run_exit_tuning_sweep.py sensitivity   # TASK 2 one-param-at-a-time (in-sample)
    python scripts/run_exit_tuning_sweep.py candidates     # TASK 3 stacked configs (in-sample + OOS)
    python scripts/run_exit_tuning_sweep.py all            # both
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Bounded concurrency: each backtest is a single CPU-bound Python process.
# 18 cores available; leave headroom. Override with SWEEP_WORKERS env.
WORKERS = int(os.environ.get("SWEEP_WORKERS", "6"))

ROOT = Path(__file__).resolve().parent.parent
BT = str(ROOT / "scripts" / "backtest_gold_standard.py")
OUT = ROOT / "journal" / "v3_eval_results" / "tuning"
OUT.mkdir(parents=True, exist_ok=True)

# Validated relaxed_C gate baseline — fixed starting point for ALL runs.
BASELINE = [
    "--puts", "--no-regime",
    "--gate-anti-chase", "off",
    "--gate-momentum", "off",
    "--gate-consecutive-loser", "off",
    "--delta-floor", "0.12",
    "--tod-buffer-min", "5",
]
IN_SAMPLE = ["--start", "2026-02-01", "--end", "2026-05-20"]
OOS = ["--start", "2025-09-08", "--end", "2025-12-07"]

HEADLINE_RE = re.compile(
    r"Headline \(excl\. losers\): P&L \$([+-]?[\d,]+) \| PF ([\d.]+) \| WR ([\d.]+)% \| (\d+) trades"
)
RUNNER_RE = re.compile(
    r"Runners \(excl\. losers\):\s+R100 (\d+) \| R200 (\d+) \| runnerPnL \$([+-]?[\d,]+) \| "
    r"largestWin ([+-]?[\d.]+)% \(\$([+-]?[\d,]+)\) \| maxDD ([\d.]+)%"
)


def parse_log(text: str) -> dict | None:
    h = HEADLINE_RE.search(text)
    if not h:
        return None
    r = RUNNER_RE.search(text)
    out = {
        "pnl": int(h.group(1).replace(",", "")),
        "pf": float(h.group(2)),
        "wr": float(h.group(3)),
        "trades": int(h.group(4)),
    }
    if r:
        out.update({
            "r100": int(r.group(1)),
            "r200": int(r.group(2)),
            "runner_pnl": int(r.group(3).replace(",", "")),
            "largest_win_pct": float(r.group(4)),
            "largest_win_dollars": int(r.group(5).replace(",", "")),
            "max_dd": float(r.group(6)),
        })
    return out


def run(name: str, extra: list[str], window: list[str]) -> dict:
    log_path = OUT / f"{name}.log"
    cmd = [sys.executable, BT, *BASELINE, *window, *extra]
    t0 = time.time()
    with open(log_path, "w") as f:
        f.write("# CMD: " + " ".join(cmd) + "\n")
        f.flush()
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    text = log_path.read_text()
    parsed = parse_log(text)
    rec = {"name": name, "extra": extra, "window": window[1], "elapsed_s": round(elapsed, 1),
           "exit_code": proc.returncode, "metrics": parsed}
    if parsed:
        print(f"[{name:28s}] PnL ${parsed['pnl']:+,} PF {parsed['pf']} WR {parsed['wr']}% "
              f"N {parsed['trades']} R100 {parsed.get('r100','?')} R200 {parsed.get('r200','?')} "
              f"maxDD {parsed.get('max_dd','?')}% ({elapsed:.0f}s)", flush=True)
    else:
        print(f"[{name:28s}] PARSE FAILED exit={proc.returncode} ({elapsed:.0f}s) see {log_path}", flush=True)
    return rec


# TASK 2: one-param-at-a-time sweep grids (in-sample window).
SENSITIVITY = {
    "grace-min": [3, 5, 8],
    "scalp-thresh": [15, 20, 25, 30],
    "soft-keep": [0.5, 0.6, 0.7],
    "adaptive-mult": [0.8, 1.0, 1.2],
    "theta-min": [90, 120, 180, 999],
    "breakeven-trigger": [15, 20, 25],
    "scaleout-trigger": [15, 20, 25],
}


def _run_pool(jobs: list[tuple[str, list[str], list[str]]]) -> dict[str, dict]:
    """jobs = list of (name, extra, window). Returns {name: rec} run with WORKERS concurrency."""
    out: dict[str, dict] = {}
    print(f"Running {len(jobs)} jobs with {WORKERS} workers...", flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(run, n, e, w): n for n, e, w in jobs}
        for fut in as_completed(futs):
            rec = fut.result()
            out[rec["name"]] = rec
    return out


def do_sensitivity() -> dict:
    jobs = [("sens_baseline", [], IN_SAMPLE)]
    for param, values in SENSITIVITY.items():
        for v in values:
            jobs.append((f"sens_{param.replace('-', '_')}_{v}", [f"--{param}", str(v)], IN_SAMPLE))
    recs = _run_pool(jobs)
    results = {"baseline": recs.get("sens_baseline"), "params": {}}
    for param, values in SENSITIVITY.items():
        results["params"][param] = [
            recs[f"sens_{param.replace('-', '_')}_{v}"] for v in values
        ]
    _save(results, "sensitivity_results.json")
    return results


def do_candidates(candidates: dict[str, list[str]]) -> dict:
    jobs = [("cand_baseline_is", [], IN_SAMPLE), ("cand_baseline_oos", [], OOS)]
    for cname, flags in candidates.items():
        jobs.append((f"cand_{cname}_is", flags, IN_SAMPLE))
        jobs.append((f"cand_{cname}_oos", flags, OOS))
    recs = _run_pool(jobs)
    results = {
        "baseline_is": recs.get("cand_baseline_is"),
        "baseline_oos": recs.get("cand_baseline_oos"),
        "candidates": {},
    }
    for cname, flags in candidates.items():
        results["candidates"][cname] = {
            "flags": flags,
            "in_sample": recs[f"cand_{cname}_is"],
            "oos": recs[f"cand_{cname}_oos"],
        }
    _save(results, "candidate_results.json")
    return results


def _save(obj, fname):
    (OUT / fname).write_text(json.dumps(obj, indent=2))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "sensitivity"
    if mode in ("sensitivity", "all"):
        do_sensitivity()
    if mode in ("candidates", "all"):
        # Candidates are filled in by the caller via candidate_config.json if present;
        # otherwise this is a no-op placeholder. See run via do_candidates() import.
        cfg_path = OUT / "candidate_config.json"
        if cfg_path.exists():
            do_candidates(json.loads(cfg_path.read_text()))
        else:
            print("No candidate_config.json — run sensitivity first, then build candidates.")
