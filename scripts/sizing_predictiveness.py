#!/usr/bin/env python3
"""Predictiveness analysis for the position-sizing experiment.

Reads a gold_standard_raw.json produced by backtest_gold_standard.py and
measures whether ML pattern-confidence (and score) predict realized per-trade
return (pnl / cost-basis). This decides whether confidence-scaled sizing has
any edge to exploit. If the relationship is null/non-monotonic, scaling is
pointless and we should instead just fix the backwards 0.80-0.90 tier.

Usage:
    python scripts/sizing_predictiveness.py <raw.json> [--label NAME]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def trade_return_pct(t: dict) -> float:
    """Realized return on cost basis. Prefer the precomputed pnl_pct; else derive."""
    if t.get("pnl_pct") is not None:
        return float(t["pnl_pct"])
    basis = t.get("effective_entry", t.get("entry", 0)) * t.get(
        "effective_contracts", t.get("contracts", 0)) * 100
    return (t["pnl"] / basis * 100) if basis else 0.0


def pearson(x: list[float], y: list[float]) -> float:
    if len(x) < 3:
        return float("nan")
    return float(np.corrcoef(np.array(x), np.array(y))[0, 1])


def spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(np.array(x)))
    ry = np.argsort(np.argsort(np.array(y)))
    return float(np.corrcoef(rx, ry)[0, 1])


def bucketize(trades, key, edges):
    """Return list of (label, count, avg_return, win_rate, avg_conf)."""
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sub = [t for t in trades if lo <= t[key] < hi]
        if not sub:
            rows.append((f"{lo:.2f}-{hi:.2f}", 0, 0.0, 0.0, 0.0))
            continue
        rets = [trade_return_pct(t) for t in sub]
        wins = sum(1 for t in sub if t["pnl"] > 0)
        rows.append((
            f"{lo:.2f}-{hi:.2f}", len(sub),
            float(np.mean(rets)), wins / len(sub) * 100,
            float(np.mean([t[key] for t in sub])),
        ))
    return rows


def analyze(raw_path: Path, label: str) -> str:
    data = json.loads(raw_path.read_text())
    trades = data.get("trade_details", [])
    # Only CALLs use the CALL pattern model confidence; PUTs use the PUT model
    # on a different scale. Analyze separately + combined.
    calls = [t for t in trades if t.get("direction") == "call"]
    puts = [t for t in trades if t.get("direction") == "put"]

    out = []
    out.append(f"## Predictiveness — {label}")
    out.append(f"Source: `{raw_path}`  |  period: {data.get('period')}  |  "
               f"N={len(trades)} ({len(calls)} call / {len(puts)} put)\n")

    for name, ts in [("ALL", trades), ("CALL", calls), ("PUT", puts)]:
        if len(ts) < 5:
            out.append(f"### {name}: too few trades ({len(ts)}) — skipped\n")
            continue
        confs = [t["pattern_conf"] for t in ts]
        scores_present = all("score" in t for t in ts)
        rets = [trade_return_pct(t) for t in ts]
        pnls = [t["pnl"] for t in ts]
        out.append(f"### {name} (N={len(ts)})")
        out.append(f"- Pearson r(conf, return%) = {pearson(confs, rets):+.3f}")
        out.append(f"- Spearman r(conf, return%) = {spearman(confs, rets):+.3f}")
        out.append(f"- Pearson r(conf, pnl$)    = {pearson(confs, pnls):+.3f}")
        out.append("")
        # Confidence buckets (CALL pattern conf typically 0.74-1.0)
        edges = [0.0, 0.74, 0.80, 0.85, 0.90, 1.01]
        rows = bucketize(ts, "pattern_conf", edges)
        out.append("| conf bucket | N | avg return % | win rate % | avg conf |")
        out.append("|---|---|---|---|---|")
        for lbl, n, ar, wr, ac in rows:
            if n == 0:
                continue
            out.append(f"| {lbl} | {n} | {ar:+.1f} | {wr:.1f} | {ac:.3f} |")
        out.append("")
        # Monotonicity check on populated buckets
        pop = [(ar) for lbl, n, ar, wr, ac in rows if n >= 5]
        if len(pop) >= 3:
            mono_up = all(pop[i] <= pop[i + 1] for i in range(len(pop) - 1))
            out.append(f"- Monotonic (return rises with conf across populated buckets, N>=5)? "
                       f"**{'YES' if mono_up else 'NO'}**")
        out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", type=str)
    ap.add_argument("--label", type=str, default="baseline")
    args = ap.parse_args()
    p = Path(args.raw)
    if not p.exists():
        print(f"not found: {p}", file=sys.stderr)
        sys.exit(1)
    print(analyze(p, args.label))


if __name__ == "__main__":
    main()
