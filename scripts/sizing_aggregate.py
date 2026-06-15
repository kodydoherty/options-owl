#!/usr/bin/env python3
"""Aggregate sizing-sweep raw JSONs into a risk-adjusted comparison table.

For each <window>_<config>_raw.json in journal/v3_eval_results/sizing_sweep/,
compute P&L, PF, Sharpe, max-DD (from the JSON) plus position-size stats
(avg / 95th-pct cost basis = effective_entry * effective_contracts * 100) and
runner counts. Emits a markdown table to stdout.

Usage:
    python scripts/sizing_aggregate.py --window is [--configs a,b,c]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "journal" / "v3_eval_results" / "sizing_sweep"

ORDER = [
    "baseline", "flat85",
    "conf_lin_60_120", "conf_lin_70_110", "conf_lin_85_115",
    "flat_mdcap2", "flat_mdcap4", "flat_mdcap6", "flat_mdcap_off",
    "baseline_mdcap2",
]


def cost_basis(t: dict) -> float:
    return t.get("effective_entry", t.get("entry", 0)) * t.get(
        "effective_contracts", t.get("contracts", 0)) * 100


def stats(window: str, names: list[str]) -> list[dict]:
    rows = []
    for name in names:
        p = OUTDIR / f"{window}_{name}_raw.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        t = d["trade_details"]
        bases = [cost_basis(x) for x in t]
        contracts = [x.get("effective_contracts", x.get("contracts", 0)) for x in t]
        runners = sum(1 for x in t if x.get("pnl_pct", 0) >= 100)
        rows.append({
            "config": name,
            "pnl": d["total_pnl"],
            "pf": d["profit_factor"],
            "sharpe": d["sharpe"],
            "maxdd": d["max_drawdown_pct"],
            "wr": d["win_rate"],
            "n": d["trades"],
            "avg_size": float(np.mean(bases)) if bases else 0,
            "p95_size": float(np.percentile(bases, 95)) if bases else 0,
            "max_size": float(np.max(bases)) if bases else 0,
            "avg_ct": float(np.mean(contracts)) if contracts else 0,
            "runners": runners,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", choices=["is", "oos"], required=True)
    ap.add_argument("--configs", type=str, default=None)
    args = ap.parse_args()
    names = args.configs.split(",") if args.configs else ORDER
    rows = stats(args.window, names)
    base = next((r for r in rows if r["config"] in ("baseline", "baseline_mdcap2")), None)

    print(f"### {args.window.upper()} sizing sweep")
    print("| config | P&L | PF | Sharpe | maxDD% | WR% | N | avg size$ | p95 size$ | max size$ | avg ct | R100 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['config']} | ${r['pnl']:+,.0f} | {r['pf']:.2f} | {r['sharpe']:.2f} | "
              f"{r['maxdd']:.1f} | {r['wr']:.0f} | {r['n']} | ${r['avg_size']:,.0f} | "
              f"${r['p95_size']:,.0f} | ${r['max_size']:,.0f} | {r['avg_ct']:.1f} | {r['runners']} |")
    if base:
        print(f"\n_baseline = {base['config']}: P&L ${base['pnl']:+,.0f}, Sharpe {base['sharpe']:.2f}, "
              f"maxDD {base['maxdd']:.1f}%, avg size ${base['avg_size']:,.0f}_")


if __name__ == "__main__":
    main()
