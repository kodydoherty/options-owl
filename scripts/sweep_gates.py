"""Gate ablation + range sweep runner for the gold-standard backtest.

Runs `scripts/backtest_gold_standard.py` across a matrix of gate
toggles/ranges, parses the machine-readable METRIC block each run prints,
captures each run's full stdout to its own file, and writes a single ranked
markdown + CSV table to journal/v3_eval_results/gate_sweep_results.md.

Every run uses the fixed base flags:
    --start 2026-02-01 --end 2026-05-20 --puts --no-regime <gate flags>

The window has UW GEX coverage and is OOS for the CALL pattern model, but
partly IN-SAMPLE for the regime / signal_quality / PUT models — so treat
absolute $ as optimistic and weight RELATIVE comparisons. (--no-regime keeps
the miscalibrated ML regime day-skip from dominating; the rule-based
directional_regime gate is still toggled as one of the ablations.)

Usage:
    python scripts/sweep_gates.py            # full matrix
    python scripts/sweep_gates.py --smoke    # tiny 5-day smoke (3 configs)
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
BACKTEST = PROJECT_DIR / "scripts" / "backtest_gold_standard.py"
OUT_DIR = PROJECT_DIR / "journal" / "v3_eval_results"
RUN_LOG_DIR = OUT_DIR / "sweep_runs"

START = "2026-02-01"
END = "2026-05-20"

# Base flags shared by every run. Regime ML OFF (miscalibrated day-skip).
BASE_FLAGS = ["--start", START, "--end", END, "--puts", "--no-regime"]


# ── Matrix definition ────────────────────────────────────────────────────────
# Each entry: (config_name, [extra flags on top of BASE_FLAGS])
def build_matrix() -> list[tuple[str, list[str]]]:
    matrix: list[tuple[str, list[str]]] = []

    # 1. Baseline — all gates at production defaults
    matrix.append(("baseline", []))

    # 2. Single-gate-OFF ablations (one at a time)
    matrix.append(("off_anti_chase", ["--gate-anti-chase", "off"]))
    matrix.append(("off_momentum", ["--gate-momentum", "off"]))
    matrix.append(("off_consecutive_loser", ["--gate-consecutive-loser", "off"]))
    matrix.append(("off_correlation_cap", ["--gate-correlation-cap", "off"]))
    matrix.append(("off_directional_regime", ["--gate-directional-regime", "off"]))
    matrix.append(("off_put_bearish", ["--gate-put-bearish", "off"]))

    # 3. Range sweeps
    for mp in (0.10, 0.15, 0.20):
        matrix.append((f"min_premium_{mp}", ["--min-premium", str(mp)]))
    for sf in (70, 72, 75, 78):
        matrix.append((f"score_floor_{sf}", ["--score-floor", str(sf)]))
    for df in (0.08, 0.12, 0.15):
        matrix.append((f"delta_floor_{df}", ["--delta-floor", str(df)]))
    for tb in (0, 5, 10):
        matrix.append((f"tod_buffer_{tb}", ["--tod-buffer-min", str(tb)]))

    # 4. Combined "relaxed" configs stacking the most promising relaxations.
    # These aim for 3-5 trades/day. (Final picks may be revised in analysis —
    # they stack the individually-cheapest gates + lower floors.)
    matrix.append((
        "relaxed_A_floors",
        ["--score-floor", "72", "--min-premium", "0.10", "--delta-floor", "0.12"],
    ))
    matrix.append((
        "relaxed_B_volume",
        ["--gate-anti-chase", "off", "--gate-momentum", "off",
         "--score-floor", "72", "--min-premium", "0.10"],
    ))
    matrix.append((
        "relaxed_C_aggressive",
        ["--gate-anti-chase", "off", "--gate-momentum", "off",
         "--gate-consecutive-loser", "off", "--score-floor", "72",
         "--min-premium", "0.10", "--delta-floor", "0.12", "--tod-buffer-min", "5"],
    ))

    return matrix


METRIC_RE = re.compile(r"^\s*METRIC\s+(\w+)\s+(\S+)\s*$")


def parse_metrics(stdout: str) -> dict:
    """Extract the METRIC lines from a run's stdout into a dict."""
    metrics: dict = {}
    in_block = False
    for line in stdout.splitlines():
        if "=== SWEEP_METRICS ===" in line:
            in_block = True
            continue
        if "=== END_SWEEP_METRICS ===" in line:
            in_block = False
            continue
        if in_block:
            m = METRIC_RE.match(line)
            if m:
                key, val = m.group(1), m.group(2)
                try:
                    metrics[key] = float(val)
                except ValueError:
                    metrics[key] = val
    return metrics


def run_config(name: str, extra_flags: list[str], smoke: bool) -> dict:
    """Run one backtest config, capture stdout to file, return parsed metrics."""
    flags = list(BASE_FLAGS)
    if smoke:
        # Replace the fixed window with a tiny 5-day window for smoke runs.
        flags = ["--days", "5", "--puts", "--no-regime"]
    flags = flags + extra_flags

    cmd = [sys.executable, str(BACKTEST), *flags]
    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RUN_LOG_DIR / f"{name}.log"

    print(f"  Running {name}: {' '.join(extra_flags) or '(defaults)'}")
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    stdout = proc.stdout
    log_path.write_text(
        f"# config: {name}\n# cmd: {' '.join(cmd)}\n"
        f"# elapsed: {elapsed:.0f}s | returncode: {proc.returncode}\n\n"
        + stdout
        + ("\n\n=== STDERR ===\n" + proc.stderr if proc.stderr else "")
    )

    metrics = parse_metrics(stdout)
    metrics["config"] = name
    metrics["flags"] = " ".join(extra_flags) or "(defaults)"
    metrics["elapsed_s"] = round(elapsed, 0)
    metrics["returncode"] = proc.returncode
    metrics["log"] = str(log_path.relative_to(PROJECT_DIR))

    if proc.returncode != 0 or "trades" not in metrics:
        print(f"    WARNING: {name} returncode={proc.returncode}, "
              f"metrics_parsed={'trades' in metrics}")
    else:
        print(f"    {name}: trades={metrics.get('trades')} "
              f"PF={metrics.get('profit_factor')} "
              f"P&L=${metrics.get('total_pnl', 0):+,.0f} "
              f"runners={metrics.get('runners_100')} ({elapsed:.0f}s)")
    return metrics


# ── Reporting ─────────────────────────────────────────────────────────────────

# Score for ranking: prioritize a balanced, profitable config that still hits
# ~3-5 trades/day and preserves runner capture. We rank primarily by total P&L
# (relative comparison), but the table surfaces everything needed to judge.
def rank_key(m: dict) -> float:
    if "trades" not in m:
        return float("-inf")
    return float(m.get("total_pnl", float("-inf")))


def write_report(results: list[dict], smoke: bool) -> Path:
    ranked = sorted(results, key=rank_key, reverse=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    md_path = OUT_DIR / ("gate_sweep_results_smoke.md" if smoke else "gate_sweep_results.md")
    csv_path = OUT_DIR / ("gate_sweep_results_smoke.csv" if smoke else "gate_sweep_results.csv")

    cols = [
        "config", "trades", "trades_per_day", "win_rate", "profit_factor",
        "total_pnl", "max_drawdown_pct", "runners_100", "big_runners_200",
        "runner_pnl", "largest_winner_pct", "largest_winner_dollars", "flags",
    ]

    # CSV
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for m in ranked:
            w.writerow([m.get(c, "") for c in cols])

    # Markdown
    lines = []
    lines.append("# Gate Ablation + Range Sweep — Gold Standard Backtest")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"\nWindow: {START} → {END}  |  Flags: `--puts --no-regime`  "
                 f"(ML regime day-skip OFF; rule-based directional_regime toggled in matrix)")
    lines.append("\n**Caveat:** This window is partly IN-SAMPLE for the regime / "
                 "signal_quality / PUT models (the CALL pattern model is OOS). Treat "
                 "absolute $ as OPTIMISTIC and weight the RELATIVE comparisons between "
                 "configs.\n")
    lines.append("Goal: ~3-5 solid profitable trades/day while preserving runner "
                 "capture (do NOT optimize trade count at the expense of runners).\n")
    lines.append("Ranked by total P&L (descending). Per-run logs: "
                 f"`{RUN_LOG_DIR.relative_to(PROJECT_DIR)}/`\n")

    hdr = ("| Rank | Config | Trades | T/Day | WR% | PF | Total P&L | MaxDD% | "
           "Runners≥100% | Big≥200% | Runner P&L | Largest Win | Flags |")
    sep = "|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    lines.append(hdr)
    lines.append(sep)
    for i, m in enumerate(ranked, 1):
        if "trades" not in m:
            lines.append(f"| {i} | {m['config']} | FAILED (rc={m.get('returncode')}) "
                         f"| | | | | | | | | | `{m.get('flags', '')}` |")
            continue
        lw = (f"{m.get('largest_winner_pct', 0):+.0f}% "
              f"(${m.get('largest_winner_dollars', 0):+,.0f})")
        lines.append(
            f"| {i} | {m['config']} | {int(m.get('trades', 0))} "
            f"| {m.get('trades_per_day', 0)} | {m.get('win_rate', 0)} "
            f"| {m.get('profit_factor', 0)} | ${m.get('total_pnl', 0):+,.0f} "
            f"| {m.get('max_drawdown_pct', 0)} | {int(m.get('runners_100', 0))} "
            f"| {int(m.get('big_runners_200', 0))} | ${m.get('runner_pnl', 0):+,.0f} "
            f"| {lw} | `{m.get('flags', '')}` |"
        )

    # ── Data-driven ANALYSIS section ──
    by_name = {m["config"]: m for m in results if "trades" in m}
    base = by_name.get("baseline")
    if base:
        lines.append("\n## Analysis\n")
        lines.append(f"Baseline (all gates at prod defaults): **{int(base['trades'])} trades** "
                     f"({base.get('trades_per_day')}/day), PF **{base.get('profit_factor')}**, "
                     f"P&L **${base.get('total_pnl', 0):+,.0f}**, "
                     f"**{int(base.get('runners_100', 0))}** runners(>=+100%).\n")

        # Per single-gate-OFF delta vs baseline
        lines.append("### Single-gate-OFF impact (vs baseline)\n")
        lines.append("| Gate turned OFF | dTrades | dP&L | dRunners | PF | Verdict |")
        lines.append("|---|---|---|---|---|---|")
        ablations = [
            ("anti_chase", "off_anti_chase"),
            ("momentum", "off_momentum"),
            ("consecutive_loser", "off_consecutive_loser"),
            ("correlation_cap", "off_correlation_cap"),
            ("directional_regime", "off_directional_regime"),
            ("put_bearish", "off_put_bearish"),
        ]
        for label, name in ablations:
            m = by_name.get(name)
            if not m:
                continue
            dt = int(m["trades"] - base["trades"])
            dp = m.get("total_pnl", 0) - base.get("total_pnl", 0)
            dr = int(m.get("runners_100", 0) - base.get("runners_100", 0))
            if dp > 5000 and dr >= 0:
                verdict = "RELAX — adds P&L + volume"
            elif dt > 5 and abs(dp) < 5000:
                verdict = "neutral P&L, adds volume"
            elif dp < -2000:
                verdict = "KEEP — gate is net-positive"
            else:
                verdict = "no-op / negligible"
            lines.append(f"| {label} | {dt:+d} | ${dp:+,.0f} | {dr:+d} "
                         f"| {m.get('profit_factor')} | {verdict} |")

        # Recommended = best total P&L that still has PF >= 1.5 and >= base runners
        candidates = [
            m for m in results if "trades" in m
            and m.get("profit_factor", 0) >= 1.5
            and m.get("runners_100", 0) >= base.get("runners_100", 0)
            and m.get("trades_per_day", 0) >= 3.0
        ]
        rec = max(candidates, key=lambda m: m.get("total_pnl", 0)) if candidates else None
        lines.append("\n### Recommended config\n")
        if rec:
            lines.append(
                f"**`{rec['config']}`** — flags: `{rec.get('flags')}`\n\n"
                f"- Trades: **{int(rec['trades'])}** ({rec.get('trades_per_day')}/day) "
                f"— hits the 3-5/day goal\n"
                f"- PF: **{rec.get('profit_factor')}** (> 1.5 floor preserved)\n"
                f"- P&L: **${rec.get('total_pnl', 0):+,.0f}** "
                f"(vs baseline ${base.get('total_pnl', 0):+,.0f})\n"
                f"- Runners >=+100%: **{int(rec.get('runners_100', 0))}** "
                f"(vs baseline {int(base.get('runners_100', 0))}) — runner capture preserved\n"
                f"- Max DD: {rec.get('max_drawdown_pct')}%\n"
            )
        else:
            lines.append("No config met PF>=1.5 + runners>=baseline + >=3 trades/day.\n")
        lines.append(
            "\n**Caveat:** absolute $ is optimistic (window partly in-sample for "
            "regime/signal_quality/PUT models; CALL model OOS). Weight the relative "
            "ordering: the *cheapest* gates to relax (most P&L + runners per trade added) "
            "win regardless of absolute scale.\n"
        )

    md_path.write_text("\n".join(lines) + "\n")
    return md_path


def main():
    ap = argparse.ArgumentParser(description="Gate ablation + range sweep runner")
    ap.add_argument("--smoke", action="store_true",
                    help="Tiny 5-day smoke test of 3 configs to verify threading")
    ap.add_argument("--report-only", action="store_true",
                    help="Skip running; rebuild the report from existing run logs in sweep_runs/")
    args = ap.parse_args()

    matrix = build_matrix()
    if args.smoke:
        matrix = [matrix[0], matrix[1], matrix[-1]]  # baseline, off_anti_chase, relaxed_C

    if args.report_only:
        # Rebuild report (incl. analysis) by re-parsing existing per-run logs.
        results = []
        for name, _extra in matrix:
            log_path = RUN_LOG_DIR / f"{name}.log"
            if not log_path.exists():
                continue
            text = log_path.read_text()
            m = parse_metrics(text)
            m["config"] = name
            m["flags"] = " ".join(_extra) or "(defaults)"
            m["returncode"] = 0
            results.append(m)
        md_path = write_report(results, args.smoke)
        print(f"Report rebuilt from {len(results)} run logs: {md_path}")
        return

    print("=" * 70)
    print(f"GATE SWEEP — {len(matrix)} configs"
          + (" (SMOKE)" if args.smoke else f"  window {START}..{END}"))
    print("=" * 70)

    results = []
    t0 = time.time()
    for name, extra in matrix:
        results.append(run_config(name, extra, args.smoke))

    md_path = write_report(results, args.smoke)
    print("=" * 70)
    print(f"Done in {time.time() - t0:.0f}s. Report: {md_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
