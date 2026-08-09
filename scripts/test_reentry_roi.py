"""Does re-entering after a losing exit actually pay — on CAPITAL-NORMALISED ROI?

THE IDEA (Kody's, 2026-08-07)
-----------------------------
We cut on a dip, and the contract then runs. Live fills show that after a losing exit,
61-75% of those same contracts go on to reach +25% from an honest re-entry at the ask,
in 3/3 months. That opportunity is real and month-robust.

WHY IT IS NOT CURRENTLY POSSIBLE
--------------------------------
Two independent blockers:
  1. SIGNAL: ml_pipeline TickerScanState.entry_emitted is one-shot per ticker per DAY,
     so the scan never re-emits after the position closes. (The RISK pipeline already
     allows it — DuplicateTickerGate only checks status='open'.)
  2. CAPITAL: on a CASH account the GFV guard caps total daily buying at ~85% of
     START-OF-DAY balance and explicitly excludes unsettled sale proceeds. Re-entry
     needs ~1.7x turnover, which cash cannot do. MARGIN_ACCOUNT=true removes this.

WHY RAW P&L IS THE WRONG TEST
-----------------------------
The 2026-07-02 discount-reentry idea looked like +82% raw and was REFUTED once capital
was normalised: it deployed 2.6x the capital for an ROI of 8.4% vs 12.1% for simply
letting trades run. A re-entry strategy manufactures P&L by deploying more money; the
only honest score is return per dollar-day deployed, under a real concurrency cap.

So this scores:
  * ROI  = P&L / capital-days deployed   (the decisive number)
  * raw P&L                              (shown, to expose the mirage)
  * both under CASH (start-of-day cap) and MARGIN (intraday recycling)
  * MAX_CONCURRENT slots enforced, so a re-entry can BLOCK a fresh signal

Usage:
  python scripts/test_reentry_roi.py --paths journal/live_exit_paths_greeks.pkl
"""

from __future__ import annotations

import argparse
import pickle
import sys
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from live_exit_sweep import ET, _downsample, base_cfg, prod_settings  # noqa: E402

from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState  # noqa: E402


def _p(s: str) -> datetime:
    d = datetime.fromisoformat(s.replace(" ", "T", 1))
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def run_leg(t: dict, start_ts, entry_px: float, contracts: int, settings):
    """Run the REAL ExitFSM from `start_ts` at `entry_px`. Returns (pnl, exit_ts, exit_px)."""
    path = [q for q in _downsample(t["path"], 5.0) if q["ts"] >= start_ts]
    if len(path) < 3 or entry_px <= 0:
        return None
    t0 = path[0]["ts"].astimezone(ET)
    st = TradeState(
        trade_id=t["tid"], ticker=t["tk"], option_type="call", entry_premium=entry_px,
        entry_time=t0, contracts=contracts, peak_premium=entry_px,
        entry_underlying_price=path[0]["up"] or 0.0,
        dte=max(0, (date.fromisoformat(t["expiry"]) - t0.date()).days),
        expiry_date=t["expiry"],
    )
    fsm = ExitFSM(base_cfg(t["tk"], "call"), settings=settings)
    for q in path[1:]:
        mid = q["mid"]
        bid = q["bid"] or mid
        ask = q["ask"] or mid
        now = q["ts"].astimezone(ET)
        act = fsm.evaluate(st, mid, bid, ask, now, current_underlying=q["up"] or 0.0,
                           minutes_to_close=max(0.0, 960 - (now.hour * 60 + now.minute)),
                           candle_data={})
        if act.should_exit:
            return ((bid - entry_px) * contracts * 100, q["ts"], bid)
    last = path[-1]
    px = last["bid"] or last["mid"]
    return ((px - entry_px) * contracts * 100, last["ts"], px)


def simulate(trades, *, allow_reentry: bool, margin: bool, max_concurrent: int,
             start_balance: float):
    """Chronological walk with slot + capital constraints. Returns metrics."""
    settings = prod_settings()
    legs = []
    for t in trades:
        legs.append({"ts": _p(t["opened_at"]), "t": t, "kind": "orig"})
    legs.sort(key=lambda x: x["ts"])

    open_slots: list = []           # (exit_ts, freed_capital)
    pnl = 0.0
    cap_days = 0.0                  # dollar-days deployed  -> the ROI denominator
    spent_today = 0.0
    cur_day = None
    blocked_slots = 0
    reentries = 0

    queue = list(legs)
    while queue:
        leg = queue.pop(0)
        t, ts = leg["t"], leg["ts"]
        day = ts.astimezone(ET).date()
        if day != cur_day:
            cur_day, spent_today = day, 0.0
        open_slots = [s for s in open_slots if s[0] > ts]
        if len(open_slots) >= max_concurrent:
            blocked_slots += 1
            continue

        if leg["kind"] == "orig":
            entry_px, contracts = t["entry"], t["contracts"]
        else:
            fut = [q for q in t["path"] if q["ts"] >= ts]
            if not fut:
                continue
            entry_px = fut[0].get("ask") or fut[0]["mid"]   # honest: pay the ask
            contracts = t["contracts"]
            if not entry_px or entry_px <= 0:
                continue

        cost = entry_px * contracts * 100
        # CASH: total daily buying capped at start-of-day (unsettled proceeds unusable).
        # MARGIN: proceeds recycle, so only the concurrency cap binds.
        if not margin and spent_today + cost > start_balance:
            continue
        spent_today += cost

        res = run_leg(t, ts, entry_px, contracts, settings)
        if res is None:
            continue
        leg_pnl, exit_ts, _ = res
        pnl += leg_pnl
        held_days = max((exit_ts - ts).total_seconds() / 86400.0, 1 / 390.0)
        cap_days += cost * held_days
        open_slots.append((exit_ts, cost))
        if leg["kind"] == "reentry":
            reentries += 1

        if allow_reentry and leg["kind"] == "orig" and leg_pnl < 0:
            queue.append({"ts": exit_ts, "t": t, "kind": "reentry"})
            queue.sort(key=lambda x: x["ts"])

    roi = (pnl / cap_days * 100) if cap_days > 0 else float("nan")
    return {"pnl": pnl, "cap_days": cap_days, "roi": roi,
            "reentries": reentries, "blocked": blocked_slots}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", required=True)
    ap.add_argument("--max-concurrent", type=int, default=8)
    ap.add_argument("--start-balance", type=float, default=21000.0)
    args = ap.parse_args()

    trades = [t for t in pickle.load(open(args.paths, "rb"))
              if t["was_webull"] and t["otype"] == "call" and t.get("opened_at")]
    print(f"{len(trades)} real-fill CALLs | MAX_CONCURRENT={args.max_concurrent} "
          f"| start balance ${args.start_balance:,.0f}\n")

    print(f"{'scenario':<34} {'P&L':>10} {'cap-days':>12} {'ROI %':>8} {'re-ent':>7} {'blocked':>8}")
    print("-" * 84)
    rows = {}
    for label, re_, mg in (
        ("baseline (no re-entry, CASH)", False, False),
        ("baseline (no re-entry, MARGIN)", False, True),
        ("RE-ENTRY on CASH", True, False),
        ("RE-ENTRY on MARGIN", True, True),
    ):
        r = simulate(trades, allow_reentry=re_, margin=mg,
                     max_concurrent=args.max_concurrent, start_balance=args.start_balance)
        rows[label] = r
        print(f"{label:<34} {r['pnl']:>10,.0f} {r['cap_days']:>12,.0f} "
              f"{r['roi']:>7.2f}% {r['reentries']:>7} {r['blocked']:>8}")

    b = rows["baseline (no re-entry, MARGIN)"]
    x = rows["RE-ENTRY on MARGIN"]
    print("\nVERDICT (margin, the only config where re-entry is possible):")
    print(f"  raw P&L   {b['pnl']:>+10,.0f} -> {x['pnl']:>+10,.0f}   ({x['pnl']-b['pnl']:+,.0f})")
    print(f"  ROI       {b['roi']:>10.2f}% -> {x['roi']:>10.2f}%   ({x['roi']-b['roi']:+.2f} pts)")
    if x["roi"] > b["roi"]:
        print("  -> ROI IMPROVES: the extra P&L is not merely bought with extra capital.")
    else:
        print("  -> ROI DEGRADES: raw P&L rises only because more capital is deployed —")
        print("     the same 'leverage mirage' that refuted discount-reentry on 2026-07-02.")


if __name__ == "__main__":
    main()
