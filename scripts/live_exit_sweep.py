"""Exit-policy SWEEP replayed on REAL live trades and REAL recorded quotes.

Companion to scripts/extract_live_exit_paths.py (which produces the pickle on the droplet).

WHY THIS HARNESS EXISTS
-----------------------
Every previous exit sweep ran on thetadata with a MODELLED fill. Two problems:
  1. thetadata stops at 2026-07-15 — it cannot see the August bleed at all.
  2. modelled fills have repeatedly flipped sweep optima (honest-fill-harness-2026-07-28).

This replays the ACTUAL trades the fleet took, against the ACTUAL bid/ask the harvester
recorded, using the REAL ExitFSM. Entry basis = the real Webull fill. Exit price = the real
recorded BID (what we could actually have sold into), not a haircut approximation.

WHAT IT CANNOT ANSWER (read before trusting a result)
-----------------------------------------------------
The trade SET is fixed. This measures "given the trades we took, what should the exits have
done" — it CANNOT model how a different exit changes which trade you enter next (an earlier
cut frees capital / a slot for a re-entry). So variants that HOLD LONGER are mildly penalised
in reality and variants that CUT EARLIER are mildly flattered. Treat a small edge as noise.

Scoring is reported two ways:
  * REAL      — actual contract counts, i.e. what the book would really have made
  * FLAT $750 — equal-weighted per trade, the pure exit-POLICY edge with sizing removed

Usage:
  python scripts/live_exit_sweep.py --paths journal/live_exit_paths.pkl --family all
  python scripts/live_exit_sweep.py --paths ... --family nevergreen --per-bot
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from options_owl.risk.exit_v5.config import (  # noqa: E402
    apply_v7_wide_trail_exits,
    get_ticker_config,
)
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState  # noqa: E402

ET = ZoneInfo("America/New_York")
FLAT = 750.0


# ── Prod baseline ─────────────────────────────────────────────────────────
# Mirrors the LIVE kody/dennis env as of 2026-08-07. Any drift here silently
# invalidates every comparison, so keep it aligned with docker-compose + settings.py.
def prod_settings(**over) -> SimpleNamespace:
    s = SimpleNamespace(
        ENABLE_V6_BREAKEVEN_RATCHET=True,
        V6_BREAKEVEN_TRIGGER_PCT=20.0,
        ENABLE_V6_SCALEOUT=True,
        V6_SCALEOUT_GAIN_PCT=20.0,
        V6_SCALEOUT_FRACTION=0.333,
        V6_SCALEOUT_MIN_CONTRACTS=3,
        ENABLE_V6_2PM_TIGHTEN=True,
        V6_2PM_TRAIL_TIGHTEN_FACTOR=0.7,
        V6_2PM_SOFT_TRAIL_BOOST=0.15,
        ENABLE_V6_PER_TICKER_CONFIG=True,
        ENABLE_V6_PREMIUM_CAP=False,
        ENABLE_V6_SPREAD_GATE=True,
        ENABLE_V6_EARLY_POP_GATE=True,
        ENABLE_V6_SIDEWAYS_SCALP=False,
        ENABLE_SCALP_TARGET=True,
        SCALP_TARGET_PCT=35.0,
        SCALP_RUNNER_CONFIRM_PCT=40.0,
        ENABLE_V7_PROFIT_LOCK=True,
        V7_PROFIT_LOCK_KEEP_FRAC=0.8,
        V7_PROFIT_LOCK_ACTIVATE_PCT=25.0,
        V7_PROFIT_LOCK_PUTS=True,
        V7_PROFIT_LOCK_PEAK_EXEMPT_PCT=0.0,
        ENABLE_0DTE_PREMIUM_HARDSTOP=True,
        PREMIUM_HARDSTOP_0DTE_PCT=25.0,
        ENABLE_MULTIDAY_CALL_HARDSTOP=True,
        MULTIDAY_CALL_HARDSTOP_PCT=25.0,
        ENABLE_MULTIDAY_PUT_HARDSTOP=True,
        MULTIDAY_PUT_HARDSTOP_PCT=25.0,
        ENABLE_STALL_CUT=True,
        STALL_CUT_MIN_MINUTES=30.0,
        STALL_CUT_LOSS_PCT=30.0,
        STALL_CUT_PEAK_PCT=10.0,
        ENABLE_EARLY_LOCK=True,
        EARLY_LOCK_ARM_PCT=12.0,
        EARLY_LOCK_FLOOR_PCT=3.0,
        ENABLE_EOD_CLOSE_ALL=True,
        ENABLE_NEVERGREEN_CUT=True,
        NEVERGREEN_CUT_LOSS_PCT=8.0,
        NEVERGREEN_MAX_PEAK_PCT=8.0,
        NEVERGREEN_MIN_MINUTES=2.0,
    )
    for k, v in over.items():
        setattr(s, k, v)
    return s


_CFG: dict[tuple, object] = {}


def base_cfg(tk: str, otype: str):
    key = (tk, otype)
    if key not in _CFG:
        _CFG[key] = apply_v7_wide_trail_exits(
            get_ticker_config(tk, use_per_ticker=True, option_type=otype),
            is_put=(otype == "put"),
        )
    return _CFG[key]


def _downsample(path: list[dict], poll_sec: float) -> list[dict]:
    """Thin the tick path to the LIVE monitor cadence.

    Fidelity, not performance. The harvester records far more ticks (~650/trade) than the
    position monitor ever evaluated (one pass per ~5s). Replaying every tick hands the FSM
    many more chances to trip a gate than it had live, which systematically over-fires every
    threshold gate and makes the replay look more trigger-happy (and more negative) than
    reality. Sampling at the real poll cadence removes that artifact.
    """
    if poll_sec <= 0 or not path:
        return path
    out = [path[0]]
    last = path[0]["ts"]
    for p in path[1:]:
        if (p["ts"] - last).total_seconds() >= poll_sec:
            out.append(p)
            last = p["ts"]
    return out


def resim_one(t: dict, settings, poll_sec: float = 5.0) -> dict:
    """Replay one trade's real tick path through the real FSM. Returns exit outcome."""
    path = _downsample(t["path"], poll_sec)
    entry = t["entry"]
    t0 = path[0]["ts"].astimezone(ET)

    fsm = ExitFSM(base_cfg(t["tk"], t["otype"]), settings=settings)
    expiry = t["expiry"]
    dte = max(0, (
        __import__("datetime").date.fromisoformat(expiry) - t0.date()
    ).days)

    st = TradeState(
        trade_id=t["tid"],
        ticker=t["tk"],
        option_type=t["otype"],
        entry_premium=entry,
        entry_time=t0,
        contracts=t["contracts"],
        peak_premium=entry,
        entry_underlying_price=path[0]["up"] or 0.0,
        dte=dte,
        expiry_date=expiry,
    )

    exit_px, reason, held_min = None, None, 0.0
    for p in path[1:]:
        mid = p["mid"]
        bid = p["bid"] if p["bid"] is not None else mid
        ask = p["ask"] if p["ask"] is not None else mid
        now = p["ts"].astimezone(ET)
        # minutes to 16:00 ET
        mtc = max(0.0, (16 * 60) - (now.hour * 60 + now.minute))
        act = fsm.evaluate(
            st, mid, bid, ask, now,
            current_underlying=p["up"] or 0.0,
            minutes_to_close=mtc,
            candle_data={},
        )
        if act.should_exit:
            # We sell into the BID — the real executable price, not the mid.
            exit_px = bid
            reason = getattr(act.reason, "value", str(act.reason))
            held_min = (now - t0).total_seconds() / 60.0
            break

    if exit_px is None:  # never triggered — mark out at the last real bid
        last = path[-1]
        exit_px = last["bid"] if last["bid"] is not None else last["mid"]
        reason = "eod_mark"
        held_min = (last["ts"].astimezone(ET) - t0).total_seconds() / 60.0

    ret_pct = (exit_px - entry) / entry * 100.0
    return {
        "bot": t["bot"],
        "tid": t["tid"],
        "tk": t["tk"],
        "otype": t["otype"],
        "date": t["opened_at"][:10],
        "opened_at": t["opened_at"],
        "ret": ret_pct,
        "real_pnl": (exit_px - entry) * t["contracts"] * 100.0,
        "flat_pnl": FLAT * ret_pct / 100.0,
        "reason": reason,
        "held_min": held_min,
        "contracts": t["contracts"],
    }


def stats(recs: list[dict]) -> dict:
    if not recs:
        return {"n": 0}
    real = sum(r["real_pnl"] for r in recs)
    flat = sum(r["flat_pnl"] for r in recs)
    g = sum(r["flat_pnl"] for r in recs if r["flat_pnl"] > 0)
    loss = -sum(r["flat_pnl"] for r in recs if r["flat_pnl"] < 0)
    pf = (g / loss) if loss > 0 else float("inf")
    wr = 100.0 * sum(1 for r in recs if r["ret"] > 0) / len(recs)
    order = sorted(recs, key=lambda r: r["opened_at"])
    cum = peak = dd = 0.0
    for r in order:
        cum += r["real_pnl"]
        peak = max(peak, cum)
        dd = min(dd, cum - peak)
    return {
        "n": len(recs), "real": real, "flat": flat, "pf": pf, "wr": wr, "dd": dd,
        "avg_hold": sum(r["held_min"] for r in recs) / len(recs),
    }


def variants(family: str):
    """Yield (label, settings). Baseline first — every delta is measured against it."""
    yield ("BASELINE (prod 2026-08-07)", prod_settings())

    if family in ("nevergreen", "all"):
        yield ("nevergreen OFF", prod_settings(ENABLE_NEVERGREEN_CUT=False))
        for lp in (12.0, 15.0, 20.0, 25.0):
            yield (f"nevergreen loss={lp:.0f}%", prod_settings(NEVERGREEN_CUT_LOSS_PCT=lp))
        for mm in (4.0, 6.0, 10.0, 15.0):
            yield (f"nevergreen min_min={mm:.0f}", prod_settings(NEVERGREEN_MIN_MINUTES=mm))
        for pk in (3.0, 5.0, 12.0):
            yield (f"nevergreen max_peak={pk:.0f}%", prod_settings(NEVERGREEN_MAX_PEAK_PCT=pk))
        # the combination the week-1 data pointed at: later + looser
        yield ("nevergreen loss=15 min_min=6",
               prod_settings(NEVERGREEN_CUT_LOSS_PCT=15.0, NEVERGREEN_MIN_MINUTES=6.0))
        yield ("nevergreen loss=20 min_min=10",
               prod_settings(NEVERGREEN_CUT_LOSS_PCT=20.0, NEVERGREEN_MIN_MINUTES=10.0))

    if family in ("stops", "all"):
        for hs in (15.0, 20.0, 30.0, 35.0, 40.0):
            yield (f"0DTE hardstop={hs:.0f}%", prod_settings(PREMIUM_HARDSTOP_0DTE_PCT=hs))
        yield ("0DTE hardstop OFF", prod_settings(ENABLE_0DTE_PREMIUM_HARDSTOP=False))
        for hs in (20.0, 30.0, 35.0):
            yield (f"MD-call hardstop={hs:.0f}%", prod_settings(MULTIDAY_CALL_HARDSTOP_PCT=hs))

    if family in ("lock", "all"):
        for kf in (0.5, 0.6, 0.7, 0.9):
            yield (f"profit-lock keep={kf}", prod_settings(V7_PROFIT_LOCK_KEEP_FRAC=kf))
        for ap in (15.0, 20.0, 35.0, 50.0):
            yield (f"profit-lock arm=+{ap:.0f}%", prod_settings(V7_PROFIT_LOCK_ACTIVATE_PCT=ap))
        yield ("profit-lock OFF", prod_settings(ENABLE_V7_PROFIT_LOCK=False))

    if family in ("earlylock", "all"):
        yield ("early-lock OFF", prod_settings(ENABLE_EARLY_LOCK=False))
        for ap in (8.0, 16.0, 20.0):
            yield (f"early-lock arm=+{ap:.0f}%", prod_settings(EARLY_LOCK_ARM_PCT=ap))
        for fp in (0.0, 6.0):
            yield (f"early-lock floor=+{fp:.0f}%", prod_settings(EARLY_LOCK_FLOOR_PCT=fp))

    if family in ("scalp", "all"):
        yield ("scalp-target OFF", prod_settings(ENABLE_SCALP_TARGET=False))
        for sp in (25.0, 50.0, 75.0):
            yield (f"scalp-target={sp:.0f}%", prod_settings(SCALP_TARGET_PCT=sp))

    if family in ("stall", "all"):
        yield ("stall-cut OFF", prod_settings(ENABLE_STALL_CUT=False))

    if family in ("scaleout", "all"):
        yield ("scaleout OFF", prod_settings(ENABLE_V6_SCALEOUT=False))
        yield ("breakeven-ratchet OFF", prod_settings(ENABLE_V6_BREAKEVEN_RATCHET=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="journal/live_exit_paths.pkl")
    ap.add_argument("--family", default="all",
                    choices=["all", "nevergreen", "stops", "lock", "earlylock",
                             "scalp", "stall", "scaleout"])
    ap.add_argument("--bots", default=None, help="Restrict to these bots (comma-sep)")
    ap.add_argument("--live-only", action="store_true",
                    help="Only trades that really hit Webull (real fills)")
    ap.add_argument("--per-bot", action="store_true")
    ap.add_argument("--poll-sec", type=float, default=5.0,
                    help="Downsample paths to the live monitor cadence (0=every tick)")
    ap.add_argument("--monthly", action="store_true",
                    help="Per-month delta vs baseline — a one-month-only win is variance")
    ap.add_argument("--since", default=None, help="Only trades opened on/after YYYY-MM-DD")
    ap.add_argument("--otype", default=None, choices=["call", "put"],
                    help="Restrict to one side. PUTs are DISABLED live (ENABLE_PUT_TRADING="
                         "false), so --otype call is the book that actually trades today.")
    args = ap.parse_args()

    with open(args.paths, "rb") as f:
        trades = pickle.load(f)

    if args.otype:
        trades = [t for t in trades if t["otype"] == args.otype]

    if args.bots:
        keep = {b.strip() for b in args.bots.split(",")}
        trades = [t for t in trades if t["bot"] in keep]
    if args.live_only:
        trades = [t for t in trades if t["was_webull"]]
    if args.since:
        trades = [t for t in trades if t["opened_at"][:10] >= args.since]

    if not trades:
        print("no trades after filters")
        return

    ds = sorted(t["opened_at"][:10] for t in trades)
    print(f"Replaying {len(trades)} real trades  {ds[0]} -> {ds[-1]}")
    bots = sorted({t["bot"] for t in trades})
    print(f"bots: {', '.join(bots)}   calls={sum(1 for t in trades if t['otype']=='call')} "
          f"puts={sum(1 for t in trades if t['otype']=='put')}")
    print()

    base_recs = None
    rows = []
    for label, s in variants(args.family):
        recs = [resim_one(t, s, args.poll_sec) for t in trades]
        st = stats(recs)
        if base_recs is None:
            base_recs = recs
            base = st
        rows.append((label, st, recs))

    print(f"{'variant':<34} {'REAL $':>10} {'vs base':>10} {'FLAT $':>10} "
          f"{'PF':>6} {'WR%':>6} {'maxDD':>10} {'hold':>7}")
    print("-" * 100)
    for label, st, _ in rows:
        d = st["real"] - base["real"]
        mark = "  <<<" if d > 0.01 * abs(base["real"]) and label != rows[0][0] else ""
        print(f"{label:<34} {st['real']:>10,.0f} {d:>+10,.0f} {st['flat']:>10,.0f} "
              f"{st['pf']:>6.2f} {st['wr']:>6.1f} {st['dd']:>10,.0f} "
              f"{st['avg_hold']:>6.1f}m{mark}")

    # Exit-reason mix for the baseline — where the P&L actually comes from.
    print("\nBASELINE exit-reason mix:")
    mix: dict[str, list] = {}
    for r in base_recs:
        mix.setdefault(r["reason"], []).append(r)
    for reason, rs in sorted(mix.items(), key=lambda kv: -len(kv[1])):
        tot = sum(x["real_pnl"] for x in rs)
        hold = sum(x["held_min"] for x in rs) / len(rs)
        print(f"  {reason:<24} n={len(rs):>4}  real=${tot:>10,.0f}  avg_hold={hold:>6.1f}m")

    if args.per_bot:
        print("\nper-bot (baseline):")
        for b in bots:
            st = stats([r for r in base_recs if r["bot"] == b])
            print(f"  {b:<8} n={st['n']:>4} real=${st['real']:>10,.0f} "
                  f"PF={st['pf']:.2f} WR={st['wr']:.0f}%")

    if args.monthly:
        # Robustness: a variant that only wins in one month is variance, not edge.
        # (This is the test that killed the GEX-regime idea — May inverted.)
        months = sorted({r["date"][:7] for r in base_recs})
        base_by_m = {m: sum(r["real_pnl"] for r in base_recs if r["date"][:7] == m)
                     for m in months}
        print(f"\nMONTHLY ROBUSTNESS (delta vs baseline, REAL $)   months={len(months)}")
        hdr = "  ".join(f"{m:>9}" for m in months)
        print(f"{'variant':<34} {hdr}  {'wins':>5}")
        print("-" * (36 + 11 * len(months) + 7))
        print(f"{'BASELINE (abs)':<34} " +
              "  ".join(f"{base_by_m[m]:>9,.0f}" for m in months))
        for label, _st, recs in rows[1:]:
            by_m = {m: sum(r["real_pnl"] for r in recs if r["date"][:7] == m)
                    for m in months}
            deltas = [by_m[m] - base_by_m[m] for m in months]
            wins = sum(1 for d in deltas if d > 0)
            cells = "  ".join(f"{d:>+9,.0f}" for d in deltas)
            flag = "  <<<" if wins == len(months) else ""
            print(f"{label:<34} {cells}  {wins}/{len(months)}{flag}")

    if args.per_bot and args.monthly:
        print("\nper-bot delta for the leading variants (REAL $):")
        for label, _st, recs in rows[1:]:
            if abs(_st["real"] - base["real"]) < 500:
                continue
            per = []
            for b in bots:
                d = (sum(r["real_pnl"] for r in recs if r["bot"] == b)
                     - sum(r["real_pnl"] for r in base_recs if r["bot"] == b))
                per.append(f"{b}={d:>+7,.0f}")
            print(f"  {label:<32} " + "  ".join(per))


if __name__ == "__main__":
    main()
