#!/usr/bin/env python3
"""Dip-confirm ENTRY TIMING tuning — fast micro-bounce trigger vs current scheme.

Offline analysis only. Reads the prior tick-study CSVs in journal/v3_eval_results/
(option_ticks.csv = ~60s premium snapshots from droplet Postgres, trades_raw.csv =
real executed trades) and simulates the achieved entry (ask) price under:

  (a) NAIVE immediate buy        — ask at first snapshot at/after signal time
  (b) CURRENT dip-confirm        — fade check (>=1% fade -> wait branch), then poll
      two variants:                for first uptick (cur > prev).
        as-coded:                  timeout -> low_water (what live code RECORDS —
                                   unfillable look-ahead; real fill is market price)
        realistic:                 timeout -> last observed ask (actual fill)
  (c) HYBRID MICRO-BOUNCE        — same fade gate (stable/rising -> enter now);
      N=1,2,3 ticks                in the wait branch, enter the INSTANT ask rises
                                   >= N ticks off the running local low since the
                                   fade check. Timeout T -> fallback buy at current
                                   ask (always fills — never misses the trade).
  (d) PURE MICRO-BOUNCE          — no fade gate; bounce trigger from t0 (shown for
                                   completeness; coarse grid punishes rising opens).

CAVEATS (also in the report):
  * Historical premium data is ~60s snapshots. Each "poll" here is ~60s, vs ~5s
    live and sub-second on the WS feed. This validates direction/magnitude only.
  * The coarse grid biases AGAINST the micro-bounce: in live ticks a bounce fires
    within seconds of the low; here it can only fire at the next minute snapshot,
    after up to 60s of adverse drift.
  * low_water timeout entries (as-coded current) are NOT fillable prices.

Tick definition: $0.01 if reference premium < $3.00 else $0.05 (standard option
minimum increments; conservative for penny-pilot names which are $0.01 everywhere).

Does NOT touch runtime code or the DB. Output: stdout summary + per-trade CSV.
"""

import csv
import statistics
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone

BASE = "/Users/kody/dev/options-owl/journal/v3_eval_results"

FADE_PCT = 1.0        # current DIP_CONFIRM_FADE_PCT
WINDOW_S = 300        # premium path horizon after signal
TIMEOUTS = [120, 180]  # micro-bounce fallback timeouts (60s grid -> 1-3 polls)
N_TICKS = [1, 2, 3]


def parse_ts(s: str) -> datetime:
    s = s.strip()
    if s.endswith("+00"):
        s = s + ":00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)  # naive DB timestamps are UTC
    return dt


def fl(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


# ---------------- load option ticks ----------------
optk: dict[tuple, list] = defaultdict(list)
with open(f"{BASE}/option_ticks.csv") as f:
    for row in csv.DictReader(f):
        ts = parse_ts(row["captured_at"])
        key = (row["ticker"], row["option_type"], round(float(row["strike"]), 2),
               row["expiry_date"])
        optk[key].append((ts, fl(row["bid"]), fl(row["ask"]), fl(row["mid"])))
for k in optk:
    optk[k].sort(key=lambda x: x[0])

# ---------------- load trades ----------------
trades = []
with open(f"{BASE}/trades_raw.csv") as f:
    for row in csv.reader(f):
        bot, tid, tk, ot, opened, ep, strike, prem, mfe, pnl, exit_reason, exit_source, contracts = row
        try:
            opened_ts = parse_ts(opened)
        except ValueError:
            continue
        if "orphan" in exit_reason.lower():
            continue
        trades.append(dict(bot=bot, id=tid, ticker=tk, otype=ot, opened=opened_ts,
                           strike=round(float(strike), 2), pnl=fl(pnl) or 0.0,
                           exit_reason=exit_reason))

print(f"Executed trades loaded: {len(trades)}")


def pick_contract(t):
    """Soonest expiry >= trade date with snapshots within +/-10 min of open."""
    candidates = []
    for (tk, ot, st, exp), series in optk.items():
        if tk != t["ticker"] or ot != t["otype"] or st != t["strike"]:
            continue
        exp_d = date.fromisoformat(exp)
        if exp_d < t["opened"].date():
            continue
        lo, hi = t["opened"] - timedelta(minutes=10), t["opened"] + timedelta(minutes=10)
        if any(lo <= x[0] <= hi for x in series):
            candidates.append((exp_d, series))
    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0])
    return candidates[0][1]


def tick_size(ref: float) -> float:
    return 0.01 if ref < 3.0 else 0.05


def micro_bounce(samples, start_low, t0, n, tick, timeout_s):
    """Enter when ask >= running_low + n*tick. samples = [(ts, ask)] after start.

    Returns (entry, fired, t_sec). Fallback: last ask within timeout (always fills).
    """
    run_low = start_low
    inside = []
    for ts_i, a in samples:
        dt_s = (ts_i - t0).total_seconds()
        if dt_s > timeout_s:
            break
        inside.append((dt_s, a))
        if a >= run_low + n * tick:
            return a, 1, dt_s
        run_low = min(run_low, a)
    if inside:
        return inside[-1][1], 0, inside[-1][0]
    return start_low, 0, 0.0


# ---------------- simulation ----------------
results = []
no_data = 0

for t in trades:
    series = pick_contract(t)
    if not series:
        no_data += 1
        continue
    hi = t["opened"] + timedelta(seconds=WINDOW_S)
    asks = [(x[0], x[2]) for x in series
            if t["opened"] <= x[0] <= hi and x[2] is not None and x[2] > 0]
    if len(asks) < 3:
        no_data += 1
        continue
    t0_ts, t0_ask = asks[0]
    if (t0_ts - t["opened"]).total_seconds() > 90:
        no_data += 1
        continue

    later = asks[1:]
    t1_ts, t1 = later[0]
    fade = (t0_ask - t1) / t0_ask * 100
    fading = fade >= FADE_PCT
    min_ask = min(a for _, a in asks)

    rec = dict(bot=t["bot"], id=t["id"], ticker=t["ticker"], otype=t["otype"],
               opened=t["opened"].isoformat(), strike=t["strike"], pnl=t["pnl"],
               n_samples=len(asks), immediate=t0_ask, fading=int(fading),
               min_ask=min_ask, tick=tick_size(t0_ask))

    # ---- (b) current dip-confirm (both timeout variants) ----
    if not fading:
        cur_ascoded = cur_real = min(t1, t0_ask)
        cur_t = (t1_ts - t["opened"]).total_seconds()
    else:
        prev = t1
        low_water = t1
        cur_ascoded = None
        cur_t = None
        for ts_i, a in later[1:7]:           # 6 polls
            low_water = min(low_water, a)
            if a > prev:
                cur_ascoded = a
                cur_t = (ts_i - t["opened"]).total_seconds()
                break
            prev = a
        if cur_ascoded is None:
            cur_ascoded = low_water           # as-coded ML fallback (unfillable)
            cur_real = prev                   # realistic: market price at timeout
            cur_t = (later[min(6, len(later) - 1)][0] - t["opened"]).total_seconds()
        else:
            cur_real = cur_ascoded
    rec["current_ascoded"] = cur_ascoded
    rec["current_real"] = cur_real
    rec["current_t"] = cur_t

    tick = rec["tick"]
    # ---- (c) HYBRID: fade gate + micro-bounce in wait branch ----
    for n in N_TICKS:
        for to in TIMEOUTS:
            if not fading:
                entry, fired, et = min(t1, t0_ask), -1, (t1_ts - t["opened"]).total_seconds()
            else:
                entry, fired, et = micro_bounce(later[1:], t1, t["opened"], n, tick, to)
            rec[f"hyb{n}_{to}"] = entry
            rec[f"hyb{n}_{to}_fired"] = fired   # -1 = stable branch (entered at fade check)
            rec[f"hyb{n}_{to}_t"] = et

    # ---- (d) PURE micro-bounce from t0 (no fade gate) ----
    for n in N_TICKS:
        entry, fired, et = micro_bounce(later, t0_ask, t["opened"], n, tick, 120)
        rec[f"pure{n}_120"] = entry
        rec[f"pure{n}_120_fired"] = fired
        rec[f"pure{n}_120_t"] = et

    results.append(rec)

print(f"Simulated: {len(results)}   no tick coverage: {no_data}")
n_fading = sum(r["fading"] for r in results)
print(f"Fading at first check (>= {FADE_PCT}%): {n_fading}/{len(results)} "
      f"(only these enter the wait branch)")


def pct_vs(a, b):
    return (a - b) / b * 100


def row(label, key, recs, base="immediate", fired_key=None):
    diffs = [pct_vs(r[key], r[base]) for r in recs]
    cheaper = sum(1 for d in diffs if d < -1e-9)
    equal = sum(1 for d in diffs if abs(d) <= 1e-9)
    fired_s = "  —  "
    if fired_key:
        wait = [r for r in recs if r.get(f"{fired_key}") != -1 and r["fading"]]
        if wait:
            f = sum(1 for r in wait if r[fired_key] == 1)
            fired_s = f"{f}/{len(wait)}"
    print(f"  {label:<26} mean={statistics.mean(diffs):+6.2f}%  "
          f"median={statistics.median(diffs):+6.2f}%  "
          f"cheaper/eq/worse={cheaper}/{equal}/{len(diffs)-cheaper-equal}  "
          f"bounce-fired(wait-branch)={fired_s}")


print("\n=== ALL TRADES: entry vs NAIVE IMMEDIATE (negative = cheaper) ===")
row("current (as-coded)", "current_ascoded", results)
row("current (realistic)", "current_real", results)
for n in N_TICKS:
    for to in TIMEOUTS:
        row(f"hybrid N={n} T={to}s", f"hyb{n}_{to}", results, fired_key=f"hyb{n}_{to}_fired")
for n in N_TICKS:
    row(f"pure  N={n} T=120s", f"pure{n}_120", results, fired_key=f"pure{n}_120_fired")

print("\n=== HYBRID vs CURRENT (realistic) — negative = micro-bounce cheaper ===")
for n in N_TICKS:
    for to in TIMEOUTS:
        row(f"hybrid N={n} T={to}s", f"hyb{n}_{to}", results, base="current_real",
            fired_key=f"hyb{n}_{to}_fired")

# ---- wait-branch only (the only place the schemes differ) ----
waiters = [r for r in results if r["fading"]]
print(f"\n=== WAIT-BRANCH ONLY (fading >= 1%, n={len(waiters)}) — vs immediate ===")
row("current (as-coded)", "current_ascoded", waiters)
row("current (realistic)", "current_real", waiters)
for n in N_TICKS:
    for to in TIMEOUTS:
        row(f"hybrid N={n} T={to}s", f"hyb{n}_{to}", waiters, fired_key=f"hyb{n}_{to}_fired")

print("\n=== WAIT-BRANCH: distance from the true in-window low (perfect bottom = 0) ===")
for label, key in [("current (realistic)", "current_real")] + \
        [(f"hybrid N={n} T={to}s", f"hyb{n}_{to}") for n in N_TICKS for to in TIMEOUTS]:
    d = [pct_vs(r[key], r["min_ask"]) for r in waiters]
    if d:
        print(f"  {label:<26} mean={statistics.mean(d):+6.2f}%  median={statistics.median(d):+6.2f}% above the low")

# fire-time distribution for the recommended config
for n in [2]:
    times = [r[f"hyb{n}_120_t"] for r in waiters if r[f"hyb{n}_120_fired"] == 1]
    nofire = [r for r in waiters if r[f"hyb{n}_120_fired"] == 0]
    if times:
        print(f"\nhybrid N={n} T=120s fire times (wait branch): "
              f"median={statistics.median(times):.0f}s  max={max(times):.0f}s  "
              f"no-fire->fallback: {len(nofire)}/{len(waiters)}")

# ---------------- per-trade CSV ----------------
out_path = f"{BASE}/dipconfirm_sim_per_trade.csv"
cols = ["bot", "id", "ticker", "otype", "opened", "strike", "pnl", "n_samples",
        "fading", "tick", "immediate", "min_ask", "current_ascoded", "current_real",
        "current_t"]
for n in N_TICKS:
    for to in TIMEOUTS:
        cols += [f"hyb{n}_{to}", f"hyb{n}_{to}_fired", f"hyb{n}_{to}_t"]
for n in N_TICKS:
    cols += [f"pure{n}_120", f"pure{n}_120_fired", f"pure{n}_120_t"]
with open(out_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in results:
        w.writerow({c: r.get(c) for c in cols})
print(f"\nWrote {out_path}")
