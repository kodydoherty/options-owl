#!/usr/bin/env python3
"""v5 scalp_and_hold backtest — clean, single-strategy, with ML feature analysis.

Goals:
  - Consistent 15-25% daily gains, not max profit
  - Let runners offset losses
  - Identify patterns for ML-based decision making
  - Analyze candle/underlying data for smarter exits
"""

import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from math import sqrt

SLIPPAGE = 0.15
PORTFOLIO = 8000

HARVESTER_DB = sys.argv[2] if len(sys.argv) > 2 else "journal/owlet-harvester/options_data.db"
SIGNALS_DB = sys.argv[1] if len(sys.argv) > 1 else "journal/owlet-kody/raw_messages.db"


# ---------------------------------------------------------------------------
# v5 scalp_and_hold — single clean implementation
# ---------------------------------------------------------------------------

def v5_simulate(entry, ticks, sig_ts, contracts, direction):
    """Pure v5 strategy. Returns (pnl, reason, hold_min, details_dict)."""
    if not ticks or entry <= 0:
        return 0, "no_data", 0, {}

    is_call = direction.lower() in ("call", "bullish", "long")
    peak = entry
    entry_underlying = None
    checkpoint_done = False

    # ML feature collection
    prices_5m = []      # price every ~5min for trajectory shape
    underlying_5m = []  # underlying every ~5min
    first_price = None
    vol_at_entry = None

    for i, tick in enumerate(ticks):
        ts, mid, bid, ask, underlying = tick
        ts_dt = datetime.fromisoformat(ts) if isinstance(ts, str) else ts
        if ts_dt.tzinfo is None:
            ts_dt = ts_dt.replace(tzinfo=timezone.utc)

        price = mid if mid and mid > 0 else ((bid + ask) / 2 if bid and ask else 0)
        if price <= 0:
            continue

        if first_price is None:
            first_price = price
        if entry_underlying is None and underlying and underlying > 0:
            entry_underlying = underlying
        if vol_at_entry is None and tick[3] and tick[2]:  # ask, bid
            vol_at_entry = (tick[3] - tick[2]) / entry * 100  # spread as % of entry

        if price > peak:
            peak = price

        elapsed = (ts_dt - sig_ts).total_seconds() / 60
        gain_pct = (price - entry) / entry * 100
        peak_gain = (peak - entry) / entry * 100
        drop_entry = max(0, (entry - price) / entry * 100)
        drop_peak = (peak - price) / peak * 100 if peak > 0 else 0
        et_hour = (ts_dt.hour - 4) % 24
        et_min = ts_dt.minute

        # Collect trajectory data every ~5min
        minute_bucket = int(elapsed // 5)
        if len(prices_5m) <= minute_bucket:
            prices_5m.append(gain_pct)
            if underlying and underlying > 0 and entry_underlying:
                underlying_5m.append((underlying - entry_underlying) / entry_underlying * 100)
            else:
                underlying_5m.append(0)

        def _exit(reason):
            pnl = (price - entry) * contracts * 100
            if pnl > 0:
                pnl *= (1 - SLIPPAGE)
            details = _build_details(elapsed, gain_pct, peak_gain, prices_5m,
                                     underlying_5m, vol_at_entry, entry_underlying,
                                     underlying, is_call)
            return pnl, reason, elapsed, details

        # ============================================================
        # TRACK 1: SCALP — fast movers that peak early and fade
        # ============================================================
        if elapsed <= 15 and peak_gain >= 20:
            # Peak was 20%+ but price has faded to <60% of peak gain
            if gain_pct < peak_gain * 0.6 and gain_pct > 0:
                return _exit("scalp_trail")

        # ============================================================
        # GRADUATED HARD STOP (active after 20min)
        # ============================================================
        if elapsed >= 20:
            if elapsed < 45:
                stop = 40
            elif elapsed < 90:
                stop = 35
            else:
                stop = 25

            if drop_entry >= stop:
                # Momentum confirmation: need underlying to confirm
                if entry_underlying and underlying and underlying > 0:
                    u_move = (underlying - entry_underlying) / entry_underlying * 100
                    against = (u_move < -0.4) if is_call else (u_move > 0.4)
                    if against:
                        pnl = (price - entry) * contracts * 100
                        details = _build_details(elapsed, gain_pct, peak_gain, prices_5m,
                                                 underlying_5m, vol_at_entry,
                                                 entry_underlying, underlying, is_call)
                        return pnl, "confirmed_stop", elapsed, details
                    # Absolute backstop
                    if drop_entry >= stop + 15:
                        pnl = (price - entry) * contracts * 100
                        details = _build_details(elapsed, gain_pct, peak_gain, prices_5m,
                                                 underlying_5m, vol_at_entry,
                                                 entry_underlying, underlying, is_call)
                        return pnl, "hard_stop", elapsed, details
                else:
                    pnl = (price - entry) * contracts * 100
                    details = _build_details(elapsed, gain_pct, peak_gain, prices_5m,
                                             underlying_5m, vol_at_entry,
                                             entry_underlying, underlying, is_call)
                    return pnl, "stop_loss", elapsed, details

        # ============================================================
        # 30-MINUTE CHECKPOINT (0% recovery rate below -15%)
        # ============================================================
        if not checkpoint_done and elapsed >= 30:
            checkpoint_done = True
            if gain_pct < -15:
                pnl = (price - entry) * contracts * 100
                details = _build_details(elapsed, gain_pct, peak_gain, prices_5m,
                                         underlying_5m, vol_at_entry,
                                         entry_underlying, underlying, is_call)
                return pnl, "checkpoint_cut", elapsed, details

        # ============================================================
        # 45-MINUTE GRACE — no trail exits before this
        # ============================================================
        if elapsed < 45:
            if et_hour >= 15 and et_min >= 45:
                return _exit("eod_cutoff")
            continue

        # ============================================================
        # SOFT TRAIL (15-50% peak gain)
        # ============================================================
        if 15 <= peak_gain < 50:
            floor = entry + (peak - entry) * 0.50
            if price <= floor:
                return _exit("soft_trail")

        # ============================================================
        # ADAPTIVE TRAIL (3 stages)
        # ============================================================
        if peak_gain >= 400:
            if drop_peak >= 30:
                return _exit("adaptive_moonshot")
        elif peak_gain >= 150:
            if drop_peak >= 45:
                return _exit("adaptive_runner")
        elif peak_gain >= 40:
            if drop_peak >= 40:
                return _exit("adaptive_active")

        # ============================================================
        # THETA BLEED (120min+ and down 30%+)
        # ============================================================
        if elapsed >= 120 and drop_entry >= 30:
            pnl = (price - entry) * contracts * 100
            details = _build_details(elapsed, gain_pct, peak_gain, prices_5m,
                                     underlying_5m, vol_at_entry,
                                     entry_underlying, underlying, is_call)
            return pnl, "theta_bleed", elapsed, details

        # ============================================================
        # EOD CUTOFF (3:45 PM ET)
        # ============================================================
        if et_hour >= 15 and et_min >= 45:
            return _exit("eod_cutoff")

    # End of data
    if ticks:
        for t in reversed(ticks):
            price = t[1] if t[1] and t[1] > 0 else 0
            if price > 0:
                pnl = (price - entry) * contracts * 100
                if pnl > 0:
                    pnl *= (1 - SLIPPAGE)
                ts_dt = datetime.fromisoformat(t[0]) if isinstance(t[0], str) else t[0]
                if ts_dt.tzinfo is None:
                    ts_dt = ts_dt.replace(tzinfo=timezone.utc)
                elapsed = (ts_dt - sig_ts).total_seconds() / 60
                peak_gain = (peak - entry) / entry * 100
                gain_pct = (price - entry) / entry * 100
                details = _build_details(elapsed, gain_pct, peak_gain, prices_5m,
                                         underlying_5m, vol_at_entry,
                                         entry_underlying, 0, is_call)
                return pnl, "eod_data_end", elapsed, details
    return 0, "no_data", 0, {}


def _build_details(elapsed, gain_pct, peak_gain, prices_5m, underlying_5m,
                   spread_pct, entry_underlying, current_underlying, is_call):
    """Build ML feature dict for analysis."""
    # Trajectory shape features
    at_5m = prices_5m[1] if len(prices_5m) > 1 else 0
    at_10m = prices_5m[2] if len(prices_5m) > 2 else 0
    at_15m = prices_5m[3] if len(prices_5m) > 3 else 0
    at_30m = prices_5m[6] if len(prices_5m) > 6 else 0

    # Underlying movement
    u_at_5m = underlying_5m[1] if len(underlying_5m) > 1 else 0
    u_at_15m = underlying_5m[3] if len(underlying_5m) > 3 else 0
    u_at_30m = underlying_5m[6] if len(underlying_5m) > 6 else 0

    # Early momentum: slope of first 3 data points
    early_slope = 0
    if len(prices_5m) >= 3:
        early_slope = (prices_5m[2] - prices_5m[0]) / 2  # gain per 5min bucket

    return {
        "at_5m": at_5m,
        "at_10m": at_10m,
        "at_15m": at_15m,
        "at_30m": at_30m,
        "u_at_5m": u_at_5m,
        "u_at_15m": u_at_15m,
        "u_at_30m": u_at_30m,
        "early_slope": early_slope,
        "spread_pct": spread_pct or 0,
        "peak_gain": peak_gain,
        "exit_gain": gain_pct,
        "hold_min": elapsed,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def build_ct(ticker, day, direction, strike):
    dt = datetime.strptime(day, "%Y-%m-%d")
    ds = dt.strftime("%y%m%d")
    cp = "C" if direction.lower() in ("call", "bullish", "long") else "P"
    si = int(strike * 1000)
    return f"O:{ticker}{ds}{cp}{si:08d}"


def main():
    sig_conn = sqlite3.connect(SIGNALS_DB)
    sig_conn.row_factory = sqlite3.Row
    harv_conn = sqlite3.connect(HARVESTER_DB)

    signals = sig_conn.execute("""
        SELECT ts.id, ts.ticker, ts.direction, ts.score, ts.strike,
               ts.atm_premium, ts.otm_premium, date(ts.created_at) as day,
               ts.created_at as sig_ts
        FROM trade_signals ts ORDER BY ts.created_at
    """).fetchall()

    # Run v5 on every signal
    results = []
    no_data = no_strike = 0

    for sig in signals:
        ticker = sig["ticker"]
        direction = sig["direction"]
        day = sig["day"]
        strike = sig["strike"]
        score = sig["score"] or 0
        premium = sig["atm_premium"] or sig["otm_premium"]

        if not strike or not premium or premium <= 0:
            no_strike += 1
            continue

        contract = build_ct(ticker, day, direction, strike)
        rows = harv_conn.execute("""
            SELECT captured_at, midpoint, bid, ask, underlying_price
            FROM harvest_snapshots WHERE contract_ticker = ? AND captured_at >= ?
            ORDER BY captured_at
        """, (contract, sig["sig_ts"])).fetchall()

        if not rows:
            no_data += 1
            continue

        first = rows[0]
        entry = (first[3] if first[3] and first[3] > 0 else first[1]) or premium
        if entry <= 0:
            entry = premium

        if score >= 95: contracts = 5
        elif score >= 90: contracts = 4
        elif score >= 85: contracts = 3
        else: contracts = 1

        sig_ts = datetime.fromisoformat(sig["sig_ts"])
        if sig_ts.tzinfo is None:
            sig_ts = sig_ts.replace(tzinfo=timezone.utc)

        # Overall peak for reference
        all_mids = [r[1] for r in rows if r[1] and r[1] > 0]
        overall_peak = max(all_mids) if all_mids else entry
        overall_peak_gain = (overall_peak - entry) / entry * 100

        # Find time to peak
        peak_ts = sig_ts
        pk = entry
        for r in rows:
            m = r[1] or 0
            if m > pk:
                pk = m
                t = datetime.fromisoformat(r[0])
                if t.tzinfo is None:
                    t = t.replace(tzinfo=timezone.utc)
                peak_ts = t
        peak_time_min = (peak_ts - sig_ts).total_seconds() / 60

        # ET hour of signal
        et_hour = (sig_ts.hour - 4) % 24

        pnl, reason, hold, details = v5_simulate(entry, rows, sig_ts, contracts, direction)

        results.append({
            "ticker": ticker, "dir": direction, "day": day, "score": score,
            "entry": entry, "contracts": contracts, "pnl": pnl,
            "reason": reason, "hold": hold,
            "peak_gain": overall_peak_gain, "peak_min": peak_time_min,
            "et_hour": et_hour,
            **details,
        })

    # =====================================================================
    # REPORT
    # =====================================================================

    print(f"{'='*80}")
    print(f"V5 SCALP_AND_HOLD BACKTEST — {len(results)} signals")
    print(f"{'='*80}")
    print(f"Portfolio: ${PORTFOLIO:,}  |  Slippage: {SLIPPAGE*100:.0f}%")
    print(f"No data: {no_data}  |  No strike: {no_strike}")

    total_pnl = sum(r["pnl"] for r in results)
    wins = [r for r in results if r["pnl"] > 0]
    losses = [r for r in results if r["pnl"] <= 0]
    wr = len(wins) / len(results) * 100
    avg_win = sum(r["pnl"] for r in wins) / len(wins) if wins else 0
    avg_loss = sum(r["pnl"] for r in losses) / len(losses) if losses else 0
    wl_ratio = abs(avg_win / avg_loss) if avg_loss else 0

    print(f"\n  Total P&L:    ${total_pnl:>+,.0f}")
    print(f"  Win Rate:     {wr:.1f}% ({len(wins)}W / {len(losses)}L)")
    print(f"  Avg Win:      ${avg_win:>+,.0f}")
    print(f"  Avg Loss:     ${avg_loss:>+,.0f}")
    print(f"  Win:Loss:     {wl_ratio:.1f}:1")
    print(f"  Avg Hold:     {sum(r['hold'] for r in results)/len(results):.0f} min")
    print(f"  Return on ${PORTFOLIO:,}: {total_pnl/PORTFOLIO*100:>+.1f}%")

    # ----- DAILY P&L -----
    print(f"\n{'='*80}")
    print("DAILY P&L")
    print(f"{'='*80}")

    daily = defaultdict(lambda: {"pnl": 0, "trades": 0, "wins": 0, "biggest_win": 0, "biggest_loss": 0})
    for r in results:
        d = daily[r["day"]]
        d["pnl"] += r["pnl"]
        d["trades"] += 1
        if r["pnl"] > 0:
            d["wins"] += 1
        d["biggest_win"] = max(d["biggest_win"], r["pnl"])
        d["biggest_loss"] = min(d["biggest_loss"], r["pnl"])

    print(f"\n{'Date':<12} {'Trades':>6} {'W/L':>7} {'WR':>5} {'P&L':>9} {'%Port':>7} "
          f"{'BigWin':>8} {'BigLoss':>9} {'Cum':>9}")
    print("-" * 85)

    cum = 0
    green_days = 0
    red_days = 0
    for day in sorted(daily.keys()):
        d = daily[day]
        cum += d["pnl"]
        l = d["trades"] - d["wins"]
        wr_d = d["wins"] / d["trades"] * 100 if d["trades"] else 0
        pct = d["pnl"] / PORTFOLIO * 100
        if d["pnl"] > 0:
            green_days += 1
        else:
            red_days += 1
        marker = ""
        if 15 <= pct <= 25:
            marker = " TARGET"
        elif pct > 25:
            marker = " RUNNER"
        elif pct < -5:
            marker = " BAD"
        print(f"{day:<12} {d['trades']:>6} {d['wins']}W/{l}L {wr_d:>4.0f}% ${d['pnl']:>8.0f} "
              f"{pct:>+6.1f}% ${d['biggest_win']:>7.0f} ${d['biggest_loss']:>8.0f} "
              f"${cum:>8.0f}{marker}")

    print(f"\n  Green days: {green_days}/{green_days+red_days} ({green_days/(green_days+red_days)*100:.0f}%)")
    print(f"  Red days:   {red_days}/{green_days+red_days}")
    print(f"  Total return: {cum/PORTFOLIO*100:>+.1f}%")

    # ----- GATE FIRE ANALYSIS -----
    print(f"\n{'='*80}")
    print("GATE FIRE BREAKDOWN")
    print(f"{'='*80}")

    gate_stats = defaultdict(lambda: {"count": 0, "pnl": 0, "wins": 0, "holds": [],
                                       "avg_peak": []})
    for r in results:
        g = gate_stats[r["reason"]]
        g["count"] += 1
        g["pnl"] += r["pnl"]
        if r["pnl"] > 0:
            g["wins"] += 1
        g["holds"].append(r["hold"])
        g["avg_peak"].append(r["peak_gain"])

    print(f"\n{'Gate':<22} {'Fires':>5} {'%':>5} {'P&L':>9} {'W/L':>8} {'WR':>5} "
          f"{'AvgHold':>7} {'AvgPeak':>8} {'Role'}")
    print("-" * 90)

    for gate, s in sorted(gate_stats.items(), key=lambda x: x[1]["count"], reverse=True):
        ct = s["count"]
        pct = ct / len(results) * 100
        wr = s["wins"] / ct * 100 if ct else 0
        ah = sum(s["holds"]) / ct if ct else 0
        ap = sum(s["avg_peak"]) / ct if ct else 0
        role = ""
        if s["pnl"] > 0 and wr > 60:
            role = "PROFIT ENGINE"
        elif s["pnl"] < 0 and wr == 0:
            role = "LOSS CUTTER"
        elif wr > 40:
            role = "MIXED"
        else:
            role = "LOSS CUTTER"
        print(f"{gate:<22} {ct:>5} {pct:>4.0f}% ${s['pnl']:>8.0f} "
              f"{s['wins']}W/{ct-s['wins']}L {wr:>4.0f}% {ah:>5.0f}m {ap:>+7.0f}% {role}")

    # ----- EVERY TRADE -----
    print(f"\n{'='*80}")
    print("ALL TRADES")
    print(f"{'='*80}")

    print(f"\n{'#':<3} {'Ticker':<7} {'Dir':<5} {'Day':<12} {'Score':>5} {'$In':>6} "
          f"{'Ct':>3} {'P&L':>8} {'Gate':<20} {'Hold':>5} {'Peak%':>6} {'PkMin':>6}")
    print("-" * 100)

    for i, r in enumerate(results):
        marker = ""
        if r["pnl"] > 500:
            marker = " ***"
        elif r["pnl"] < -300:
            marker = " !!!"
        print(f"{i+1:<3} {r['ticker']:<7} {r['dir'][:4]:<5} {r['day']:<12} {r['score']:>5} "
              f"${r['entry']:>5.2f} {r['contracts']:>3} ${r['pnl']:>+7.0f} "
              f"{r['reason']:<20} {r['hold']:>4.0f}m {r['peak_gain']:>+5.0f}% "
              f"{r['peak_min']:>5.0f}m{marker}")

    # =====================================================================
    # ML FEATURE ANALYSIS — what predicts winners vs losers?
    # =====================================================================
    print(f"\n\n{'='*80}")
    print("ML FEATURE ANALYSIS — WHAT PREDICTS WINNERS?")
    print(f"{'='*80}")

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0

    w_feats = [r for r in results if r["pnl"] > 0]
    l_feats = [r for r in results if r["pnl"] <= 0]

    features = [
        ("Premium gain at 5min", "at_5m", "%"),
        ("Premium gain at 10min", "at_10m", "%"),
        ("Premium gain at 15min", "at_15m", "%"),
        ("Premium gain at 30min", "at_30m", "%"),
        ("Underlying move at 5min", "u_at_5m", "%"),
        ("Underlying move at 15min", "u_at_15m", "%"),
        ("Underlying move at 30min", "u_at_30m", "%"),
        ("Early slope (gain/5min)", "early_slope", ""),
        ("Bid-ask spread %", "spread_pct", "%"),
        ("Score", "score", ""),
        ("ET hour of signal", "et_hour", ""),
        ("Overall peak gain", "peak_gain", "%"),
        ("Time to peak", "peak_min", "m"),
    ]

    print(f"\n{'Feature':<30} {'Winners':>10} {'Losers':>10} {'Gap':>10} {'Signal?':>8}")
    print("-" * 75)

    for name, key, unit in features:
        w_vals = [r.get(key, 0) or 0 for r in w_feats]
        l_vals = [r.get(key, 0) or 0 for r in l_feats]
        w_avg = avg(w_vals)
        l_avg = avg(l_vals)
        gap = w_avg - l_avg
        # Simple significance: is gap > 1 stddev of combined?
        all_vals = w_vals + l_vals
        if all_vals:
            mean = avg(all_vals)
            variance = sum((x - mean)**2 for x in all_vals) / len(all_vals)
            std = sqrt(variance) if variance > 0 else 1
            significant = abs(gap) > std * 0.5
        else:
            significant = False
        sig_str = "YES" if significant else "weak"
        print(f"{name:<30} {w_avg:>+9.1f}{unit} {l_avg:>+9.1f}{unit} "
              f"{gap:>+9.1f} {sig_str:>8}")

    # ----- PREDICTIVE THRESHOLDS -----
    print(f"\n{'='*80}")
    print("PREDICTIVE THRESHOLDS — If we knew X at time Y, what's the WR?")
    print(f"{'='*80}")

    # Premium at 5min
    print(f"\nPremium gain at 5 minutes:")
    for thresh in [-20, -10, -5, 0, 5, 10, 20]:
        above = [r for r in results if (r.get("at_5m") or 0) >= thresh]
        below = [r for r in results if (r.get("at_5m") or 0) < thresh]
        wr_a = sum(1 for r in above if r["pnl"] > 0) / len(above) * 100 if above else 0
        wr_b = sum(1 for r in below if r["pnl"] > 0) / len(below) * 100 if below else 0
        pnl_a = sum(r["pnl"] for r in above)
        pnl_b = sum(r["pnl"] for r in below)
        print(f"  >= {thresh:>+3}%: {len(above):>3} trades, WR={wr_a:>5.1f}%, P&L=${pnl_a:>+8.0f} "
              f" |  < {thresh:>+3}%: {len(below):>3} trades, WR={wr_b:>5.1f}%, P&L=${pnl_b:>+8.0f}")

    # Underlying at 5min
    print(f"\nUnderlying move at 5 minutes (confirms thesis):")
    for thresh in [-0.5, -0.2, 0, 0.1, 0.2, 0.5]:
        # For calls, positive underlying = good. For puts, negative = good.
        # Normalize: "thesis confirmed" = underlying moved in our direction
        above = []
        below = []
        for r in results:
            u = r.get("u_at_5m") or 0
            is_call = r["dir"].lower() in ("call", "bullish", "long")
            thesis_move = u if is_call else -u  # normalize to "thesis direction"
            if thesis_move >= thresh:
                above.append(r)
            else:
                below.append(r)
        wr_a = sum(1 for r in above if r["pnl"] > 0) / len(above) * 100 if above else 0
        wr_b = sum(1 for r in below if r["pnl"] > 0) / len(below) * 100 if below else 0
        pnl_a = sum(r["pnl"] for r in above)
        pnl_b = sum(r["pnl"] for r in below)
        print(f"  Thesis >= {thresh:>+4.1f}%: {len(above):>3} trades, WR={wr_a:>5.1f}%, P&L=${pnl_a:>+8.0f} "
              f" |  below: {len(below):>3} trades, WR={wr_b:>5.1f}%, P&L=${pnl_b:>+8.0f}")

    # Early slope
    print(f"\nEarly premium slope (gain% per 5min bucket, first 15min):")
    for thresh in [-5, -2, 0, 2, 5, 10]:
        above = [r for r in results if (r.get("early_slope") or 0) >= thresh]
        below = [r for r in results if (r.get("early_slope") or 0) < thresh]
        wr_a = sum(1 for r in above if r["pnl"] > 0) / len(above) * 100 if above else 0
        wr_b = sum(1 for r in below if r["pnl"] > 0) / len(below) * 100 if below else 0
        pnl_a = sum(r["pnl"] for r in above)
        pnl_b = sum(r["pnl"] for r in below)
        print(f"  >= {thresh:>+3}: {len(above):>3} trades, WR={wr_a:>5.1f}%, P&L=${pnl_a:>+8.0f} "
              f" |  < {thresh:>+3}: {len(below):>3} trades, WR={wr_b:>5.1f}%, P&L=${pnl_b:>+8.0f}")

    # Time of day
    print(f"\nSignal time (ET hour):")
    for hour in range(9, 16):
        h_trades = [r for r in results if r["et_hour"] == hour]
        if not h_trades:
            continue
        h_wr = sum(1 for r in h_trades if r["pnl"] > 0) / len(h_trades) * 100
        h_pnl = sum(r["pnl"] for r in h_trades)
        print(f"  {hour}:00 ET: {len(h_trades):>3} trades, WR={h_wr:>5.1f}%, P&L=${h_pnl:>+8.0f}")

    # Score
    print(f"\nSignal score:")
    for lo, hi in [(78, 84), (85, 89), (90, 94), (95, 100)]:
        s_trades = [r for r in results if lo <= r["score"] <= hi]
        if not s_trades:
            continue
        s_wr = sum(1 for r in s_trades if r["pnl"] > 0) / len(s_trades) * 100
        s_pnl = sum(r["pnl"] for r in s_trades)
        print(f"  {lo}-{hi}: {len(s_trades):>3} trades, WR={s_wr:>5.1f}%, P&L=${s_pnl:>+8.0f}")

    # =====================================================================
    # ML OPPORTUNITIES — where could ML help?
    # =====================================================================
    print(f"\n\n{'='*80}")
    print("WHERE ML + CANDLE DATA CAN HELP")
    print(f"{'='*80}")

    # 1. Checkpoint improvement
    cp_trades = [r for r in results if r["reason"] == "checkpoint_cut"]
    non_cp = [r for r in results if r["reason"] != "checkpoint_cut" and r.get("at_30m", 0) and r["at_30m"] < -10]
    print(f"""
1. SMARTER 30-MIN CHECKPOINT (currently cuts {len(cp_trades)} trades)
   Current: simple threshold (down >15% = cut)
   ML could: use underlying movement + premium trajectory + candle patterns
   to predict recovery probability instead of a fixed threshold.

   Trades cut at checkpoint: {len(cp_trades)}, P&L from cuts: ${sum(r['pnl'] for r in cp_trades):>+,.0f}
   Trades between -10% and -15% at 30min (gray zone): {len(non_cp)}
   ML features: premium trajectory shape, underlying momentum, bid-ask spread,
   volume trend, RSI from candle_cache, time of day""")

    # 2. Scalp vs hold decision
    scalp_trades = [r for r in results if r["reason"] == "scalp_trail"]
    print(f"""
2. SCALP VS HOLD DECISION (currently {len(scalp_trades)} scalps)
   Current: if peak >20% and faded to <60% in first 15min, scalp
   ML could: predict whether this is a "early peak" (take profit) or
   "dip before bigger move" (hold for more)

   Scalp trades: {len(scalp_trades)}, P&L: ${sum(r['pnl'] for r in scalp_trades):>+,.0f}
   ML features: volume at entry vs avg, RSI divergence, underlying trend
   direction from 1h/4h candles, pattern recognition (hammer/engulfing)""")

    # 3. Stop confirmation
    stop_trades = [r for r in results if "stop" in r["reason"]]
    print(f"""
3. SMARTER STOP LOSS (currently {len(stop_trades)} stops)
   Current: premium down X% + underlying confirms
   ML could: multi-factor thesis validation using candle data:
   - Is the underlying in a higher-TF downtrend (4h bearish)?
   - Is RSI oversold (potential bounce)?
   - Is volume declining (sellers exhausted)?
   - Is there a support level nearby?

   Stop trades: {len(stop_trades)}, P&L: ${sum(r['pnl'] for r in stop_trades):>+,.0f}
   We already have candle_cache.py with RSI, OBV, patterns for 5 timeframes.
   ENRG gate already does multi-TF voting — extend it to all stop decisions.""")

    # 4. EOD runner detection
    eod_trades = [r for r in results if r["reason"] == "eod_cutoff"]
    print(f"""
4. RUNNER DETECTION — LET WINNERS RUN LONGER
   EOD exits: {len(eod_trades)}, P&L: ${sum(r['pnl'] for r in eod_trades):>+,.0f}
   These are trades that survived all gates until 3:45 PM — likely strong trends.
   ML could: at the 1hr mark, predict if this is a "trend day" trade that
   will keep running, and widen the trail accordingly.

   Candle features: higher-TF trend strength, volume acceleration,
   new highs vs new lows in last 30min, SPY/QQQ correlation""")

    # 5. Market regime
    print(f"""
5. MARKET REGIME DETECTION (Yank's idea)
   Use multi-TF candle data at trade entry to classify:
   - UPTREND: SPY/QQQ 1h+4h candles bullish, volume confirming
     → Wide stops, long hold, let calls run, tight puts
   - FLAT: SPY/QQQ in range, low ATR
     → Quick scalps, tight trails, grab 15-20% and move on
   - DOWNTREND: SPY/QQQ 1h+4h bearish
     → Tight stops on calls, let puts run

   Infrastructure already exists: candle_cache.py fetches RSI, OBV,
   patterns across 5 timeframes. Just need a regime classifier.""")


if __name__ == "__main__":
    main()
