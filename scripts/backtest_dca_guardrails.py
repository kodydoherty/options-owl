"""Backtest DCA guardrails to mitigate MSFT-like double-down losses.

Tests:
  1. Post-DCA timeout: if not breakeven within N min after DCA, cut
  2. Post-DCA tight stop: if drops X% below DCA price, cut immediately
  3. DCA time-of-day: only DCA before 1PM ET (theta kills afternoon DCA)
  4. DCA peak gate: only DCA if trade previously hit +X%
  5. Combinations of above

Usage:
    python scripts/backtest_dca_guardrails.py
"""

from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from types import SimpleNamespace

from options_owl.risk.exit_v5.config import V5Config, get_ticker_config, AdaptiveTier
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState

_V6_SETTINGS = SimpleNamespace(
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
)

SIGNALS_DB = str(PROJECT_DIR / "journal" / "owlet-kody" / "raw_messages.db")
HARVESTER_DB = str(PROJECT_DIR / "journal" / "owlet-harvester" / "options_data.db")
PORTFOLIO = 8000

SCORE_TIERS = [
    (135, 1.00), (120, 0.85), (100, 0.85), (90, 0.50), (78, 0.25),
]


def load_signals():
    conn = sqlite3.connect(SIGNALS_DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT id, ticker, direction, sentiment, score,
               atm_premium, otm_premium, strike, expiry,
               entry_price, created_at
        FROM trade_signals WHERE score >= 70 ORDER BY created_at
    """).fetchall()
    signals = []
    for r in rows:
        sig = dict(r)
        sig["premium"] = sig["atm_premium"] or sig["otm_premium"]
        sent = (sig.get("sentiment") or sig.get("direction") or "bullish").lower()
        sig["option_type"] = "put" if sent in ("bearish", "put") else "call"
        if sig["premium"] and sig["premium"] > 0 and sig["strike"]:
            signals.append(sig)
    conn.close()
    return signals


def build_contract_ticker(ticker, expiry, strike, option_type):
    if not expiry:
        return ""
    try:
        exp_dt = datetime.strptime(expiry, "%Y-%m-%d")
    except ValueError:
        return ""
    exp_str = exp_dt.strftime("%y%m%d")
    ot = "C" if option_type.lower() in ("call", "bullish", "c") else "P"
    strike_int = int(strike * 1000)
    return f"O:{ticker}{exp_str}{ot}{strike_int:08d}"


def load_ticks(harvester_conn, signal):
    ticker = signal["ticker"]
    strike = signal["strike"]
    created_at = signal["created_at"]
    option_type = signal["option_type"]
    sig_date = created_at[:10]
    sig_dt = datetime.strptime(sig_date, "%Y-%m-%d").date()
    candidates = [sig_dt]
    for delta in range(1, 6):
        d = sig_dt + timedelta(days=delta)
        if d.weekday() < 5:
            candidates.append(d)
            if len(candidates) >= 4:
                break
    for exp_date in candidates:
        expiry = exp_date.strftime("%Y-%m-%d")
        ct = build_contract_ticker(ticker, expiry, strike, option_type)
        if not ct:
            continue
        rows = harvester_conn.execute("""
            SELECT captured_at, midpoint, bid, ask, underlying_price,
                   implied_volatility, delta, gamma, theta, vega, day_volume
            FROM harvest_snapshots
            WHERE contract_ticker = ? AND captured_at >= ?
            ORDER BY captured_at
        """, (ct, created_at)).fetchall()
        if rows and len(rows) >= 10:
            signal["_dte"] = (exp_date - sig_dt).days
            signal["_expiry_date"] = expiry
            break
    else:
        return None
    df = pd.DataFrame(rows, columns=[
        "captured_at", "midpoint", "bid", "ask", "underlying_price",
        "iv", "delta", "gamma", "theta", "vega", "volume"
    ])
    df["premium"] = df["midpoint"].where(df["midpoint"] > 0, (df["bid"] + df["ask"]) / 2)
    df["premium"] = df["premium"].where(df["premium"] > 0, np.nan)
    df = df.dropna(subset=["premium"])
    if len(df) < 10:
        return None
    df["ts"] = pd.to_datetime(df["captured_at"])
    df = df.sort_values("ts").reset_index(drop=True)
    return df


def size_contracts(score, entry_premium):
    deployable = PORTFOLIO * 0.75
    per_slot = deployable / 4
    position_cap = PORTFOLIO * 0.15
    score_mult = 0.25
    for threshold, mult in SCORE_TIERS:
        if score >= threshold:
            score_mult = mult
            break
    cost_per = entry_premium * 100
    scaled_target = per_slot * score_mult
    raw = int(scaled_target / cost_per) if cost_per > 0 else 1
    cap = int(position_cap / cost_per) if cost_per > 0 else 1
    return max(1, min(raw, cap))


def momentum_blocked(df, direction):
    is_call = direction in ("bullish", "call")
    window = min(15, len(df))
    ups = []
    for i in range(window):
        u = df["underlying_price"].iloc[i]
        if u and u > 0:
            ups.append(float(u))
    if len(ups) < 5:
        return False
    h1, h2 = ups[:len(ups)//2], ups[len(ups)//2:]
    pct = (sum(h2)/len(h2) - sum(h1)/len(h1)) / (sum(h1)/len(h1)) * 100
    ps = df["premium"].iloc[0]
    p5 = df["premium"].iloc[min(4, len(df)-1)]
    pf = (p5 - ps) / ps * 100 if ps > 0 else 0
    neg = 0
    if is_call and pct < -0.05:
        neg += 1
    elif not is_call and pct > 0.05:
        neg += 1
    if pf < -5:
        neg += 1
    against = 0
    for i in range(max(0, window-3), window):
        if i == 0:
            continue
        pu, cu = df["underlying_price"].iloc[i-1], df["underlying_price"].iloc[i]
        if pu and cu:
            if is_call and cu < pu:
                against += 1
            elif not is_call and cu > pu:
                against += 1
    if against >= 3:
        neg += 1
    return neg >= 2


def simulate(df, entry_premium, contracts, direction, dte, expiry_date,
             ticker="SIM", dca_mode="none",
             dca_timeout_min=0, dca_stop_pct=0, dca_before_hour_et=0,
             dca_min_peak_pct=0):
    """
    dca_mode: "none" | "always"
    dca_timeout_min: if >0, exit DCA'd trade if not breakeven within N min
    dca_stop_pct: if >0, exit if drops X% below DCA fill price
    dca_before_hour_et: if >0, only allow DCA before this ET hour
    dca_min_peak_pct: if >0, only DCA if trade previously hit +X%
    """
    if entry_premium <= 0:
        return {"pnl": 0, "reason": "no_data", "hold": 0, "exit_prem": 0,
                "peak_gain": 0, "dca_fired": False}

    cfg = get_ticker_config(ticker, use_per_ticker=True)
    fsm = ExitFSM(cfg, settings=_V6_SETTINGS)
    option_type = "put" if direction in ("bearish", "put") else "call"

    entry_ts = df["ts"].iloc[0]
    if hasattr(entry_ts, 'to_pydatetime'):
        entry_ts = entry_ts.to_pydatetime()
    if entry_ts.tzinfo is not None:
        entry_ts = entry_ts.replace(tzinfo=None)

    first_underlying = 0.0
    for i in range(min(5, len(df))):
        u = df["underlying_price"].iloc[i]
        if u and u > 0:
            first_underlying = float(u)
            break

    state = TradeState(
        trade_id=1, ticker=ticker, option_type=option_type,
        entry_premium=entry_premium, entry_time=entry_ts,
        contracts=contracts, peak_premium=entry_premium,
        entry_underlying_price=first_underlying,
        dte=dte, expiry_date=expiry_date or "",
    )

    locked_pnl = 0.0
    remaining = contracts
    dca_fired = False
    dca_time = None
    dca_fill_price = 0.0
    max_gain_seen = 0.0
    original_entry = entry_premium

    for idx in range(1, len(df)):
        premium = df["premium"].iloc[idx]
        if np.isnan(premium) or premium <= 0:
            continue

        raw_bid = df["bid"].iloc[idx]
        raw_ask = df["ask"].iloc[idx]
        bid = float(raw_bid) if raw_bid and not pd.isna(raw_bid) else premium
        ask = float(raw_ask) if raw_ask and not pd.isna(raw_ask) else premium

        now = df["ts"].iloc[idx]
        if hasattr(now, 'to_pydatetime'):
            now = now.to_pydatetime()
        if now.tzinfo is not None:
            now = now.replace(tzinfo=None)

        underlying = df["underlying_price"].iloc[idx] or 0.0
        et_hour = now.hour - 4
        if et_hour < 0:
            et_hour += 24
        minutes_to_close = max(0, (16 * 60) - (et_hour * 60 + now.minute))
        elapsed_min = (now - entry_ts).total_seconds() / 60

        gain_from_original = (premium - original_entry) / original_entry * 100
        max_gain_seen = max(max_gain_seen, gain_from_original)

        # ── Post-DCA guardrails ──────────────────────────────────────────
        if dca_fired and dca_time is not None:
            min_since_dca = (now - dca_time).total_seconds() / 60

            # Guardrail 1: DCA timeout — not breakeven after N min, cut
            if dca_timeout_min > 0 and min_since_dca >= dca_timeout_min:
                # Check if we're still below the new averaged entry
                if premium < entry_premium:
                    pnl = locked_pnl + (premium - entry_premium) * remaining * 100
                    pk = (state.peak_premium - entry_premium) / entry_premium * 100
                    return {
                        "pnl": pnl, "reason": "dca_timeout",
                        "hold": elapsed_min, "exit_prem": premium,
                        "peak_gain": pk, "dca_fired": True,
                    }

            # Guardrail 2: DCA stop — dropped X% below DCA fill price
            if dca_stop_pct > 0 and dca_fill_price > 0:
                drop_from_dca = (dca_fill_price - premium) / dca_fill_price * 100
                if drop_from_dca >= dca_stop_pct:
                    pnl = locked_pnl + (premium - entry_premium) * remaining * 100
                    pk = (state.peak_premium - entry_premium) / entry_premium * 100
                    return {
                        "pnl": pnl, "reason": "dca_stop",
                        "hold": elapsed_min, "exit_prem": premium,
                        "peak_gain": pk, "dca_fired": True,
                    }

        # ── DCA trigger ──────────────────────────────────────────────────
        if not dca_fired and dca_mode != "none" and elapsed_min >= 5 and remaining >= 1:
            dip_pct = (entry_premium - premium) / entry_premium * 100
            if 15 <= dip_pct <= 35:
                should_dca = True

                # Gate: time of day
                if dca_before_hour_et > 0 and et_hour >= dca_before_hour_et:
                    should_dca = False

                # Gate: previous profitability
                if dca_min_peak_pct > 0 and max_gain_seen < dca_min_peak_pct:
                    should_dca = False

                if should_dca:
                    dca_fired = True
                    dca_time = now
                    dca_fill_price = premium
                    dca_qty = remaining
                    total_cost = entry_premium * remaining + premium * dca_qty
                    remaining += dca_qty
                    entry_premium = total_cost / remaining
                    state.entry_premium = entry_premium
                    state.contracts = remaining
                    state.peak_premium = max(state.peak_premium, entry_premium)
                    continue

        # ── Normal FSM ───────────────────────────────────────────────────
        action = fsm.evaluate(
            state, premium, bid, ask, now,
            current_underlying=underlying,
            minutes_to_close=minutes_to_close,
        )

        if action.should_exit:
            if action.contracts_to_close > 0 and action.contracts_to_close < remaining:
                closed = action.contracts_to_close
                locked_pnl += (premium - entry_premium) * closed * 100
                remaining -= closed
                state.contracts = remaining
                continue

            peak_gain = (state.peak_premium - entry_premium) / entry_premium * 100
            pnl = locked_pnl + (premium - entry_premium) * remaining * 100
            return {
                "pnl": pnl, "reason": action.reason.value,
                "hold": elapsed_min, "exit_prem": premium,
                "peak_gain": peak_gain, "dca_fired": dca_fired,
            }

    last_prem = df["premium"].iloc[-1]
    last_ts = df["ts"].iloc[-1]
    if hasattr(last_ts, 'to_pydatetime'):
        last_ts = last_ts.to_pydatetime()
    if last_ts.tzinfo is not None:
        last_ts = last_ts.replace(tzinfo=None)
    elapsed = (last_ts - entry_ts).total_seconds() / 60
    peak_gain = (state.peak_premium - entry_premium) / entry_premium * 100
    pnl = locked_pnl + (last_prem - entry_premium) * remaining * 100
    return {
        "pnl": pnl, "reason": "eod_data_end", "hold": elapsed,
        "exit_prem": last_prem, "peak_gain": peak_gain, "dca_fired": dca_fired,
    }


def main():
    signals = load_signals()
    print(f"Loaded {len(signals)} signals")

    harvester_conn = sqlite3.connect(HARVESTER_DB)
    trades = []
    no_data = 0
    for sig in signals:
        ticker = sig["ticker"]
        direction = (sig["direction"] or "bullish").lower()
        score = sig["score"] or 80
        if score < 78:
            continue
        df = load_ticks(harvester_conn, sig)
        if df is None:
            no_data += 1
            continue
        dte = sig.get("_dte", 0)
        expiry_date = sig.get("_expiry_date", "")
        first_ask = df["ask"].iloc[0]
        first_mid = df["premium"].iloc[0]
        adj_entry = first_ask if first_ask and first_ask > 0 else first_mid
        if adj_entry <= 0:
            adj_entry = sig["premium"]
        contracts = size_contracts(score, adj_entry)
        blocked = momentum_blocked(df, direction)
        trades.append({
            "ticker": ticker, "direction": direction, "score": score,
            "day": sig["created_at"][:10], "df": df, "entry": adj_entry,
            "contracts": contracts, "dte": dte, "expiry_date": expiry_date,
            "blocked": blocked,
        })
    harvester_conn.close()
    print(f"{len(trades)} trades, {no_data} skipped")

    # ── Baselines ────────────────────────────────────────────────────────

    def run_scenario(label, **kwargs):
        results = []
        dca_count = 0
        for t in trades:
            if t["blocked"]:
                results.append({"pnl": 0, "dca_fired": False, "reason": "blocked",
                                "ticker": t["ticker"], "day": t["day"], "peak_gain": 0})
                continue
            r = simulate(t["df"], t["entry"], t["contracts"], t["direction"],
                         t["dte"], t["expiry_date"], ticker=t["ticker"], **kwargs)
            r["ticker"] = t["ticker"]
            r["day"] = t["day"]
            results.append(r)
            if r.get("dca_fired"):
                dca_count += 1
        pnls = [r["pnl"] for r in results]
        total = sum(pnls)
        wins = sum(1 for p in pnls if p > 0)
        losses = len(pnls) - wins
        wr = wins / len(pnls) * 100
        max_l = min(pnls)
        return {
            "label": label, "total": total, "wins": wins, "losses": losses,
            "wr": wr, "max_loss": max_l, "dca_count": dca_count,
            "results": results,
        }

    scenarios = []

    # No DCA baseline
    scenarios.append(run_scenario("no_dca", dca_mode="none"))
    # DCA always (current behavior)
    scenarios.append(run_scenario("dca_always", dca_mode="always"))

    # ── Post-DCA timeout sweep ───────────────────────────────────────────

    print(f"\n{'=' * 100}")
    print(f"POST-DCA TIMEOUT: cut if not breakeven within N minutes after DCA")
    print(f"{'=' * 100}")

    for timeout in [5, 8, 10, 15, 20, 30]:
        scenarios.append(run_scenario(
            f"dca+timeout_{timeout}m",
            dca_mode="always", dca_timeout_min=timeout,
        ))

    # ── Post-DCA stop sweep ──────────────────────────────────────────────

    print(f"\n{'=' * 100}")
    print(f"POST-DCA STOP: cut if drops X% below DCA fill price")
    print(f"{'=' * 100}")

    for stop in [10, 15, 20, 25, 30]:
        scenarios.append(run_scenario(
            f"dca+stop_{stop}%",
            dca_mode="always", dca_stop_pct=stop,
        ))

    # ── Time-of-day gate ─────────────────────────────────────────────────

    print(f"\n{'=' * 100}")
    print(f"DCA TIME GATE: only DCA before X:00 ET")
    print(f"{'=' * 100}")

    for hour in [11, 12, 13, 14]:
        scenarios.append(run_scenario(
            f"dca+before_{hour}ET",
            dca_mode="always", dca_before_hour_et=hour,
        ))

    # ── Peak gate ────────────────────────────────────────────────────────

    print(f"\n{'=' * 100}")
    print(f"DCA PEAK GATE: only DCA if trade previously hit +X%")
    print(f"{'=' * 100}")

    for peak in [5, 10, 15, 20]:
        scenarios.append(run_scenario(
            f"dca+peak_{peak}%",
            dca_mode="always", dca_min_peak_pct=peak,
        ))

    # ── Combos ───────────────────────────────────────────────────────────

    print(f"\n{'=' * 100}")
    print(f"BEST COMBOS")
    print(f"{'=' * 100}")

    # Combo: timeout + stop
    for timeout in [10, 15, 20]:
        for stop in [15, 20, 25]:
            scenarios.append(run_scenario(
                f"dca+t{timeout}m+s{stop}%",
                dca_mode="always", dca_timeout_min=timeout, dca_stop_pct=stop,
            ))

    # Combo: time gate + stop
    for hour in [12, 13]:
        for stop in [15, 20]:
            scenarios.append(run_scenario(
                f"dca+b{hour}ET+s{stop}%",
                dca_mode="always", dca_before_hour_et=hour, dca_stop_pct=stop,
            ))

    # Combo: timeout + stop + time gate
    for timeout in [10, 15]:
        for stop in [15, 20]:
            for hour in [12, 13]:
                scenarios.append(run_scenario(
                    f"dca+t{timeout}m+s{stop}%+b{hour}ET",
                    dca_mode="always", dca_timeout_min=timeout,
                    dca_stop_pct=stop, dca_before_hour_et=hour,
                ))

    # ── Results table ────────────────────────────────────────────────────

    no_dca_pnl = scenarios[0]["total"]
    dca_always_pnl = scenarios[1]["total"]

    scenarios.sort(key=lambda s: s["total"], reverse=True)

    print(f"\n{'=' * 100}")
    print(f"ALL SCENARIOS RANKED")
    print(f"{'=' * 100}")
    print(f"\n{'Scenario':<30} {'Total P&L':>12} {'vs NoDCA':>10} {'vs DCA':>10} "
          f"{'Win%':>6} {'DCA#':>5} {'MaxLoss':>10}")
    print("-" * 90)
    for s in scenarios:
        d1 = s["total"] - no_dca_pnl
        d2 = s["total"] - dca_always_pnl
        print(f"{s['label']:<30} ${s['total']:>10,.0f} ${d1:>+9,.0f} ${d2:>+9,.0f} "
              f"{s['wr']:>5.1f}% {s['dca_count']:>5} ${s['max_loss']:>8,.0f}")

    # ── DCA trade detail for top 3 ──────────────────────────────────────

    print(f"\n{'=' * 100}")
    print(f"DCA TRADE DETAIL — TOP 3 vs DCA_ALWAYS")
    print(f"{'=' * 100}")

    dca_always_results = scenarios[0]["results"] if scenarios[0]["label"] == "dca_always" else None
    for s in [sc for sc in scenarios if sc["label"] != "no_dca"][:3]:
        if s["label"] == "dca_always":
            continue
        print(f"\n--- {s['label']} (${s['total']:,.0f}) ---")

        dca_trades_idx = [i for i, r in enumerate(s["results"]) if r.get("dca_fired")]
        if not dca_trades_idx:
            print("  No DCA trades")
            continue

        # Find DCA always results for comparison
        always_s = next((sc for sc in scenarios if sc["label"] == "dca_always"), None)
        if not always_s:
            continue

        print(f"  {'Day':<12} {'Tkr':<6} {'Peak%':>6} {'Always':>10} {'This':>10} {'Delta':>8} {'Reason'}")
        print(f"  {'-' * 75}")

        always_total = 0
        this_total = 0
        for i in dca_trades_idx:
            a = always_s["results"][i]
            t = s["results"][i]
            always_total += a["pnl"]
            this_total += t["pnl"]
            d = t["pnl"] - a["pnl"]
            if abs(d) > 1:
                print(f"  {t.get('day',''):<12} {t.get('ticker',''):<6} "
                      f"{a.get('peak_gain',0):>5.0f}% "
                      f"${a['pnl']:>8,.0f} ${t['pnl']:>8,.0f} ${d:>+7,.0f} {t.get('reason','')}")

        print(f"  DCA trades: always=${always_total:,.0f} this=${this_total:,.0f} "
              f"delta=${this_total-always_total:+,.0f}")


if __name__ == "__main__":
    main()
