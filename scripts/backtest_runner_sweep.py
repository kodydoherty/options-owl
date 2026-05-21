"""Sweep soft trail + adaptive trail params to capture more runner profits.

Key question: can we keep more of the peak gains without cutting winners early?
"""

from __future__ import annotations
import sqlite3, sys
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np, pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from options_owl.risk.exit_v5.config import V5Config, AdaptiveTier
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState

SIGNALS_DB = str(PROJECT_DIR / "journal" / "owlet-kody" / "raw_messages.db")
HARVESTER_DB = str(PROJECT_DIR / "journal" / "owlet-harvester" / "options_data.db")
PORTFOLIO = 8000


def load_signals():
    conn = sqlite3.connect(SIGNALS_DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT id, ticker, direction, sentiment, score,
               atm_premium, otm_premium, strike, expiry, created_at
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
            SELECT captured_at, midpoint, bid, ask, underlying_price
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
    df = pd.DataFrame(rows, columns=["captured_at", "midpoint", "bid", "ask", "underlying_price"])
    df["premium"] = df["midpoint"].where(df["midpoint"] > 0, (df["bid"] + df["ask"]) / 2)
    df["premium"] = df["premium"].where(df["premium"] > 0, np.nan)
    df = df.dropna(subset=["premium"])
    if len(df) < 10:
        return None
    df["ts"] = pd.to_datetime(df["captured_at"])
    df = df.sort_values("ts").reset_index(drop=True)
    return df


def simulate(df, entry_premium, contracts, direction, dte, expiry_date, ticker, cfg):
    if entry_premium <= 0:
        return {"pnl": 0, "reason": "no_data", "hold": 0, "peak_gain": 0}
    fsm = ExitFSM(cfg)
    option_type = "put" if direction in ("bearish", "put") else "call"
    entry_ts = df["ts"].iloc[0]
    if hasattr(entry_ts, 'to_pydatetime'):
        entry_ts = entry_ts.to_pydatetime()
    if entry_ts.tzinfo is not None:
        entry_ts = entry_ts.replace(tzinfo=None)
    first_u = 0.0
    for i in range(min(5, len(df))):
        u = df["underlying_price"].iloc[i]
        if u and u > 0:
            first_u = float(u)
            break
    state = TradeState(
        trade_id=1, ticker=ticker, option_type=option_type,
        entry_premium=entry_premium, entry_time=entry_ts,
        contracts=contracts, peak_premium=entry_premium,
        entry_underlying_price=first_u, dte=dte, expiry_date=expiry_date or "",
    )
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
        mtc = max(0, (16 * 60) - (et_hour * 60 + now.minute))
        action = fsm.evaluate(state, premium, bid, ask, now,
                              current_underlying=underlying, minutes_to_close=mtc)
        if action.should_exit:
            pnl = (premium - entry_premium) * contracts * 100
            peak_gain = (state.peak_premium - entry_premium) / entry_premium * 100
            return {"pnl": pnl, "reason": action.reason.value,
                    "hold": (now - entry_ts).total_seconds() / 60, "peak_gain": peak_gain}
    last_prem = df["premium"].iloc[-1]
    pnl = (last_prem - entry_premium) * contracts * 100
    peak_gain = (state.peak_premium - entry_premium) / entry_premium * 100
    return {"pnl": pnl, "reason": "eod_data_end",
            "hold": (df["ts"].iloc[-1].to_pydatetime().replace(tzinfo=None) - entry_ts).total_seconds() / 60,
            "peak_gain": peak_gain}


def main():
    signals = load_signals()
    harvester_conn = sqlite3.connect(HARVESTER_DB)
    trade_data = []
    for sig in signals:
        df = load_ticks(harvester_conn, sig)
        if df is None:
            continue
        ticker = sig["ticker"]
        direction = (sig["direction"] or "bullish").lower()
        score = sig["score"] or 80
        first_ask = df["ask"].iloc[0]
        first_mid = df["premium"].iloc[0]
        adj_entry = first_ask if first_ask and first_ask > 0 else first_mid
        if adj_entry <= 0:
            adj_entry = sig["premium"]
        cost_per = adj_entry * 100
        sm = 1.0 if score >= 95 else 0.75 if score >= 90 else 0.50 if score >= 85 else 0.25
        per_slot = PORTFOLIO * 0.75 / 5
        raw = int(per_slot * sm / cost_per) if cost_per > 0 else 1
        pc = int(PORTFOLIO * 0.15 / cost_per) if cost_per > 0 else 1
        contracts = max(1, min(raw, pc, 20))
        trade_data.append({
            "df": df, "entry_premium": adj_entry, "contracts": contracts,
            "direction": direction, "dte": sig.get("_dte", 0),
            "expiry_date": sig.get("_expiry_date", ""), "ticker": ticker,
            "score": score, "day": sig["created_at"][:10],
        })
    harvester_conn.close()
    print(f"Loaded {len(trade_data)} trades\n")

    configs = {
        "A default": V5Config(),
        "B keep=70%": V5Config(soft_trail_keep_pct=0.70),
        "C band=15-50,keep70": V5Config(soft_trail_band_low_pct=15.0, soft_trail_keep_pct=0.70),
        "D band=20-60,keep70": V5Config(
            soft_trail_band_low_pct=20.0, soft_trail_band_high_pct=60.0, soft_trail_keep_pct=0.70,
        ),
        "E tighter adaptive HV": V5Config(
            adaptive_highvol_tiers=(AdaptiveTier(400, 30), AdaptiveTier(150, 45), AdaptiveTier(40, 45)),
        ),
        "F keep70+tighter adapt": V5Config(
            soft_trail_keep_pct=0.70,
            adaptive_highvol_tiers=(AdaptiveTier(400, 30), AdaptiveTier(150, 45), AdaptiveTier(40, 45)),
            adaptive_standard_tiers=(AdaptiveTier(300, 22), AdaptiveTier(100, 35), AdaptiveTier(30, 30)),
            adaptive_index_tiers=(AdaptiveTier(300, 22), AdaptiveTier(100, 35), AdaptiveTier(30, 30)),
        ),
        "G active@50 (later)": V5Config(
            adaptive_highvol_tiers=(AdaptiveTier(400, 35), AdaptiveTier(200, 50), AdaptiveTier(50, 45)),
            adaptive_standard_tiers=(AdaptiveTier(300, 25), AdaptiveTier(150, 40), AdaptiveTier(50, 35)),
        ),
        "H keep70+active@50": V5Config(
            soft_trail_keep_pct=0.70,
            adaptive_highvol_tiers=(AdaptiveTier(400, 35), AdaptiveTier(200, 50), AdaptiveTier(50, 45)),
            adaptive_standard_tiers=(AdaptiveTier(300, 25), AdaptiveTier(150, 40), AdaptiveTier(50, 35)),
            adaptive_index_tiers=(AdaptiveTier(300, 25), AdaptiveTier(100, 35), AdaptiveTier(30, 30)),
        ),
        "I keep70+band15+act@50": V5Config(
            soft_trail_keep_pct=0.70,
            soft_trail_band_low_pct=15.0,
            adaptive_highvol_tiers=(AdaptiveTier(400, 35), AdaptiveTier(200, 50), AdaptiveTier(50, 45)),
            adaptive_standard_tiers=(AdaptiveTier(300, 25), AdaptiveTier(150, 40), AdaptiveTier(50, 35)),
            adaptive_index_tiers=(AdaptiveTier(300, 25), AdaptiveTier(100, 35), AdaptiveTier(30, 30)),
        ),
        # Wider scalp (don't cut small runners at 60% fade)
        "J scalp_fade=0.5": V5Config(scalp_fade_ratio=0.50, soft_trail_keep_pct=0.70),
        # Profit target for all 0DTE at 35% (not just index)
        "K profit_all_35%": V5Config(
            profit_target_index_0dte_pct=35.0,
            soft_trail_keep_pct=0.70,
        ),
    }

    print(f"{'Config':<27} {'P&L':>9} {'Win%':>6} {'W/L':>8} {'AvgHold':>7} "
          f"{'Soft':>5} {'Adapt':>5} {'Scalp':>5} {'ProfT':>5} {'AvgPeak':>7}")
    print("-" * 100)

    for name, cfg in configs.items():
        total_pnl = 0
        wins = losses = soft_n = adap_n = scalp_n = pt_n = 0
        total_hold = 0
        total_peak = 0

        for td in trade_data:
            r = simulate(
                td["df"], td["entry_premium"], td["contracts"],
                td["direction"], td["dte"], td["expiry_date"], td["ticker"], cfg,
            )
            total_pnl += r["pnl"]
            total_hold += r["hold"]
            total_peak += r["peak_gain"]
            if r["pnl"] > 0:
                wins += 1
            else:
                losses += 1
            if r["reason"] == "soft_trail":
                soft_n += 1
            elif r["reason"] == "adaptive_trail":
                adap_n += 1
            elif r["reason"] == "scalp_trail":
                scalp_n += 1
            elif r["reason"] == "profit_target":
                pt_n += 1

        n = wins + losses
        wr = wins / n * 100 if n > 0 else 0
        avg_hold = total_hold / n if n > 0 else 0
        avg_peak = total_peak / n if n > 0 else 0
        print(f"{name:<27} ${total_pnl:>7,.0f} {wr:>5.1f}% {wins:>3}/{losses:<3} "
              f"{avg_hold:>5.0f}m {soft_n:>5} {adap_n:>5} {scalp_n:>5} {pt_n:>5} "
              f"{avg_peak:>5.0f}%")

    # Show trade-by-trade comparison for best config vs default
    best_name = max(configs.keys(), key=lambda k: sum(
        simulate(td["df"], td["entry_premium"], td["contracts"],
                 td["direction"], td["dte"], td["expiry_date"], td["ticker"], configs[k])["pnl"]
        for td in trade_data
    ))

    print(f"\n{'=' * 100}")
    print(f"BEST CONFIG: {best_name}")
    print(f"{'=' * 100}")

    # Show where the best differs from default
    default_cfg = configs["A default"]
    best_cfg = configs[best_name]

    print(f"\n{'Day':<12} {'Ticker':<6} {'Dir':<5} {'Def$':>8} {'DefReason':<18} "
          f"{'Best$':>8} {'BestReason':<18} {'Delta':>8}")
    print("-" * 100)

    diffs = []
    for td in trade_data:
        r_def = simulate(td["df"], td["entry_premium"], td["contracts"],
                         td["direction"], td["dte"], td["expiry_date"], td["ticker"], default_cfg)
        r_best = simulate(td["df"], td["entry_premium"], td["contracts"],
                          td["direction"], td["dte"], td["expiry_date"], td["ticker"], best_cfg)
        if abs(r_def["pnl"] - r_best["pnl"]) > 5.0:
            delta = r_best["pnl"] - r_def["pnl"]
            diffs.append((delta, td, r_def, r_best))

    diffs.sort(key=lambda x: x[0], reverse=True)
    for delta, td, r_def, r_best in diffs[:25]:
        print(f"{td['day']:<12} {td['ticker']:<6} {td['direction'][:4]:<5} "
              f"${r_def['pnl']:>7,.0f} {r_def['reason']:<18} "
              f"${r_best['pnl']:>7,.0f} {r_best['reason']:<18} "
              f"${delta:>+7,.0f}")


if __name__ == "__main__":
    main()
