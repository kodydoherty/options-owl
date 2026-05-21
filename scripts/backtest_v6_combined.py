"""V6 Combined Backtest — All research findings in one integrated strategy.

Combines:
  1. Per-ticker optimal FSM configs (from per_ticker_tuning backtest)
  2. Break-even ratchet: once +20% gain, stop moves to break-even
  3. 2PM trail tightening: after 2PM ET, tighten all adaptive trails 30%
  4. Premium $5 cap: reject single-stock entries > $5
  5. Spread-cost gate: reject if bid-ask spread > 15% of premium
  6. Scale-out at +20%: sell 1/3 of contracts at first +20% gain
  7. MID_TRADE_DCA: add contracts at 8-20min dip (for tickers where it helps)

Compares V6 vs baseline V5 (DEFAULT config for all tickers).

Usage:
    python scripts/backtest_v6_combined.py
"""

from __future__ import annotations

import sqlite3
import sys
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from options_owl.risk.exit_v5.config import (
    V5Config, AdaptiveTier, TickerCategory, INDEX_TICKERS, categorize_ticker,
)
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState
from options_owl.risk.exit_v5.types import ExitReason

SIGNALS_DB = str(PROJECT_DIR / "journal" / "owlet-kody" / "raw_messages.db")
HARVESTER_DB = str(PROJECT_DIR / "journal" / "owlet-harvester" / "options_data.db")

PORTFOLIO = 8000

# ── Per-ticker optimal configs (from backtest_per_ticker_tuning.py) ──────────

def _base():
    return V5Config()

TICKER_CONFIGS = {
    # NVDA: EARLY_PROFIT (+$1,142 improvement)
    "NVDA": replace(_base(),
        profit_target_index_0dte_pct=20.0,
        soft_trail_band_low_pct=8.0,
        soft_trail_keep_pct=0.70,
    ),
    # GOOGL: WIDE_STOP (+$912)
    "GOOGL": replace(_base(),
        tight_stop_0dte_pct=45.0,
        backstop_0dte_pct=75.0,
        checkpoint_drop_pct=40.0,
    ),
    # TSLA: LONG_GRACE (+$822)
    "TSLA": replace(_base(), grace_period_min=8.0),
    # IWM: WIDE_STOP (+$800)
    "IWM": replace(_base(),
        tight_stop_0dte_pct=45.0,
        backstop_0dte_pct=75.0,
        checkpoint_drop_pct=40.0,
    ),
    # QQQ: LONG_GRACE (+$516)
    "QQQ": replace(_base(), grace_period_min=8.0),
    # META: DEFENSIVE (+$384)
    "META": replace(_base(),
        tight_stop_0dte_pct=25.0,
        backstop_0dte_pct=50.0,
        checkpoint_drop_pct=20.0,
        adaptive_highvol_tiers=(
            AdaptiveTier(400, 25), AdaptiveTier(150, 40), AdaptiveTier(40, 35),
        ),
        theta_bleed_min=90.0,
        theta_bleed_drop_pct=20.0,
    ),
    # AAPL: DEFENSIVE (+$291)
    "AAPL": replace(_base(),
        tight_stop_0dte_pct=25.0,
        backstop_0dte_pct=50.0,
        checkpoint_drop_pct=20.0,
        adaptive_highvol_tiers=(
            AdaptiveTier(400, 25), AdaptiveTier(150, 40), AdaptiveTier(40, 35),
        ),
        theta_bleed_min=90.0,
        theta_bleed_drop_pct=20.0,
    ),
    # AMZN: TIGHT_TRAIL (+$261)
    "AMZN": replace(_base(),
        adaptive_highvol_tiers=(
            AdaptiveTier(400, 25), AdaptiveTier(150, 40), AdaptiveTier(40, 35),
        ),
        adaptive_standard_tiers=(
            AdaptiveTier(300, 20), AdaptiveTier(100, 30), AdaptiveTier(30, 25),
        ),
    ),
    # AVGO: EARLY_PROFIT (+$210)
    "AVGO": replace(_base(),
        profit_target_index_0dte_pct=20.0,
        soft_trail_band_low_pct=8.0,
        soft_trail_keep_pct=0.70,
    ),
    # MSFT: EARLY_PROFIT (+$48)
    "MSFT": replace(_base(),
        profit_target_index_0dte_pct=20.0,
        soft_trail_band_low_pct=8.0,
        soft_trail_keep_pct=0.70,
    ),
    # MSTR: TIGHT+QUICK (+$40)
    "MSTR": replace(_base(),
        adaptive_highvol_tiers=(
            AdaptiveTier(400, 25), AdaptiveTier(150, 40), AdaptiveTier(40, 35),
        ),
        scalp_peak_threshold_pct=15.0,
        scalp_fade_ratio=0.50,
        soft_trail_keep_pct=0.70,
    ),
}

# Tickers where MID_TRADE_DCA helps (from backtest_dca_delayed.py)
DCA_TICKERS = {"MSFT", "IWM", "SPY", "QQQ", "AMZN", "NVDA"}

# ── Data loading ─────────────────────────────────────────────────────────────

def load_signals():
    conn = sqlite3.connect(SIGNALS_DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT id, ticker, direction, sentiment, score,
               atm_premium, otm_premium, strike, expiry,
               entry_price, created_at
        FROM trade_signals
        WHERE score >= 78
        ORDER BY created_at
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


def compute_contracts(entry_premium, score):
    deployable = PORTFOLIO * 0.75
    per_slot = deployable / 5
    position_cap = PORTFOLIO * 0.15
    if score >= 95: sm = 1.0
    elif score >= 90: sm = 0.75
    elif score >= 85: sm = 0.50
    else: sm = 0.25
    cost_per = entry_premium * 100
    scaled = per_slot * sm
    raw = int(scaled / cost_per) if cost_per > 0 else 1
    pos_cap = int(position_cap / cost_per) if cost_per > 0 else 1
    return max(1, min(raw, pos_cap, 20))


# ── Simulation ───────────────────────────────────────────────────────────────


def simulate_v6(df, entry_premium, contracts, direction, dte, expiry_date,
                ticker, cfg, enable_v6=False):
    """Run FSM with optional V6 enhancements.

    V6 enhancements (when enable_v6=True):
      - Break-even ratchet: once +20%, hard floor at entry
      - 2PM trail tightening: after 2PM ET, tighten adaptive trail 30%
      - Scale-out at +20%: sell 1/3 of contracts
      - MID_TRADE_DCA: add contracts at 8-20min dip for eligible tickers
    """
    if entry_premium <= 0:
        return {"pnl": 0, "reason": "no_data", "hold": 0, "exit_prem": 0,
                "peak_gain": 0, "contracts_final": contracts,
                "scaleout_pnl": 0, "dca_fired": False, "entry_filtered": False}

    option_type = "put" if direction in ("bearish", "put") else "call"
    is_call = option_type in ("call", "bullish")

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

    # V6: scale-out tracking
    remaining_contracts = contracts
    scaleout_pnl = 0.0
    scaled_out = False

    # V6: DCA tracking
    dca_fired = False
    total_cost = entry_premium * contracts * 100
    avg_entry = entry_premium

    # V6: break-even ratchet
    hit_20pct = False

    fsm = ExitFSM(cfg)
    state = TradeState(
        trade_id=1, ticker=ticker, option_type=option_type,
        entry_premium=entry_premium, entry_time=entry_ts,
        contracts=remaining_contracts, peak_premium=entry_premium,
        entry_underlying_price=first_underlying,
        dte=dte, expiry_date=expiry_date or "",
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
        elapsed_min = (now - entry_ts).total_seconds() / 60.0

        # ET hour (timestamps are UTC, ET = UTC-4)
        et_hour = now.hour - 4
        if et_hour < 0:
            et_hour += 24
        et_minute = now.minute
        minutes_to_close = max(0, (16 * 60) - (et_hour * 60 + et_minute))

        gain = (premium - avg_entry) / avg_entry * 100 if avg_entry > 0 else 0

        if enable_v6:
            # ── V6: MID_TRADE_DCA (8-20min dip, underlying OK) ──
            if (not dca_fired and ticker in DCA_TICKERS
                    and 8.0 <= elapsed_min <= 20.0):
                dip_pct = (entry_premium - premium) / entry_premium * 100
                if 15.0 <= dip_pct <= 35.0:
                    underlying_ok = True
                    if first_underlying > 0 and underlying > 0:
                        u_move = (underlying - first_underlying) / first_underlying * 100
                        if is_call and u_move < -0.5:
                            underlying_ok = False
                        elif not is_call and u_move > 0.5:
                            underlying_ok = False
                    if underlying_ok:
                        dca_fired = True
                        add_ct = max(1, contracts)
                        total_cost += premium * add_ct * 100
                        remaining_contracts += add_ct
                        avg_entry = total_cost / (remaining_contracts * 100)
                        state.entry_premium = avg_entry
                        state.contracts = remaining_contracts
                        state.peak_premium = max(avg_entry, premium)
                        gain = (premium - avg_entry) / avg_entry * 100

            # ── V6: Scale-out at +20% (sell 1/3) ──
            if not scaled_out and gain >= 20 and remaining_contracts >= 3:
                scaleout_ct = remaining_contracts // 3
                scaleout_pnl = (premium - avg_entry) * scaleout_ct * 100
                remaining_contracts -= scaleout_ct
                scaled_out = True
                state.contracts = remaining_contracts

            # ── V6: Break-even ratchet ──
            if gain >= 20:
                hit_20pct = True
            if hit_20pct and premium < avg_entry:
                # Exit at break-even (actually slightly below due to poll interval)
                elapsed = (now - entry_ts).total_seconds() / 60
                peak_gain = (state.peak_premium - avg_entry) / avg_entry * 100 if avg_entry > 0 else 0
                pnl = (premium - avg_entry) * remaining_contracts * 100 + scaleout_pnl
                return {"pnl": pnl, "reason": "breakeven_ratchet",
                        "hold": elapsed, "exit_prem": premium,
                        "peak_gain": peak_gain,
                        "contracts_final": remaining_contracts,
                        "scaleout_pnl": scaleout_pnl,
                        "dca_fired": dca_fired, "entry_filtered": False}

            # ── V6: 2PM trail tightening ──
            # Dynamically create tighter config after 2PM ET
            if et_hour >= 14 and cfg is not fsm.cfg:
                pass  # already tightened
            elif et_hour >= 14:
                # Tighten adaptive tiers by 30%
                cat = categorize_ticker(ticker)
                base_tiers = cfg.get_adaptive_tiers(cat)
                tight_tiers = tuple(
                    AdaptiveTier(t.min_peak_gain, t.trail_width * 0.7)
                    for t in base_tiers
                )
                # Also tighten soft trail keep
                tight_cfg = replace(cfg,
                    soft_trail_keep_pct=min(0.80, cfg.soft_trail_keep_pct + 0.15),
                )
                if cat == TickerCategory.HIGH_VOL:
                    tight_cfg = replace(tight_cfg, adaptive_highvol_tiers=tight_tiers)
                elif cat == TickerCategory.INDEX:
                    tight_cfg = replace(tight_cfg, adaptive_index_tiers=tight_tiers)
                else:
                    tight_cfg = replace(tight_cfg, adaptive_standard_tiers=tight_tiers)
                fsm = ExitFSM(tight_cfg)

        # Run FSM evaluation
        action = fsm.evaluate(
            state, premium, bid, ask, now,
            current_underlying=underlying,
            minutes_to_close=minutes_to_close,
        )

        if action.should_exit:
            elapsed = (now - entry_ts).total_seconds() / 60
            peak_gain = (state.peak_premium - avg_entry) / avg_entry * 100 if avg_entry > 0 else 0
            pnl = (premium - avg_entry) * remaining_contracts * 100 + scaleout_pnl
            return {"pnl": pnl, "reason": action.reason.value,
                    "hold": elapsed, "exit_prem": premium,
                    "peak_gain": peak_gain,
                    "contracts_final": remaining_contracts,
                    "scaleout_pnl": scaleout_pnl,
                    "dca_fired": dca_fired, "entry_filtered": False}

    # End of data
    last_prem = df["premium"].iloc[-1]
    last_ts = df["ts"].iloc[-1]
    if hasattr(last_ts, 'to_pydatetime'):
        last_ts = last_ts.to_pydatetime()
    if last_ts.tzinfo is not None:
        last_ts = last_ts.replace(tzinfo=None)
    elapsed = (last_ts - entry_ts).total_seconds() / 60
    peak_gain = (state.peak_premium - avg_entry) / avg_entry * 100 if avg_entry > 0 else 0
    pnl = (last_prem - avg_entry) * remaining_contracts * 100 + scaleout_pnl
    return {"pnl": pnl, "reason": "eod_data_end",
            "hold": elapsed, "exit_prem": last_prem,
            "peak_gain": peak_gain,
            "contracts_final": remaining_contracts,
            "scaleout_pnl": scaleout_pnl,
            "dca_fired": dca_fired, "entry_filtered": False}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    signals = load_signals()
    print(f"Loaded {len(signals)} signals from DB")

    harvester_conn = sqlite3.connect(HARVESTER_DB)

    baseline_results = []
    v6_results = []
    no_data = 0
    filtered = 0

    for sig in signals:
        ticker = sig["ticker"]
        direction = (sig["direction"] or "bullish").lower()
        score = sig["score"] or 80
        day = sig["created_at"][:10]

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

        first_bid = df["bid"].iloc[0]
        first_ask_val = df["ask"].iloc[0]

        contracts = compute_contracts(adj_entry, score)
        meta = {"ticker": ticker, "day": day, "score": score,
                "entry": adj_entry, "contracts": contracts,
                "direction": direction, "dte": dte}

        # ── BASELINE: default V5Config for all tickers ──
        base_result = simulate_v6(
            df, adj_entry, contracts, direction, dte, expiry_date,
            ticker=ticker, cfg=V5Config(), enable_v6=False,
        )
        base_result.update(meta)
        baseline_results.append(base_result)

        # ── V6: entry filters + per-ticker config + enhancements ──

        # Entry filter 1: Premium $5 cap for non-index tickers
        is_index = ticker in INDEX_TICKERS
        if not is_index and adj_entry > 5.0:
            v6_result = {"pnl": 0, "reason": "premium_cap_filtered",
                         "hold": 0, "exit_prem": 0, "peak_gain": 0,
                         "contracts_final": 0, "scaleout_pnl": 0,
                         "dca_fired": False, "entry_filtered": True}
            v6_result.update(meta)
            v6_results.append(v6_result)
            filtered += 1
            continue

        # Entry filter 2: Spread-cost gate (reject if spread > 15% of premium)
        if (first_bid and first_ask_val and not pd.isna(first_bid)
                and not pd.isna(first_ask_val)
                and first_ask_val > 0 and first_bid > 0):
            spread = first_ask_val - first_bid
            spread_pct = spread / adj_entry * 100
            if spread_pct > 15:
                v6_result = {"pnl": 0, "reason": "spread_filtered",
                             "hold": 0, "exit_prem": 0, "peak_gain": 0,
                             "contracts_final": 0, "scaleout_pnl": 0,
                             "dca_fired": False, "entry_filtered": True}
                v6_result.update(meta)
                v6_results.append(v6_result)
                filtered += 1
                continue

        # Per-ticker config
        ticker_cfg = TICKER_CONFIGS.get(ticker, V5Config())

        v6_result = simulate_v6(
            df, adj_entry, contracts, direction, dte, expiry_date,
            ticker=ticker, cfg=ticker_cfg, enable_v6=True,
        )
        v6_result.update(meta)
        v6_results.append(v6_result)

    harvester_conn.close()

    if not baseline_results:
        print("No results")
        return

    n_trades = len(baseline_results)

    # ── Summary ──────────────────────────────────────────────────────────

    df_base = pd.DataFrame(baseline_results)
    df_v6 = pd.DataFrame(v6_results)

    # Exclude filtered trades from V6 win-rate calc
    df_v6_traded = df_v6[~df_v6["entry_filtered"]]

    b_pnl = df_base["pnl"].sum()
    v_pnl = df_v6["pnl"].sum()
    b_wins = (df_base["pnl"] > 0).sum()
    v_wins = (df_v6_traded["pnl"] > 0).sum() if len(df_v6_traded) > 0 else 0
    b_wr = b_wins / len(df_base) * 100
    v_wr = v_wins / len(df_v6_traded) * 100 if len(df_v6_traded) > 0 else 0

    print(f"\n{'=' * 110}")
    print(f"V6 COMBINED BACKTEST — {n_trades} signals, {no_data} no data, {filtered} filtered by V6")
    print(f"{'=' * 110}")

    print(f"\n{'':>30} {'BASELINE (V5)':>15} {'V6 COMBINED':>15} {'DELTA':>15}")
    print("-" * 78)
    print(f"{'Total P&L':>30} ${b_pnl:>13,.2f} ${v_pnl:>13,.2f} ${v_pnl - b_pnl:>+13,.2f}")
    print(f"{'Trades Executed':>30} {len(df_base):>15} {len(df_v6_traded):>15} {len(df_v6_traded) - len(df_base):>+15}")
    print(f"{'Win Rate':>30} {b_wr:>14.1f}% {v_wr:>14.1f}% {v_wr - b_wr:>+14.1f}%")
    b_avg_w = df_base[df_base['pnl'] > 0]['pnl'].mean() if b_wins > 0 else 0
    v_avg_w = df_v6_traded[df_v6_traded['pnl'] > 0]['pnl'].mean() if v_wins > 0 else 0
    b_avg_l = df_base[df_base['pnl'] <= 0]['pnl'].mean() if (df_base['pnl'] <= 0).sum() > 0 else 0
    v_avg_l = df_v6_traded[df_v6_traded['pnl'] <= 0]['pnl'].mean() if (df_v6_traded['pnl'] <= 0).sum() > 0 else 0
    print(f"{'Avg Win':>30} ${b_avg_w:>13,.2f} ${v_avg_w:>13,.2f} ${v_avg_w - b_avg_w:>+13,.2f}")
    print(f"{'Avg Loss':>30} ${b_avg_l:>13,.2f} ${v_avg_l:>13,.2f} ${v_avg_l - b_avg_l:>+13,.2f}")
    print(f"{'Max Win':>30} ${df_base['pnl'].max():>13,.2f} ${df_v6['pnl'].max():>13,.2f}")
    print(f"{'Max Loss':>30} ${df_base['pnl'].min():>13,.2f} ${df_v6['pnl'].min():>13,.2f}")

    # ── V6 features breakdown ────────────────────────────────────────────

    be_count = (df_v6["reason"] == "breakeven_ratchet").sum()
    dca_count = df_v6["dca_fired"].sum()
    so_count = (df_v6["scaleout_pnl"] != 0).sum()
    filt_premium = (df_v6["reason"] == "premium_cap_filtered").sum()
    filt_spread = (df_v6["reason"] == "spread_filtered").sum()

    print(f"\n--- V6 Feature Activity ---")
    print(f"  Premium cap filtered:  {filt_premium} trades blocked")
    print(f"  Spread gate filtered:  {filt_spread} trades blocked")
    print(f"  Break-even ratchets:   {be_count} exits")
    print(f"  Scale-outs at +20%:    {so_count} partial fills (${df_v6['scaleout_pnl'].sum():,.2f} locked)")
    print(f"  DCA additions:         {dca_count} adds")

    # ── Per-ticker comparison ────────────────────────────────────────────

    print(f"\n{'=' * 110}")
    print("PER-TICKER P&L COMPARISON")
    print(f"{'=' * 110}")

    tickers = sorted(set(r["ticker"] for r in baseline_results))
    print(f"\n{'Ticker':<8} {'Trades':>6} {'Baseline':>12} {'V6':>12} {'Delta':>12} {'Filt':>5} {'DCA':>4} {'BE':>4} {'SO':>4} {'Config Used':<18}")
    print("-" * 100)

    total_b = 0
    total_v = 0
    for t in tickers:
        t_base = sum(r["pnl"] for r in baseline_results if r["ticker"] == t)
        t_v6 = sum(r["pnl"] for r in v6_results if r["ticker"] == t)
        t_n = sum(1 for r in baseline_results if r["ticker"] == t)
        t_filt = sum(1 for r in v6_results if r["ticker"] == t and r.get("entry_filtered"))
        t_dca = sum(1 for r in v6_results if r["ticker"] == t and r.get("dca_fired"))
        t_be = sum(1 for r in v6_results if r["ticker"] == t and r["reason"] == "breakeven_ratchet")
        t_so = sum(1 for r in v6_results if r["ticker"] == t and r.get("scaleout_pnl", 0) != 0)
        delta = t_v6 - t_base
        cfg_name = "per-ticker" if t in TICKER_CONFIGS else "DEFAULT"
        if t in TICKER_CONFIGS:
            # Identify the config name
            names = {"NVDA": "EARLY_PROFIT", "GOOGL": "WIDE_STOP", "TSLA": "LONG_GRACE",
                     "IWM": "WIDE_STOP", "QQQ": "LONG_GRACE", "META": "DEFENSIVE",
                     "AAPL": "DEFENSIVE", "AMZN": "TIGHT_TRAIL", "AVGO": "EARLY_PROFIT",
                     "MSFT": "EARLY_PROFIT", "MSTR": "TIGHT+QUICK"}
            cfg_name = names.get(t, "CUSTOM")

        marker = " +" if delta > 50 else (" -" if delta < -50 else "")
        total_b += t_base
        total_v += t_v6
        print(f"{t:<8} {t_n:>6} ${t_base:>10,.2f} ${t_v6:>10,.2f} ${delta:>+10,.2f} {t_filt:>5} {t_dca:>4} {t_be:>4} {t_so:>4} {cfg_name:<18}{marker}")

    print(f"\n{'TOTAL':<8} {n_trades:>6} ${total_b:>10,.2f} ${total_v:>10,.2f} ${total_v - total_b:>+10,.2f}")

    # ── Per-trade detail ─────────────────────────────────────────────────

    print(f"\n{'=' * 110}")
    print("PER-TRADE DETAIL")
    print(f"{'=' * 110}")
    print(f"\n{'Day':<12} {'Ticker':<6} {'Dir':<5} {'Score':>5} {'Entry':>7} {'Ct':>3} "
          f"{'Base P&L':>10} {'V6 P&L':>10} {'Delta':>10} {'V6 Reason':<20} {'Notes'}")
    print("-" * 120)

    for i in range(len(baseline_results)):
        b = baseline_results[i]
        v = v6_results[i]
        delta = v["pnl"] - b["pnl"]
        notes = []
        if v.get("entry_filtered"):
            notes.append("FILTERED")
        if v.get("dca_fired"):
            notes.append("DCA")
        if v.get("scaleout_pnl", 0) != 0:
            notes.append(f"SO:${v['scaleout_pnl']:.0f}")
        if v["reason"] == "breakeven_ratchet":
            notes.append("BE_RATCHET")
        note_str = " ".join(notes)
        marker = ""
        if delta > 50: marker = " +"
        elif delta < -50: marker = " -"

        print(f"{b['day']:<12} {b['ticker']:<6} {b['direction'][:4]:<5} {b['score']:>5} "
              f"${b['entry']:>5.2f} {b['contracts']:>3} "
              f"${b['pnl']:>8,.2f} ${v['pnl']:>8,.2f} ${delta:>+8,.2f} "
              f"{v['reason']:<20} {note_str}{marker}")

    # ── Exit reason breakdown ────────────────────────────────────────────

    print(f"\n{'=' * 110}")
    print("EXIT REASON BREAKDOWN — V6")
    print(f"{'=' * 110}")
    print(f"\n{'Reason':<25} {'Count':>6} {'Total P&L':>12} {'Avg P&L':>10} {'Win%':>6}")
    print("-" * 62)
    for reason, group in df_v6.groupby("reason"):
        gpnl = group["pnl"]
        gwins = (gpnl > 0).sum()
        gwr = gwins / len(gpnl) * 100 if len(gpnl) > 0 else 0
        print(f"{reason:<25} {len(gpnl):>6} ${gpnl.sum():>10,.2f} ${gpnl.mean():>8,.2f} {gwr:>5.0f}%")

    # ── Daily P&L ────────────────────────────────────────────────────────

    print(f"\n{'=' * 110}")
    print("DAILY P&L COMPARISON")
    print(f"{'=' * 110}")

    daily_b = df_base.groupby("day")["pnl"].sum()
    daily_v = df_v6.groupby("day")["pnl"].sum()
    all_days = sorted(set(daily_b.index) | set(daily_v.index))

    cum_b = 0
    cum_v = 0
    print(f"\n{'Day':<12} {'Base Day':>10} {'V6 Day':>10} {'Delta':>10} {'Base Cum':>10} {'V6 Cum':>10}")
    print("-" * 65)
    for day in all_days:
        b_day = daily_b.get(day, 0)
        v_day = daily_v.get(day, 0)
        cum_b += b_day
        cum_v += v_day
        delta = v_day - b_day
        print(f"{day:<12} ${b_day:>8,.2f} ${v_day:>8,.2f} ${delta:>+8,.2f} ${cum_b:>8,.2f} ${cum_v:>8,.2f}")


if __name__ == "__main__":
    main()
