"""V6 Production Build — Full Signal-by-Signal Report.

Shows every signal with entry, exit, P&L, exit reason, and V6 features fired.
Organized by day with daily subtotals and running cumulative P&L.

Uses the ACTUAL production ExitFSM + DCA (same code that runs on the droplet).

Usage:
    python scripts/backtest_v6_full_report.py
"""

from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from options_owl.risk.exit_v5.config import (
    INDEX_TICKERS,
    TICKER_CONFIGS,
    V5Config,
    get_ticker_config,
)
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState
from options_owl.risk.exit_v5.types import ExitReason

SIGNALS_DB = str(PROJECT_DIR / "journal" / "owlet-kody" / "raw_messages.db")
HARVESTER_DB = str(PROJECT_DIR / "journal" / "owlet-harvester" / "options_data.db")
PORTFOLIO = 8000


def _v6_settings():
    return SimpleNamespace(
        ENABLE_V6_PER_TICKER_CONFIG=True,
        ENABLE_V6_BREAKEVEN_RATCHET=True,
        V6_BREAKEVEN_TRIGGER_PCT=20.0,
        ENABLE_V6_2PM_TIGHTEN=True,
        V6_2PM_TRAIL_TIGHTEN_FACTOR=0.7,
        V6_2PM_SOFT_TRAIL_BOOST=0.15,
        ENABLE_V6_PREMIUM_CAP=True,
        V6_PREMIUM_CAP=5.0,
        ENABLE_V6_SPREAD_GATE=True,
        V6_MAX_SPREAD_PCT=15.0,
        ENABLE_V6_SCALEOUT=True,
        V6_SCALEOUT_GAIN_PCT=20.0,
        V6_SCALEOUT_FRACTION=0.333,
        V6_SCALEOUT_MIN_CONTRACTS=3,
        ENABLE_V6_DCA=True,
        V6_DCA_TICKERS="MSFT,IWM,SPY,QQQ,AMZN,NVDA",
        V6_DCA_MIN_MINUTES=8.0,
        V6_DCA_MAX_MINUTES=20.0,
        V6_DCA_MIN_DIP_PCT=15.0,
        V6_DCA_MAX_DIP_PCT=35.0,
        V6_DCA_UNDERLYING_THRESHOLD=0.5,
    )


def _v5_baseline_settings():
    return SimpleNamespace(
        ENABLE_V6_PER_TICKER_CONFIG=False,
        ENABLE_V6_BREAKEVEN_RATCHET=False,
        ENABLE_V6_2PM_TIGHTEN=False,
        ENABLE_V6_SCALEOUT=False,
        ENABLE_V6_DCA=False,
    )


# ── Data loading ─────────────────────────────────────────────────────────


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


# ── Simulation ───────────────────────────────────────────────────────────


def simulate(df, entry_premium, contracts, direction, dte, expiry_date,
             ticker, settings):
    """Run production V6 FSM + DCA. Returns detailed result dict."""
    if entry_premium <= 0:
        return {
            "pnl": 0, "reason": "no_data", "hold_min": 0, "peak_gain": 0,
            "exit_prem": 0, "entry_prem": entry_premium,
            "scaleout_pnl": 0, "dca_fired": False, "dca_add": 0,
            "final_contracts": contracts, "entry_time": "", "exit_time": "",
        }

    option_type = "put" if direction in ("bearish", "put") else "call"
    is_call = option_type in ("call", "bullish")

    use_per_ticker = getattr(settings, "ENABLE_V6_PER_TICKER_CONFIG", False)
    cfg = get_ticker_config(ticker, use_per_ticker=use_per_ticker)
    fsm = ExitFSM(cfg, settings=settings)

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

    remaining_contracts = contracts
    scaleout_pnl = 0.0
    avg_entry = entry_premium
    total_cost = entry_premium * contracts * 100

    enable_dca = getattr(settings, "ENABLE_V6_DCA", False)
    dca_tickers_str = getattr(settings, "V6_DCA_TICKERS", "")
    dca_tickers = {t.strip().upper() for t in dca_tickers_str.split(",") if t.strip()}
    dca_fired = False
    dca_add = 0

    def _result(pnl, reason, exit_prem, now):
        elapsed = (now - entry_ts).total_seconds() / 60
        peak_gain = (state.peak_premium - avg_entry) / avg_entry * 100 if avg_entry > 0 else 0
        return {
            "pnl": round(pnl, 2), "reason": reason,
            "hold_min": round(elapsed, 1),
            "peak_gain": round(peak_gain, 1),
            "exit_prem": round(exit_prem, 4),
            "entry_prem": round(avg_entry, 4),
            "scaleout_pnl": round(scaleout_pnl, 2),
            "dca_fired": dca_fired, "dca_add": dca_add,
            "final_contracts": remaining_contracts,
            "entry_time": entry_ts.strftime("%H:%M") if entry_ts else "",
            "exit_time": now.strftime("%H:%M") if now else "",
        }

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
        et_hour = now.hour - 4
        if et_hour < 0:
            et_hour += 24
        et_minute = now.minute
        minutes_to_close = max(0, (16 * 60) - (et_hour * 60 + et_minute))

        # V6 DCA
        if enable_dca and not dca_fired and ticker in dca_tickers:
            dca_min = getattr(settings, "V6_DCA_MIN_MINUTES", 8.0)
            dca_max = getattr(settings, "V6_DCA_MAX_MINUTES", 20.0)
            dca_min_dip = getattr(settings, "V6_DCA_MIN_DIP_PCT", 15.0)
            dca_max_dip = getattr(settings, "V6_DCA_MAX_DIP_PCT", 35.0)
            u_thresh = getattr(settings, "V6_DCA_UNDERLYING_THRESHOLD", 0.5)

            if dca_min <= elapsed_min <= dca_max and avg_entry > 0:
                dip_pct = (avg_entry - premium) / avg_entry * 100
                if dca_min_dip <= dip_pct <= dca_max_dip:
                    underlying_ok = True
                    if first_underlying > 0 and underlying > 0:
                        u_move = (underlying - first_underlying) / first_underlying * 100
                        if is_call and u_move < -u_thresh:
                            underlying_ok = False
                        elif not is_call and u_move > u_thresh:
                            underlying_ok = False
                    if underlying_ok:
                        dca_fired = True
                        dca_add = contracts
                        total_cost += premium * dca_add * 100
                        remaining_contracts += dca_add
                        avg_entry = total_cost / (remaining_contracts * 100)
                        state.entry_premium = avg_entry
                        state.contracts = remaining_contracts
                        state.peak_premium = max(avg_entry, premium)

        action = fsm.evaluate(
            state, premium, bid, ask, now,
            current_underlying=underlying,
            minutes_to_close=minutes_to_close,
        )

        if action.should_exit:
            if action.contracts_to_close > 0:
                close_qty = min(action.contracts_to_close, remaining_contracts)
                scaleout_pnl += (premium - avg_entry) * close_qty * 100
                remaining_contracts -= close_qty
                state.contracts = remaining_contracts
                if remaining_contracts <= 0:
                    return _result(scaleout_pnl, action.reason.value, premium, now)
                continue

            pnl = (premium - avg_entry) * remaining_contracts * 100 + scaleout_pnl
            return _result(pnl, action.reason.value, premium, now)

    last_prem = df["premium"].iloc[-1]
    last_ts = df["ts"].iloc[-1]
    if hasattr(last_ts, 'to_pydatetime'):
        last_ts = last_ts.to_pydatetime()
    if last_ts.tzinfo is not None:
        last_ts = last_ts.replace(tzinfo=None)
    pnl = (last_prem - avg_entry) * remaining_contracts * 100 + scaleout_pnl
    return _result(pnl, "eod_data_end", last_prem, last_ts)


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    signals = load_signals()
    harvester_conn = sqlite3.connect(HARVESTER_DB)
    v6_settings = _v6_settings()
    v5_settings = _v5_baseline_settings()

    results = []
    no_data_signals = []

    for sig in signals:
        ticker = sig["ticker"]
        direction = (sig["direction"] or "bullish").lower()
        score = sig["score"] or 80
        day = sig["created_at"][:10]
        sig_time = sig["created_at"][11:16] if len(sig["created_at"]) > 11 else ""

        df = load_ticks(harvester_conn, sig)
        if df is None:
            no_data_signals.append({
                "day": day, "time": sig_time, "ticker": ticker,
                "direction": direction, "score": score,
                "strike": sig["strike"], "premium": sig["premium"],
            })
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

        # Check V6 entry filters
        is_index = ticker in INDEX_TICKERS
        filtered_reason = None
        if not is_index and adj_entry > 5.0:
            filtered_reason = "premium_cap"
        elif (first_bid and first_ask_val and not pd.isna(first_bid)
                and not pd.isna(first_ask_val)
                and first_ask_val > 0 and first_bid > 0):
            spread_pct = (first_ask_val - first_bid) / adj_entry * 100
            if spread_pct > 15:
                filtered_reason = f"spread_{spread_pct:.0f}%"

        # V5 baseline
        v5 = simulate(df, adj_entry, contracts, direction, dte, expiry_date,
                       ticker, v5_settings)

        # V6 production
        if filtered_reason:
            v6 = {
                "pnl": 0, "reason": f"FILTERED:{filtered_reason}",
                "hold_min": 0, "peak_gain": 0, "exit_prem": 0,
                "entry_prem": adj_entry, "scaleout_pnl": 0,
                "dca_fired": False, "dca_add": 0,
                "final_contracts": 0, "entry_time": "", "exit_time": "",
            }
        else:
            v6 = simulate(df, adj_entry, contracts, direction, dte, expiry_date,
                           ticker, v6_settings)

        # Build per-ticker config label
        cfg_label = "DEFAULT"
        if ticker in TICKER_CONFIGS:
            cfg_names = {
                "NVDA": "EARLY_PROFIT", "GOOGL": "WIDE_STOP", "TSLA": "LONG_GRACE",
                "IWM": "WIDE_STOP", "QQQ": "LONG_GRACE", "META": "DEFENSIVE",
                "AAPL": "DEFENSIVE", "AMZN": "TIGHT_TRAIL", "AVGO": "EARLY_PROFIT",
                "MSFT": "EARLY_PROFIT", "MSTR": "TIGHT+QUICK",
            }
            cfg_label = cfg_names.get(ticker, "CUSTOM")

        results.append({
            "day": day, "sig_time": sig_time, "ticker": ticker,
            "dir": direction[:4].upper(), "score": score,
            "strike": sig["strike"], "contracts": contracts,
            "entry": adj_entry, "dte": dte,
            # V5
            "v5_pnl": v5["pnl"], "v5_reason": v5["reason"],
            "v5_exit_prem": v5["exit_prem"], "v5_hold": v5["hold_min"],
            "v5_peak": v5["peak_gain"],
            # V6
            "v6_pnl": v6["pnl"], "v6_reason": v6["reason"],
            "v6_exit_prem": v6["exit_prem"], "v6_hold": v6["hold_min"],
            "v6_peak": v6["peak_gain"],
            "v6_scaleout": v6["scaleout_pnl"],
            "v6_dca": v6["dca_fired"], "v6_dca_add": v6["dca_add"],
            "v6_final_ct": v6["final_contracts"],
            "v6_entry_time": v6["entry_time"], "v6_exit_time": v6["exit_time"],
            "cfg": cfg_label,
            "delta": v6["pnl"] - v5["pnl"],
        })

    harvester_conn.close()

    if not results:
        print("No results")
        return

    # ══════════════════════════════════════════════════════════════════════
    # HEADER
    # ══════════════════════════════════════════════════════════════════════

    total_v5 = sum(r["v5_pnl"] for r in results)
    total_v6 = sum(r["v6_pnl"] for r in results)
    traded = [r for r in results if not r["v6_reason"].startswith("FILTERED")]
    v6_wins = sum(1 for r in traded if r["v6_pnl"] > 0)
    v6_losses = sum(1 for r in traded if r["v6_pnl"] <= 0)
    v5_wins = sum(1 for r in results if r["v5_pnl"] > 0)
    v5_losses = sum(1 for r in results if r["v5_pnl"] <= 0)
    filtered_count = sum(1 for r in results if r["v6_reason"].startswith("FILTERED"))
    dca_count = sum(1 for r in results if r["v6_dca"])

    w = 130
    print("=" * w)
    print("V6 PRODUCTION BUILD — FULL BACKTEST REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  Portfolio: ${PORTFOLIO:,}")
    print("=" * w)
    print()
    print(f"  {'':>30} {'V5 BASELINE':>14} {'V6 PROD':>14} {'DELTA':>14}")
    print(f"  {'-' * 72}")
    print(f"  {'Total P&L':>30} ${total_v5:>12,.2f} ${total_v6:>12,.2f} ${total_v6 - total_v5:>+12,.2f}")
    print(f"  {'Signals':>30} {len(results):>14} {len(traded):>14} {-filtered_count:>+14} filtered")
    print(f"  {'Win Rate':>30} {v5_wins}/{v5_wins+v5_losses} ({v5_wins/(v5_wins+v5_losses)*100:.0f}%){'':<5}"
          f"{v6_wins}/{v6_wins+v6_losses} ({v6_wins/(v6_wins+v6_losses)*100:.0f}%)" if traded else "")
    avg_v5_win = np.mean([r["v5_pnl"] for r in results if r["v5_pnl"] > 0]) if v5_wins else 0
    avg_v5_loss = np.mean([r["v5_pnl"] for r in results if r["v5_pnl"] <= 0]) if v5_losses else 0
    avg_v6_win = np.mean([r["v6_pnl"] for r in traded if r["v6_pnl"] > 0]) if v6_wins else 0
    avg_v6_loss = np.mean([r["v6_pnl"] for r in traded if r["v6_pnl"] <= 0]) if v6_losses else 0
    print(f"  {'Avg Win':>30} ${avg_v5_win:>12,.2f} ${avg_v6_win:>12,.2f}")
    print(f"  {'Avg Loss':>30} ${avg_v5_loss:>12,.2f} ${avg_v6_loss:>12,.2f}")
    print(f"  {'DCA Fires':>30} {'N/A':>14} {dca_count:>14}")
    print(f"  {'Filtered (cap/spread)':>30} {'N/A':>14} {filtered_count:>14}")
    print(f"  {'No Data (skipped)':>30} {len(no_data_signals):>14}")
    print()

    # ══════════════════════════════════════════════════════════════════════
    # PER-TRADE DETAIL, GROUPED BY DAY
    # ══════════════════════════════════════════════════════════════════════

    days = sorted(set(r["day"] for r in results))
    cum_v5 = 0.0
    cum_v6 = 0.0

    for day in days:
        day_results = [r for r in results if r["day"] == day]
        day_v5 = sum(r["v5_pnl"] for r in day_results)
        day_v6 = sum(r["v6_pnl"] for r in day_results)
        cum_v5 += day_v5
        cum_v6 += day_v6
        day_wins = sum(1 for r in day_results if r["v6_pnl"] > 0 and not r["v6_reason"].startswith("FILTERED"))
        day_losses = sum(1 for r in day_results if r["v6_pnl"] <= 0 and not r["v6_reason"].startswith("FILTERED"))
        day_traded = day_wins + day_losses

        print("=" * w)
        wr_str = f"{day_wins}/{day_traded} ({day_wins/day_traded*100:.0f}%)" if day_traded else "N/A"
        print(f"  {day}  |  {len(day_results)} signals  |  "
              f"V5: ${day_v5:>+9,.2f}  |  V6: ${day_v6:>+9,.2f}  |  "
              f"Delta: ${day_v6 - day_v5:>+9,.2f}  |  WR: {wr_str}  |  "
              f"Cum V6: ${cum_v6:>+10,.2f}")
        print("=" * w)

        # Column headers
        print(f"  {'#':>3} {'Time':>5} {'Ticker':<6} {'Dir':<5} {'Scr':>3} "
              f"{'Strike':>8} {'Ct':>3} {'Entry':>7} {'DTE':>3} "
              f"{'V5 P&L':>10} {'V5 Exit':>18} "
              f"{'V6 P&L':>10} {'V6 Exit':>18} "
              f"{'Delta':>9} {'Notes'}")
        print(f"  {'-' * (w - 2)}")

        for i, r in enumerate(day_results, 1):
            # Build notes column
            notes = []
            if r["v6_reason"].startswith("FILTERED"):
                notes.append(r["v6_reason"].replace("FILTERED:", ""))
            if r["v6_dca"]:
                notes.append(f"DCA+{r['v6_dca_add']}")
            if r["v6_scaleout"] != 0:
                notes.append(f"SO:${r['v6_scaleout']:.0f}")
            if r["cfg"] != "DEFAULT":
                notes.append(r["cfg"])

            # Win/loss marker
            if r["v6_pnl"] > 50:
                marker = "W"
            elif r["v6_pnl"] < -50:
                marker = "L"
            else:
                marker = " "

            # Format exit info with hold time
            v5_exit_str = f"{r['v5_reason']}"
            if r["v5_hold"] > 0:
                v5_exit_str = f"{r['v5_reason']}({r['v5_hold']:.0f}m)"

            v6_exit_str = f"{r['v6_reason']}"
            if r["v6_hold"] > 0:
                v6_exit_str = f"{r['v6_reason']}({r['v6_hold']:.0f}m)"

            # Truncate exit strings
            v5_exit_str = v5_exit_str[:18]
            v6_exit_str = v6_exit_str[:18]

            note_str = " ".join(notes)

            print(f" {marker}{i:>3} {r['sig_time']:>5} {r['ticker']:<6} {r['dir']:<5} {r['score']:>3} "
                  f"${r['strike']:>7.2f} {r['contracts']:>3} ${r['entry']:>5.2f} {r['dte']:>3} "
                  f"${r['v5_pnl']:>8,.2f} {v5_exit_str:<18} "
                  f"${r['v6_pnl']:>8,.2f} {v6_exit_str:<18} "
                  f"${r['delta']:>+7,.2f} {note_str}")

        # Day subtotal
        print(f"  {'-' * (w - 2)}")
        print(f"  {'DAY TOTAL':>49} ${day_v5:>8,.2f} {'':18} "
              f"${day_v6:>8,.2f} {'':18} ${day_v6 - day_v5:>+7,.2f}")
        print()

    # ══════════════════════════════════════════════════════════════════════
    # DAILY SUMMARY TABLE
    # ══════════════════════════════════════════════════════════════════════

    print("=" * w)
    print("DAILY P&L SUMMARY")
    print("=" * w)
    print(f"\n  {'Day':<12} {'Signals':>7} {'Filt':>5} {'V5 Day':>10} {'V6 Day':>10} "
          f"{'Delta':>10} {'V5 Cum':>10} {'V6 Cum':>10} {'V6 WR':>8}")
    print(f"  {'-' * 85}")

    cum_v5 = 0
    cum_v6 = 0
    for day in days:
        day_results = [r for r in results if r["day"] == day]
        day_v5 = sum(r["v5_pnl"] for r in day_results)
        day_v6 = sum(r["v6_pnl"] for r in day_results)
        cum_v5 += day_v5
        cum_v6 += day_v6
        filt = sum(1 for r in day_results if r["v6_reason"].startswith("FILTERED"))
        day_traded = [r for r in day_results if not r["v6_reason"].startswith("FILTERED")]
        day_w = sum(1 for r in day_traded if r["v6_pnl"] > 0)
        wr = f"{day_w}/{len(day_traded)}" if day_traded else "-"
        print(f"  {day:<12} {len(day_results):>7} {filt:>5} ${day_v5:>8,.2f} ${day_v6:>8,.2f} "
              f"${day_v6 - day_v5:>+8,.2f} ${cum_v5:>8,.2f} ${cum_v6:>8,.2f} {wr:>8}")

    print(f"  {'-' * 85}")
    print(f"  {'TOTAL':<12} {len(results):>7} {filtered_count:>5} ${total_v5:>8,.2f} ${total_v6:>8,.2f} "
          f"${total_v6 - total_v5:>+8,.2f}")

    # ══════════════════════════════════════════════════════════════════════
    # PER-TICKER SUMMARY
    # ══════════════════════════════════════════════════════════════════════

    print(f"\n{'=' * w}")
    print("PER-TICKER SUMMARY")
    print(f"{'=' * w}")
    print(f"\n  {'Ticker':<7} {'Cfg':<14} {'N':>3} {'Filt':>4} {'DCA':>4} "
          f"{'V5 P&L':>10} {'V6 P&L':>10} {'Delta':>10} {'V6 WR':>8} {'V6 AvgW':>8} {'V6 AvgL':>8}")
    print(f"  {'-' * 95}")

    tickers = sorted(set(r["ticker"] for r in results))
    for t in tickers:
        tr = [r for r in results if r["ticker"] == t]
        t_traded = [r for r in tr if not r["v6_reason"].startswith("FILTERED")]
        t_v5 = sum(r["v5_pnl"] for r in tr)
        t_v6 = sum(r["v6_pnl"] for r in tr)
        t_filt = sum(1 for r in tr if r["v6_reason"].startswith("FILTERED"))
        t_dca = sum(1 for r in tr if r["v6_dca"])
        t_w = sum(1 for r in t_traded if r["v6_pnl"] > 0)
        wr = f"{t_w}/{len(t_traded)}" if t_traded else "-"
        avg_w = np.mean([r["v6_pnl"] for r in t_traded if r["v6_pnl"] > 0]) if t_w else 0
        avg_l = np.mean([r["v6_pnl"] for r in t_traded if r["v6_pnl"] <= 0]) if len(t_traded) - t_w > 0 else 0
        cfg_label = "DEFAULT"
        if t in TICKER_CONFIGS:
            cfg_names = {
                "NVDA": "EARLY_PROFIT", "GOOGL": "WIDE_STOP", "TSLA": "LONG_GRACE",
                "IWM": "WIDE_STOP", "QQQ": "LONG_GRACE", "META": "DEFENSIVE",
                "AAPL": "DEFENSIVE", "AMZN": "TIGHT_TRAIL", "AVGO": "EARLY_PROFIT",
                "MSFT": "EARLY_PROFIT", "MSTR": "TIGHT+QUICK",
            }
            cfg_label = cfg_names.get(t, "CUSTOM")
        print(f"  {t:<7} {cfg_label:<14} {len(tr):>3} {t_filt:>4} {t_dca:>4} "
              f"${t_v5:>8,.2f} ${t_v6:>8,.2f} ${t_v6 - t_v5:>+8,.2f} {wr:>8} ${avg_w:>6,.0f} ${avg_l:>6,.0f}")

    print(f"  {'-' * 95}")
    print(f"  {'TOTAL':<7} {'':14} {len(results):>3} {filtered_count:>4} {dca_count:>4} "
          f"${total_v5:>8,.2f} ${total_v6:>8,.2f} ${total_v6 - total_v5:>+8,.2f}")

    # ══════════════════════════════════════════════════════════════════════
    # EXIT REASON BREAKDOWN
    # ══════════════════════════════════════════════════════════════════════

    print(f"\n{'=' * w}")
    print("V6 EXIT REASON BREAKDOWN")
    print(f"{'=' * w}")
    print(f"\n  {'Reason':<22} {'Count':>6} {'Win':>4} {'Loss':>5} {'WR':>6} "
          f"{'Total P&L':>12} {'Avg P&L':>10} {'Avg Hold':>9}")
    print(f"  {'-' * 78}")

    df_all = pd.DataFrame(results)
    for reason, group in df_all.groupby("v6_reason"):
        cnt = len(group)
        wins = (group["v6_pnl"] > 0).sum()
        losses = cnt - wins
        wr = wins / cnt * 100 if cnt else 0
        total = group["v6_pnl"].sum()
        avg = group["v6_pnl"].mean()
        avg_hold = group["v6_hold"].mean()
        print(f"  {reason:<22} {cnt:>6} {wins:>4} {losses:>5} {wr:>5.0f}% "
              f"${total:>10,.2f} ${avg:>8,.2f} {avg_hold:>7.0f}m")

    # ══════════════════════════════════════════════════════════════════════
    # V6 FEATURE IMPACT
    # ══════════════════════════════════════════════════════════════════════

    print(f"\n{'=' * w}")
    print("V6 FEATURE IMPACT SUMMARY")
    print(f"{'=' * w}")

    # DCA
    dca_trades = [r for r in results if r["v6_dca"]]
    dca_pnl = sum(r["v6_pnl"] for r in dca_trades)
    dca_v5_pnl = sum(r["v5_pnl"] for r in dca_trades)
    print(f"\n  DCA ({len(dca_trades)} fires):")
    if dca_trades:
        print(f"    V5 P&L on same trades: ${dca_v5_pnl:>+10,.2f}")
        print(f"    V6 P&L on same trades: ${dca_pnl:>+10,.2f}")
        print(f"    DCA benefit:           ${dca_pnl - dca_v5_pnl:>+10,.2f}")

    # Per-ticker config
    custom_cfg = [r for r in results if r["cfg"] != "DEFAULT"
                  and not r["v6_reason"].startswith("FILTERED")]
    custom_v5 = sum(r["v5_pnl"] for r in custom_cfg)
    custom_v6 = sum(r["v6_pnl"] for r in custom_cfg)
    print(f"\n  Per-Ticker Configs ({len(custom_cfg)} trades with custom config):")
    print(f"    V5 P&L (default cfg): ${custom_v5:>+10,.2f}")
    print(f"    V6 P&L (custom cfg):  ${custom_v6:>+10,.2f}")
    print(f"    Config benefit:       ${custom_v6 - custom_v5:>+10,.2f}")

    # Filtered trades
    filt_trades = [r for r in results if r["v6_reason"].startswith("FILTERED")]
    filt_v5 = sum(r["v5_pnl"] for r in filt_trades)
    print(f"\n  Entry Filters ({len(filt_trades)} blocked):")
    print(f"    V5 P&L (would have traded): ${filt_v5:>+10,.2f}")
    print(f"    V6 P&L (blocked at $0):     ${0:>+10,.2f}")
    print(f"    Filter benefit:             ${-filt_v5:>+10,.2f}")

    # Scale-out
    so_trades = [r for r in results if r["v6_scaleout"] != 0]
    print(f"\n  Scale-Out ({len(so_trades)} fires):")
    if so_trades:
        print(f"    Locked profit: ${sum(r['v6_scaleout'] for r in so_trades):>+10,.2f}")

    print()

    # ══════════════════════════════════════════════════════════════════════
    # SKIPPED SIGNALS (no harvester data)
    # ══════════════════════════════════════════════════════════════════════

    if no_data_signals:
        print(f"{'=' * w}")
        print(f"SKIPPED SIGNALS — No Harvester Data ({len(no_data_signals)} signals)")
        print(f"{'=' * w}")
        for s in no_data_signals:
            print(f"  {s['day']} {s['time']} {s['ticker']:<6} {s['direction'][:4]:<5} "
                  f"score={s['score']} ${s['strike']:.2f} prem=${s['premium']:.2f}")
        print()


if __name__ == "__main__":
    main()
