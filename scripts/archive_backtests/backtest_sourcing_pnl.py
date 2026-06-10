"""Backtest the sourcing agent with V5 FSM exit engine — full P&L simulation.

Generates signals from historical 1-min bars using the sourcing scorer,
then replays each signal through the production ExitFSM against option tick data.

Score modes:
  --mode raw       Use raw score as-is (default). Missing alpha sources = lower scores.
  --mode rescaled  Rescale technical-only score to 0-100 (accounts for missing data).
  --sweep          Test multiple thresholds and report summary for each.

Outputs:
  - Per-trade results table
  - Daily P&L summary with cumulative equity curve
  - Chart saved as PNG

Usage:
    python scripts/backtest_sourcing_pnl.py                          # last 30 days
    python scripts/backtest_sourcing_pnl.py --days 60                # last 60 days
    python scripts/backtest_sourcing_pnl.py --start 2026-03-01       # from date
    python scripts/backtest_sourcing_pnl.py --ticker NVDA            # single ticker
    python scripts/backtest_sourcing_pnl.py --threshold 70           # score threshold
    python scripts/backtest_sourcing_pnl.py --portfolio 20000        # starting balance
    python scripts/backtest_sourcing_pnl.py --mode rescaled          # rescale for missing alpha
    python scripts/backtest_sourcing_pnl.py --sweep                  # test thresholds 45-75
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from options_owl.risk.exit_v5.config import get_ticker_config
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState
from options_owl.sourcing.data.indicator_engine import compute_indicators
from options_owl.sourcing.filters.penalty_veto import check_penalty_veto
from options_owl.sourcing.filters.quality_gate import check_quality_gate
from options_owl.sourcing.scoring.engine import compute_score
from options_owl.sourcing.scoring.types import Direction, SignalContext, SignalState

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HIST_DB = str(PROJECT_DIR / "journal" / "historical_0dte.db")

TICKERS = [
    "SPY", "QQQ", "NVDA", "TSLA", "META", "AAPL", "AMZN",
    "GOOGL", "MSFT", "AMD", "MSTR", "PLTR",
]

# Scan every 3 5m bars (= 15 min real time) — matches 3-min scan cycle with data gaps
SCAN_INTERVAL_BARS = 3

# Per-ticker 0DTE expiry schedules (same as production smart entry)
# True = has 0DTE options that day
_0DTE_DAILY = {"SPY", "QQQ"}
_0DTE_MWF = {"NVDA", "TSLA", "META", "AAPL", "AMZN", "GOOGL", "MSFT", "AVGO"}
_0DTE_FRIDAY = {"AMD", "PLTR", "MSTR"}  # weekly only

# Production V6 settings (matches docker-compose.yml)
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
    ENABLE_V6_PREMIUM_CAP=True,
    V6_PREMIUM_CAP=6.0,
    V6_PREMIUM_CAP_MID=7.0,
    V6_PREMIUM_CAP_HIGH=9.0,
    ENABLE_V6_SPREAD_GATE=True,
    V6_MAX_SPREAD_PCT=15.0,
    ENABLE_V6_EARLY_POP_GATE=True,
    ENABLE_V6_DCA=True,
    V6_DCA_TICKERS="MSFT,IWM,SPY,QQQ,AMZN,NVDA",
    V6_DCA_MIN_MINUTES=8.0,
    V6_DCA_MAX_MINUTES=20.0,
    V6_DCA_MIN_DIP_PCT=15.0,
    V6_DCA_MAX_DIP_PCT=35.0,
    V6_DCA_UNDERLYING_THRESHOLD=0.5,
)

# ---------------------------------------------------------------------------
# Score rescaling for backtest
# ---------------------------------------------------------------------------
# In production, the full score range is 0-100 across 5 tiers:
#   Tier 1 Direction:    0-40  (pure technicals — EMA, VWAP, ADX, MACD)
#   Tier 2 Timing:       0-30  (volume, RSI, MACD momentum, BB, ATR)
#   Tier 3 Amplifiers:   0-15  (squeeze 0-5, OBV 0-3, multi_tf 0-3, alpha 0-4)
#   Tier 4 Risk:        -15-0  (RSI overextend, wide BB, low ADX, spread)
#   Tier 5 Calibration:  0-15  (time-of-day, session, day-of-week)
#
# In backtest, these are ALWAYS 0 (no historical data):
#   - Alpha source bonus: 0-4 (insider, congress, sentiment)
#   - Options spread penalty: 0-4 (no bid/ask data)
#   - Multi-TF: gets default 1 instead of potential 3 (no 15m candles)
#
# Max achievable in backtest: ~92 (vs 100 production)
# Practical ceiling: ~80 (technicals rarely all align perfectly)
#
# Rescaling: score_rescaled = raw_score * (100 / MAX_TECHNICAL_ONLY)

MAX_TECHNICAL_ONLY = 92  # max score achievable without alpha/options data


def rescale_score(raw_score: int) -> int:
    """Rescale a technical-only score to the full 0-100 range."""
    return min(100, int(raw_score * 100 / MAX_TECHNICAL_ONLY))


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class SourcingSignal:
    """A signal generated by the sourcing scoring engine."""
    date: str
    time_str: str  # HH:MM in UTC
    timestamp_ms: int
    ticker: str
    direction: str  # "CALL" / "PUT"
    score: int  # effective score (raw or rescaled depending on mode)
    raw_score: int  # always the unmodified scorer output
    underlying_price: float
    ema_cross: float
    rsi: float
    volume_ratio: float


@dataclass
class TradeResult:
    """Result of a single trade through the V5 FSM."""
    date: str
    ticker: str
    direction: str
    score: int
    entry_premium: float
    contracts: int
    exit_premium: float
    pnl: float
    peak_gain_pct: float
    hold_minutes: float
    exit_reason: str
    dte: int


# ---------------------------------------------------------------------------
# Candle loading + aggregation (from historical DB)
# ---------------------------------------------------------------------------

def load_1m_bars(conn: sqlite3.Connection, ticker: str, date: str) -> list[dict]:
    rows = conn.execute(
        """SELECT timestamp, open, high, low, close, volume
           FROM underlying_bars
           WHERE ticker = ? AND date = ?
           ORDER BY timestamp ASC""",
        (ticker, date),
    ).fetchall()
    return [
        {"timestamp": r[0], "open": r[1], "high": r[2], "low": r[3],
         "close": r[4], "volume": r[5] or 0}
        for r in rows
    ]


def aggregate_to_5m(bars_1m: list[dict]) -> list[dict]:
    if not bars_1m:
        return []
    candles = []
    bucket: list[dict] = []
    bucket_start = None
    for bar in bars_1m:
        bucket_ts = (bar["timestamp"] // (5 * 60 * 1000)) * (5 * 60 * 1000)
        if bucket_start is None:
            bucket_start = bucket_ts
        if bucket_ts != bucket_start:
            if bucket:
                candles.append(_flush_bucket(bucket))
            bucket = [bar]
            bucket_start = bucket_ts
        else:
            bucket.append(bar)
    if bucket:
        candles.append(_flush_bucket(bucket))
    return candles


def _flush_bucket(bars: list[dict]) -> dict:
    return {
        "open": bars[0]["open"],
        "high": max(b["high"] for b in bars),
        "low": min(b["low"] for b in bars),
        "close": bars[-1]["close"],
        "volume": sum(b["volume"] for b in bars),
    }


# ---------------------------------------------------------------------------
# Direction inference (same as scanner.py)
# ---------------------------------------------------------------------------

def _infer_direction(indicators) -> Direction:
    bullish = bearish = 0
    if indicators.ema_cross_strength > 0.05:
        bullish += 2
    elif indicators.ema_cross_strength < -0.05:
        bearish += 2
    if indicators.macd_line > 0:
        bullish += 1
    elif indicators.macd_line < 0:
        bearish += 1
    if indicators.vwap > 0 and indicators.last_close > indicators.vwap:
        bullish += 1
    elif indicators.vwap > 0 and indicators.last_close < indicators.vwap:
        bearish += 1
    return Direction.CALL if bullish >= bearish else Direction.PUT


# ---------------------------------------------------------------------------
# Signal generation (sourcing engine on historical data)
# ---------------------------------------------------------------------------

def generate_signals_for_day(
    conn: sqlite3.Connection,
    ticker: str,
    date: str,
    score_threshold: int,
    score_mode: str = "raw",
) -> list[SourcingSignal]:
    """Run the sourcing scoring engine on a single day for a ticker.

    score_mode: "raw" = use scorer output as-is, "rescaled" = rescale for missing alpha.
    """
    bars_1m = load_1m_bars(conn, ticker, date)
    if len(bars_1m) < 50:
        return []

    candles_5m = aggregate_to_5m(bars_1m)
    if len(candles_5m) < 15:
        return []

    signals = []
    for scan_idx in range(SCAN_INTERVAL_BARS, len(candles_5m), SCAN_INTERVAL_BARS):
        window = candles_5m[max(0, scan_idx - 78):scan_idx]
        if len(window) < 10:
            continue

        indicators = compute_indicators(window)
        direction = _infer_direction(indicators)

        # Build scan time from bar timestamps
        bar_idx = min(scan_idx * 5, len(bars_1m) - 1)
        ts_ms = bars_1m[bar_idx]["timestamp"]
        scan_dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)

        # Only trade during market hours (9:33 AM - 3:57 PM ET = 13:33 - 19:57 UTC)
        utc_hour = scan_dt.hour
        utc_minute = scan_dt.minute
        utc_total_min = utc_hour * 60 + utc_minute
        if utc_total_min < 13 * 60 + 33 or utc_total_min > 19 * 60 + 57:
            continue
        scan_time_str = scan_dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
        time_display = scan_dt.strftime("%H:%M")

        ctx = SignalContext(
            ticker=ticker,
            scan_time=scan_time_str,
            state=SignalState.INDICATED,
            direction=direction,
            candles_5m=window,
            indicators=indicators,
        )

        scored = compute_score(ctx)
        raw_score = scored.score
        effective_score = rescale_score(raw_score) if score_mode == "rescaled" else raw_score
        ctx.score_total = effective_score
        ctx.state = SignalState.SCORED

        if scored.rejected:
            continue
        if effective_score < score_threshold:
            continue
        if not check_quality_gate(ctx, score_threshold):
            continue
        if check_penalty_veto(ctx):
            continue

        signals.append(SourcingSignal(
            date=date,
            time_str=time_display,
            timestamp_ms=ts_ms,
            ticker=ticker,
            direction=direction.value,
            score=effective_score,
            raw_score=raw_score,
            underlying_price=window[-1]["close"],
            ema_cross=round(indicators.ema_cross_strength, 3),
            rsi=round(indicators.rsi9, 1),
            volume_ratio=round(indicators.volume_ratio, 2),
        ))

    return signals


# ---------------------------------------------------------------------------
# Option tick matching (find contract in option_bars)
# ---------------------------------------------------------------------------

def _get_dte_for_ticker(ticker: str, weekday: int) -> tuple[int, str]:
    """Return (dte, expiry_date_offset_days) for this ticker on this weekday.

    weekday: 0=Mon, 4=Fri
    Returns dte (0 for same-day, 1 for next day, etc.)
    """
    if ticker in _0DTE_DAILY:
        return 0, "same"
    if ticker in _0DTE_MWF:
        # Mon/Wed/Fri = 0DTE, Tue = 1DTE (Wed expiry), Thu = 1DTE (Fri expiry)
        if weekday in (0, 2, 4):
            return 0, "same"
        elif weekday == 1:
            return 1, "next"
        else:  # Thu
            return 1, "next"
    if ticker in _0DTE_FRIDAY:
        # Weekly only — Friday = 0DTE, otherwise DTE = days until Friday
        days_to_fri = (4 - weekday) % 7
        if days_to_fri == 0:
            return 0, "same"
        return days_to_fri, "friday"
    # Unknown — assume 0DTE
    return 0, "same"


def build_contract_ticker(ticker: str, expiry_date: str, strike: float, option_type: str) -> str:
    """Build OCC-style contract ticker: O:TICKER YYMMDD C/P SSSSSSSS."""
    try:
        exp_dt = datetime.strptime(expiry_date, "%Y-%m-%d")
    except ValueError:
        return ""
    exp_str = exp_dt.strftime("%y%m%d")
    ot = "C" if option_type.upper() in ("CALL", "C") else "P"
    strike_int = int(strike * 1000)
    return f"O:{ticker}{exp_str}{ot}{strike_int:08d}"


def find_contract_from_trading_days(
    conn: sqlite3.Connection, ticker: str, date: str, direction: str,
) -> tuple[str, float] | None:
    """Look up the pre-computed ATM contract from trading_days table.

    Returns (contract_ticker, strike) or None.
    """
    col = "atm_call_ticker" if direction == "CALL" else "atm_put_ticker"
    row = conn.execute(
        f"SELECT {col}, atm_strike FROM trading_days WHERE date = ? AND ticker = ?",
        (date, ticker),
    ).fetchone()
    if row and row[0]:
        return row[0], row[1]
    return None


def load_option_ticks(conn: sqlite3.Connection, contract_ticker: str,
                      after_ts_ms: int) -> pd.DataFrame | None:
    """Load option tick data for a contract after a given timestamp."""
    rows = conn.execute(
        """SELECT timestamp, open, high, low, close, volume, vwap
           FROM option_bars
           WHERE contract_ticker = ? AND timestamp >= ?
           ORDER BY timestamp ASC""",
        (contract_ticker, after_ts_ms),
    ).fetchall()

    if not rows or len(rows) < 5:
        return None

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume", "vwap"])
    df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    # Use midpoint of open/close as premium proxy (1-min bars)
    df["premium"] = df["close"]
    df["bid"] = df["low"]  # approximate
    df["ask"] = df["high"]  # approximate
    df["underlying_price"] = 0.0  # not available in option_bars
    return df


# ---------------------------------------------------------------------------
# FSM simulation (reused from backtest_v5_production.py)
# ---------------------------------------------------------------------------

def simulate_fsm(df: pd.DataFrame, entry_premium: float, contracts: int,
                 direction: str, dte: int, expiry_date: str,
                 ticker: str) -> TradeResult | None:
    """Run the production ExitFSM against option tick data."""
    if entry_premium <= 0:
        return None

    cfg = get_ticker_config(ticker, use_per_ticker=True)
    fsm = ExitFSM(cfg, settings=_V6_SETTINGS)

    entry_ts = df["ts"].iloc[0]
    if hasattr(entry_ts, "to_pydatetime"):
        entry_ts = entry_ts.to_pydatetime()
    if entry_ts.tzinfo is not None:
        entry_ts = entry_ts.replace(tzinfo=None)

    option_type = "put" if direction == "PUT" else "call"

    state = TradeState(
        trade_id=1,
        ticker=ticker,
        option_type=option_type,
        entry_premium=entry_premium,
        entry_time=entry_ts,
        contracts=contracts,
        peak_premium=entry_premium,
        entry_underlying_price=0.0,
        dte=dte,
        expiry_date=expiry_date,
    )

    locked_pnl = 0.0
    remaining = contracts

    for idx in range(1, len(df)):
        premium = df["premium"].iloc[idx]
        if np.isnan(premium) or premium <= 0:
            continue

        bid = float(df["bid"].iloc[idx]) if not pd.isna(df["bid"].iloc[idx]) else premium
        ask = float(df["ask"].iloc[idx]) if not pd.isna(df["ask"].iloc[idx]) else premium

        now = df["ts"].iloc[idx]
        if hasattr(now, "to_pydatetime"):
            now = now.to_pydatetime()
        if now.tzinfo is not None:
            now = now.replace(tzinfo=None)

        # ET = UTC - 4 (approximate)
        et_hour = now.hour - 4
        if et_hour < 0:
            et_hour += 24
        minutes_to_close = max(0, (16 * 60) - (et_hour * 60 + now.minute))

        action = fsm.evaluate(
            state, premium, bid, ask, now,
            current_underlying=0.0,
            minutes_to_close=minutes_to_close,
        )

        if action.should_exit:
            if action.contracts_to_close > 0 and action.contracts_to_close < remaining:
                closed = action.contracts_to_close
                locked_pnl += (premium - entry_premium) * closed * 100
                remaining -= closed
                state.contracts = remaining
                continue

            elapsed = (now - entry_ts).total_seconds() / 60
            peak_gain = (state.peak_premium - entry_premium) / entry_premium * 100
            pnl = locked_pnl + (premium - entry_premium) * remaining * 100
            return TradeResult(
                date=df["ts"].iloc[0].strftime("%Y-%m-%d") if hasattr(df["ts"].iloc[0], "strftime") else "",
                ticker=ticker,
                direction=direction,
                score=0,
                entry_premium=entry_premium,
                contracts=contracts,
                exit_premium=premium,
                pnl=pnl,
                peak_gain_pct=peak_gain,
                hold_minutes=elapsed,
                exit_reason=action.reason.value,
                dte=dte,
            )

    # End of data — force close
    last_prem = df["premium"].iloc[-1]
    last_ts = df["ts"].iloc[-1]
    if hasattr(last_ts, "to_pydatetime"):
        last_ts = last_ts.to_pydatetime()
    if last_ts.tzinfo is not None:
        last_ts = last_ts.replace(tzinfo=None)
    elapsed = (last_ts - entry_ts).total_seconds() / 60
    peak_gain = (state.peak_premium - entry_premium) / entry_premium * 100
    pnl = locked_pnl + (last_prem - entry_premium) * remaining * 100
    return TradeResult(
        date=df["ts"].iloc[0].strftime("%Y-%m-%d") if hasattr(df["ts"].iloc[0], "strftime") else "",
        ticker=ticker, direction=direction, score=0,
        entry_premium=entry_premium, contracts=contracts,
        exit_premium=last_prem, pnl=pnl, peak_gain_pct=peak_gain,
        hold_minutes=elapsed, exit_reason="eod_data_end", dte=dte,
    )


# ---------------------------------------------------------------------------
# Position sizing (matches production flat 85%)
# ---------------------------------------------------------------------------

def compute_contracts(portfolio: float, entry_premium: float, score: int,
                      max_concurrent: int = 5, score_floor: int = 60) -> int:
    """Flat 85% budget allocation for all qualifying scores.

    Note: sourcing scores are 0-100 (vs Discord 78-177), so the floor
    is the sourcing threshold, not the Discord floor.
    """
    if score < score_floor or entry_premium <= 0:
        return 0

    max_risk_pct = 0.75
    max_position_pct = 0.15
    budget_mult = 0.85  # flat for all qualifying scores

    deployable = portfolio * max_risk_pct
    per_slot = deployable / max_concurrent
    scaled_target = per_slot * budget_mult
    cost_per = entry_premium * 100
    position_cap = portfolio * max_position_pct

    raw_contracts = int(scaled_target / cost_per)
    cap_contracts = int(position_cap / cost_per)
    return max(1, min(raw_contracts, cap_contracts))


# ---------------------------------------------------------------------------
# Main backtest
# ---------------------------------------------------------------------------

def run_backtest(
    ticker_filter: str | None,
    start_date: str,
    end_date: str,
    score_threshold: int,
    portfolio_start: float,
    max_concurrent: int,
    score_mode: str = "raw",
    quiet: bool = False,
) -> dict:
    hist_conn = sqlite3.connect(HIST_DB)

    tickers = [ticker_filter.upper()] if ticker_filter else TICKERS

    dates = [
        r[0] for r in hist_conn.execute(
            "SELECT DISTINCT date FROM trading_days WHERE date BETWEEN ? AND ? ORDER BY date",
            (start_date, end_date),
        ).fetchall()
    ]

    if not dates:
        if not quiet:
            print("No trading days found in date range.")
        hist_conn.close()
        return {"pnl": 0, "trades": 0, "win_rate": 0}

    mode_label = "rescaled" if score_mode == "rescaled" else "raw"
    if not quiet:
        print(f"\n{'='*80}")
        print(f"SOURCING AGENT P&L BACKTEST (score mode: {mode_label})")
        print(f"{'='*80}")
        print(f"Period:     {start_date} → {end_date} ({len(dates)} trading days)")
        print(f"Tickers:    {', '.join(tickers)}")
        print(f"Threshold:  {score_threshold}")
        print(f"Portfolio:  ${portfolio_start:,.0f}")
        print(f"Sizing:     Flat 85% | max_concurrent={max_concurrent} | max_pos=15%")
        print(f"Exit:       V5 FSM with V6 enhancements (production settings)")
        print(f"{'='*80}\n")

    all_results: list[TradeResult] = []
    signals_generated = 0
    signals_no_option_data = 0
    signals_premium_capped = 0
    active_trades: list[str] = []  # track concurrent positions by date
    portfolio = portfolio_start

    # Cooldown: prevent same ticker+direction within 90 min
    last_signal: dict[str, int] = {}  # "TICKER_DIR" -> timestamp_ms

    for date_idx, date in enumerate(dates):
        weekday = datetime.strptime(date, "%Y-%m-%d").weekday()
        day_signals = []

        # Generate signals for all tickers on this day
        for ticker in tickers:
            sigs = generate_signals_for_day(hist_conn, ticker, date, score_threshold, score_mode)
            for sig in sigs:
                signals_generated += 1
                cooldown_key = f"{sig.ticker}_{sig.direction}"
                last_ts = last_signal.get(cooldown_key, 0)
                if sig.timestamp_ms - last_ts < 90 * 60 * 1000:
                    continue  # cooldown
                day_signals.append(sig)

        # Sort by score descending (best signals first)
        day_signals.sort(key=lambda s: s.score, reverse=True)

        # Process signals — respect max concurrent
        day_active = 0
        for sig in day_signals:
            if day_active >= max_concurrent:
                break

            # Look up ATM contract from trading_days (pre-computed)
            dte, _ = _get_dte_for_ticker(sig.ticker, weekday)
            lookup_date = date  # 0DTE: same day
            if dte > 0:
                # For non-0DTE, try expiry date first, fall back to same day
                exp_date = datetime.strptime(date, "%Y-%m-%d") + timedelta(days=dte)
                lookup_date = exp_date.strftime("%Y-%m-%d")

            result = find_contract_from_trading_days(
                hist_conn, sig.ticker, date, sig.direction,
            )
            if result is None:
                signals_no_option_data += 1
                continue
            ct, strike = result
            expiry_str = date  # trading_days contracts are same-day expiry

            df = load_option_ticks(hist_conn, ct, sig.timestamp_ms)
            if df is None:
                signals_no_option_data += 1
                continue

            # Entry premium = first ask (worst fill, conservative)
            entry_premium = float(df["ask"].iloc[0])
            if entry_premium <= 0:
                entry_premium = float(df["premium"].iloc[0])
            if entry_premium <= 0:
                signals_no_option_data += 1
                continue

            # V6 premium cap
            cap = _V6_SETTINGS.V6_PREMIUM_CAP
            if sig.score >= 150:
                cap = _V6_SETTINGS.V6_PREMIUM_CAP_HIGH
            elif sig.score >= 120:
                cap = _V6_SETTINGS.V6_PREMIUM_CAP_MID
            if entry_premium > cap:
                signals_premium_capped += 1
                continue

            # Position sizing (use score_threshold as floor since sourcing uses 0-100 scale)
            contracts = compute_contracts(portfolio, entry_premium, sig.score, max_concurrent,
                                          score_floor=score_threshold)
            if contracts <= 0:
                continue

            # Run through FSM
            result = simulate_fsm(
                df, entry_premium, contracts, sig.direction,
                dte, expiry_str, sig.ticker,
            )
            if result is None:
                continue

            result.score = sig.score
            result.date = date
            all_results.append(result)
            day_active += 1

            # Update portfolio
            portfolio += result.pnl

            # Record cooldown
            last_signal[f"{sig.ticker}_{sig.direction}"] = sig.timestamp_ms

        # Progress
        if not quiet and ((date_idx + 1) % 20 == 0 or date_idx == len(dates) - 1):
            print(f"  [{date_idx+1}/{len(dates)}] {date} | "
                  f"{len(all_results)} trades | portfolio=${portfolio:,.0f}")

    hist_conn.close()

    if not all_results:
        if not quiet:
            print("\nNo trades executed — check score threshold and option data availability.")
        return {"pnl": 0, "trades": 0, "win_rate": 0, "signals": signals_generated}

    # ---------------------------------------------------------------------------
    # Results
    # ---------------------------------------------------------------------------

    df_results = pd.DataFrame([vars(r) for r in all_results])
    pnls = df_results["pnl"]
    wins = (pnls > 0).sum()
    losses = (pnls <= 0).sum()
    total_pnl = pnls.sum()
    win_rate = wins / len(pnls) * 100

    summary = {
        "pnl": total_pnl,
        "trades": len(all_results),
        "win_rate": win_rate,
        "wins": int(wins),
        "losses": int(losses),
        "signals": signals_generated,
        "no_option_data": signals_no_option_data,
        "premium_capped": signals_premium_capped,
        "avg_win": float(pnls[pnls > 0].mean()) if wins > 0 else 0,
        "avg_loss": float(pnls[pnls <= 0].mean()) if losses > 0 else 0,
        "max_drawdown": float((portfolio_start + df_results.groupby("date")["pnl"].sum().cumsum()).min() - portfolio_start),
    }

    if quiet:
        hist_conn.close()
        return summary

    print(f"\n{'='*80}")
    print(f"RESULTS — Sourcing Agent + V5 FSM (score mode: {mode_label})")
    print(f"{'='*80}")
    print(f"Signals generated:  {signals_generated}")
    print(f"No option data:     {signals_no_option_data}")
    print(f"Premium capped:     {signals_premium_capped}")
    print(f"Trades executed:    {len(all_results)}")
    print(f"")
    print(f"Starting portfolio: ${portfolio_start:,.0f}")
    print(f"Ending portfolio:   ${portfolio:,.0f}")
    print(f"Total P&L:          ${total_pnl:,.2f}")
    print(f"Return:             {total_pnl/portfolio_start*100:.1f}%")
    print(f"Win Rate:           {win_rate:.1f}% ({wins}W / {losses}L)")
    if wins > 0:
        print(f"Avg Win:            ${pnls[pnls > 0].mean():,.2f}")
    if losses > 0:
        print(f"Avg Loss:           ${pnls[pnls <= 0].mean():,.2f}")
    print(f"Avg Hold:           {df_results['hold_minutes'].mean():.0f} min")
    print(f"Max Win:            ${pnls.max():,.2f}")
    print(f"Max Loss:           ${pnls.min():,.2f}")

    # --- Exit reason breakdown ---
    print(f"\n{'Reason':<25} {'Count':>6} {'Total P&L':>12} {'Avg P&L':>10} {'Win%':>6}")
    print("-" * 62)
    for reason, group in df_results.groupby("exit_reason"):
        gpnl = group["pnl"]
        gwins = (gpnl > 0).sum()
        gwr = gwins / len(gpnl) * 100
        print(f"{reason:<25} {len(gpnl):>6} ${gpnl.sum():>10,.2f} ${gpnl.mean():>8,.2f} {gwr:>5.0f}%")

    # --- Per-ticker breakdown ---
    print(f"\n{'Ticker':<8} {'Trades':>6} {'P&L':>12} {'Win%':>6} {'Avg':>10}")
    print("-" * 45)
    for ticker, group in df_results.groupby("ticker"):
        gpnl = group["pnl"]
        gwins = (gpnl > 0).sum()
        gwr = gwins / len(gpnl) * 100
        print(f"{ticker:<8} {len(gpnl):>6} ${gpnl.sum():>10,.2f} {gwr:>5.0f}% ${gpnl.mean():>8,.2f}")

    # --- Direction breakdown ---
    for d in ["CALL", "PUT"]:
        subset = df_results[df_results["direction"] == d]
        if len(subset) > 0:
            sw = (subset["pnl"] > 0).sum()
            swr = sw / len(subset) * 100
            print(f"\n{d}: {len(subset)} trades, ${subset['pnl'].sum():,.2f} total, {swr:.0f}% win rate")

    # --- Per-trade detail ---
    print(f"\n{'Day':<12} {'Ticker':<6} {'Dir':<5} {'Score':>5} {'Entry':>7} {'Ct':>3} "
          f"{'Exit':>7} {'P&L':>9} {'Peak%':>6} {'Hold':>5} {'Reason':<20}")
    print("-" * 105)
    for r in all_results:
        print(f"{r.date:<12} {r.ticker:<6} {r.direction[:4]:<5} {r.score:>5} "
              f"${r.entry_premium:>5.2f} {r.contracts:>3} ${r.exit_premium:>5.2f} "
              f"${r.pnl:>8.2f} {r.peak_gain_pct:>5.0f}% {r.hold_minutes:>4.0f}m {r.exit_reason:<20}")

    # --- Daily P&L ---
    daily = df_results.groupby("date").agg(
        trades=("pnl", "count"),
        pnl=("pnl", "sum"),
        wins=("pnl", lambda x: (x > 0).sum()),
    ).reset_index()
    daily["cum_pnl"] = daily["pnl"].cumsum()
    daily["equity"] = portfolio_start + daily["cum_pnl"]
    daily["win_rate"] = daily["wins"] / daily["trades"] * 100

    print(f"\n{'Day':<12} {'Trades':>6} {'Day P&L':>10} {'Cum P&L':>10} {'Equity':>10} {'W/L':>6} {'Win%':>6}")
    print("-" * 70)
    for _, row in daily.iterrows():
        losses_d = row["trades"] - row["wins"]
        print(f"{row['date']:<12} {row['trades']:>6} ${row['pnl']:>8,.2f} "
              f"${row['cum_pnl']:>8,.2f} ${row['equity']:>8,.0f} "
              f"{int(row['wins'])}/{int(losses_d):>2} {row['win_rate']:>5.0f}%")

    # --- Chart ---
    _save_chart(daily, all_results, portfolio_start, total_pnl, win_rate)

    hist_conn.close()
    return summary


def _save_chart(daily: pd.DataFrame, results: list, portfolio_start: float,
                total_pnl: float, win_rate: float) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), height_ratios=[2, 1, 1])
        fig.suptitle("Sourcing Agent + V5 FSM — Portfolio Performance",
                     fontsize=14, fontweight="bold")

        chart_dates = pd.to_datetime(daily["date"])

        # Top: equity curve
        ax1 = axes[0]
        ax1.plot(chart_dates, daily["equity"], "b-o", markersize=4, linewidth=2)
        ax1.axhline(y=portfolio_start, color="gray", linestyle="--", alpha=0.5, label=f"Start ${portfolio_start:,.0f}")
        ax1.fill_between(chart_dates, daily["equity"], portfolio_start,
                         where=daily["equity"] >= portfolio_start, alpha=0.15, color="green")
        ax1.fill_between(chart_dates, daily["equity"], portfolio_start,
                         where=daily["equity"] < portfolio_start, alpha=0.15, color="red")
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.set_title(f"P&L: ${total_pnl:,.2f} ({total_pnl/portfolio_start*100:.1f}%) | "
                      f"Win Rate: {win_rate:.0f}% | {len(results)} trades")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)

        # Middle: daily P&L bars
        ax2 = axes[1]
        colors = ["green" if p >= 0 else "red" for p in daily["pnl"]]
        ax2.bar(chart_dates, daily["pnl"], color=colors, alpha=0.7, width=0.8)
        ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax2.set_ylabel("Daily P&L ($)")
        ax2.grid(True, alpha=0.3)

        # Bottom: trades per day
        ax3 = axes[2]
        ax3.bar(chart_dates, daily["trades"], color="steelblue", alpha=0.7, width=0.8)
        ax3.set_ylabel("Trades/Day")
        ax3.set_xlabel("Date")
        ax3.grid(True, alpha=0.3)

        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(daily) // 15)))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        plt.tight_layout()
        chart_path = str(PROJECT_DIR / "sourcing_pnl.png")
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        print(f"\nChart saved: {chart_path}")
        plt.close()

    except ImportError:
        print("\nmatplotlib not installed — skipping chart (pip install matplotlib)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_sweep(
    ticker_filter: str | None,
    start_date: str,
    end_date: str,
    portfolio_start: float,
    max_concurrent: int,
    score_mode: str,
) -> None:
    """Test multiple thresholds and print comparison table."""
    thresholds = [45, 50, 55, 60, 65, 70, 75]

    print(f"\n{'='*90}")
    print(f"THRESHOLD SWEEP — score mode: {score_mode}")
    print(f"Period: {start_date} → {end_date} | Portfolio: ${portfolio_start:,.0f}")
    print(f"{'='*90}")
    print(f"{'Threshold':>9} {'Trades':>7} {'P&L':>12} {'Return':>8} {'Win%':>6} "
          f"{'AvgWin':>9} {'AvgLoss':>9} {'Signals':>8} {'MaxDD':>10}")
    print("-" * 90)

    for t in thresholds:
        result = run_backtest(
            ticker_filter=ticker_filter,
            start_date=start_date,
            end_date=end_date,
            score_threshold=t,
            portfolio_start=portfolio_start,
            max_concurrent=max_concurrent,
            score_mode=score_mode,
            quiet=True,
        )
        pnl = result["pnl"]
        ret = pnl / portfolio_start * 100
        print(f"{t:>9} {result['trades']:>7} ${pnl:>10,.2f} {ret:>7.1f}% {result['win_rate']:>5.1f}% "
              f"${result.get('avg_win', 0):>7,.0f} ${result.get('avg_loss', 0):>7,.0f} "
              f"{result.get('signals', 0):>8} ${result.get('max_drawdown', 0):>8,.0f}")

    print(f"\nNote: 'rescaled' mode scales technical-only scores to 0-100 (compensates for missing alpha data)")


def main():
    parser = argparse.ArgumentParser(description="Backtest sourcing agent with V5 FSM exit engine")
    parser.add_argument("--days", type=int, default=30, help="Days to backtest (default: 30)")
    parser.add_argument("--start", type=str, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, help="End date YYYY-MM-DD")
    parser.add_argument("--ticker", type=str, help="Single ticker")
    parser.add_argument("--threshold", type=int, default=60, help="Score threshold (default: 60)")
    parser.add_argument("--portfolio", type=float, default=20000, help="Starting portfolio (default: $20,000)")
    parser.add_argument("--max-concurrent", type=int, default=5, help="Max concurrent trades (default: 5)")
    parser.add_argument("--mode", choices=["raw", "rescaled"], default="raw",
                        help="Score mode: raw (as-is) or rescaled (adjust for missing alpha)")
    parser.add_argument("--sweep", action="store_true",
                        help="Test multiple thresholds (45-75) and print comparison")
    args = parser.parse_args()

    if args.start:
        start_date = args.start
    else:
        start_date = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")
    end_date = args.end or datetime.now().strftime("%Y-%m-%d")

    if args.sweep:
        run_sweep(
            ticker_filter=args.ticker,
            start_date=start_date,
            end_date=end_date,
            portfolio_start=args.portfolio,
            max_concurrent=args.max_concurrent,
            score_mode=args.mode,
        )
    else:
        run_backtest(
            ticker_filter=args.ticker,
            start_date=start_date,
            end_date=end_date,
            score_threshold=args.threshold,
            portfolio_start=args.portfolio,
            max_concurrent=args.max_concurrent,
            score_mode=args.mode,
        )


if __name__ == "__main__":
    main()
