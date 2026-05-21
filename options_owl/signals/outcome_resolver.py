"""Resolve trade signal outcomes by analyzing price bars after signal entry."""

from __future__ import annotations

from datetime import datetime

from loguru import logger

from options_owl.models.signals import Direction, PriceBar, ResolvedSignal, TradeOutcome


def resolve_signal(
    signal: dict,
    bars: list[PriceBar],
    signal_id: int,
) -> ResolvedSignal:
    """Walk price bars and determine if a trade signal hit T1, T2, stop, or expired.

    Args:
        signal: A trade_signals row dict from the DB.
        bars: Intraday price bars for the signal's trading day.
        signal_id: The DB id of the trade signal.
    """
    entry = signal["entry_price"]
    t1 = signal.get("target_1")
    t2 = signal.get("target_2")
    stop = signal.get("stop_price")
    direction = signal["direction"]
    is_put = direction in ("put", Direction.PUT)

    if not bars:
        return ResolvedSignal(signal_id=signal_id, outcome=TradeOutcome.UNKNOWN)

    # Filter bars to after signal creation time if we have timestamp info.
    # Discord timestamps are real UTC; yfinance bars are US/Eastern labeled as UTC.
    # Convert signal UTC → ET-naive so the comparison is apples-to-apples.
    signal_time = signal.get("created_at")
    if signal_time:
        if isinstance(signal_time, str):
            try:
                signal_dt = datetime.fromisoformat(signal_time)
                # Convert UTC signal time to US/Eastern (naive) to match yfinance bars
                from zoneinfo import ZoneInfo

                if signal_dt.tzinfo is not None:
                    # Real Discord timestamp with timezone — convert UTC→ET
                    signal_dt_et = signal_dt.astimezone(ZoneInfo("America/New_York")).replace(tzinfo=None)
                else:
                    # Naive timestamp (tests, manual entries) — use as-is
                    signal_dt_et = signal_dt

                bars = [b for b in bars if b.timestamp.replace(tzinfo=None) >= signal_dt_et]
            except ValueError:
                pass

    if not bars:
        return ResolvedSignal(signal_id=signal_id, outcome=TradeOutcome.UNKNOWN)

    # Track max favorable/adverse excursion
    max_favorable = 0.0
    max_adverse = 0.0
    t1_hit = False
    t1_hit_time: datetime | None = None
    t2_hit = False
    t2_hit_time: datetime | None = None
    stop_hit = False
    stop_hit_time: datetime | None = None
    hit_price: float | None = None

    for bar in bars:
        if is_put:
            # For puts: favorable = price going DOWN from entry
            favorable_pct = (entry - bar.low) / entry * 100
            adverse_pct = (bar.high - entry) / entry * 100
        else:
            # For calls: favorable = price going UP from entry
            favorable_pct = (bar.high - entry) / entry * 100
            adverse_pct = (entry - bar.low) / entry * 100

        max_favorable = max(max_favorable, favorable_pct)
        max_adverse = max(max_adverse, adverse_pct)

        # Check targets
        if t1 and not t1_hit:
            if is_put and bar.low <= t1:
                t1_hit = True
                t1_hit_time = bar.timestamp
            elif not is_put and bar.high >= t1:
                t1_hit = True
                t1_hit_time = bar.timestamp

        if t2 and not t2_hit:
            if is_put and bar.low <= t2:
                t2_hit = True
                t2_hit_time = bar.timestamp
            elif not is_put and bar.high >= t2:
                t2_hit = True
                t2_hit_time = bar.timestamp

        # Check stop
        if stop and not stop_hit:
            if is_put and bar.high >= stop:
                stop_hit = True
                stop_hit_time = bar.timestamp
            elif not is_put and bar.low <= stop:
                stop_hit = True
                stop_hit_time = bar.timestamp

    # Determine outcome — priority: stop before T1 if same bar, then T2 > T1
    if stop_hit and t1_hit:
        # Both hit — which came first?
        if stop_hit_time and t1_hit_time and stop_hit_time < t1_hit_time:
            outcome = TradeOutcome.STOP_HIT
            hit_price = stop
        elif stop_hit_time and t1_hit_time and t1_hit_time < stop_hit_time:
            outcome = TradeOutcome.T2_HIT if t2_hit else TradeOutcome.T1_HIT
            hit_price = t2 if t2_hit else t1
        else:
            # Same bar — be conservative, assume stop hit
            outcome = TradeOutcome.STOP_HIT
            hit_price = stop
    elif t2_hit:
        outcome = TradeOutcome.T2_HIT
        hit_price = t2
    elif t1_hit:
        outcome = TradeOutcome.T1_HIT
        hit_price = t1
    elif stop_hit:
        outcome = TradeOutcome.STOP_HIT
        hit_price = stop
    else:
        outcome = TradeOutcome.EXPIRED

    # Calculate PnL on underlying
    if hit_price:
        if is_put:
            pnl_pct = (entry - hit_price) / entry * 100
        else:
            pnl_pct = (hit_price - entry) / entry * 100
    elif bars:
        # Use last bar close as exit price
        last_close = bars[-1].close
        if is_put:
            pnl_pct = (entry - last_close) / entry * 100
        else:
            pnl_pct = (last_close - entry) / entry * 100
    else:
        pnl_pct = 0.0

    # Estimate option PnL using simple delta approximation
    # ATM delta ~0.50, OTM delta ~0.30
    atm_premium = signal.get("atm_premium")
    otm_premium = signal.get("otm_premium")
    underlying_move = abs(hit_price - entry) if hit_price else abs(bars[-1].close - entry) if bars else 0

    pnl_atm = None
    if atm_premium and atm_premium > 0:
        atm_option_move = underlying_move * 0.50
        if outcome == TradeOutcome.STOP_HIT:
            pnl_atm = -atm_option_move / atm_premium * 100
        else:
            pnl_atm = atm_option_move / atm_premium * 100

    pnl_otm = None
    if otm_premium and otm_premium > 0:
        otm_option_move = underlying_move * 0.30
        if outcome == TradeOutcome.STOP_HIT:
            pnl_otm = -otm_option_move / otm_premium * 100
        else:
            pnl_otm = otm_option_move / otm_premium * 100

    hit_time = t2_hit_time or t1_hit_time or stop_hit_time

    logger.info(
        f"Resolved signal {signal_id}: {signal['ticker']} {direction} → {outcome.value} "
        f"(pnl={pnl_pct:+.2f}%, mfe={max_favorable:.2f}%, mae={max_adverse:.2f}%)"
    )

    return ResolvedSignal(
        signal_id=signal_id,
        outcome=outcome,
        hit_price=hit_price,
        hit_time=hit_time,
        pnl_underlying_pct=round(pnl_pct, 4),
        pnl_atm_est=round(pnl_atm, 2) if pnl_atm is not None else None,
        pnl_otm_est=round(pnl_otm, 2) if pnl_otm is not None else None,
        max_favorable_pct=round(max_favorable, 4),
        max_adverse_pct=round(max_adverse, 4),
    )
