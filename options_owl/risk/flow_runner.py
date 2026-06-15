"""Serve-time P(runner) for UW flow trades (Stage D, gated behind ENABLE_V7_RUNNER_TILT).

Assembles the live feature vector via compute_option_features_from_live (the SINGLE source of
truth shared with training — no skew), then predicts runner_score from the per-ticker/GENERIC
runner model. Returns None on ANY missing data so the conviction multiplier safely falls back to
the validated cluster/premium/ask sizing — a bad/zero feature vector NEVER reaches live sizing.

DEFAULT OFF. Must be validated against live market data (serve features ~ training distribution)
before activating on real money — wired flag-gated for that validation, not auto-on.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo

from loguru import logger

ET = ZoneInfo("America/New_York")


async def get_market_tide_bias(settings) -> float | None:
    """B2: live market-wide whale tide bias = net_call_premium - net_put_premium AS OF now (intraday
    cumulative). >0 = bullish tide, <0 = bearish. None on failure (caller treats as no-gate). One call."""
    try:
        import httpx
        api_key = getattr(settings, "UNUSUAL_WHALES_API_KEY", "") or ""
        if not api_key:
            return None
        today = datetime.now(ET).strftime("%Y-%m-%d")
        url = "https://api.unusualwhales.com/api/market/market-tide"
        async with httpx.AsyncClient(timeout=12) as client:
            r = await client.get(url, headers={"Authorization": f"Bearer {api_key}"}, params={"date": today})
        if r.status_code != 200:
            return None
        ticks = r.json().get("data", []) or []
        if not ticks:
            return None
        last = ticks[-1]  # latest tick = cumulative net premium so far today (no lookahead)
        return float(last.get("net_call_premium") or 0) - float(last.get("net_put_premium") or 0)
    except Exception as exc:
        logger.warning(f"MARKET_TIDE: fetch failed: {exc}")
        return None


async def compute_flow_p_runner(signal, settings) -> float | None:
    """Best-effort P(runner) for a UW-flow signal. None if data/model unavailable (safe no-op)."""
    try:
        from options_owl.collectors.polygon_options import (
            build_option_contract_ticker,
            polygon_intraday_1m,
            polygon_option_snapshot_greeks,
        )
        from options_owl.sourcing.scoring.ml_gates.signal_model import (
            compute_option_features_from_live,
            predict_entry_confidence,
        )

        api_key = getattr(settings, "POLYGON_API_KEY", "") or ""
        ticker = (signal.ticker or "").upper()
        strike = signal.strike or signal.atm_strike or 0
        expiry = signal.expiry or ""
        otype = "call" if str(signal.direction).lower().endswith("call") else "put"
        is_call = otype == "call"
        if not (api_key and ticker and strike and expiry):
            return None

        today = datetime.now(ET).strftime("%Y-%m-%d")
        now_et = datetime.now(ET)
        minutes_since_open = max(0, (now_et.hour - 9) * 60 + now_et.minute - 30)
        contract = build_option_contract_ticker(ticker, strike, expiry, otype)

        # Fetch snapshot (greeks) + option bars + underlying bars — all timeout-bounded.
        snap, opt_bars, und_bars = await asyncio.gather(
            asyncio.wait_for(polygon_option_snapshot_greeks(api_key, ticker, strike, expiry, otype), timeout=12),
            asyncio.wait_for(polygon_intraday_1m(api_key, contract, today), timeout=12),
            asyncio.wait_for(polygon_intraday_1m(api_key, ticker, today), timeout=12),
            return_exceptions=True,
        )
        if isinstance(snap, Exception) or not snap or snap.get("delta", 0) <= 0:
            return None  # greeks are required; no greeks => no reliable P(runner)
        opt_bars = [] if isinstance(opt_bars, Exception) else (opt_bars or [])
        und_bars = [] if isinstance(und_bars, Exception) else (und_bars or [])

        premium = snap["mid"] or signal.atm_premium or (opt_bars[-1]["close"] if opt_bars else 0)
        if premium <= 0:
            return None
        # histories: premium/volume EXCLUDE current; underlying INCLUDES current (matches training)
        premium_history = [b["close"] for b in opt_bars[:-1]] if len(opt_bars) > 1 else []
        volume_history = [int(b["volume"]) for b in opt_bars[:-1]] if len(opt_bars) > 1 else []
        underlying_history = [b["close"] for b in und_bars] if und_bars else []
        underlying_price = (underlying_history[-1] if underlying_history
                            else (signal.entry_price or 0))

        features = compute_option_features_from_live(
            ticker=ticker, premium=premium, bid=snap["bid"], ask=snap["ask"],
            iv=snap["iv"], delta=snap["delta"], theta=snap["theta"], vega=snap["vega"],
            volume=snap["volume"], underlying_price=underlying_price,
            minutes_since_open=minutes_since_open, is_call=is_call,
            premium_history=premium_history, volume_history=volume_history,
            underlying_history=underlying_history,
            bid_size=snap["bid_size"], ask_size=snap["ask_size"],
        )
        res = predict_entry_confidence(ticker, features, otype.upper())
        score = res.get("runner_score")
        if score is None or res.get("model_source") == "none":
            return None
        logger.info(f"FLOW_P_RUNNER: {ticker} {otype} p_runner={score:.3f} "
                    f"(model={res.get('model_source')}, snap_iv={snap['iv']:.2f} delta={snap['delta']:.2f})")
        return float(score)
    except Exception as exc:
        logger.warning(f"FLOW_P_RUNNER: failed for {getattr(signal, 'ticker', '?')}: {exc}")
        return None
