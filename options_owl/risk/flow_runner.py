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


# ── runner_v1 (ml_v3) serve path — the VALIDATED model (AUC 0.74), CALLS only ────────────────
# Distinct from compute_flow_p_runner above, which scores the WEAK signal_ml_v2 model. runner_v1 has
# its own 18-feature schema; features here replicate scripts/runner_separation_realdata.py
# :build_runner_v1_features VERBATIM (the validated, training-matched builder). None on ANY missing
# data so a bad/zero vector NEVER reaches live sizing. Gated by ENABLE_RUNNER_V1_SIZING (default off).
_RUNNER_V1 = None
_RUNNER_V1_META = None


def _load_runner_v1():
    global _RUNNER_V1, _RUNNER_V1_META
    if _RUNNER_V1 is not None:
        return _RUNNER_V1, _RUNNER_V1_META
    import json
    from pathlib import Path

    import lightgbm as lgb
    base = Path(__file__).resolve().parents[2] / "journal" / "models" / "ml_v3"
    _RUNNER_V1 = lgb.Booster(model_file=str(base / "runner_v1.lgb"))
    _RUNNER_V1_META = json.load(open(base / "runner_v1_meta.json"))
    return _RUNNER_V1, _RUNNER_V1_META


def _score_runner_v1(feat: dict) -> float:
    """Score the 18-feature vector — pandas frame in meta order, ticker/day_of_week as category
    (matches scripts/runner_separation_realdata.py:score_runner_v1, the validated scorer)."""
    import pandas as pd
    model, meta = _load_runner_v1()
    cols = meta["features"]
    df = pd.DataFrame([{c: feat.get(c) for c in cols}], columns=cols)
    for c in meta["cat_features"]:
        df[c] = df[c].astype("category")
    return float(model.predict(df)[0])


async def compute_runner_v1_p(signal, settings) -> float | None:
    """Best-effort P(runner) from runner_v1 for a CALL entry. None if data/model unavailable or PUT
    (the model is ATM-CALL only; abstains on delta<=0). Safe no-op → conviction mult unchanged."""
    try:
        import math

        import httpx

        from options_owl.collectors.polygon_options import (
            build_option_contract_ticker,
            polygon_intraday_1m,
            polygon_option_snapshot_greeks,
        )

        otype = "call" if str(signal.direction).lower().endswith("call") else "put"
        if otype != "call":
            return None  # runner_v1 is a CALL model
        api_key = getattr(settings, "POLYGON_API_KEY", "") or ""
        ticker = (signal.ticker or "").upper()
        strike = signal.strike or signal.atm_strike or 0
        expiry = signal.expiry or ""
        if not (api_key and ticker and strike and expiry):
            return None

        now_et = datetime.now(ET)
        today = now_et.strftime("%Y-%m-%d")
        entry_min = max(0, (now_et.hour - 9) * 60 + now_et.minute - 30)
        contract = build_option_contract_ticker(ticker, strike, expiry, otype)

        snap, opt_bars, und_bars = await asyncio.gather(
            asyncio.wait_for(polygon_option_snapshot_greeks(api_key, ticker, strike, expiry, otype), timeout=12),
            asyncio.wait_for(polygon_intraday_1m(api_key, contract, today), timeout=12),
            asyncio.wait_for(polygon_intraday_1m(api_key, ticker, today), timeout=12),
            return_exceptions=True,
        )
        if isinstance(snap, Exception) or not snap or (snap.get("delta") or 0) <= 0:
            return None  # greeks required; delta<=0 → not a tradeable call
        opt_bars = [] if isinstance(opt_bars, Exception) else (opt_bars or [])
        und_bars = [] if isinstance(und_bars, Exception) else (und_bars or [])

        entry_prem = snap.get("mid") or signal.atm_premium or (opt_bars[-1]["close"] if opt_bars else 0)
        und_closes = [b["close"] for b in und_bars if b.get("close")]
        und_now = und_closes[-1] if und_closes else (signal.entry_price or 0)
        day_open = und_closes[0] if und_closes else 0
        if entry_prem <= 0 or und_now <= 0 or day_open <= 0:
            return None

        # prior-day OHLC (gap_pct + prior_range_pct) — one daily-aggs call (no lookahead, prior day closed)
        prior_close = prior_high = prior_low = 0.0
        try:
            url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/prev"
            async with httpx.AsyncClient(timeout=8) as client:
                r = await client.get(url, params={"apiKey": api_key, "adjusted": "true"})
            res = (r.json().get("results") or [{}])[0] if r.status_code == 200 else {}
            prior_close = float(res.get("c") or 0)
            prior_high = float(res.get("h") or 0)
            prior_low = float(res.get("l") or 0)
        except Exception:
            pass
        gap_pct = (day_open / prior_close - 1) * 100 if prior_close > 0 else 0.0
        prior_range_pct = (prior_high - prior_low) / prior_close * 100 if prior_close > 0 else 0.0

        bid = snap.get("bid") or 0.0
        ask = snap.get("ask") or 0.0
        spread_pct = (ask - bid) / ask * 100 if ask > 0 else 0.0
        # underlying micro-features (last-N bars including current — matches the validated builder)
        recent5 = und_closes[-5:]
        und_slope_5 = (recent5[-1] / recent5[0] - 1) * 100 if len(recent5) >= 2 and recent5[0] > 0 else 0.0
        r15 = und_closes[-15:]
        if len(r15) >= 5:
            import numpy as np
            arr = np.array(r15, dtype=float)
            und_rvol_15 = float(np.std(np.diff(arr) / arr[:-1]) * 100)
        else:
            und_rvol_15 = 0.0
        opt_vol_5 = float(sum(int(b.get("volume") or 0) for b in opt_bars[-5:]))

        try:
            dte = max(0, (datetime.strptime(expiry, "%Y-%m-%d").date() - now_et.date()).days)
        except Exception:
            dte = 0

        feat = {
            "entry_premium": float(entry_prem),
            "log_premium": float(math.log(entry_prem)),
            "delta": float(snap["delta"]),
            "iv": float(snap.get("iv") or 0.0),
            "vega": float(snap.get("vega") or 0.0),
            "theta": float(snap.get("theta") or 0.0),
            "moneyness": float(strike / und_now),
            "spread_pct": float(spread_pct),
            "und_move_pct": float((und_now / day_open - 1) * 100),
            "und_slope_5": float(und_slope_5),
            "und_rvol_15": float(und_rvol_15),
            "opt_vol_5": opt_vol_5,
            "gap_pct": float(gap_pct),
            "prior_range_pct": float(prior_range_pct),
            "dte": int(dte),
            "entry_min": int(entry_min),
            "ticker": ticker,
            "day_of_week": now_et.weekday(),
        }
        p = _score_runner_v1(feat)
        if p is None or not (0.0 <= p <= 1.0):
            return None
        # Observe-first: log P(runner) + key inputs so the LIVE distribution can be verified vs backtest
        # before the sizing is trusted (the meta warns thresholds may shift on new data).
        logger.info(
            f"RUNNER_V1: {ticker} CALL p_runner={p:.3f} | delta={snap['delta']:.2f} iv={snap.get('iv') or 0:.2f} "
            f"mny={strike/und_now:.3f} dte={dte} entry_min={entry_min} gap={gap_pct:+.2f}% rng={prior_range_pct:.2f}%"
        )
        return float(p)
    except Exception as exc:
        logger.warning(f"RUNNER_V1: failed for {getattr(signal, 'ticker', '?')}: {exc}")
        return None
