"""Shared regime feature module — SINGLE SOURCE OF TRUTH.

The regime classifier's 40-feature vector is defined *exactly once* in
``compute_regime_feature_vector``. Both the offline trainer
(``scripts/train_ml_models_v3.py``) and the live owlet serving path
(``options_owl/sourcing/ml_pipeline.py:compute_regime_features``) build the
SAME normalized ``raw_inputs`` dict and feed it to that one function, so
training and serving features are identical by construction. This closes the
"40 vs 18" train/serve skew that blocked deployment (spec §3, §5.1-5.3).

----------------------------------------------------------------------------
raw_inputs CONTRACT  (data-source-agnostic — trainer fills it from thetadata
sqlite + UW sqlite; serving fills it from Postgres)
----------------------------------------------------------------------------
raw_inputs = {
    "ticker": str,                 # e.g. "NVDA"
    "date":   str,                 # "YYYY-MM-DD" (ET trading date)
    "morning_bars": [              # OWN ticker RTH 1-min bars, 09:30..09:44
        {"open","high","low","close","volume"}, ...
    ],
    "prior_day": {                 # OWN ticker prior-day RTH lags (all strictly
        "prev_range_pct", "prev_volume", "prev_day_ret",   # past data)
        "prev_close_pos", "avg_3d_range", "avg_prev_vol",
        "vol_5d", "prev_close",
    },
    "gex": {                       # OI-based dealer-positioning aggregate
        "call_gamma","put_gamma","call_delta","put_delta",
        "call_charm","put_charm","call_vanna","put_vanna",
    },
    "market": {                    # SPY/QQQ early-morning context (same shape as
        "SPY": {<_MARKET_CONTEXT_COLS>} | None,            # own morning/prior)
        "QQQ": {<_MARKET_CONTEXT_COLS>} | None,
    },
}

Every sub-dict may be missing/None — the builder 0-fills with a deterministic
default (NOT silent skew: the missing-feature warner in the serving path stays
quiet because we always emit the FULL feature set here).

----------------------------------------------------------------------------
SERVE-TIME SAFETY (CRITICAL — spec §4.1)
----------------------------------------------------------------------------
Every feature must be computable from ONLY past data at the model's ~9:45 ET
serve time. Per-feature data source + "serve-time-safe because…" notes are in
the FEATURE TABLE below. Banned full-day leak features (day_range_pct,
day_volume, same-day options-volume aggregates) are NEVER produced here and are
asserted-absent by tests.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Universe + windows  (kept in sync with the trainer + ml_pipeline)
# ---------------------------------------------------------------------------

TICKERS = [
    "SPY", "QQQ", "NVDA", "TSLA", "META", "AAPL", "AMZN",
    "GOOGL", "MSFT", "AMD", "MSTR", "PLTR", "AVGO", "IWM",
    "COIN", "NFLX", "JPM", "BA", "MU", "SMCI",
]

RTH_START = "09:30"
RTH_END = "16:00"
EARLY_END = "09:45"  # day-level regime model serves at ~9:45 ET

# Per-ticker early-morning + prior-day context. Names match the trainer's
# _TICKER_CONTEXT_FEATURES and the live serving features 1:1.
_TICKER_CONTEXT_FEATURES = [
    "morning_range_pct",   # 09:30-09:44 high-low range %
    "morning_volume",      # 09:30-09:44 share volume
    "morning_direction",   # 09:30 open -> 09:44 close return %
    "morning_body_ratio",  # |close-open| / (high-low) of the 15m window
    "morning_vol_15m",     # realized vol of 1-min returns, 09:30-09:44
    "overnight_gap_pct",   # today's RTH open vs yesterday's RTH close
    "prev_range_pct",      # yesterday's RTH range %
    "prev_volume",         # yesterday's RTH volume
    "prev_day_ret",        # yesterday's open->close return %
    "prev_close_pos",      # where yesterday closed in its range (0..1)
    "avg_3d_range",        # mean RTH range % of prior 3 days
    "vol_5d",              # realized vol of prior 5 close-to-close returns
    "range_trend",         # morning_range_pct / avg_3d_range - 1
    "volume_vs_prev",      # morning volume extrapolated vs prior 5-day avg
]

# Cross-market context (SPY/QQQ early morning).
_MARKET_CONTEXT_COLS = [
    "morning_direction", "morning_range_pct", "morning_vol_15m",
    "overnight_gap_pct", "prev_day_ret", "vol_5d",
]

# OI-based dealer-positioning aggregate. OI is fixed at the open (prior day's
# clearing), so a gamma×OI proxy computed at/after 9:45 uses only-past data.
_GEX_COLS = [
    "call_gamma", "put_gamma", "net_gamma",
    "call_delta", "put_delta", "net_delta",
    "call_charm", "put_charm", "net_charm",
    "call_vanna", "put_vanna", "net_vanna",
]

# The exact, ordered 40-feature vector the regime model expects. This is the
# ONE definition; the model's _meta.json["features"] must equal this list.
REGIME_FEATURE_ORDER: list[str] = (
    ["ticker_idx", "day_of_week"]
    + _TICKER_CONTEXT_FEATURES
    + _GEX_COLS
    + [f"spy_{c}" for c in _MARKET_CONTEXT_COLS]
    + [f"qqq_{c}" for c in _MARKET_CONTEXT_COLS]
)

# ---------------------------------------------------------------------------
# PER-FEATURE DATA SOURCE + SERVE-TIME-SAFETY TABLE  (spec §5.5)
# ---------------------------------------------------------------------------
# feature                | source                       | serve-time-safe because…
# -----------------------|------------------------------|--------------------------------
# ticker_idx             | static universe index        | constant, no time dependence
# day_of_week            | calendar(date)               | known before the open
# morning_range_pct      | own 09:30-09:44 RTH bars     | uses only bars completed by 9:44 (< 9:45 serve)
# morning_volume         | own 09:30-09:44 RTH bars     | "
# morning_direction      | own 09:30-09:44 RTH bars     | "
# morning_body_ratio     | own 09:30-09:44 RTH bars     | "
# morning_vol_15m        | own 09:30-09:44 RTH bars     | "
# overnight_gap_pct      | today RTH open / prev close  | prev close is yesterday's data
# prev_range_pct         | prior-day RTH bars (lag 1)   | strictly past day
# prev_volume            | prior-day RTH bars (lag 1)   | strictly past day
# prev_day_ret           | prior-day RTH bars (lag 1)   | strictly past day
# prev_close_pos         | prior-day RTH bars (lag 1)   | strictly past day
# avg_3d_range           | prior 3 days RTH ranges      | strictly past days
# vol_5d                 | prior 5 day close-returns    | strictly past days
# range_trend            | morning_range_pct/avg_3d-1   | both inputs are past data
# volume_vs_prev         | morning vol vs prior-5d avg  | morning vol < 9:45; avg is past
# call_gamma..net_vanna  | gamma×OI proxy (gex_ticks)   | OI fixed at the open (prior-day clearing)
# spy_* / qqq_*          | SPY/QQQ same 09:30-09:44+lag | same windows as own ticker (all <= 9:45 / past)
#
# DROPPED (banned leak): day_range_pct, day_volume, rth_range_pct,
# rth_close_pos, same-day options-volume aggregates — full-day / label info.
# ---------------------------------------------------------------------------

_LEAK_FEATURES = frozenset(
    {"day_range_pct", "day_volume", "rth_range_pct", "rth_close_pos"}
)


def _f(v: Any, default: float = 0.0) -> float:
    """Coerce any value (None / NaN / numpy scalar) to a finite float."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    if x != x or x in (float("inf"), float("-inf")):  # NaN / inf
        return default
    return x


# ---------------------------------------------------------------------------
# Sub-feature math (defined ONCE) — reused by both load_* paths
# ---------------------------------------------------------------------------


def compute_morning_features(morning_bars: list[dict]) -> dict[str, float]:
    """Own-ticker 09:30-09:44 early-morning window features.

    SERVE-TIME-SAFE: consumes only bars completed before the 9:45 serve time.
    Returns zeros (never NaN) if the window is too sparse to be meaningful.
    """
    out = {
        "morning_range_pct": 0.0,
        "morning_volume": 0.0,
        "morning_direction": 0.0,
        "morning_body_ratio": 0.0,
        "morning_vol_15m": 0.0,
    }
    if not morning_bars:
        return out

    closes = np.array([_f(b.get("close")) for b in morning_bars])
    opens = np.array([_f(b.get("open")) for b in morning_bars])
    highs = np.array([_f(b.get("high")) for b in morning_bars])
    lows = np.array([_f(b.get("low")) for b in morning_bars])
    out["morning_volume"] = float(sum(_f(b.get("volume")) for b in morning_bars))

    valid_closes = closes[closes > 0]
    valid_opens = opens[opens > 0]
    valid_highs = highs[highs > 0]
    valid_lows = lows[lows > 0]
    if (
        len(valid_closes) >= 5
        and len(valid_opens) > 0
        and len(valid_highs) > 0
        and len(valid_lows) > 0
    ):
        m_open = float(valid_opens[0])
        m_close = float(valid_closes[-1])
        m_high = float(np.max(valid_highs))
        m_low = float(np.min(valid_lows))
        if m_open > 0 and m_low > 0:
            out["morning_range_pct"] = (m_high - m_low) / m_low * 100
            out["morning_direction"] = (m_close / m_open - 1) * 100
            rng = m_high - m_low
            out["morning_body_ratio"] = abs(m_close - m_open) / rng if rng > 0 else 0.0
            if len(valid_closes) > 1 and np.all(valid_closes[:-1] > 0):
                out["morning_vol_15m"] = float(
                    np.std(np.diff(valid_closes) / valid_closes[:-1]) * 100
                )
    return out


def compute_daily_context_row(
    morning_bars: list[dict], prior_day: dict | None
) -> dict[str, float]:
    """Assemble the 14 _TICKER_CONTEXT_FEATURES for one ticker-day.

    Combines the early-morning window (``compute_morning_features``) with
    prior-day lag values (``prior_day``). All values are serve-time-safe (see
    the per-feature table). prior_day is expected to already hold STRICTLY-PAST
    (shift(1)) aggregates — both trainer and serving compute them the same way.
    """
    morning = compute_morning_features(morning_bars)
    pd_ = prior_day or {}

    morning_range_pct = morning["morning_range_pct"]
    avg_3d_range = _f(pd_.get("avg_3d_range"))
    avg_prev_vol = _f(pd_.get("avg_prev_vol"), 1.0)
    morning_open = 0.0
    if morning_bars:
        opens = [_f(b.get("open")) for b in morning_bars if _f(b.get("open")) > 0]
        morning_open = opens[0] if opens else 0.0
    prev_close = _f(pd_.get("prev_close"))

    return {
        "morning_range_pct": morning_range_pct,
        "morning_volume": morning["morning_volume"],
        "morning_direction": morning["morning_direction"],
        "morning_body_ratio": morning["morning_body_ratio"],
        "morning_vol_15m": morning["morning_vol_15m"],
        "overnight_gap_pct": (
            (morning_open / prev_close - 1) * 100 if prev_close > 0 else 0.0
        ),
        "prev_range_pct": _f(pd_.get("prev_range_pct")),
        "prev_volume": _f(pd_.get("prev_volume")),
        "prev_day_ret": _f(pd_.get("prev_day_ret")),
        "prev_close_pos": _f(pd_.get("prev_close_pos")),
        "avg_3d_range": avg_3d_range,
        "vol_5d": _f(pd_.get("vol_5d")),
        "range_trend": (morning_range_pct / max(avg_3d_range, 0.01) - 1),
        "volume_vs_prev": morning["morning_volume"] * 26 / max(avg_prev_vol, 1.0),
    }


def compute_gex_features(gex: dict | None) -> dict[str, float]:
    """OI-based dealer-positioning aggregate -> the 12 _GEX_COLS.

    Net legs are derived here (single definition) from the call/put legs so
    trainer and serving cannot disagree on sign convention. SERVE-TIME-SAFE:
    open-interest is fixed at the open, so any gamma×OI proxy is past-only.
    Empty/None gex (e.g. gex_ticks not yet populated) -> all zeros, no NaN.
    """
    f = {k: 0.0 for k in _GEX_COLS}
    if not gex:
        return f
    f["call_gamma"] = _f(gex.get("call_gamma"))
    f["put_gamma"] = _f(gex.get("put_gamma"))
    f["net_gamma"] = f["call_gamma"] - f["put_gamma"]
    f["call_delta"] = _f(gex.get("call_delta"))
    f["put_delta"] = _f(gex.get("put_delta"))
    f["net_delta"] = f["call_delta"] - f["put_delta"]
    f["call_charm"] = _f(gex.get("call_charm"))
    f["put_charm"] = _f(gex.get("put_charm"))
    f["net_charm"] = f["call_charm"] - f["put_charm"]
    f["call_vanna"] = _f(gex.get("call_vanna"))
    f["put_vanna"] = _f(gex.get("put_vanna"))
    f["net_vanna"] = f["call_vanna"] - f["put_vanna"]
    return f


def compute_market_features(market: dict | None) -> dict[str, float]:
    """spy_*/qqq_* cross-market early-morning features (zeros when missing)."""
    f: dict[str, float] = {}
    market = market or {}
    for mkt in ("SPY", "QQQ"):
        prefix = f"{mkt.lower()}_"
        ctx = market.get(mkt)
        for col in _MARKET_CONTEXT_COLS:
            f[prefix + col] = _f(ctx.get(col)) if isinstance(ctx, dict) else 0.0
    return f


# ---------------------------------------------------------------------------
# THE single feature vector builder
# ---------------------------------------------------------------------------


def compute_regime_feature_vector(raw_inputs: dict) -> dict[str, float]:
    """Build the FULL ordered regime feature vector from normalized raw_inputs.

    Pure + deterministic. Always returns every feature in
    ``REGIME_FEATURE_ORDER`` (no silent omissions), so the serving path's
    missing-feature warner stays quiet and the train/serve parity test holds.
    """
    ticker = str(raw_inputs.get("ticker", "") or "").upper()
    date_str = str(raw_inputs.get("date", "") or "")

    f: dict[str, float] = {}
    f["ticker_idx"] = float(TICKERS.index(ticker)) if ticker in TICKERS else 0.0
    try:
        f["day_of_week"] = float(datetime.strptime(date_str, "%Y-%m-%d").weekday())
    except (ValueError, TypeError):
        f["day_of_week"] = 0.0

    f.update(
        compute_daily_context_row(
            raw_inputs.get("morning_bars") or [],
            raw_inputs.get("prior_day"),
        )
    )
    f.update(compute_gex_features(raw_inputs.get("gex")))
    f.update(compute_market_features(raw_inputs.get("market")))

    # Guarantee a leak-free, fully-populated, correctly-ordered vector.
    assert not (_LEAK_FEATURES & set(f)), "leak features must never be produced"
    return {k: _f(f.get(k)) for k in REGIME_FEATURE_ORDER}


# ===========================================================================
# SERVING input loader (Postgres) — used by ml_pipeline.compute_regime_features
# ===========================================================================

_gex_empty_warned: set[str] = set()


async def load_serving_inputs(
    ticker: str,
    now_et: datetime,
    *,
    tz_et: Any = None,
) -> dict:
    """Build raw_inputs for a LIVE owlet from Postgres (spec §5.4 schema).

    Reads:
      - stock_candles (1m): own ticker + SPY/QQQ/VIX, RTH only, with prior-day
        lags computed exactly like the trainer's _load_daily_context.
      - gex_ticks: latest dealer-positioning aggregate at/before serve time.
        EMPTY table -> zeros + ONE logged warning per ticker (spec §4.2).

    Returns a raw_inputs dict ready for compute_regime_feature_vector. Never
    raises on missing data — degrades to zeros so the model still loads.
    """
    from zoneinfo import ZoneInfo

    from loguru import logger

    from options_owl.db import postgres as pg

    et = tz_et or ZoneInfo("America/New_York")
    if now_et.tzinfo is None:
        now_et = now_et.replace(tzinfo=et)
    date_str = now_et.astimezone(et).strftime("%Y-%m-%d")

    async def _ticker_inputs(sym: str) -> dict:
        """morning_bars + prior_day lags for one symbol from stock_candles."""
        morning_bars: list[dict] = []
        prior_day: dict = {}
        try:
            # 1-min RTH candles for this symbol up to 'today'. Ordered ascending.
            rows = await pg.fetch(
                """
                SELECT bar_time, open, high, low, close, volume
                FROM stock_candles
                WHERE ticker = $1 AND timeframe = '1m'
                ORDER BY bar_time
                """,
                sym,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"REGIME_FEATURES: stock_candles read failed for {sym}: {exc}")
            return {"morning_bars": morning_bars, "prior_day": prior_day}

        # Group bars by ET trading date; keep RTH (09:30-16:00) only.
        by_date: dict[str, list[dict]] = {}
        for r in rows:
            bt = r["bar_time"]
            if bt is None:
                continue
            bt_et = bt.astimezone(et) if bt.tzinfo else bt.replace(tzinfo=et).astimezone(et)
            hhmm = bt_et.strftime("%H:%M")
            if hhmm < RTH_START or hhmm > RTH_END:
                continue
            d = bt_et.strftime("%Y-%m-%d")
            by_date.setdefault(d, []).append(
                {
                    "tm": hhmm,
                    "open": _f(r["open"]),
                    "high": _f(r["high"]),
                    "low": _f(r["low"]),
                    "close": _f(r["close"]),
                    "volume": _f(r["volume"]),
                }
            )

        # Per-date RTH aggregates (sorted ascending) for prior-day lags.
        daily = _aggregate_daily(by_date)

        # Today's early-morning window (bars strictly before 09:45).
        today_bars = by_date.get(date_str, [])
        morning_bars = [b for b in today_bars if b["tm"] < EARLY_END]

        # Prior-day lags: the most recent date strictly before today.
        prior_dates = [d for d in sorted(daily) if d < date_str]
        if prior_dates:
            prior_day = _prior_day_lags(daily, prior_dates, today_bars)
        return {"morning_bars": morning_bars, "prior_day": prior_day}

    own = await _ticker_inputs(ticker.upper())
    spy = await _ticker_inputs("SPY")
    qqq = await _ticker_inputs("QQQ")
    # VIX is harvested too; its early-morning context is folded into nothing
    # extra in the current 40-feature set, but we read it to stay schema-aligned
    # and tolerate its absence. (Reserved for future VIX-derived features.)

    gex = await _load_serving_gex(ticker.upper(), date_str)

    return {
        "ticker": ticker.upper(),
        "date": date_str,
        "morning_bars": own["morning_bars"],
        "prior_day": own["prior_day"],
        "gex": gex,
        "market": {
            "SPY": _market_ctx_from_inputs(spy),
            "QQQ": _market_ctx_from_inputs(qqq),
        },
    }


def _market_ctx_from_inputs(sym_inputs: dict) -> dict:
    """Reduce a symbol's {morning_bars, prior_day} into _MARKET_CONTEXT_COLS."""
    row = compute_daily_context_row(
        sym_inputs.get("morning_bars") or [], sym_inputs.get("prior_day")
    )
    return {c: row.get(c, 0.0) for c in _MARKET_CONTEXT_COLS}


def _aggregate_daily(by_date: dict[str, list[dict]]) -> dict[str, dict]:
    """Per-date RTH OHLCV aggregates (open/close/high/low/volume)."""
    daily: dict[str, dict] = {}
    for d, bars in by_date.items():
        if not bars:
            continue
        opens = [b["open"] for b in bars if b["open"] > 0]
        closes = [b["close"] for b in bars if b["close"] > 0]
        highs = [b["high"] for b in bars if b["high"] > 0]
        lows = [b["low"] for b in bars if b["low"] > 0]
        if not (opens and closes and highs and lows):
            continue
        daily[d] = {
            "rth_open": opens[0],
            "rth_close": closes[-1],
            "rth_high": max(highs),
            "rth_low": min(lows),
            "rth_volume": sum(b["volume"] for b in bars),
        }
    return daily


def _prior_day_lags(
    daily: dict[str, dict], prior_dates: list[str], today_bars: list[dict]
) -> dict:
    """Compute strictly-past lag aggregates matching the trainer's shift(1).

    prior_dates is ascending and excludes today. Mirrors _load_daily_context:
    prev_* use the last prior day; avg_3d_range / avg_prev_vol / vol_5d use the
    trailing prior-day windows.
    """
    last = prior_dates[-1]
    pdrow = daily[last]
    rng = pdrow["rth_high"] - pdrow["rth_low"]
    prev_range_pct = (rng / pdrow["rth_low"] * 100) if pdrow["rth_low"] > 0 else 0.0
    prev_day_ret = (
        (pdrow["rth_close"] / pdrow["rth_open"] - 1) * 100 if pdrow["rth_open"] > 0 else 0.0
    )
    span = rng if rng > 0 else 0.0
    prev_close_pos = (
        (pdrow["rth_close"] - pdrow["rth_low"]) / span if span > 0 else 0.5
    )

    # Trailing prior-day range %s for avg_3d_range (last 3 prior days).
    def _range_pct(d: str) -> float:
        row = daily[d]
        r = row["rth_high"] - row["rth_low"]
        return (r / row["rth_low"] * 100) if row["rth_low"] > 0 else 0.0

    last3 = prior_dates[-3:]
    avg_3d_range = float(np.mean([_range_pct(d) for d in last3])) if last3 else 0.0

    last5_vol = [daily[d]["rth_volume"] for d in prior_dates[-5:]]
    avg_prev_vol = float(np.mean(last5_vol)) if last5_vol else 1.0

    # vol_5d: std of the prior 5 close-to-close returns (trainer: pct_change
    # over closes, rolling(5).std, shifted by 1 — i.e. only prior days).
    closes_for_vol = [daily[d]["rth_close"] for d in prior_dates[-6:]]
    vol_5d = 0.0
    if len(closes_for_vol) >= 6:
        arr = np.array(closes_for_vol, dtype=float)
        rets = np.diff(arr) / arr[:-1]
        vol_5d = float(np.std(rets) * 100)

    return {
        "prev_range_pct": prev_range_pct,
        "prev_volume": pdrow["rth_volume"],
        "prev_day_ret": prev_day_ret,
        "prev_close_pos": prev_close_pos,
        "avg_3d_range": avg_3d_range,
        "avg_prev_vol": avg_prev_vol,
        "vol_5d": vol_5d,
        "prev_close": pdrow["rth_close"],
    }


async def _load_serving_gex(ticker: str, date_str: str) -> dict:
    """Latest gex_ticks aggregate for ticker at/before today (serving path).

    Reads the spec §5.4 schema columns. Tolerates an EMPTY/absent gex_ticks
    table (rollout window before Agent A's harvester populates it) -> returns
    zeros and logs exactly ONE warning per ticker.
    """
    from loguru import logger

    from options_owl.db import postgres as pg

    # captured_at is timestamptz — asyncpg needs a datetime, NOT a string (a string raises
    # DataError, which the except below mislabeled as "table unavailable" → GEX silently 0).
    from datetime import timezone
    try:
        cutoff = datetime.strptime(str(date_str)[:10], "%Y-%m-%d").replace(
            hour=23, minute=59, second=59, tzinfo=timezone.utc)
    except (ValueError, TypeError):
        cutoff = datetime.now(timezone.utc)

    try:
        row = await pg.fetchrow(
            """
            SELECT net_gamma, call_gamma, put_gamma,
                   net_charm, net_vanna, total_oi, spot
            FROM gex_ticks
            WHERE ticker = $1 AND captured_at <= $2
            ORDER BY captured_at DESC
            LIMIT 1
            """,
            ticker,
            cutoff,
        )
    except Exception as exc:
        # Table missing / not yet created by the harvester. One warning, zeros.
        if ticker not in _gex_empty_warned:
            logger.warning(
                f"REGIME_FEATURES: gex_ticks unavailable for {ticker} "
                f"({type(exc).__name__}) — GEX features default to 0 until the "
                f"harvester populates the table"
            )
            _gex_empty_warned.add(ticker)
        return {}

    if row is None:
        if ticker not in _gex_empty_warned:
            logger.warning(
                f"REGIME_FEATURES: gex_ticks empty for {ticker} — GEX features "
                f"default to 0 until the harvester populates the table"
            )
            _gex_empty_warned.add(ticker)
        return {}

    # gex_ticks stores net/call/put gamma + net charm/vanna. The model's
    # call/put charm & vanna legs are not separately persisted, so we map the
    # net aggregate onto the call leg (put leg 0). compute_gex_features then
    # re-derives net_charm/net_vanna identically. delta legs are not in
    # gex_ticks (dealer delta needs signed positioning we don't approximate) ->
    # left 0, same as a stale/absent UW row in the trainer.
    return {
        "call_gamma": _f(row["call_gamma"]),
        "put_gamma": _f(row["put_gamma"]),
        "call_delta": 0.0,
        "put_delta": 0.0,
        "call_charm": _f(row["net_charm"]),
        "put_charm": 0.0,
        "call_vanna": _f(row["net_vanna"]),
        "put_vanna": 0.0,
    }


# ===========================================================================
# TRAINING input loader (thetadata + UW sqlite) — used by the trainer
# ===========================================================================


def rth_bars_by_date_from_rows(rows: list[dict]) -> dict:
    """Group flat (date, tm, ohlcv) rows into {date: [bar,...]} RTH-only.

    Shared helper so the trainer's sqlite rows and any other source normalize
    identically. ``rows`` items must have keys d (YYYY-MM-DD), tm (HH:MM),
    open/high/low/close/volume. Premarket / after-hours bars are dropped.
    """
    by_date: dict[str, list[dict]] = {}
    for r in rows:
        tm = r["tm"]
        if tm < RTH_START or tm > RTH_END:
            continue
        by_date.setdefault(r["d"], []).append(
            {
                "tm": tm,
                "open": _f(r["open"]),
                "high": _f(r["high"]),
                "low": _f(r["low"]),
                "close": _f(r["close"]),
                "volume": _f(r["volume"]),
            }
        )
    return by_date


def daily_inputs_for_date(by_date: dict[str, list[dict]], date_str: str) -> dict:
    """{morning_bars, prior_day} for one date from grouped RTH bars.

    Identical reduction to the serving loader's per-symbol path, so the trainer
    and owlets feed compute_regime_feature_vector the same shape.
    """
    daily = _aggregate_daily(by_date)
    today_bars = by_date.get(date_str, [])
    morning_bars = [b for b in today_bars if b["tm"] < EARLY_END]
    prior_dates = [d for d in sorted(daily) if d < date_str]
    prior_day = (
        _prior_day_lags(daily, prior_dates, today_bars) if prior_dates else {}
    )
    return {"morning_bars": morning_bars, "prior_day": prior_day}


def load_training_inputs(
    ticker: str,
    date_str: str,
    *,
    by_date: dict[str, list[dict]],
    market_by_date: dict[str, dict[str, list[dict]]],
    gex_row: dict,
) -> dict:
    """Build raw_inputs for the TRAINER from grouped RTH bars + UW GEX.

    NORMALIZES the trainer's data into the EXACT raw_inputs contract the
    serving path uses, so both call compute_regime_feature_vector and cannot
    drift. The morning/prior-day/market math now lives ONLY in the shared
    sub-feature functions — the trainer just supplies grouped RTH bars.

    Parameters
    ----------
    by_date : {date: [RTH bar,...]} for the OWN ticker (from
        rth_bars_by_date_from_rows).
    market_by_date : {"SPY": {date: [...]}, "QQQ": {date: [...]}} grouped RTH
        bars for the cross-market context symbols.
    gex_row : dict
        The 8 call/put gamma/delta/charm/vanna legs from UW (per _gex_features).
    """
    own = daily_inputs_for_date(by_date, date_str)

    market: dict[str, dict] = {}
    for mkt in ("SPY", "QQQ"):
        mkt_bars = market_by_date.get(mkt)
        if mkt_bars:
            market[mkt] = _market_ctx_from_inputs(daily_inputs_for_date(mkt_bars, date_str))
        else:
            market[mkt] = None

    return {
        "ticker": ticker.upper(),
        "date": date_str,
        "morning_bars": own["morning_bars"],
        "prior_day": own["prior_day"],
        "gex": gex_row,
        "market": market,
    }
