"""Production ML pipeline — bridges backtest models to live trading.

Autonomous signal generation pipeline (no Discord dependency):
  1. Load ML models at startup (pattern_entry, entry_timing, regime_classifier, stop_calibration)
  2. Run regime filter at 9:45 AM ET — skip entire day if market conditions are bad
  3. Scan every minute 9:35-11:00 ET for each of 14 tickers
  4. Emit qualifying signals to PostgreSQL for consumption by trading bots

Feature computation is copied EXACTLY from backtest_gold_standard.py.
The model was trained on these exact features — do not refactor.

Usage:
    python -m options_owl.sourcing.ml_pipeline
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
from loguru import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = PROJECT_DIR / "journal" / "models" / "ml_v3"
UW_DB_PATH = PROJECT_DIR / "journal" / "uw_historical.db"

TICKERS = [
    "SPY", "QQQ", "NVDA", "TSLA", "META", "AAPL", "AMZN",
    "GOOGL", "MSFT", "AMD", "MSTR", "PLTR", "AVGO", "IWM",
    # New tickers (added 2026-05-29) — diversification beyond tech
    "COIN", "NFLX", "JPM", "BA", "MU", "SMCI",
]

# Tickers excluded from sourcing (net losers in concurrent backtest, 2026-05-30)
# MSFT: 22% WR. COIN: 55% WR, -$8.9K. AVGO: 71% WR but -$3.6K avg loss. MU: flat.
EXCLUDED_TICKERS = {"MSFT", "COIN", "AVGO", "MU"}

# Default thresholds (overridable via settings)
# DEFAULT_PATTERN_THRESHOLD is only the last-resort fallback — the runtime
# threshold comes from pattern_entry_meta.json "best_threshold" (the model's
# validated optimal operating point) unless ML_PATTERN_THRESHOLD is set in env.
DEFAULT_PATTERN_THRESHOLD = 0.74
DEFAULT_ENTRY_THRESHOLD = 0.80
DEFAULT_REGIME_THRESHOLD = 0.19
DEFAULT_SCAN_START_MIN = 5
DEFAULT_SCAN_END_MIN = 90

# Entry gates
PREMIUM_FLOOR = 0.20
PREMIUM_CAP = 6.0
SPREAD_GATE_PCT = 15.0


# ---------------------------------------------------------------------------
# Settings (read from environment, with defaults)
# ---------------------------------------------------------------------------


@dataclass
class MLPipelineSettings:
    """Configuration for the ML pipeline. Loaded from environment."""

    ENABLE_ML_PIPELINE: bool = False
    # 0.0 = use the model's own best_threshold from pattern_entry_meta.json.
    # Set explicitly (env ML_PATTERN_THRESHOLD) only to override the model.
    ML_PATTERN_THRESHOLD: float = 0.0
    ML_ENTRY_THRESHOLD: float = DEFAULT_ENTRY_THRESHOLD
    PUT_ENTRY_TIMING_THRESHOLD: float = 0.0  # disabled — model is net destructive for PUTs (blocks $53K profit to avoid $12K losses)
    ML_REGIME_THRESHOLD: float = DEFAULT_REGIME_THRESHOLD
    ML_SCAN_START_MIN: int = DEFAULT_SCAN_START_MIN
    ML_SCAN_END_MIN: int = DEFAULT_SCAN_END_MIN
    POLYGON_API_KEY: str = ""
    DATABASE_URL: str = "postgresql://owl:owl_dev_2026@postgres:5432/options_owl"
    AGENT_ID: str = "owlet_sourcing_ml"
    SHARED_CANDLE_DB: str = ""


def load_settings_from_env() -> MLPipelineSettings:
    """Build settings from environment variables."""
    import os

    def _bool(key: str, default: bool) -> bool:
        val = os.getenv(key, "")
        if not val:
            return default
        return val.lower() in ("true", "1", "yes")

    def _float(key: str, default: float) -> float:
        val = os.getenv(key, "")
        return float(val) if val else default

    def _int(key: str, default: int) -> int:
        val = os.getenv(key, "")
        return int(val) if val else default

    return MLPipelineSettings(
        ENABLE_ML_PIPELINE=_bool("ENABLE_ML_PIPELINE", False),
        ML_PATTERN_THRESHOLD=_float("ML_PATTERN_THRESHOLD", 0.0),  # 0 = model meta best_threshold
        ML_ENTRY_THRESHOLD=_float("ML_ENTRY_THRESHOLD", DEFAULT_ENTRY_THRESHOLD),
        PUT_ENTRY_TIMING_THRESHOLD=_float("PUT_ENTRY_TIMING_THRESHOLD", 0.0),
        ML_REGIME_THRESHOLD=_float("ML_REGIME_THRESHOLD", DEFAULT_REGIME_THRESHOLD),
        ML_SCAN_START_MIN=_int("ML_SCAN_START_MIN", DEFAULT_SCAN_START_MIN),
        ML_SCAN_END_MIN=_int("ML_SCAN_END_MIN", DEFAULT_SCAN_END_MIN),
        POLYGON_API_KEY=os.getenv("POLYGON_API_KEY", ""),
        DATABASE_URL=os.getenv(
            "DATABASE_URL",
            "postgresql://owl:owl_dev_2026@postgres:5432/options_owl",
        ),
        AGENT_ID=os.getenv("AGENT_ID", "owlet_sourcing_ml"),
        SHARED_CANDLE_DB=os.getenv("SHARED_CANDLE_DB", ""),
    )


# ---------------------------------------------------------------------------
# 1-Minute Candle Buffer
# ---------------------------------------------------------------------------


@dataclass
class MinuteBar:
    """A single 1-minute OHLCV bar."""

    timestamp: datetime  # bar open time (ET)
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float = 0.0  # Polygon AM-bar vwap (0 if not provided by the source)


class CandleBuffer:
    """Aggregates sporadic streaming quotes into clean 1-minute OHLCV bars.

    Bridges the gap between live WebSocket data (sporadic ticks at random
    intervals) and the model's expectation (clean, aligned 1-minute bars).
    """

    def __init__(self, max_bars: int = 120):
        self._bars: dict[str, deque[MinuteBar]] = {}
        self._building: dict[str, dict] = {}  # ticker -> partial bar being built
        self._max_bars = max_bars

    def ingest_tick(self, ticker: str, price: float, volume: float, ts: datetime) -> None:
        """Ingest a price tick and aggregate into 1-minute bars.

        Parameters
        ----------
        ticker : str
            Underlying or option ticker.
        price : float
            Trade/quote price.
        volume : float
            Trade volume (0 for quotes).
        ts : datetime
            Tick timestamp (must be tz-aware, ET preferred).
        """
        if price <= 0:
            return

        ticker = ticker.upper()
        # Truncate to minute boundary
        bar_time = ts.replace(second=0, microsecond=0)

        if ticker not in self._building:
            self._building[ticker] = {
                "bar_time": bar_time,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": volume,
            }
            return

        current = self._building[ticker]

        # Same minute — update OHLCV
        if current["bar_time"] == bar_time:
            current["high"] = max(current["high"], price)
            current["low"] = min(current["low"], price)
            current["close"] = price
            current["volume"] += volume
            return

        # New minute — finalize previous bar and start new one
        self._finalize_bar(ticker, current)
        self._building[ticker] = {
            "bar_time": bar_time,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": volume,
        }

    def _finalize_bar(self, ticker: str, bar_data: dict) -> None:
        """Convert a partial bar dict into a MinuteBar and append to history."""
        bar = MinuteBar(
            timestamp=bar_data["bar_time"],
            open=bar_data["open"],
            high=bar_data["high"],
            low=bar_data["low"],
            close=bar_data["close"],
            volume=bar_data["volume"],
        )
        if ticker not in self._bars:
            self._bars[ticker] = deque(maxlen=self._max_bars)
        self._bars[ticker].append(bar)

    def flush_current(self, ticker: str) -> None:
        """Force-finalize the current partial bar (call at scan time)."""
        ticker = ticker.upper()
        if ticker in self._building:
            self._finalize_bar(ticker, self._building[ticker])
            # Reset building state for next bar
            del self._building[ticker]

    def get_bars(self, ticker: str, count: int = 90) -> list[MinuteBar]:
        """Return the last `count` finalized 1-minute bars for a ticker."""
        ticker = ticker.upper()
        bars = self._bars.get(ticker)
        if not bars:
            return []
        return list(bars)[-count:]

    def get_numpy_arrays(self, ticker: str, count: int = 90) -> dict[str, np.ndarray] | None:
        """Return numpy arrays suitable for feature computation.

        Returns dict with keys: closes, volumes, highs, lows, or None if
        insufficient data.
        """
        bars = self.get_bars(ticker, count)
        if len(bars) < 5:
            return None

        return {
            "closes": np.array([b.close for b in bars], dtype=np.float64),
            "highs": np.array([b.high for b in bars], dtype=np.float64),
            "lows": np.array([b.low for b in bars], dtype=np.float64),
            "volumes": np.array([b.volume for b in bars], dtype=np.float64),
            "opens": np.array([b.open for b in bars], dtype=np.float64),
        }

    def ingest_polygon_minute_bars(
        self, ticker: str, bars: list[tuple[float, float, float, float, float, float, float]]
    ) -> None:
        """Ingest pre-aggregated minute bars from Polygon WS (AM.* events).

        Each tuple: (timestamp_ms, open, high, low, close, volume, vwap).
        """
        ticker = ticker.upper()
        for ts_ms, o, h, low, c, v, vw in bars:
            try:
                bar_time = datetime.fromtimestamp(ts_ms / 1000.0, tz=ET).replace(
                    second=0, microsecond=0
                )
            except (OSError, ValueError):
                continue

            if c <= 0:
                continue

            # Capture vwap (previously discarded) — needed if models are
            # retrained on real session VWAP instead of trailing-window mean.
            bar = MinuteBar(
                timestamp=bar_time, open=o, high=h, low=low, close=c,
                volume=v, vwap=float(vw or 0),
            )
            if ticker not in self._bars:
                self._bars[ticker] = deque(maxlen=self._max_bars)
            self._bars[ticker].append(bar)


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------


@dataclass
class MLModels:
    """Container for loaded LightGBM models and their metadata."""

    pattern_model: Any = None
    pattern_features: list[str] = field(default_factory=list)
    pattern_meta: dict = field(default_factory=dict)

    entry_model: Any = None
    entry_features: list[str] = field(default_factory=list)

    regime_model: Any = None
    regime_features: list[str] = field(default_factory=list)

    stop_model: Any = None
    stop_features: list[str] = field(default_factory=list)

    signal_model: Any = None
    signal_features: list[str] = field(default_factory=list)

    # Dedicated PUT pattern model (trained on PUT chain data)
    put_pattern_model: Any = None
    put_pattern_features: list[str] = field(default_factory=list)
    put_pattern_meta: dict = field(default_factory=dict)
    put_pattern_threshold: float = 0.40

    # Dedicated PUT entry timing model (trained on PUT chain data)
    put_entry_model: Any = None
    put_entry_features: list[str] = field(default_factory=list)
    put_entry_threshold: float = 0.80

    # V2 direction-specific models (per-ticker PUT/CALL)
    put_models: dict = field(default_factory=dict)  # {ticker: (model, features, threshold)}
    call_models: dict = field(default_factory=dict)  # {ticker: (model, features, threshold)}


def _check_feature_contract(name: str, model, meta_features: list, required: bool = False) -> bool:
    """Assert meta['features'] == booster.feature_name(). Fail loudly on mismatch.

    A mismatch silently corrupts every prediction (wrong column order/content).
    For required models this raises; for optional models it returns False so
    the caller can disable the model.
    """
    try:
        booster_features = list(model.feature_name())
    except Exception:
        return True  # cannot introspect — skip validation
    if list(meta_features) != booster_features:
        msg = (
            f"ML_PIPELINE: FEATURE CONTRACT MISMATCH for {name} — "
            f"meta has {len(meta_features)} features, booster has {len(booster_features)}. "
            f"meta_only={sorted(set(meta_features) - set(booster_features))} "
            f"booster_only={sorted(set(booster_features) - set(meta_features))}"
        )
        if required:
            raise ValueError(msg)
        logger.error(f"{msg} — model DISABLED")
        return False
    return True


def load_models() -> MLModels:
    """Load all ML models from MODEL_DIR. Returns MLModels container."""
    try:
        import lightgbm as lgb
    except ImportError:
        logger.error("lightgbm not installed — ML pipeline cannot start")
        raise

    models = MLModels()

    # Pattern entry (required)
    pattern_path = MODEL_DIR / "pattern_entry.txt"
    pattern_meta_path = MODEL_DIR / "pattern_entry_meta.json"
    if not pattern_path.exists():
        raise FileNotFoundError(f"Pattern model not found at {pattern_path}")

    models.pattern_model = lgb.Booster(model_file=str(pattern_path))
    with open(pattern_meta_path) as f:
        models.pattern_meta = json.load(f)
    models.pattern_features = models.pattern_meta["features"]
    _check_feature_contract("pattern_entry", models.pattern_model,
                            models.pattern_features, required=True)
    logger.info(f"ML_PIPELINE: Loaded pattern_entry (AUC={models.pattern_meta['auc']:.4f})")

    # Entry timing (optional but recommended)
    entry_path = MODEL_DIR / "entry_timing.txt"
    if entry_path.exists():
        models.entry_model = lgb.Booster(model_file=str(entry_path))
        models.entry_features = models.entry_model.feature_name()
        logger.info(
            f"ML_PIPELINE: Loaded entry_timing ({len(models.entry_features)} features)"
        )
    else:
        logger.warning("ML_PIPELINE: No entry_timing model — skipping entry quality gate")

    # Regime classifier (optional)
    regime_path = MODEL_DIR / "regime_classifier.txt"
    if regime_path.exists():
        models.regime_model = lgb.Booster(model_file=str(regime_path))
        models.regime_features = models.regime_model.feature_name()
        logger.info(
            f"ML_PIPELINE: Loaded regime_classifier ({len(models.regime_features)} features)"
        )
    else:
        logger.warning("ML_PIPELINE: No regime model — no daily pre-filter")

    # Stop calibration (optional)
    stop_path = MODEL_DIR / "stop_calibration.txt"
    if stop_path.exists():
        models.stop_model = lgb.Booster(model_file=str(stop_path))
        models.stop_features = models.stop_model.feature_name()
        logger.info(
            f"ML_PIPELINE: Loaded stop_calibration ({len(models.stop_features)} features)"
        )

    # Signal quality (optional, for ranking)
    signal_path = MODEL_DIR / "signal_quality.txt"
    if signal_path.exists():
        models.signal_model = lgb.Booster(model_file=str(signal_path))
        models.signal_features = models.signal_model.feature_name()
        logger.info(
            f"ML_PIPELINE: Loaded signal_quality ({len(models.signal_features)} features)"
        )

    # Dedicated PUT pattern model (optional — uses CALL model as fallback)
    put_pattern_path = MODEL_DIR / "put_pattern_v1.lgb"
    put_pattern_meta_path = MODEL_DIR / "put_pattern_v1_meta.json"
    if put_pattern_path.exists() and put_pattern_meta_path.exists():
        models.put_pattern_model = lgb.Booster(model_file=str(put_pattern_path))
        with open(put_pattern_meta_path) as f:
            models.put_pattern_meta = json.load(f)
        models.put_pattern_features = models.put_pattern_meta.get("features", [])
        if not models.put_pattern_features:
            logger.error("ML_PIPELINE: PUT pattern meta missing 'features' — disabling PUT model")
            models.put_pattern_model = None
        elif not _check_feature_contract("put_pattern_v1", models.put_pattern_model,
                                         models.put_pattern_features):
            models.put_pattern_model = None
        else:
            models.put_pattern_threshold = models.put_pattern_meta.get("best_threshold", 0.40)
            logger.info(
                f"ML_PIPELINE: Loaded put_pattern_v1 "
                f"(AUC={models.put_pattern_meta.get('auc', 0):.4f}, "
                f"threshold={models.put_pattern_threshold:.2f})"
            )
    else:
        logger.warning("ML_PIPELINE: No PUT pattern model — using CALL model for PUT signals")

    # Dedicated PUT entry timing model (optional — PUTs skip entry timing if missing)
    put_entry_path = MODEL_DIR / "put_entry_timing.txt"
    put_entry_meta_path = MODEL_DIR / "put_entry_timing_meta.json"
    if put_entry_path.exists() and put_entry_meta_path.exists():
        models.put_entry_model = lgb.Booster(model_file=str(put_entry_path))
        with open(put_entry_meta_path) as f:
            put_entry_meta = json.load(f)
        models.put_entry_features = put_entry_meta.get("features", [])
        if not models.put_entry_features:
            logger.error("ML_PIPELINE: PUT entry timing meta missing 'features' — disabling")
            models.put_entry_model = None
        elif not _check_feature_contract("put_entry_timing", models.put_entry_model,
                                         models.put_entry_features):
            models.put_entry_model = None
        else:
            meta_threshold = put_entry_meta.get("best_threshold", 0.80)
            models.put_entry_threshold = meta_threshold
            logger.info(
                f"ML_PIPELINE: Loaded put_entry_timing "
                f"(AUC={put_entry_meta.get('auc', 0):.4f}, "
                f"meta_threshold={meta_threshold:.2f}) "
                f"— runtime threshold from PUT_ENTRY_TIMING_THRESHOLD setting"
            )
    else:
        logger.info("ML_PIPELINE: No PUT entry timing model — PUTs skip entry timing")

    # V2 direction-specific models (per-ticker PUT signal models)
    v2_dir = PROJECT_DIR / "journal" / "models" / "signal_ml_v2"
    if v2_dir.exists():
        for ticker in TICKERS:
            for direction in ["PUT", "CALL"]:
                model_path = v2_dir / f"signal_{ticker}_{direction}.lgb"
                meta_path = v2_dir / f"signal_{ticker}_{direction}_meta.json"
                if model_path.exists() and meta_path.exists():
                    try:
                        m = lgb.Booster(model_file=str(model_path))
                        with open(meta_path) as mf:
                            meta = json.load(mf)
                        features = meta.get("features", m.feature_name())
                        if not _check_feature_contract(
                            f"signal_{ticker}_{direction}", m, features
                        ):
                            continue
                        threshold = meta.get("optimal_threshold", 0.5)
                        target = models.put_models if direction == "PUT" else models.call_models
                        target[ticker] = (m, features, threshold)
                    except Exception as exc:
                        logger.debug(f"ML_PIPELINE: Failed to load {ticker}_{direction}: {exc}")

        put_count = len(models.put_models)
        call_count = len(models.call_models)
        if put_count > 0 or call_count > 0:
            logger.info(
                f"ML_PIPELINE: Loaded V2 direction models: {put_count} PUT, {call_count} CALL"
            )

    return models


# ---------------------------------------------------------------------------
# Feature Computation — EXACT copy from backtest_gold_standard.py
# ---------------------------------------------------------------------------


def compute_pattern_features(
    closes: np.ndarray,
    volumes: np.ndarray,
    ivs: np.ndarray,
    deltas: np.ndarray,
    thetas: np.ndarray,
    underlyings: np.ndarray,
    bids: np.ndarray,
    asks: np.ndarray,
    idx: int,
    opening_price: float,
) -> dict | None:
    """Compute trailing features for pattern model at position idx.

    EXACT copy of compute_pattern_features() from backtest_gold_standard.py.
    """
    if idx < 5:
        return None

    w5_start = max(0, idx - 5)
    w10_start = max(0, idx - 10)

    pre5 = closes[w5_start:idx]
    pre10 = closes[w10_start:idx]
    pre5_v = volumes[w5_start:idx]
    pre5_iv = ivs[w5_start:idx]
    pre5_u = underlyings[w5_start:idx]

    valid5 = pre5[~np.isnan(pre5)]
    valid10 = pre10[~np.isnan(pre10)]
    valid5_v = pre5_v[~np.isnan(pre5_v)]
    valid5_iv = pre5_iv[~np.isnan(pre5_iv)]
    valid5_u = pre5_u[~np.isnan(pre5_u)]

    if len(valid5) < 3 or valid5[0] <= 0:
        return None

    current = closes[idx]
    if np.isnan(current) or current <= 0:
        return None

    f: dict[str, float] = {}
    f["prem_slope_5"] = (valid5[-1] / valid5[0] - 1) * 100
    f["prem_slope_10"] = (
        (valid10[-1] / valid10[0] - 1) * 100
        if len(valid10) >= 5 and valid10[0] > 0
        else f["prem_slope_5"]
    )

    if len(valid5) >= 4:
        mid = len(valid5) // 2
        first_rate = (valid5[mid] / valid5[0] - 1) * 100 if valid5[0] > 0 else 0
        second_rate = (valid5[-1] / valid5[mid] - 1) * 100 if valid5[mid] > 0 else 0
        f["prem_accel"] = second_rate - first_rate
    else:
        f["prem_accel"] = 0

    last3 = valid5[-3:] if len(valid5) >= 3 else valid5
    f["prem_stabilizing"] = (
        (max(last3) - min(last3)) / max(last3) * 100 if max(last3) > 0 else 0
    )

    if len(valid5) >= 3 and all(c > 0 for c in valid5[:-1]):
        returns = np.diff(valid5) / valid5[:-1]
        f["prem_volatility"] = float(np.std(returns) * 100)
    else:
        f["prem_volatility"] = 0

    f["volume_avg_5"] = float(np.mean(valid5_v)) if len(valid5_v) > 0 else 0
    w20_start = max(0, idx - 20)
    vol20 = volumes[w20_start:idx]
    vol20_valid = vol20[~np.isnan(vol20)]
    avg20 = float(np.mean(vol20_valid)) if len(vol20_valid) > 0 else 1
    f["volume_ratio"] = f["volume_avg_5"] / max(avg20, 1)

    if len(valid5_v) >= 3:
        f["volume_trend"] = float(valid5_v[-1] / max(valid5_v[0], 1))
    else:
        f["volume_trend"] = 1.0

    if len(valid5_iv) >= 2:
        f["iv_change_5"] = float(valid5_iv[-1] - valid5_iv[0])
        f["iv_level"] = float(valid5_iv[-1])
    else:
        f["iv_change_5"] = 0
        f["iv_level"] = 0

    if len(valid5_u) >= 2 and valid5_u[0] > 0:
        f["und_slope_5"] = (valid5_u[-1] / valid5_u[0] - 1) * 100
    else:
        f["und_slope_5"] = 0

    f["drop_from_open"] = (current / opening_price - 1) * 100 if opening_price > 0 else 0

    bid = bids[idx] if idx < len(bids) else 0
    ask = asks[idx] if idx < len(asks) else 0
    f["spread_pct"] = (ask - bid) / ask * 100 if ask > 0 and bid >= 0 else 0
    f["delta"] = float(deltas[idx]) if idx < len(deltas) and not np.isnan(deltas[idx]) else 0
    f["theta"] = float(thetas[idx]) if idx < len(thetas) and not np.isnan(thetas[idx]) else 0
    f["minutes_since_open"] = idx
    f["premium"] = float(current)

    return f


def compute_put_pattern_features(
    closes: np.ndarray,
    volumes: np.ndarray,
    ivs: np.ndarray,
    deltas: np.ndarray,
    thetas: np.ndarray,
    underlyings: np.ndarray,
    bids: np.ndarray,
    asks: np.ndarray,
    idx: int,
    opening_price: float,
    vegas: np.ndarray | None = None,
    bid_sizes: np.ndarray | None = None,
    ask_sizes: np.ndarray | None = None,
    call_ivs: np.ndarray | None = None,
    call_volumes: np.ndarray | None = None,
) -> dict | None:
    """Compute 27 features for PUT pattern model V2 at position idx.

    EXACT copy of compute_put_features() from scripts/train_put_pattern.py.
    """
    if idx < 5:
        return None

    w5_start = max(0, idx - 5)
    w10_start = max(0, idx - 10)
    w15_start = max(0, idx - 15)

    pre5 = closes[w5_start:idx]
    pre10 = closes[w10_start:idx]
    pre5_v = volumes[w5_start:idx]
    pre5_iv = ivs[w5_start:idx]
    pre5_u = underlyings[w5_start:idx]

    valid5 = pre5[~np.isnan(pre5)]
    valid10 = pre10[~np.isnan(pre10)]
    valid5_v = pre5_v[~np.isnan(pre5_v)]
    valid5_iv = pre5_iv[~np.isnan(pre5_iv)]
    valid5_u = pre5_u[~np.isnan(pre5_u)]

    if len(valid5) < 3 or valid5[0] <= 0:
        return None

    current = closes[idx]
    if np.isnan(current) or current <= 0:
        return None

    f: dict[str, float] = {}

    # Premium trajectory
    f["prem_slope_5"] = (valid5[-1] / valid5[0] - 1) * 100
    f["prem_slope_10"] = (
        (valid10[-1] / valid10[0] - 1) * 100
        if len(valid10) >= 5 and valid10[0] > 0
        else f["prem_slope_5"]
    )

    if len(valid5) >= 4:
        mid = len(valid5) // 2
        first_rate = (valid5[mid] / valid5[0] - 1) * 100 if valid5[0] > 0 else 0
        second_rate = (valid5[-1] / valid5[mid] - 1) * 100 if valid5[mid] > 0 else 0
        f["prem_accel"] = second_rate - first_rate
    else:
        f["prem_accel"] = 0

    last3 = valid5[-3:] if len(valid5) >= 3 else valid5
    f["prem_stabilizing"] = (
        (max(last3) - min(last3)) / max(last3) * 100 if max(last3) > 0 else 0
    )

    if len(valid5) >= 3 and all(c > 0 for c in valid5[:-1]):
        returns = np.diff(valid5) / valid5[:-1]
        f["prem_volatility"] = float(np.std(returns) * 100)
    else:
        f["prem_volatility"] = 0

    # Volume
    f["volume_avg_5"] = float(np.mean(valid5_v)) if len(valid5_v) > 0 else 0
    w20_start = max(0, idx - 20)
    vol20 = volumes[w20_start:idx]
    vol20_valid = vol20[~np.isnan(vol20)]
    avg20 = float(np.mean(vol20_valid)) if len(vol20_valid) > 0 else 1
    f["volume_ratio"] = f["volume_avg_5"] / max(avg20, 1)

    if len(valid5_v) >= 3:
        f["volume_trend"] = float(valid5_v[-1] / max(valid5_v[0], 1))
    else:
        f["volume_trend"] = 1.0

    # IV
    if len(valid5_iv) >= 2:
        f["iv_change_5"] = float(valid5_iv[-1] - valid5_iv[0])
        f["iv_level"] = float(valid5_iv[-1])
    else:
        f["iv_change_5"] = 0
        f["iv_level"] = 0

    # IV acceleration
    if len(valid5_iv) >= 4:
        mid_iv = len(valid5_iv) // 2
        first_iv_rate = valid5_iv[mid_iv] - valid5_iv[0]
        second_iv_rate = valid5_iv[-1] - valid5_iv[mid_iv]
        f["iv_accel"] = float(second_iv_rate - first_iv_rate)
    else:
        f["iv_accel"] = 0.0

    # Underlying slopes (5, 10, 15 candles)
    if len(valid5_u) >= 2 and valid5_u[0] > 0:
        f["und_slope_5"] = (valid5_u[-1] / valid5_u[0] - 1) * 100
    else:
        f["und_slope_5"] = 0

    pre10_u = underlyings[w10_start:idx]
    pre15_u = underlyings[w15_start:idx]
    valid10_u = pre10_u[~np.isnan(pre10_u)]
    valid15_u = pre15_u[~np.isnan(pre15_u)]
    f["und_slope_10"] = (
        (valid10_u[-1] / valid10_u[0] - 1) * 100
        if len(valid10_u) >= 5 and valid10_u[0] > 0
        else f["und_slope_5"]
    )
    f["und_slope_15"] = (
        (valid15_u[-1] / valid15_u[0] - 1) * 100
        if len(valid15_u) >= 5 and valid15_u[0] > 0
        else f["und_slope_10"]
    )

    # Underlying momentum (RSI-like: ratio of down vs total moves)
    if len(valid5_u) >= 3:
        diffs = np.diff(valid5_u)
        up_sum = float(np.sum(diffs[diffs > 0]))
        down_sum = float(-np.sum(diffs[diffs < 0]))
        f["und_momentum"] = down_sum / max(up_sum + down_sum, 1e-8) * 100
    else:
        f["und_momentum"] = 50.0

    # Consecutive underlying down candles
    if len(valid5_u) >= 3 and valid5_u[0] > 0:
        down_count = 0
        for i in range(len(valid5_u) - 1, 0, -1):
            if valid5_u[i] < valid5_u[i - 1]:
                down_count += 1
            else:
                break
        f["consec_underlying_down"] = down_count
    else:
        f["consec_underlying_down"] = 0

    f["drop_from_open"] = (current / opening_price - 1) * 100 if opening_price > 0 else 0

    bid = bids[idx] if idx < len(bids) else 0
    ask = asks[idx] if idx < len(asks) else 0
    f["spread_pct"] = (ask - bid) / ask * 100 if ask > 0 and bid >= 0 else 0
    f["delta"] = float(deltas[idx]) if idx < len(deltas) and not np.isnan(deltas[idx]) else 0
    f["theta"] = float(thetas[idx]) if idx < len(thetas) and not np.isnan(thetas[idx]) else 0

    # Vega
    if vegas is not None and idx < len(vegas) and not np.isnan(vegas[idx]):
        f["vega"] = float(vegas[idx])
    else:
        f["vega"] = 0.0

    f["minutes_since_open"] = idx
    f["premium"] = float(current)

    # Candle range (intrabar volatility — approximated from close spread)
    # In production we don't have per-candle high/low, so use bid-ask range
    f["candle_range_pct"] = (ask - bid) / max(current, 0.01) * 100 if bid >= 0 else 0.0

    # Bid/ask size imbalance
    if bid_sizes is not None and ask_sizes is not None and idx < len(bid_sizes):
        bs = bid_sizes[idx] if not np.isnan(bid_sizes[idx]) else 0
        as_ = ask_sizes[idx] if not np.isnan(ask_sizes[idx]) else 0
        total = bs + as_
        f["bid_size_ratio"] = bs / max(total, 1)
    else:
        f["bid_size_ratio"] = 0.5

    # IV skew (PUT IV / CALL IV)
    if call_ivs is not None and idx < len(call_ivs):
        put_iv = valid5_iv[-1] if len(valid5_iv) > 0 else 0
        call_iv = call_ivs[idx] if not np.isnan(call_ivs[idx]) else 0
        f["iv_skew"] = put_iv / call_iv if call_iv > 0 else 1.0
    else:
        f["iv_skew"] = 1.0

    # PUT volume / CALL volume
    if call_volumes is not None and idx < len(call_volumes):
        call_vol = call_volumes[idx] if not np.isnan(call_volumes[idx]) else 0
        put_vol = volumes[idx] if not np.isnan(volumes[idx]) else 0
        f["put_call_volume_ratio"] = put_vol / max(call_vol, 1)
    else:
        f["put_call_volume_ratio"] = 1.0

    return f


def compute_entry_timing_features(
    closes: np.ndarray,
    volumes: np.ndarray,
    bids_arr: np.ndarray,
    asks_arr: np.ndarray,
    bid_sizes: np.ndarray,
    ask_sizes: np.ndarray,
    ivs: np.ndarray,
    deltas: np.ndarray,
    thetas: np.ndarray,
    vegas: np.ndarray,
    underlyings: np.ndarray,
    stock_closes: np.ndarray,
    stock_highs: np.ndarray,
    stock_lows: np.ndarray,
    idx: int,
    entry_features: list[str],
) -> dict | None:
    """Compute entry_timing model features at position idx.

    EXACT copy of compute_entry_timing_features() from backtest_gold_standard.py.
    """
    lookback = 15
    if idx < lookback + 1:
        return None

    entry_price = closes[idx]
    if np.isnan(entry_price) or entry_price <= 0:
        return None

    f: dict[str, float] = {}

    # Time
    f["minutes_since_open"] = idx
    f["hour_bucket"] = idx // 60
    f["is_first_30min"] = 1 if idx <= 30 else 0

    # Premium action
    prices = closes[max(0, idx - lookback) : idx + 1]
    valid_prices = prices[~np.isnan(prices) & (prices > 0)]
    if len(valid_prices) < 3:
        return None

    f["premium"] = float(entry_price)
    f["premium_change_5m"] = (
        float(
            (valid_prices[-1] / valid_prices[max(-6, -len(valid_prices))] - 1) * 100
        )
        if valid_prices[max(-6, -len(valid_prices))] > 0
        else 0
    )
    f["premium_change_10m"] = (
        float(
            (valid_prices[-1] / valid_prices[max(-11, -len(valid_prices))] - 1) * 100
        )
        if valid_prices[max(-11, -len(valid_prices))] > 0
        else 0
    )
    f["premium_change_15m"] = (
        float((valid_prices[-1] / valid_prices[0] - 1) * 100)
        if valid_prices[0] > 0
        else 0
    )

    if len(valid_prices) > 2 and all(valid_prices[:-1] > 0):
        returns = np.diff(valid_prices) / valid_prices[:-1]
        f["premium_volatility"] = float(np.std(returns) * 100)
    else:
        f["premium_volatility"] = 0

    # Volume
    vols = volumes[max(0, idx - lookback) : idx + 1]
    valid_vols = vols[~np.isnan(vols)]
    f["current_volume"] = float(volumes[idx]) if not np.isnan(volumes[idx]) else 0
    avg_vol = float(np.mean(valid_vols[:-1])) if len(valid_vols) > 1 else 1
    f["volume_ratio"] = float(f["current_volume"] / max(avg_vol, 1))
    if len(valid_vols) > 5 and np.std(valid_vols[:-1]) > 0:
        f["volume_zscore"] = float(
            (valid_vols[-1] - np.mean(valid_vols[:-1])) / np.std(valid_vols[:-1])
        )
    else:
        f["volume_zscore"] = 0

    # Bid/ask
    bid = float(bids_arr[idx]) if not np.isnan(bids_arr[idx]) else 0
    ask = float(asks_arr[idx]) if not np.isnan(asks_arr[idx]) else 0
    mid = (bid + ask) / 2 if (bid + ask) > 0 else entry_price
    f["spread"] = float(ask - bid) if ask > bid else 0
    f["spread_pct"] = float(f["spread"] / mid * 100) if mid > 0 else 0
    f["bid_size"] = (
        float(bid_sizes[idx])
        if idx < len(bid_sizes) and not np.isnan(bid_sizes[idx])
        else 0
    )
    f["ask_size"] = (
        float(ask_sizes[idx])
        if idx < len(ask_sizes) and not np.isnan(ask_sizes[idx])
        else 0
    )
    f["size_imbalance"] = float(
        (f["bid_size"] - f["ask_size"]) / max(f["bid_size"] + f["ask_size"], 1)
    )

    # Greeks
    f["iv"] = float(ivs[idx]) if not np.isnan(ivs[idx]) else 0
    f["delta"] = float(abs(deltas[idx])) if not np.isnan(deltas[idx]) else 0
    f["theta"] = float(thetas[idx]) if not np.isnan(thetas[idx]) else 0
    f["vega"] = (
        float(vegas[idx]) if idx < len(vegas) and not np.isnan(vegas[idx]) else 0
    )

    iv_window = ivs[max(0, idx - lookback) : idx + 1]
    valid_iv = iv_window[~np.isnan(iv_window)]
    f["iv_change_15m"] = float(valid_iv[-1] - valid_iv[0]) if len(valid_iv) > 3 else 0

    f["underlying_price"] = (
        float(underlyings[idx]) if not np.isnan(underlyings[idx]) else 0
    )

    # Underlying price action (from stock data)
    s_idx = min(idx, len(stock_closes) - 1)
    if s_idx > 5 and len(stock_closes) > 5:
        s_window = stock_closes[max(0, s_idx - lookback) : s_idx + 1]
        s_valid = s_window[~np.isnan(s_window) & (s_window > 0)]
        if len(s_valid) > 1:
            f["underlying_change_5m"] = float(
                (s_valid[-1] / s_valid[max(-6, -len(s_valid))] - 1) * 100
            )
            f["underlying_change_15m"] = float(
                (s_valid[-1] / s_valid[0] - 1) * 100
            )
            if len(s_valid) > 2 and all(s_valid[:-1] > 0):
                f["underlying_volatility"] = float(
                    np.std(np.diff(s_valid) / s_valid[:-1]) * 100
                )
            else:
                f["underlying_volatility"] = 0
        else:
            f["underlying_change_5m"] = 0
            f["underlying_change_15m"] = 0
            f["underlying_volatility"] = 0

        # Daily trend
        s_all = stock_closes[: s_idx + 1]
        s_all_valid = s_all[~np.isnan(s_all) & (s_all > 0)]
        if len(s_all_valid) > 10 and s_all_valid[0] > 0:
            f["daily_trend_pct"] = float(
                (s_all_valid[-1] / s_all_valid[0] - 1) * 100
            )
        else:
            f["daily_trend_pct"] = 0

        if len(s_all_valid) > 1:
            day_lo = s_all_valid.min()
            day_hi = s_all_valid.max()
            f["daily_range_position"] = (
                float((s_all_valid[-1] - day_lo) / (day_hi - day_lo))
                if day_hi > day_lo
                else 0.5
            )
        else:
            f["daily_range_position"] = 0.5

        # ATR
        if s_idx > 14 and len(stock_highs) > 14:
            h_window = stock_highs[max(0, s_idx - 14) : s_idx]
            l_window = stock_lows[max(0, s_idx - 14) : s_idx]
            h_valid = h_window[~np.isnan(h_window)]
            l_valid = l_window[~np.isnan(l_window)]
            if (
                len(h_valid) >= 14
                and len(l_valid) >= 14
                and s_all_valid[-1] > 0
            ):
                f["atr_pct"] = float(
                    np.mean(h_valid[-14:] - l_valid[-14:]) / s_all_valid[-1] * 100
                )
            else:
                f["atr_pct"] = 0
        else:
            f["atr_pct"] = 0
    else:
        for k in [
            "underlying_change_5m",
            "underlying_change_15m",
            "underlying_volatility",
            "daily_trend_pct",
            "daily_range_position",
            "atr_pct",
        ]:
            f[k] = 0

    # Premium drop from recent peak (top feature)
    recent = closes[max(0, idx - 10) : idx + 1]
    valid_recent = recent[~np.isnan(recent) & (recent > 0)]
    if len(valid_recent) > 0:
        f["prem_drop_from_recent_peak"] = float(
            (closes[idx] / np.max(valid_recent) - 1) * 100
        )
    else:
        f["prem_drop_from_recent_peak"] = 0

    # Decline deceleration
    if len(valid_recent) >= 3:
        first_half = valid_recent[: len(valid_recent) // 2]
        second_half = valid_recent[len(valid_recent) // 2 :]
        if (
            len(first_half) > 0
            and len(second_half) > 0
            and first_half[0] > 0
            and second_half[0] > 0
        ):
            first_change = (first_half[-1] / first_half[0] - 1) * 100
            second_change = (second_half[-1] / second_half[0] - 1) * 100
            f["decline_deceleration"] = float(second_change - first_change)
        else:
            f["decline_deceleration"] = 0
    else:
        f["decline_deceleration"] = 0

    # Return only features the model expects (warn once if any are missing
    # from the computed dict instead of silently zero-filling)
    from options_owl.sourcing.scoring.ml_gates.signal_model import _warn_missing_features

    _warn_missing_features("entry_timing_features", entry_features, f)
    return {k: f.get(k, 0) for k in entry_features}


async def compute_regime_features(
    ticker: str,
    now_et: datetime,
    regime_features: list[str],
) -> dict | None:
    """Compute the FULL regime feature vector for live serving (Postgres).

    Delegates to the SHARED feature module (regime_features.py) — the single
    source of truth also used by scripts/train_ml_models_v3.py. This produces
    the model's complete feature set (40 features) instead of the old 18, so
    there is no train/serve skew and no silent zero-fill: load_serving_inputs
    reads Postgres (stock_candles SPY/QQQ/VIX + lags, gex_ticks) and
    compute_regime_feature_vector builds every feature in REGIME_FEATURE_ORDER.

    Returns the feature dict keyed in the model's expected order, or None if the
    early-morning window is too sparse to score.
    """
    from options_owl.sourcing.features.regime_features import (
        REGIME_FEATURE_ORDER,
        compute_regime_feature_vector,
        load_serving_inputs,
    )
    from options_owl.sourcing.scoring.ml_gates.signal_model import (
        _warn_missing_features,
    )

    raw_inputs = await load_serving_inputs(ticker, now_et, tz_et=ET)

    # Require a usable early-morning window (>= 5 bars) — same gate as training.
    morning = raw_inputs.get("morning_bars") or []
    if len([b for b in morning if b.get("close", 0) > 0]) < 5:
        return None

    f = compute_regime_feature_vector(raw_inputs)

    # The shared module ALWAYS emits the full REGIME_FEATURE_ORDER, so this
    # warner must stay quiet. It still guards against a future model/meta drift.
    _warn_missing_features("regime_classifier", regime_features, f)
    # Honor the loaded model's feature order (REGIME_FEATURE_ORDER by default).
    return {k: f.get(k, 0.0) for k in (regime_features or REGIME_FEATURE_ORDER)}


def _build_v2_signal_features(
    state: "TickerScanState",
    idx: int,
    v2_features: list[str],
    direction: str,
) -> dict | None:
    """Build V2 signal model features from TickerScanState data.

    Thin adapter: slices the accumulated per-minute arrays into the history
    windows expected by the SHARED feature builder
    (signal_model.compute_option_features_from_live) — the single source of
    truth that matches train_option_signals_v2.py exactly. Do NOT reimplement
    feature math here.
    """
    if idx < 5:
        return None

    from options_owl.sourcing.scoring.ml_gates.signal_model import (
        _warn_missing_features,
        compute_option_features_from_live,
    )

    arrays = state.to_numpy()
    closes = arrays["closes"]
    volumes = arrays["volumes"]
    ivs = arrays["ivs"]
    deltas = arrays["deltas"]
    thetas = arrays["thetas"]
    vegas = arrays["vegas"]
    underlyings = arrays["underlyings"]
    bids = arrays["bids"]
    asks = arrays["asks"]
    bid_sizes = arrays["bid_sizes"]
    ask_sizes = arrays["ask_sizes"]

    if idx >= len(closes) or closes[idx] <= 0:
        return None

    lookback = 15
    start = max(0, idx - lookback)

    # Current snapshot values
    premium = float(closes[idx])
    bid = float(bids[idx]) if idx < len(bids) else 0.0
    ask = float(asks[idx]) if idx < len(asks) else 0.0
    iv = float(ivs[idx]) if idx < len(ivs) else 0.0
    delta = float(deltas[idx]) if idx < len(deltas) else 0.0
    theta = float(thetas[idx]) if idx < len(thetas) else 0.0
    vega = float(vegas[idx]) if idx < len(vegas) else 0.0
    volume = int(volumes[idx]) if idx < len(volumes) else 0
    underlying_price = float(underlyings[idx]) if idx < len(underlyings) else 0.0
    bid_size = float(bid_sizes[idx]) if idx < len(bid_sizes) else 0.0
    ask_size = float(ask_sizes[idx]) if idx < len(ask_sizes) else 0.0

    # Trailing histories (see compute_option_features_from_live docstring for
    # inclusion conventions): premium/volume EXCLUDE current; spread/IV/
    # underlying INCLUDE current as last element.
    premium_history = [float(v) for v in closes[start:idx] if v > 0]
    volume_history = [int(v) for v in volumes[start:idx]]
    underlying_history = [float(v) for v in underlyings[start:idx + 1] if v > 0]
    spread_history = [
        float(a - b)
        for a, b in zip(asks[start:idx + 1], bids[start:idx + 1])
        if a > 0 and b > 0 and a >= b
    ]
    iv_history = [float(v) for v in ivs[start:idx + 1] if v > 0]

    f = compute_option_features_from_live(
        ticker="",  # ticker not used in feature math
        premium=premium,
        bid=bid,
        ask=ask,
        iv=iv,
        delta=delta,
        theta=theta,
        vega=vega,
        volume=volume,
        underlying_price=underlying_price,
        minutes_since_open=idx,  # state accumulates one entry per minute
        is_call=direction == "CALL",
        premium_history=premium_history,
        volume_history=volume_history,
        underlying_history=underlying_history,
        bid_size=bid_size,
        ask_size=ask_size,
        spread_history=spread_history,
        iv_history=iv_history,
    )

    _warn_missing_features("v2_signal_pipeline", v2_features, f)
    return {k: f.get(k, 0) for k in v2_features}


# ---------------------------------------------------------------------------
# Live Data Fetching
# ---------------------------------------------------------------------------


async def fetch_live_option_chain(
    api_key: str, ticker: str, expiry: str
) -> list[dict]:
    """Fetch ATM call option chain from Polygon for a given ticker/expiry.

    Returns list of dicts with keys: strike, bid, ask, mid, volume,
    open_interest, last_price, option_type, iv, delta, theta, vega.

    Near-expiry fallback (2026-06-15 fix): not every ticker has a 0DTE today (e.g. AMD/JPM/MSTR
    expire Friday). If the requested expiry has 0 contracts, walk forward over the next few business
    days to the ticker's nearest available expiry. Liquid 0DTE names return on the first call.
    """
    from datetime import datetime as _dt
    from datetime import timedelta as _td

    from options_owl.collectors.polygon_options import polygon_option_chain

    try:
        base = _dt.strptime(expiry, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        base = _dt.now(tz=ET).date()
    candidates, d = [expiry] if expiry else [], base
    for _ in range(6):  # walk up to ~1 week ahead to hit the nearest weekly expiry
        d = d + _td(days=1)
        if d.weekday() < 5:
            candidates.append(d.strftime("%Y-%m-%d"))

    seen: set[str] = set()
    for exp in candidates:
        if not exp or exp in seen:
            continue
        seen.add(exp)
        try:
            chain = await asyncio.wait_for(
                polygon_option_chain(api_key, ticker, exp, option_type="call"), timeout=10)
            if chain:  # first non-empty expiry wins
                return chain
        except asyncio.TimeoutError:
            logger.warning(f"ML_PIPELINE: Polygon chain timeout for {ticker} {exp}")
        except Exception as exc:
            logger.warning(f"ML_PIPELINE: Polygon chain error for {ticker} {exp}: {exc}")
    return []


async def fetch_live_underlying_price(api_key: str, ticker: str) -> float | None:
    """Fetch current underlying price — tries Redis first, falls back to Polygon REST."""
    # Try Redis first (harvester publishes prices every minute bar)
    try:
        from options_owl.db import redis_client
        if redis_client.is_connected():
            result = await redis_client.get_price(ticker, max_age=90)
            if result is not None:
                price, age = result
                logger.debug(
                    f"ML_PIPELINE: {ticker} price from Redis: ${price:.2f} ({age:.0f}s old)"
                )
                return price
    except Exception:
        pass

    # Fallback: Polygon REST snapshot
    try:
        import httpx

        url = f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}"
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, params={"apiKey": api_key})
            if resp.status_code == 200:
                data = resp.json()
                ticker_data = data.get("ticker", {})
                last_trade = ticker_data.get("lastTrade", {})
                price = last_trade.get("p")
                if price and price > 0:
                    return float(price)
                # Fallback: day close
                day = ticker_data.get("day", {})
                close = day.get("c")
                if close and close > 0:
                    return float(close)
        return None
    except Exception as exc:
        logger.debug(f"ML_PIPELINE: underlying price fetch failed for {ticker}: {exc}")
        return None


async def fetch_option_snapshot_data(
    api_key: str, ticker: str, strike: float, expiry: str
) -> dict | None:
    """Fetch full option snapshot (bid/ask/greeks) for a specific contract.

    Returns dict with keys: bid, ask, mid, iv, delta, theta, vega, volume,
    underlying_price, bid_size, ask_size.
    """
    try:
        import httpx
        from options_owl.collectors.polygon_options import build_option_contract_ticker

        contract = build_option_contract_ticker(ticker, strike, expiry, "call")
        url = f"https://api.polygon.io/v3/snapshot/options/{ticker.upper()}/{contract}"

        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, params={"apiKey": api_key})
            if resp.status_code != 200:
                return None

            result = resp.json().get("results", {})
            quote = result.get("last_quote", {})
            greeks = result.get("greeks", {})
            day = result.get("day", {})
            details = result.get("details", {})

            bid = float(quote.get("bid", 0) or 0)
            ask = float(quote.get("ask", 0) or 0)
            mid = round((bid + ask) / 2.0, 2) if bid > 0 and ask > 0 else 0

            return {
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "iv": float(greeks.get("implied_volatility", 0) or 0),
                "delta": float(greeks.get("delta", 0) or 0),
                "theta": float(greeks.get("theta", 0) or 0),
                "vega": float(greeks.get("vega", 0) or 0),
                "volume": int(day.get("volume", 0) or 0),
                "underlying_price": float(
                    result.get("underlying_asset", {}).get("price", 0) or 0
                ),
                "bid_size": float(quote.get("bid_size", 0) or 0),
                "ask_size": float(quote.get("ask_size", 0) or 0),
                "open_interest": int(result.get("open_interest", 0) or 0),
                "strike": float(details.get("strike_price", strike) or strike),
            }
    except Exception as exc:
        logger.debug(f"ML_PIPELINE: option snapshot error for {ticker}: {exc}")
        return None


def get_todays_expiry() -> str:
    """Get today's date formatted for Polygon option queries."""
    now_et = datetime.now(tz=ET)
    return now_et.strftime("%Y-%m-%d")


def find_atm_strike(chain: list[dict], underlying_price: float) -> dict | None:
    """Find the ATM call from the chain closest to underlying price."""
    if not chain or underlying_price <= 0:
        return None

    calls = [c for c in chain if c.get("option_type", "").lower() == "call"]
    if not calls:
        calls = chain  # fallback if all are calls already

    best = min(calls, key=lambda c: abs(c.get("strike", 0) - underlying_price))
    return best


async def fetch_gex_data(ticker: str) -> dict | None:
    """Fetch GEX data from UW historical DB (previous day close)."""
    if not UW_DB_PATH.exists():
        return None

    today = datetime.now(tz=ET).strftime("%Y-%m-%d")

    def _sync() -> dict | None:
        try:
            conn = sqlite3.connect(str(UW_DB_PATH))
            conn.execute("PRAGMA busy_timeout = 5000")
            row = conn.execute(
                "SELECT call_gamma, put_gamma, call_delta, put_delta "
                "FROM greek_exposure WHERE ticker=? AND date<? "
                "ORDER BY date DESC LIMIT 1",
                (ticker, today),
            ).fetchone()
            conn.close()
            if row:
                return {
                    "call_gamma": float(row[0] or 0),
                    "put_gamma": float(row[1] or 0),
                    "call_delta": float(row[2] or 0),
                    "put_delta": float(row[3] or 0),
                }
            return None
        except Exception as exc:
            logger.debug(f"ML_PIPELINE: GEX fetch failed for {ticker}: {exc}")
            return None

    try:
        return await asyncio.wait_for(asyncio.to_thread(_sync), timeout=10)
    except asyncio.TimeoutError:
        logger.warning(f"ML_PIPELINE: GEX fetch timed out for {ticker}")
        return None


async def fetch_morning_stock_bars(
    api_key: str, ticker: str, date_str: str
) -> list[dict]:
    """Fetch first 15 minutes of stock OHLCV bars from Polygon REST."""
    try:
        import httpx

        # Polygon /v2/aggs/ticker/{ticker}/range/1/minute/{from}/{to}
        url = (
            f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/"
            f"{date_str}/{date_str}"
        )
        params = {
            "apiKey": api_key,
            "adjusted": "true",
            "sort": "asc",
            "limit": 15,
        }
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, params=params)
            if resp.status_code != 200:
                return []

            data = resp.json()
            results = data.get("results", [])
            bars = []
            for r in results:
                bars.append(
                    {
                        "open": float(r.get("o", 0)),
                        "high": float(r.get("h", 0)),
                        "low": float(r.get("l", 0)),
                        "close": float(r.get("c", 0)),
                        "volume": float(r.get("v", 0)),
                    }
                )
            return bars
    except Exception as exc:
        logger.debug(f"ML_PIPELINE: morning bars fetch failed for {ticker}: {exc}")
        return []


async def fetch_prev_day_stats(
    api_key: str, ticker: str, date_str: str
) -> dict | None:
    """Fetch previous day stats from Polygon for regime model features."""
    try:
        import httpx

        # Get previous close from snapshot
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/prev"
        params = {"apiKey": api_key, "adjusted": "true"}
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, params=params)
            if resp.status_code != 200:
                return None

            data = resp.json()
            results = data.get("results", [])
            if not results:
                return None

            r = results[0]
            prev_high = float(r.get("h", 0))
            prev_low = float(r.get("l", 0))
            prev_close = float(r.get("c", 0))
            prev_volume = float(r.get("v", 0))
            prev_range_pct = (
                (prev_high - prev_low) / prev_low * 100
                if prev_low > 0
                else 0
            )

            return {
                "prev_range_pct": prev_range_pct,
                "prev_volume": prev_volume,
                "prev_close": prev_close,
                "avg_3d_range": prev_range_pct,  # approximate (single day)
                "avg_prev_vol": prev_volume,     # approximate (single day)
            }
    except Exception as exc:
        logger.debug(f"ML_PIPELINE: prev day fetch failed for {ticker}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Option Data Accumulator (builds numpy arrays from live snapshots)
# ---------------------------------------------------------------------------


@dataclass
class TickerScanState:
    """Accumulated minute-by-minute option data for a single ticker-day."""

    closes: list[float] = field(default_factory=list)
    volumes: list[float] = field(default_factory=list)
    ivs: list[float] = field(default_factory=list)
    deltas: list[float] = field(default_factory=list)
    thetas: list[float] = field(default_factory=list)
    vegas: list[float] = field(default_factory=list)
    underlyings: list[float] = field(default_factory=list)
    bids: list[float] = field(default_factory=list)
    asks: list[float] = field(default_factory=list)
    bid_sizes: list[float] = field(default_factory=list)
    ask_sizes: list[float] = field(default_factory=list)
    stock_closes: list[float] = field(default_factory=list)
    stock_highs: list[float] = field(default_factory=list)
    stock_lows: list[float] = field(default_factory=list)
    strike: float = 0.0
    expiry: str = ""
    opening_price: float = 0.0
    entry_emitted: bool = False  # prevent duplicate entries for same ticker/day
    last_append_minute: int = -1  # prevent duplicate minute entries
    strike_resolved_price: float = 0.0  # underlying price when strike was resolved

    data_changed: bool = False  # set True when snapshot data is new or updated

    def append_snapshot(self, snap: dict, minute: int) -> None:
        """Append or update a single minute's option + underlying snapshot.

        If called again for the same minute (e.g. 15s scan interval within a
        1-minute window), OVERWRITES the last entry with fresh Redis data so
        all bots evaluate the latest snapshot regardless of scan phase offset.
        """
        mid = snap.get("mid", 0) or snap.get("close", 0)

        if minute == self.last_append_minute and self.closes:
            # Same minute — overwrite last entry with fresher data
            old_mid = self.closes[-1]
            self.closes[-1] = mid
            self.volumes[-1] = snap.get("volume") or 0
            self.ivs[-1] = snap.get("iv") or 0
            self.deltas[-1] = abs(snap.get("delta") or 0)
            self.thetas[-1] = snap.get("theta") or 0
            self.vegas[-1] = snap.get("vega") or 0
            self.underlyings[-1] = snap.get("underlying_price", 0)
            self.bids[-1] = snap.get("bid", 0)
            self.asks[-1] = snap.get("ask", 0)
            self.bid_sizes[-1] = snap.get("bid_size", 0)
            self.ask_sizes[-1] = snap.get("ask_size", 0)
            self.stock_closes[-1] = snap.get("underlying_price", 0)
            self.stock_highs[-1] = snap.get("underlying_high", snap.get("underlying_price", 0))
            self.stock_lows[-1] = snap.get("underlying_low", snap.get("underlying_price", 0))
            # Mark changed only if the mid price actually moved
            self.data_changed = abs(mid - old_mid) > 1e-6
            return

        if minute < self.last_append_minute:
            return  # stale minute — skip

        self.last_append_minute = minute
        self.data_changed = True

        self.closes.append(mid)
        self.volumes.append(snap.get("volume") or 0)
        self.ivs.append(snap.get("iv") or 0)
        self.deltas.append(abs(snap.get("delta") or 0))
        self.thetas.append(snap.get("theta") or 0)
        self.vegas.append(snap.get("vega") or 0)
        self.underlyings.append(snap.get("underlying_price", 0))
        self.bids.append(snap.get("bid", 0))
        self.asks.append(snap.get("ask", 0))
        self.bid_sizes.append(snap.get("bid_size", 0))
        self.ask_sizes.append(snap.get("ask_size", 0))

        # Stock bar data (underlying candle)
        self.stock_closes.append(snap.get("underlying_price", 0))
        self.stock_highs.append(snap.get("underlying_high", snap.get("underlying_price", 0)))
        self.stock_lows.append(snap.get("underlying_low", snap.get("underlying_price", 0)))

        if self.opening_price <= 0 and self.closes[-1] > 0:
            self.opening_price = self.closes[-1]

    def to_numpy(self) -> dict[str, np.ndarray]:
        """Convert accumulated lists to numpy arrays for feature computation."""
        return {
            "closes": np.array(self.closes, dtype=np.float64),
            "volumes": np.array(self.volumes, dtype=np.float64),
            "ivs": np.array(self.ivs, dtype=np.float64),
            "deltas": np.array(self.deltas, dtype=np.float64),
            "thetas": np.array(self.thetas, dtype=np.float64),
            "vegas": np.array(self.vegas, dtype=np.float64),
            "underlyings": np.array(self.underlyings, dtype=np.float64),
            "bids": np.array(self.bids, dtype=np.float64),
            "asks": np.array(self.asks, dtype=np.float64),
            "bid_sizes": np.array(self.bid_sizes, dtype=np.float64),
            "ask_sizes": np.array(self.ask_sizes, dtype=np.float64),
            "stock_closes": np.array(self.stock_closes, dtype=np.float64),
            "stock_highs": np.array(self.stock_highs, dtype=np.float64),
            "stock_lows": np.array(self.stock_lows, dtype=np.float64),
        }


# ---------------------------------------------------------------------------
# Signal Emission
# ---------------------------------------------------------------------------


async def emit_signal_to_pg(
    ticker: str,
    direction: str,
    pattern_conf: float,
    entry_conf: float | None,
    premium: float,
    strike: float,
    expiry: str,
    stop_pct: float | None,
    signal_quality: float | None,
    extra: dict | None = None,
    threshold: float = DEFAULT_PATTERN_THRESHOLD,
) -> None:
    """Fire-and-forget: emit an ML signal to PostgreSQL for trading bots.

    Uses the existing pg.emit_ml_signal() interface. Never blocks the scan loop.
    """
    try:
        from options_owl.db import postgres as pg

        if not pg.is_connected():
            logger.debug("ML_PIPELINE: PG not connected — signal not emitted")
            return

        signal_data = {
            "ticker": ticker,
            "direction": direction,
            # Map 0-1 confidence to 0-100 score. NOTE: ML scores live on a
            # different scale than Discord scores (capped at 100). Downstream,
            # ScoreGate applies settings.ML_MIN_SCORE (not MIN_SCORE) to
            # ML_SOURCING signals — the model threshold above is the real gate,
            # so there is no dead band between threshold*100 and MIN_SCORE.
            "score": int(pattern_conf * 100),
            "ml_confidence": pattern_conf,
            "ml_threshold": threshold,
            "ml_model_source": "ml_v3_pipeline",
            "ml_runner_score": signal_quality,
            "premium": premium,
            "strike": strike,
            "expiry_date": expiry,
            "indicators": {
                "pattern_conf": round(pattern_conf, 4),
                "entry_conf": round(entry_conf, 4) if entry_conf is not None else None,
                "stop_pct": round(stop_pct, 2) if stop_pct is not None else None,
                "signal_quality": (
                    round(signal_quality, 3) if signal_quality is not None else None
                ),
                **(extra or {}),
            },
            "score_breakdown": {},
            "emitted_at": datetime.now(tz=timezone.utc),
        }

        signal_id = await asyncio.wait_for(
            pg.emit_ml_signal(signal_data), timeout=10
        )
        if signal_id:
            _entry_str = f"{entry_conf:.3f}" if entry_conf is not None else "N/A"
            logger.info(
                f"ML_PIPELINE: SIGNAL EMITTED #{signal_id} {ticker} {direction} "
                f"pattern={pattern_conf:.3f} entry={_entry_str} "
                f"premium=${premium:.2f} strike=${strike:.0f}"
            )
    except asyncio.TimeoutError:
        logger.warning(f"ML_PIPELINE: PG emit timed out for {ticker}")
    except Exception as exc:
        logger.warning(f"ML_PIPELINE: PG emit failed for {ticker}: {exc}")


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------


async def run_regime_filter(
    models: MLModels, settings: MLPipelineSettings
) -> bool:
    """Run SPY regime filter at 9:45 AM ET. Returns True if trading is allowed today.

    This is the market-wide check — if SPY regime says "bad day", skip everything.
    Individual tickers are further filtered by check_ticker_regime().
    """
    if models.regime_model is None:
        logger.info("ML_PIPELINE: No regime model loaded — allowing all days")
        return True

    if settings.ML_REGIME_THRESHOLD <= 0:
        logger.info("ML_PIPELINE: Regime threshold=0 — allowing all days")
        return True

    score = await _compute_regime_score_for_ticker(
        "SPY", models, settings
    )
    if score is None:
        return True

    # Share via Redis for cross-agent coordination
    try:
        from options_owl.db import redis_client
        today = get_todays_expiry()
        skip = score < settings.ML_REGIME_THRESHOLD
        await redis_client.set_regime_score(today, score, skip)
    except Exception:
        pass

    if score < settings.ML_REGIME_THRESHOLD:
        logger.info(
            f"ML_PIPELINE: REGIME SKIP (SPY) — score={score:.3f} "
            f"< threshold={settings.ML_REGIME_THRESHOLD}. Skipping entire day."
        )
        return False

    logger.info(
        f"ML_PIPELINE: REGIME PASS (SPY) — score={score:.3f} "
        f">= threshold={settings.ML_REGIME_THRESHOLD}. Trading allowed."
    )
    return True


# Cache per-ticker regime scores for the day (computed once at 9:45)
_ticker_regime_cache: dict[str, float] = {}


async def check_ticker_regime(
    ticker: str, models: MLModels, settings: MLPipelineSettings
) -> bool:
    """Per-ticker regime check. Returns True if this ticker is allowed today.

    Unlike the global SPY filter (which skips the entire day), this checks
    whether each individual ticker looks like it will trend. A ticker can
    be in a good regime even if SPY is marginal, or vice versa.

    Uses a slightly lower threshold (0.15) since per-ticker predictions
    are noisier than SPY aggregate.
    """
    if models.regime_model is None:
        return True

    # Use cached score if already computed today
    cache_key = f"{ticker}:{get_todays_expiry()}"
    if cache_key in _ticker_regime_cache:
        score = _ticker_regime_cache[cache_key]
    else:
        score = await _compute_regime_score_for_ticker(ticker, models, settings)
        if score is None:
            return True
        _ticker_regime_cache[cache_key] = score
        logger.debug(f"ML_PIPELINE: {ticker} regime score={score:.3f}")

    # Per-ticker threshold is lower — more permissive since we still have
    # the pattern/entry models to filter bad setups
    per_ticker_threshold = max(settings.ML_REGIME_THRESHOLD - 0.04, 0.10)
    if score < per_ticker_threshold:
        logger.info(
            f"ML_PIPELINE: {ticker} REGIME SKIP — score={score:.3f} "
            f"< {per_ticker_threshold:.2f}"
        )
        return False
    return True


async def _compute_regime_score_for_ticker(
    ticker: str, models: MLModels, settings: MLPipelineSettings
) -> float | None:
    """Compute regime score for a single ticker.

    Features come from the SHARED feature module via compute_regime_features
    (Postgres-backed: stock_candles + gex_ticks). The 15s timeout protects the
    monitor loop from a hung DB read (CLAUDE.md external-I/O rule).
    """
    now_et = datetime.now(tz=ET)
    try:
        feat = await asyncio.wait_for(
            compute_regime_features(ticker, now_et, models.regime_features),
            timeout=15,
        )
    except asyncio.TimeoutError:
        logger.warning(f"ML_PIPELINE: regime feature build timed out (15s) for {ticker}")
        return None
    except Exception as exc:
        logger.warning(f"ML_PIPELINE: regime feature build failed for {ticker}: {exc}")
        return None

    if feat is None:
        logger.warning(
            f"ML_PIPELINE: Insufficient morning data for {ticker} regime — allowing"
        )
        return None

    X = np.array(
        [[feat.get(f, 0) for f in models.regime_features]], dtype=np.float32
    )
    return float(models.regime_model.predict(X)[0])


def resolve_pattern_threshold(settings: MLPipelineSettings, models: MLModels) -> float:
    """Runtime pattern threshold: explicit env override, else the model's own
    validated best_threshold from its meta, else the legacy default."""
    if settings.ML_PATTERN_THRESHOLD > 0:
        return settings.ML_PATTERN_THRESHOLD
    return float(models.pattern_meta.get("best_threshold", DEFAULT_PATTERN_THRESHOLD))


async def scan_ticker_minute(
    ticker: str,
    minute: int,
    state: TickerScanState,
    models: MLModels,
    settings: MLPipelineSettings,
) -> bool:
    """Run ML models for a single ticker at a single minute.

    Returns True if a signal was emitted.
    """
    if state.entry_emitted:
        return False

    arrays = state.to_numpy()
    idx = len(state.closes) - 1  # current index

    if idx < 5:
        return False

    # Step 1: Pattern model (sourcing)
    feat = compute_pattern_features(
        arrays["closes"],
        arrays["volumes"],
        arrays["ivs"],
        arrays["deltas"],
        arrays["thetas"],
        arrays["underlyings"],
        arrays["bids"],
        arrays["asks"],
        idx,
        state.opening_price,
    )
    if feat is None:
        return False

    from options_owl.sourcing.scoring.ml_gates.signal_model import _warn_missing_features

    _warn_missing_features("pattern_entry", models.pattern_features, feat)
    X_pattern = np.array(
        [[feat.get(f, 0) for f in models.pattern_features]], dtype=np.float32
    )
    pattern_conf = float(models.pattern_model.predict(X_pattern)[0])

    pattern_threshold = resolve_pattern_threshold(settings, models)
    if pattern_conf < pattern_threshold:
        return False

    logger.info(
        f"ML_PIPELINE: {ticker} min={minute} PATTERN PASS "
        f"conf={pattern_conf:.3f} >= {pattern_threshold}"
    )

    # Step 2: Entry timing model (quality gate)
    entry_conf = None
    et_feat = None
    if models.entry_model and models.entry_features:
        et_feat = compute_entry_timing_features(
            arrays["closes"],
            arrays["volumes"],
            arrays["bids"],
            arrays["asks"],
            arrays["bid_sizes"],
            arrays["ask_sizes"],
            arrays["ivs"],
            arrays["deltas"],
            arrays["thetas"],
            arrays["vegas"],
            arrays["underlyings"],
            arrays["stock_closes"],
            arrays["stock_highs"],
            arrays["stock_lows"],
            idx,
            models.entry_features,
        )
        if et_feat is not None:
            X_entry = np.array(
                [[et_feat.get(f, 0) for f in models.entry_features]],
                dtype=np.float32,
            )
            entry_conf = float(models.entry_model.predict(X_entry)[0])
            if entry_conf < settings.ML_ENTRY_THRESHOLD:
                logger.info(
                    f"ML_PIPELINE: {ticker} min={minute} ENTRY BLOCKED "
                    f"conf={entry_conf:.3f} < {settings.ML_ENTRY_THRESHOLD}"
                )
                return False

    # Step 3: Entry gates
    current_ask = state.asks[-1] if state.asks else 0
    current_bid = state.bids[-1] if state.bids else 0
    current_mid = state.closes[-1] if state.closes else 0
    premium = current_ask if current_ask > 0 else current_mid

    if premium <= 0:
        return False

    # Premium floor
    if premium < PREMIUM_FLOOR:
        logger.debug(f"ML_PIPELINE: {ticker} premium ${premium:.2f} < floor ${PREMIUM_FLOOR}")
        return False

    # Premium cap
    if premium > PREMIUM_CAP:
        logger.debug(f"ML_PIPELINE: {ticker} premium ${premium:.2f} > cap ${PREMIUM_CAP}")
        return False

    # Spread gate
    if current_bid > 0 and premium > 0:
        spread_pct = (premium - current_bid) / premium * 100
        if spread_pct > SPREAD_GATE_PCT:
            logger.debug(
                f"ML_PIPELINE: {ticker} spread {spread_pct:.1f}% > gate {SPREAD_GATE_PCT}%"
            )
            return False

    # Step 4: Stop calibration (optional)
    stop_pct = None
    if models.stop_model and et_feat is not None:
        X_stop = np.array(
            [[et_feat.get(f, 0) for f in models.stop_features]], dtype=np.float32
        )
        stop_pct = float(models.stop_model.predict(X_stop)[0])
        # Clamp to reasonable range
        stop_pct = max(15.0, min(55.0, stop_pct))

    # Step 5: Signal quality ranking (optional, for downstream sorting)
    signal_quality = None
    if models.signal_model and models.signal_features:
        sq_feat = compute_entry_timing_features(
            arrays["closes"],
            arrays["volumes"],
            arrays["bids"],
            arrays["asks"],
            arrays["bid_sizes"],
            arrays["ask_sizes"],
            arrays["ivs"],
            arrays["deltas"],
            arrays["thetas"],
            arrays["vegas"],
            arrays["underlyings"],
            arrays["stock_closes"],
            arrays["stock_highs"],
            arrays["stock_lows"],
            idx,
            models.signal_features,
        )
        if sq_feat is not None:
            X_sq = np.array(
                [[sq_feat.get(f, 0) for f in models.signal_features]],
                dtype=np.float32,
            )
            signal_quality = float(models.signal_model.predict(X_sq)[0])

    # Determine direction from underlying price action
    # If underlying is declining from open, signal is PUT; otherwise CALL
    direction = "CALL"
    if len(state.underlyings) >= 3:
        current_underlying = state.underlyings[-1]
        open_underlying = state.underlyings[0]
        if current_underlying > 0 and open_underlying > 0:
            move_pct = (current_underlying - open_underlying) / open_underlying * 100
            if move_pct < -0.15:  # underlying down 0.15%+ from open → PUT
                direction = "PUT"

    # V2 direction-specific model validation (if available)
    # Use per-ticker PUT/CALL model to confirm the direction signal
    v2_models = models.put_models if direction == "PUT" else models.call_models
    if ticker in v2_models:
        v2_model, v2_features, v2_threshold = v2_models[ticker]
        # Build V2 feature vector from available data
        v2_feat = _build_v2_signal_features(
            state, idx, v2_features, direction,
        )
        if v2_feat is not None:
            X_v2 = np.array(
                [[v2_feat.get(f, 0) for f in v2_features]], dtype=np.float32
            )
            v2_conf = float(v2_model.predict(X_v2)[0])
            if v2_conf < v2_threshold * 0.8:  # 20% below optimal threshold = weak signal
                logger.info(
                    f"ML_PIPELINE: {ticker} V2 {direction} model BLOCKED — "
                    f"conf={v2_conf:.3f} < {v2_threshold * 0.8:.3f}"
                )
                return False
            logger.info(
                f"ML_PIPELINE: {ticker} V2 {direction} model CONFIRMED — "
                f"conf={v2_conf:.3f} >= {v2_threshold * 0.8:.3f}"
            )

    # Emit signal
    state.entry_emitted = True
    asyncio.create_task(
        emit_signal_to_pg(
            ticker=ticker,
            direction=direction,
            pattern_conf=pattern_conf,
            entry_conf=entry_conf,
            premium=premium,
            strike=state.strike,
            expiry=state.expiry,
            stop_pct=stop_pct,
            signal_quality=signal_quality,
            threshold=pattern_threshold,
        )
    )
    return True


async def scan_all_tickers(
    models: MLModels,
    settings: MLPipelineSettings,
    ticker_states: dict[str, TickerScanState],
    candle_buffer: CandleBuffer,
    minute: int,
) -> int:
    """Scan all tickers for the current minute. Returns count of signals emitted."""
    active_tickers = [t for t in TICKERS if t not in EXCLUDED_TICKERS]
    expiry = get_todays_expiry()
    signals_emitted = 0

    for ticker in active_tickers:
        try:
            # Initialize state for this ticker if needed
            if ticker not in ticker_states:
                ticker_states[ticker] = TickerScanState(expiry=expiry)

            state = ticker_states[ticker]

            if state.entry_emitted:
                continue

            # Per-ticker regime check (runs once per ticker per day at minute 15+)
            if minute >= 15 and models.regime_model is not None:
                ticker_ok = await check_ticker_regime(ticker, models, settings)
                if not ticker_ok:
                    continue

            # Fetch live option snapshot for ATM strike
            underlying_price = await asyncio.wait_for(
                fetch_live_underlying_price(settings.POLYGON_API_KEY, ticker),
                timeout=10,
            )
            if not underlying_price or underlying_price <= 0:
                logger.debug(f"ML_PIPELINE: {ticker} no underlying price")
                continue

            # Resolve ATM strike on first scan
            if state.strike <= 0:
                chain = await fetch_live_option_chain(
                    settings.POLYGON_API_KEY, ticker, expiry
                )
                if not chain:
                    # Try next business day expiry
                    tomorrow = (
                        datetime.now(tz=ET) + timedelta(days=1)
                    ).strftime("%Y-%m-%d")
                    chain = await fetch_live_option_chain(
                        settings.POLYGON_API_KEY, ticker, tomorrow
                    )
                    if chain:
                        expiry_used = tomorrow
                    else:
                        logger.debug(f"ML_PIPELINE: {ticker} no option chain")
                        continue
                else:
                    expiry_used = expiry

                atm = find_atm_strike(chain, underlying_price)
                if not atm:
                    continue
                state.strike = atm.get("strike", 0)
                state.expiry = expiry_used
                logger.info(
                    f"ML_PIPELINE: {ticker} ATM strike=${state.strike:.0f} "
                    f"expiry={state.expiry}"
                )

            # Fetch full option snapshot with greeks
            snap = await asyncio.wait_for(
                fetch_option_snapshot_data(
                    settings.POLYGON_API_KEY, ticker, state.strike, state.expiry
                ),
                timeout=10,
            )
            if not snap:
                # Append a minimal snapshot so we don't skip the minute entirely
                snap = {
                    "mid": 0,
                    "bid": 0,
                    "ask": 0,
                    "iv": 0,
                    "delta": 0,
                    "theta": 0,
                    "vega": 0,
                    "volume": 0,
                    "underlying_price": underlying_price,
                    "bid_size": 0,
                    "ask_size": 0,
                }
            else:
                # Fill in underlying price if missing from snapshot
                if snap.get("underlying_price", 0) <= 0:
                    snap["underlying_price"] = underlying_price

            state.append_snapshot(snap, minute)

            # Run ML models
            emitted = await scan_ticker_minute(
                ticker, minute, state, models, settings
            )
            if emitted:
                signals_emitted += 1

        except asyncio.TimeoutError:
            logger.warning(f"ML_PIPELINE: {ticker} scan timed out at minute {minute}")
        except Exception as exc:
            logger.exception(f"ML_PIPELINE: {ticker} scan error at minute {minute}: {exc}")

    return signals_emitted


# ---------------------------------------------------------------------------
# Main Loop
# ---------------------------------------------------------------------------


def _is_market_open() -> bool:
    """Check if US equity market is currently open (weekday 9:30-4:00 ET)."""
    now = datetime.now(tz=ET)
    if now.weekday() >= 5:
        return False
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= now <= market_close


def _minutes_since_open() -> int:
    """Minutes since 9:30 AM ET. Negative if before open."""
    now = datetime.now(tz=ET)
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    return int((now - market_open).total_seconds() / 60)


async def run_ml_pipeline(settings: MLPipelineSettings | None = None) -> None:
    """Main entry point for the production ML pipeline.

    1. Load models at startup
    2. Wait for market open
    3. Run regime filter at 9:45 AM ET
    4. Scan every minute 9:35-11:00 ET
    5. Reset state at EOD, loop for next day
    """
    if settings is None:
        settings = load_settings_from_env()

    # Load models
    models = load_models()

    pattern_threshold = resolve_pattern_threshold(settings, models)
    threshold_source = "env" if settings.ML_PATTERN_THRESHOLD > 0 else "model_meta"
    logger.info(
        f"ML_PIPELINE: Starting | pattern_t={pattern_threshold} ({threshold_source}) "
        f"entry_t={settings.ML_ENTRY_THRESHOLD} regime_t={settings.ML_REGIME_THRESHOLD} "
        f"scan={settings.ML_SCAN_START_MIN}-{settings.ML_SCAN_END_MIN}min"
    )

    # Initialize PG connection pool
    try:
        from options_owl.db import postgres as pg

        await pg.init_pool(settings.DATABASE_URL)
        logger.info("ML_PIPELINE: PostgreSQL connection pool initialized")
    except Exception as exc:
        logger.warning(f"ML_PIPELINE: PG init failed — signals will not be emitted: {exc}")

    candle_buffer = CandleBuffer()

    while True:
        # Wait for market open
        while not _is_market_open():
            await asyncio.sleep(30)

        today = get_todays_expiry()
        logger.info(f"ML_PIPELINE: Market open — starting scan day {today}")

        # Reset per-day state
        ticker_states: dict[str, TickerScanState] = {}
        regime_checked = False
        regime_allowed = True
        _ticker_regime_cache.clear()

        try:
            while _is_market_open():
                minute = _minutes_since_open()

                # Regime filter at minute 15 (9:45 AM ET)
                if not regime_checked and minute >= 15:
                    regime_checked = True
                    regime_allowed = await run_regime_filter(models, settings)
                    if not regime_allowed:
                        # Skip entire day — wait for market close
                        logger.info("ML_PIPELINE: Day skipped by regime filter. Waiting for close.")
                        while _is_market_open():
                            await asyncio.sleep(60)
                        break

                # Only scan during the configured window
                if minute < settings.ML_SCAN_START_MIN:
                    await asyncio.sleep(10)
                    continue

                if minute > settings.ML_SCAN_END_MIN:
                    logger.info(
                        f"ML_PIPELINE: Scan window closed (minute {minute} > {settings.ML_SCAN_END_MIN}). "
                        f"Waiting for EOD."
                    )
                    # Done scanning for today — wait for market close
                    while _is_market_open():
                        await asyncio.sleep(60)
                    break

                # Scan all tickers
                scan_start = time.monotonic()
                signals = await scan_all_tickers(
                    models, settings, ticker_states, candle_buffer, minute
                )
                scan_elapsed = time.monotonic() - scan_start

                if signals > 0:
                    logger.info(
                        f"ML_PIPELINE: Scan minute {minute} complete — "
                        f"{signals} signals emitted ({scan_elapsed:.1f}s)"
                    )
                else:
                    logger.debug(
                        f"ML_PIPELINE: Scan minute {minute} — no signals ({scan_elapsed:.1f}s)"
                    )

                # Sleep until next minute (aligned to clock)
                now = datetime.now(tz=ET)
                next_minute = (now + timedelta(minutes=1)).replace(second=5, microsecond=0)
                sleep_secs = max(0, (next_minute - now).total_seconds())
                await asyncio.sleep(sleep_secs)

        except Exception as exc:
            logger.exception(f"ML_PIPELINE: Unhandled error in scan loop: {exc}")

        # EOD summary
        total_signals = sum(1 for s in ticker_states.values() if s.entry_emitted)
        total_scanned = sum(len(s.closes) for s in ticker_states.values())
        logger.info(
            f"ML_PIPELINE: EOD {today} — {total_signals} signals emitted, "
            f"{total_scanned} total snapshots collected"
        )

        # Wait for next market day
        logger.info("ML_PIPELINE: Market closed. Sleeping until next open.")
        await asyncio.sleep(60)  # brief pause before re-entering wait loop


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the ML pipeline as a standalone process."""
    asyncio.run(run_ml_pipeline())


if __name__ == "__main__":
    main()
