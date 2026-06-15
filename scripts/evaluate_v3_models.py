"""Evaluate V3 ML models — isolation tests + combo strategies + full reports.

Tests each V3 model against pre-computed sweep candidates (from sweep_combined_scoring.py)
to measure how much each model improves end-to-end trading results.

Models evaluated:
  1. entry_timing   — gate: only enter when model says "near the low"
  2. exit_timing    — replace FSM exit: model says HOLD vs SELL each minute
  3. regime         — pre-market gate: skip chop days entirely
  4. ticker_select  — pre-market gate: only trade tickers model picks
  5. stop_calibrate — modify stops: use model-predicted stop width instead of fixed
  6. signal_quality — gate: only enter when predicted magnitude is high enough

Each test runs against the SAME candidate pool with the SAME portfolio simulation,
so results are directly comparable.

Usage:
    python scripts/evaluate_v3_models.py                    # full evaluation + report
    python scripts/evaluate_v3_models.py --model regime     # single model
    python scripts/evaluate_v3_models.py --combos           # combo strategies only
    python scripts/evaluate_v3_models.py --report-only      # regenerate report from cached results
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import lightgbm as lgb
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from options_owl.risk.exit_v5.config import V5Config, get_ticker_config
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState
from options_owl.sourcing.scoring.ml_gates.signal_model import (
    compute_option_features_from_live,
)

ET = ZoneInfo("America/New_York")

THETADATA_DB = str(PROJECT_DIR / "journal" / "thetadata_options.db")
MODEL_DIR = PROJECT_DIR / "journal" / "models" / "ml_v3"
CANDIDATES_CACHE = PROJECT_DIR / "journal" / "sweep_candidates.json"
RESULTS_DIR = PROJECT_DIR / "journal" / "v3_eval_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = [
    "SPY", "QQQ", "NVDA", "TSLA", "META", "AAPL", "AMZN",
    "GOOGL", "MSFT", "AMD", "MSTR", "PLTR", "AVGO", "IWM",
]

# Portfolio settings (match sweep_combined_scoring.py)
PORTFOLIO_START = 23_000
MAX_CONCURRENT = 4
MAX_POSITION_PCT = 0.15
MAX_RISK_PCT = 0.75
GFV_BUFFER_PCT = 15.0
DAILY_LOSS_CB_PCT = 15.0
MAX_SAME_DIRECTION = 3

# V6 settings for FSM re-simulation
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
    ENABLE_V6_SIDEWAYS_SCALP=True,
    ENABLE_SCALP_TARGET=True,
    SCALP_TARGET_PCT=25.0,
    SCALP_RUNNER_CONFIRM_PCT=40.0,
)

# Stop configs for stop_calibrate model comparison
STOP_CONFIGS = {
    20.0: "ultra_tight",
    30.0: "tight",
    40.0: "moderate",
    50.0: "wide",
    65.0: "wide",   # 65% maps to "wide" bucket
}


# ── Model Loaders ──────────────────────────────────────────────────────────


def load_v3_model(name: str) -> tuple[lgb.Booster | None, dict | None]:
    """Load a V3 model + its metadata. Returns (model, meta) or (None, None)."""
    model_path = MODEL_DIR / f"{name}.txt"
    meta_path = MODEL_DIR / f"{name}_meta.json"

    if not model_path.exists():
        # Try alternate names
        alt_names = {
            "regime": "regime_classifier",
            "ticker_select": "ticker_selection",
            "stop_calibrate": "stop_calibration",
        }
        alt = alt_names.get(name)
        if alt:
            model_path = MODEL_DIR / f"{alt}.txt"
            meta_path = MODEL_DIR / f"{alt}_meta.json"

    if not model_path.exists():
        return None, None

    model = lgb.Booster(model_file=str(model_path))
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    return model, meta


def load_all_v3_models() -> dict[str, tuple[lgb.Booster | None, dict | None]]:
    """Load all available V3 models."""
    models = {}
    for name in ["entry_timing", "exit_timing", "regime", "ticker_select",
                  "stop_calibrate", "signal_quality"]:
        model, meta = load_v3_model(name)
        if model is not None:
            models[name] = (model, meta)
            print(f"  Loaded {name}: {meta.get('auc', meta.get('mae', 'N/A'))}")
        else:
            print(f"  {name}: NOT FOUND (skipping)")
    return models


# ── Candidate Loading ──────────────────────────────────────────────────────


def load_candidates() -> list[dict]:
    """Load pre-computed sweep candidates."""
    if not CANDIDATES_CACHE.exists():
        print(f"ERROR: No candidates cache at {CANDIDATES_CACHE}")
        print("Run: python scripts/sweep_combined_scoring.py --phase1-only")
        sys.exit(1)

    with open(CANDIDATES_CACHE) as f:
        candidates = json.load(f)

    print(f"  Loaded {len(candidates)} candidates from cache")
    return candidates


# ── Portfolio Simulation ───────────────────────────────────────────────────


@dataclass
class SimResult:
    """Result of a portfolio simulation."""
    name: str
    description: str
    trades: int = 0
    wins: int = 0
    total_pnl: float = 0.0
    final_portfolio: float = PORTFOLIO_START
    profit_factor: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe: float = 0.0
    daily_pnls: dict = field(default_factory=dict)
    per_ticker: dict = field(default_factory=dict)
    blocked_by_model: int = 0
    passed_model: int = 0
    trade_details: list = field(default_factory=list)


def simulate_portfolio(candidates: list[dict], name: str, description: str,
                       stop_config_name: str = "wide",
                       filter_fn=None,
                       stop_override_fn=None) -> SimResult:
    """Run portfolio simulation with optional model-based filter/override.

    Args:
        candidates: Pre-computed candidate trades
        name: Strategy name
        description: Strategy description
        stop_config_name: Which pre-computed stop config to use for P&L
        filter_fn: Optional function(candidate) -> bool. If returns False, skip trade.
        stop_override_fn: Optional function(candidate) -> str. Returns stop config name override.
    """
    res = SimResult(name=name, description=description)
    portfolio = PORTFOLIO_START
    peak_portfolio = portfolio
    max_dd = 0.0
    pnl_list = []

    # Daily state
    open_today = {}
    open_dirs = {}
    daily_spent = {}
    daily_realized = {}
    daily_cb = {}

    for c in candidates:
        day = c["day"]
        ticker = c["ticker"]
        session = c["session"]
        direction = c["direction"]

        # Init daily state
        if day not in open_today:
            open_today[day] = []
            open_dirs[day] = []
            daily_spent[day] = 0.0
            daily_realized[day] = 0.0
            daily_cb[day] = False

        # Circuit breaker
        if daily_cb[day]:
            continue

        # Standard gates (afternoon veto, spread)
        if session == "early_afternoon":
            continue
        spread = c.get("spread_pct")
        if spread is not None and spread > 30:
            continue

        # Concurrent check
        if len(open_today[day]) >= MAX_CONCURRENT:
            continue
        if ticker in open_today[day]:
            continue

        # Correlation guard
        same_dir = sum(1 for d in open_dirs[day] if d == direction)
        if same_dir >= MAX_SAME_DIRECTION:
            continue

        # Model-based filter
        if filter_fn is not None:
            if not filter_fn(c):
                res.blocked_by_model += 1
                continue
            res.passed_model += 1

        # Determine stop config
        active_stop = stop_config_name
        if stop_override_fn is not None:
            active_stop = stop_override_fn(c)

        # Position sizing (confidence-weighted)
        entry_premium = c["entry_premium"]
        deployable = portfolio * MAX_RISK_PCT
        per_slot = deployable / MAX_CONCURRENT
        position_cap = portfolio * MAX_POSITION_PCT
        cost_per = entry_premium * 100

        sod_balance = portfolio
        gfv_limit = sod_balance * (1 - GFV_BUFFER_PCT / 100)
        gfv_remaining = gfv_limit - daily_spent[day]
        if gfv_remaining < cost_per:
            continue

        conf = c.get("ml_confidence", 0.8)
        if conf >= 0.90:
            mult = 0.95
        elif conf >= 0.80:
            mult = 0.60
        else:
            mult = 1.00

        scaled = per_slot * mult
        raw_ct = int(scaled / cost_per) if cost_per > 0 else 1
        cap_ct = int(position_cap / cost_per) if cost_per > 0 else 1
        gfv_ct = int(gfv_remaining / cost_per) if cost_per > 0 else 1
        contracts = max(1, min(raw_ct, cap_ct, gfv_ct))

        trade_cost = contracts * cost_per
        daily_spent[day] += trade_cost

        # P&L from pre-computed results
        stop_data = c.get("stop_results", {}).get(active_stop)
        if stop_data:
            pnl_pc = stop_data["pnl_per_contract"]
        else:
            pnl_pc = c["pnl_per_contract"]
        trade_pnl = pnl_pc * contracts
        portfolio += trade_pnl
        pnl_list.append(trade_pnl)

        open_today[day].append(ticker)
        open_dirs[day].append(direction)

        is_win = trade_pnl > 0
        res.trades += 1
        if is_win:
            res.wins += 1

        if day not in res.daily_pnls:
            res.daily_pnls[day] = 0
        res.daily_pnls[day] += trade_pnl

        if ticker not in res.per_ticker:
            res.per_ticker[ticker] = {"trades": 0, "wins": 0, "pnl": 0.0}
        res.per_ticker[ticker]["trades"] += 1
        if is_win:
            res.per_ticker[ticker]["wins"] += 1
        res.per_ticker[ticker]["pnl"] += trade_pnl

        res.trade_details.append({
            "day": day, "ticker": ticker, "direction": direction,
            "entry_premium": entry_premium, "contracts": contracts,
            "pnl": round(trade_pnl, 2), "stop": active_stop,
        })

        # Circuit breaker
        daily_realized[day] += trade_pnl
        if daily_realized[day] < 0:
            loss_pct = abs(daily_realized[day]) / sod_balance * 100
            if loss_pct >= DAILY_LOSS_CB_PCT:
                daily_cb[day] = True

        # Drawdown tracking
        if portfolio > peak_portfolio:
            peak_portfolio = portfolio
        dd = (peak_portfolio - portfolio) / peak_portfolio * 100
        if dd > max_dd:
            max_dd = dd

    # Compute summary stats
    res.total_pnl = portfolio - PORTFOLIO_START
    res.final_portfolio = portfolio
    res.max_drawdown_pct = max_dd
    res.win_rate = res.wins / res.trades * 100 if res.trades > 0 else 0

    wins_list = [p for p in pnl_list if p > 0]
    losses_list = [p for p in pnl_list if p <= 0]
    res.avg_win = np.mean(wins_list) if wins_list else 0
    res.avg_loss = np.mean(losses_list) if losses_list else 0
    gross_profit = sum(wins_list)
    gross_loss = abs(sum(losses_list))
    res.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    daily_returns = list(res.daily_pnls.values())
    if len(daily_returns) > 1 and np.std(daily_returns) > 0:
        res.sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)

    return res


# ── Model-Specific Filters ────────────────────────────────────────────────


def make_entry_timing_filter(model, meta, threshold=0.5):
    """Filter: only enter when entry_timing model says we're near the low."""
    features = meta.get("features", [])

    def filter_fn(c):
        # Build features from candidate data
        feat_dict = _candidate_to_features(c, features)
        if feat_dict is None:
            return True  # pass through if we can't compute features
        X = np.array([[feat_dict.get(f, 0) for f in features]], dtype=np.float32)
        pred = model.predict(X)[0]
        return pred >= threshold

    return filter_fn


def make_regime_filter(model, meta, threshold=0.5):
    """Filter: skip chop days (regime model predicts 'trending' < threshold)."""
    features = meta.get("features", [])

    # Cache predictions per day (regime is daily, not per-trade)
    _cache = {}

    def filter_fn(c):
        day = c["day"]
        if day not in _cache:
            feat_dict = _candidate_to_regime_features(c, features)
            if feat_dict is None:
                _cache[day] = True  # pass through
            else:
                X = np.array([[feat_dict.get(f, 0) for f in features]], dtype=np.float32)
                pred = model.predict(X)[0]
                _cache[day] = pred >= threshold
        return _cache[day]

    return filter_fn


def make_ticker_select_filter(model, meta, threshold=0.5):
    """Filter: only trade tickers the model says will be profitable today."""
    features = meta.get("features", [])

    _cache = {}

    def filter_fn(c):
        key = (c["day"], c["ticker"])
        if key not in _cache:
            feat_dict = _candidate_to_ticker_features(c, features)
            if feat_dict is None:
                _cache[key] = True
            else:
                X = np.array([[feat_dict.get(f, 0) for f in features]], dtype=np.float32)
                pred = model.predict(X)[0]
                _cache[key] = pred >= threshold
        return _cache[key]

    return filter_fn


def make_signal_quality_filter(model, meta, min_magnitude=20.0):
    """Filter: only enter when predicted peak gain magnitude >= min_magnitude%."""
    features = meta.get("features", [])

    def filter_fn(c):
        feat_dict = _candidate_to_features(c, features)
        if feat_dict is None:
            return True
        X = np.array([[feat_dict.get(f, 0) for f in features]], dtype=np.float32)
        pred_magnitude = model.predict(X)[0]
        return pred_magnitude >= min_magnitude

    return filter_fn


def make_stop_calibrate_override(model, meta):
    """Override: use model-predicted stop width instead of fixed."""
    features = meta.get("features", [])

    def override_fn(c):
        feat_dict = _candidate_to_features(c, features)
        if feat_dict is None:
            return "wide"  # fallback to default
        X = np.array([[feat_dict.get(f, 0) for f in features]], dtype=np.float32)
        pred_stop = model.predict(X)[0]
        # Map predicted stop % to nearest config bucket
        if pred_stop <= 25:
            return "ultra_tight"
        elif pred_stop <= 35:
            return "tight"
        elif pred_stop <= 45:
            return "moderate"
        else:
            return "wide"

    return override_fn


# ── Feature Extraction from Candidates ──────────────────────────────────


def _candidate_to_features(c: dict, feature_names: list[str]) -> dict | None:
    """Extract features from a candidate dict matching model's feature names."""
    f = {}

    # Map candidate fields to common feature names
    f["minutes_since_open"] = c.get("minutes_since_open", 0)
    f["is_call"] = 1 if c.get("direction", "call") == "call" else 0
    f["ticker_idx"] = TICKERS.index(c["ticker"]) if c["ticker"] in TICKERS else 0
    f["day_of_week"] = datetime.strptime(c["day"], "%Y-%m-%d").weekday()
    f["premium"] = c.get("entry_premium", 0)
    f["iv"] = 0  # not in candidates
    f["delta"] = 0
    f["theta"] = 0
    f["vega"] = 0
    f["bid"] = 0
    f["ask"] = c.get("entry_premium", 0)
    f["spread_pct"] = c.get("spread_pct", 0) or 0
    f["volume"] = 0
    f["underlying_price"] = 0
    f["ml_confidence"] = c.get("ml_confidence", 0)
    f["tech_score"] = c.get("tech_score", 0)
    f["premium_hist_len"] = c.get("premium_hist_len", 0)
    f["dte"] = c.get("dte", 0)

    # Indicator-based features
    f["volume_ratio"] = c.get("volume_ratio", 1.0)
    f["atr14"] = c.get("atr14", 0)
    f["adx"] = c.get("adx", 0)
    f["momentum_5m_pct"] = c.get("momentum_5m_pct", 0)

    # Fill any remaining features with 0
    for name in feature_names:
        if name not in f:
            f[name] = 0

    return f


def _candidate_to_regime_features(c: dict, feature_names: list[str]) -> dict | None:
    """Extract regime features from a candidate (daily-level)."""
    f = {}
    f["day_of_week"] = datetime.strptime(c["day"], "%Y-%m-%d").weekday()
    f["ticker_idx"] = TICKERS.index(c["ticker"]) if c["ticker"] in TICKERS else 0

    # These features need DB lookup — use cached indicator data from candidates
    f["atr14"] = c.get("atr14", 0)
    f["adx"] = c.get("adx", 0)
    f["volume_ratio"] = c.get("volume_ratio", 1.0)
    f["momentum_5m_pct"] = c.get("momentum_5m_pct", 0)

    for name in feature_names:
        if name not in f:
            f[name] = 0
    return f


def _candidate_to_ticker_features(c: dict, feature_names: list[str]) -> dict | None:
    """Extract ticker selection features from a candidate."""
    f = {}
    f["ticker_idx"] = TICKERS.index(c["ticker"]) if c["ticker"] in TICKERS else 0
    f["day_of_week"] = datetime.strptime(c["day"], "%Y-%m-%d").weekday()
    f["opening_premium"] = c.get("entry_premium", 0)
    f["ml_confidence"] = c.get("ml_confidence", 0)
    f["tech_score"] = c.get("tech_score", 0)
    f["spread_pct"] = c.get("spread_pct", 0) or 0
    f["atr14"] = c.get("atr14", 0)
    f["adx"] = c.get("adx", 0)
    f["volume_ratio"] = c.get("volume_ratio", 1.0)

    for name in feature_names:
        if name not in f:
            f[name] = 0
    return f


# ── Evaluation Runner ──────────────────────────────────────────────────────


def evaluate_baseline(candidates: list[dict]) -> SimResult:
    """Run baseline: best known config (0.8 tech + 0.2 ML, threshold 50, wide stops)."""
    # Apply combined scoring filter (baseline)
    def baseline_filter(c):
        score = 0.8 * c.get("tech_score", 0) + 0.2 * c.get("ml_confidence", 0) * 100
        return score >= 50

    return simulate_portfolio(
        candidates, "BASELINE",
        "Best known: 0.8 tech + 0.2 ML, threshold 50, wide stops",
        stop_config_name="wide",
        filter_fn=baseline_filter,
    )


def evaluate_ml_only(candidates: list[dict]) -> SimResult:
    """Run ML-only baseline: 1.0 ML, threshold 50, wide stops."""
    def ml_filter(c):
        score = c.get("ml_confidence", 0) * 100
        return score >= 50

    return simulate_portfolio(
        candidates, "ML_ONLY",
        "ML-only: 1.0 ML, threshold 50, wide stops",
        stop_config_name="wide",
        filter_fn=ml_filter,
    )


def evaluate_no_filter(candidates: list[dict]) -> SimResult:
    """Run unfiltered: take every candidate."""
    return simulate_portfolio(
        candidates, "UNFILTERED",
        "No filter: take every ML-approved candidate, wide stops",
        stop_config_name="wide",
    )


def evaluate_model_isolation(models: dict, candidates: list[dict]) -> list[SimResult]:
    """Test each V3 model in isolation against the baseline."""
    results = []

    # Entry timing at various thresholds
    if "entry_timing" in models:
        model, meta = models["entry_timing"]
        for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
            filt = make_entry_timing_filter(model, meta, threshold=thresh)
            r = simulate_portfolio(
                candidates,
                f"entry_timing_t{int(thresh*100)}",
                f"Entry timing gate (threshold={thresh}): only enter near predicted low",
                stop_config_name="wide",
                filter_fn=filt,
            )
            results.append(r)

    # Regime at various thresholds
    if "regime" in models:
        model, meta = models["regime"]
        for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
            filt = make_regime_filter(model, meta, threshold=thresh)
            r = simulate_portfolio(
                candidates,
                f"regime_t{int(thresh*100)}",
                f"Regime gate (threshold={thresh}): skip chop days",
                stop_config_name="wide",
                filter_fn=filt,
            )
            results.append(r)

    # Ticker selection at various thresholds
    if "ticker_select" in models:
        model, meta = models["ticker_select"]
        for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
            filt = make_ticker_select_filter(model, meta, threshold=thresh)
            r = simulate_portfolio(
                candidates,
                f"ticker_select_t{int(thresh*100)}",
                f"Ticker selection gate (threshold={thresh}): skip unprofitable tickers",
                stop_config_name="wide",
                filter_fn=filt,
            )
            results.append(r)

    # Signal quality at various magnitude thresholds
    if "signal_quality" in models:
        model, meta = models["signal_quality"]
        for min_mag in [10, 20, 30, 40, 50]:
            filt = make_signal_quality_filter(model, meta, min_magnitude=min_mag)
            r = simulate_portfolio(
                candidates,
                f"signal_quality_m{min_mag}",
                f"Signal quality gate (min_magnitude={min_mag}%): skip low-magnitude moves",
                stop_config_name="wide",
                filter_fn=filt,
            )
            results.append(r)

    # Stop calibration (replaces fixed stop with model-predicted)
    if "stop_calibrate" in models:
        model, meta = models["stop_calibrate"]
        override = make_stop_calibrate_override(model, meta)
        r = simulate_portfolio(
            candidates,
            "stop_calibrate",
            "Stop calibration: model-predicted stop width per trade",
            stop_config_name="wide",  # fallback only
            stop_override_fn=override,
        )
        results.append(r)

    return results


def evaluate_combos(models: dict, candidates: list[dict]) -> list[SimResult]:
    """Test combinations of V3 models."""
    results = []

    # Define best threshold per model (from isolation tests — use reasonable defaults)
    best_thresholds = {
        "entry_timing": 0.5,
        "regime": 0.5,
        "ticker_select": 0.5,
        "signal_quality": 30,
    }

    # Build individual filters
    filters = {}
    if "entry_timing" in models:
        m, meta = models["entry_timing"]
        filters["entry_timing"] = make_entry_timing_filter(m, meta, best_thresholds["entry_timing"])
    if "regime" in models:
        m, meta = models["regime"]
        filters["regime"] = make_regime_filter(m, meta, best_thresholds["regime"])
    if "ticker_select" in models:
        m, meta = models["ticker_select"]
        filters["ticker_select"] = make_ticker_select_filter(m, meta, best_thresholds["ticker_select"])
    if "signal_quality" in models:
        m, meta = models["signal_quality"]
        filters["signal_quality"] = make_signal_quality_filter(m, meta, best_thresholds["signal_quality"])

    stop_override = None
    if "stop_calibrate" in models:
        m, meta = models["stop_calibrate"]
        stop_override = make_stop_calibrate_override(m, meta)

    available = list(filters.keys())

    # Test all pairs
    for combo in combinations(available, 2):
        combo_name = " + ".join(combo)

        def make_combo_filter(names):
            fns = [filters[n] for n in names]
            def combined(c):
                return all(fn(c) for fn in fns)
            return combined

        r = simulate_portfolio(
            candidates,
            f"combo_{'_'.join(combo)}",
            f"Combo: {combo_name}",
            stop_config_name="wide",
            filter_fn=make_combo_filter(combo),
            stop_override_fn=stop_override if "stop_calibrate" in combo else None,
        )
        results.append(r)

    # Test all triples
    if len(available) >= 3:
        for combo in combinations(available, 3):
            combo_name = " + ".join(combo)

            def make_combo_filter(names):
                fns = [filters[n] for n in names]
                def combined(c):
                    return all(fn(c) for fn in fns)
                return combined

            r = simulate_portfolio(
                candidates,
                f"combo_{'_'.join(combo)}",
                f"Combo: {combo_name}",
                stop_config_name="wide",
                filter_fn=make_combo_filter(combo),
                stop_override_fn=stop_override,
            )
            results.append(r)

    # Kitchen sink: all filters + stop calibration
    if len(available) >= 2:
        def all_filter(c):
            return all(fn(c) for fn in filters.values())

        r = simulate_portfolio(
            candidates,
            "combo_ALL_MODELS",
            f"All models combined: {' + '.join(available)}",
            stop_config_name="wide",
            filter_fn=all_filter,
            stop_override_fn=stop_override,
        )
        results.append(r)

    # Baseline + stop calibration only (no extra filters)
    if stop_override:
        def baseline_filter(c):
            score = 0.8 * c.get("tech_score", 0) + 0.2 * c.get("ml_confidence", 0) * 100
            return score >= 50

        r = simulate_portfolio(
            candidates,
            "baseline_with_stop_cal",
            "Baseline (0.8 tech + 0.2 ML, t50) + model-predicted stops",
            stop_config_name="wide",
            filter_fn=baseline_filter,
            stop_override_fn=stop_override,
        )
        results.append(r)

    return results


# ── Report Generation ──────────────────────────────────────────────────────


def generate_report(baselines: list[SimResult], isolation: list[SimResult],
                    combos: list[SimResult], models: dict) -> str:
    """Generate a comprehensive markdown report."""
    lines = []
    lines.append("# V3 ML Model Evaluation Report")
    lines.append(f"Generated: {datetime.now(ET).strftime('%Y-%m-%d %H:%M ET')}")
    lines.append("")

    # Model inventory
    lines.append("## Model Inventory")
    lines.append("")
    lines.append("| Model | Status | Key Metric |")
    lines.append("|---|---|---|")
    for name in ["entry_timing", "exit_timing", "regime", "ticker_select",
                  "stop_calibrate", "signal_quality"]:
        if name in models:
            _, meta = models[name]
            if "auc" in meta:
                metric = f"AUC={meta['auc']:.3f}"
            elif "mae" in meta:
                metric = f"MAE={meta['mae']:.2f}"
            elif "correlation" in meta:
                metric = f"Corr={meta['correlation']:.3f}"
            else:
                metric = "trained"
            n_train = meta.get("n_train", "?")
            lines.append(f"| {name} | TRAINED | {metric} (n={n_train}) |")
        else:
            lines.append(f"| {name} | NOT FOUND | - |")
    lines.append("")

    # Baseline comparison
    lines.append("## Baseline Results")
    lines.append("")
    lines.append("| Strategy | Trades | WR% | P&L | PF | Sharpe | MaxDD |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in baselines:
        pnl_str = f"${r.total_pnl:+,.0f}"
        lines.append(
            f"| {r.name} | {r.trades} | {r.win_rate:.1f}% | {pnl_str} | "
            f"{r.profit_factor:.2f} | {r.sharpe:.2f} | {r.max_drawdown_pct:.1f}% |"
        )
    lines.append("")

    # Isolation results
    if isolation:
        lines.append("## Isolation Tests (each model alone)")
        lines.append("")

        # Group by model type
        model_groups = defaultdict(list)
        for r in isolation:
            model_type = r.name.split("_t")[0].split("_m")[0]
            model_groups[model_type].append(r)

        for model_type, group in model_groups.items():
            lines.append(f"### {model_type}")
            lines.append("")
            lines.append("| Config | Trades | Blocked | WR% | P&L | PF | Sharpe | MaxDD |")
            lines.append("|---|---|---|---|---|---|---|---|")

            # Find baseline for delta comparison
            baseline_pnl = baselines[0].total_pnl if baselines else 0

            for r in group:
                pnl_str = f"${r.total_pnl:+,.0f}"
                delta = r.total_pnl - baseline_pnl
                delta_str = f"({'+' if delta >= 0 else ''}{delta:,.0f})"
                lines.append(
                    f"| {r.name} | {r.trades} | {r.blocked_by_model} | "
                    f"{r.win_rate:.1f}% | {pnl_str} {delta_str} | "
                    f"{r.profit_factor:.2f} | {r.sharpe:.2f} | {r.max_drawdown_pct:.1f}% |"
                )
            lines.append("")

    # Combo results
    if combos:
        lines.append("## Combo Strategies")
        lines.append("")
        lines.append("| Strategy | Trades | WR% | P&L | PF | Sharpe | MaxDD |")
        lines.append("|---|---|---|---|---|---|---|")

        baseline_pnl = baselines[0].total_pnl if baselines else 0
        # Sort by P&L descending
        sorted_combos = sorted(combos, key=lambda x: x.total_pnl, reverse=True)
        for r in sorted_combos:
            pnl_str = f"${r.total_pnl:+,.0f}"
            delta = r.total_pnl - baseline_pnl
            delta_str = f"({'+' if delta >= 0 else ''}{delta:,.0f})"
            lines.append(
                f"| {r.name} | {r.trades} | {r.win_rate:.1f}% | "
                f"{pnl_str} {delta_str} | "
                f"{r.profit_factor:.2f} | {r.sharpe:.2f} | {r.max_drawdown_pct:.1f}% |"
            )
        lines.append("")

    # Best strategies summary
    all_results = baselines + isolation + combos
    if all_results:
        lines.append("## Rankings")
        lines.append("")

        lines.append("### By P&L (top 10)")
        lines.append("")
        lines.append("| # | Strategy | Trades | WR% | P&L | PF | Sharpe |")
        lines.append("|---|---|---|---|---|---|---|")
        by_pnl = sorted(all_results, key=lambda x: x.total_pnl, reverse=True)[:10]
        for i, r in enumerate(by_pnl):
            lines.append(
                f"| {i+1} | {r.name} | {r.trades} | {r.win_rate:.1f}% | "
                f"${r.total_pnl:+,.0f} | {r.profit_factor:.2f} | {r.sharpe:.2f} |"
            )
        lines.append("")

        lines.append("### By Sharpe (top 10, min 20 trades)")
        lines.append("")
        lines.append("| # | Strategy | Trades | WR% | P&L | PF | Sharpe |")
        lines.append("|---|---|---|---|---|---|---|")
        by_sharpe = sorted(
            [r for r in all_results if r.trades >= 20],
            key=lambda x: x.sharpe, reverse=True
        )[:10]
        for i, r in enumerate(by_sharpe):
            lines.append(
                f"| {i+1} | {r.name} | {r.trades} | {r.win_rate:.1f}% | "
                f"${r.total_pnl:+,.0f} | {r.profit_factor:.2f} | {r.sharpe:.2f} |"
            )
        lines.append("")

        lines.append("### By Profit Factor (top 10, min 20 trades)")
        lines.append("")
        lines.append("| # | Strategy | Trades | WR% | P&L | PF | Sharpe |")
        lines.append("|---|---|---|---|---|---|---|")
        by_pf = sorted(
            [r for r in all_results if r.trades >= 20],
            key=lambda x: x.profit_factor, reverse=True
        )[:10]
        for i, r in enumerate(by_pf):
            lines.append(
                f"| {i+1} | {r.name} | {r.trades} | {r.win_rate:.1f}% | "
                f"${r.total_pnl:+,.0f} | {r.profit_factor:.2f} | {r.sharpe:.2f} |"
            )
        lines.append("")

    # Per-ticker breakdown for best strategy
    if all_results:
        best = max(all_results, key=lambda x: x.sharpe if x.trades >= 20 else -999)
        lines.append(f"## Per-Ticker Breakdown: {best.name}")
        lines.append("")
        lines.append("| Ticker | Trades | WR% | P&L | Profitable? |")
        lines.append("|---|---|---|---|---|")
        for ticker in sorted(best.per_ticker.keys()):
            t = best.per_ticker[ticker]
            wr = t["wins"] / t["trades"] * 100 if t["trades"] > 0 else 0
            profitable = "Yes" if t["pnl"] > 0 else "**No**"
            lines.append(
                f"| {ticker} | {t['trades']} | {wr:.0f}% | ${t['pnl']:+,.0f} | {profitable} |"
            )
        lines.append("")

    # Key findings
    lines.append("## Key Findings")
    lines.append("")
    if all_results:
        baseline_r = baselines[0] if baselines else None
        best_pnl = max(all_results, key=lambda x: x.total_pnl)
        best_sharpe = max(
            [r for r in all_results if r.trades >= 20],
            key=lambda x: x.sharpe, default=None
        )

        if baseline_r:
            lines.append(f"- **Baseline P&L**: ${baseline_r.total_pnl:+,.0f} ({baseline_r.trades} trades, {baseline_r.win_rate:.1f}% WR)")
        lines.append(f"- **Best P&L**: {best_pnl.name} at ${best_pnl.total_pnl:+,.0f} ({best_pnl.trades} trades)")
        if best_sharpe:
            lines.append(f"- **Best risk-adjusted**: {best_sharpe.name} Sharpe={best_sharpe.sharpe:.2f} ({best_sharpe.trades} trades, ${best_sharpe.total_pnl:+,.0f})")

        # Which models helped vs hurt
        if isolation and baseline_r:
            helpers = []
            hurters = []
            for r in isolation:
                delta = r.total_pnl - baseline_r.total_pnl
                if delta > 0:
                    helpers.append((r.name, delta))
                elif delta < -100:
                    hurters.append((r.name, delta))

            if helpers:
                helpers.sort(key=lambda x: -x[1])
                lines.append(f"- **Models that HELPED** (vs baseline):")
                for name, delta in helpers[:5]:
                    lines.append(f"  - {name}: +${delta:,.0f}")
            if hurters:
                hurters.sort(key=lambda x: x[1])
                lines.append(f"- **Models that HURT** (vs baseline):")
                for name, delta in hurters[:5]:
                    lines.append(f"  - {name}: ${delta:,.0f}")
    lines.append("")

    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Evaluate V3 ML models")
    parser.add_argument("--model", type=str, default=None,
                        help="Evaluate single model (default: all)")
    parser.add_argument("--combos", action="store_true",
                        help="Run combo strategies only")
    parser.add_argument("--report-only", action="store_true",
                        help="Regenerate report from cached results")
    args = parser.parse_args()

    print("V3 ML Model Evaluation")
    print("=" * 70)

    # Load models
    print("\nLoading V3 models...")
    models = load_all_v3_models()
    if not models:
        print("\nNo V3 models found. Training may still be in progress.")
        print(f"Check: ls {MODEL_DIR}/")
        sys.exit(1)

    # Load candidates
    print("\nLoading candidates...")
    candidates = load_candidates()

    # Run baselines
    print("\nRunning baselines...")
    baselines = [
        evaluate_baseline(candidates),
        evaluate_ml_only(candidates),
        evaluate_no_filter(candidates),
    ]
    for r in baselines:
        print(f"  {r.name}: {r.trades} trades, {r.win_rate:.1f}% WR, "
              f"${r.total_pnl:+,.0f}, PF={r.profit_factor:.2f}, Sharpe={r.sharpe:.2f}")

    # Isolation tests
    isolation_results = []
    if not args.combos:
        print("\nRunning isolation tests...")
        if args.model:
            # Filter to single model
            filtered = {k: v for k, v in models.items() if k == args.model}
            isolation_results = evaluate_model_isolation(filtered, candidates)
        else:
            isolation_results = evaluate_model_isolation(models, candidates)

        for r in isolation_results:
            delta = r.total_pnl - baselines[0].total_pnl
            print(f"  {r.name}: {r.trades} trades, {r.win_rate:.1f}% WR, "
                  f"${r.total_pnl:+,.0f} (delta={'+' if delta >= 0 else ''}{delta:,.0f}), "
                  f"blocked={r.blocked_by_model}")

    # Combo tests
    combo_results = []
    if args.combos or not args.model:
        if len(models) >= 2:
            print("\nRunning combo strategies...")
            combo_results = evaluate_combos(models, candidates)
            for r in combo_results:
                delta = r.total_pnl - baselines[0].total_pnl
                print(f"  {r.name}: {r.trades} trades, "
                      f"${r.total_pnl:+,.0f} (delta={'+' if delta >= 0 else ''}{delta:,.0f})")

    # Generate report
    print("\nGenerating report...")
    report = generate_report(baselines, isolation_results, combo_results, models)

    report_path = RESULTS_DIR / "v3_evaluation_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report saved to {report_path}")

    # Also save raw results as JSON for later analysis
    raw_results = {
        "generated": datetime.now(ET).isoformat(),
        "n_candidates": len(candidates),
        "models_available": list(models.keys()),
        "baselines": [_result_to_dict(r) for r in baselines],
        "isolation": [_result_to_dict(r) for r in isolation_results],
        "combos": [_result_to_dict(r) for r in combo_results],
    }
    json_path = RESULTS_DIR / "v3_evaluation_raw.json"
    with open(json_path, "w") as f:
        json.dump(raw_results, f, indent=2)
    print(f"Raw results saved to {json_path}")

    # Print report to stdout
    print("\n" + "=" * 70)
    print(report)


def _result_to_dict(r: SimResult) -> dict:
    return {
        "name": r.name,
        "description": r.description,
        "trades": r.trades,
        "wins": r.wins,
        "win_rate": round(r.win_rate, 1),
        "total_pnl": round(r.total_pnl, 2),
        "final_portfolio": round(r.final_portfolio, 2),
        "profit_factor": round(r.profit_factor, 2),
        "sharpe": round(r.sharpe, 2),
        "max_drawdown_pct": round(r.max_drawdown_pct, 1),
        "avg_win": round(r.avg_win, 2),
        "avg_loss": round(r.avg_loss, 2),
        "blocked_by_model": r.blocked_by_model,
        "passed_model": r.passed_model,
        "per_ticker": r.per_ticker,
    }


if __name__ == "__main__":
    main()
