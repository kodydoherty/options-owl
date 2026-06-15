"""Gold Standard End-to-End Backtest.

Full autonomous pipeline — no Discord dependency:
  1. Pattern entry model scans every minute 9:30-11:00 (sourcing)
  2. Entry timing model filters bad entries (quality gate)
  3. V5 FSM handles all exits (category-aware, DTE-aware)
  4. Full portfolio simulation (sizing, GFV, circuit breaker, concurrent limits)

Usage:
    python scripts/backtest_gold_standard.py                    # last 60 trading days
    python scripts/backtest_gold_standard.py --days 90          # last 90 days
    python scripts/backtest_gold_standard.py --no-entry-filter  # pattern model only
    python scripts/backtest_gold_standard.py --sweep            # sweep entry thresholds
    python scripts/backtest_gold_standard.py --puts             # CALL + PUT with SPY gate
    python scripts/backtest_gold_standard.py --puts-only        # PUT only (SPY gate)
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
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import lightgbm as lgb
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from options_owl.risk.exit_v5.config import V5Config, AdaptiveTier, get_ticker_config, get_max_otm_distance
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState
from options_owl.risk.vinny_strategy import score_to_contracts
from options_owl.sourcing.features.regime_features import (
    EARLY_END as REGIME_EARLY_END,
    compute_regime_feature_vector,
    load_training_inputs,
    rth_bars_by_date_from_rows,
)

# Silence loguru INFO spam from score_to_contracts (one SIZING line per trade)
from loguru import logger as _loguru_logger
_loguru_logger.disable("options_owl.risk.vinny_strategy")

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

THETADATA_DB = str(PROJECT_DIR / "journal" / "thetadata_options.db")
MODEL_DIR = PROJECT_DIR / "journal" / "models" / "ml_v3"
REPORT_DIR = PROJECT_DIR / "journal" / "v3_eval_results"

TICKERS = [
    "SPY", "QQQ", "NVDA", "TSLA", "META", "AAPL", "AMZN",
    "GOOGL", "MSFT", "AMD", "MSTR", "PLTR", "AVGO", "IWM",
    # New tickers (added 2026-05-28) — diversification beyond tech
    "COIN", "NFLX", "JPM", "BA", "MU", "SMCI",
]

# Exclude tickers that are net losers in backtest
EXCLUDED_TICKERS = {"MSFT", "COIN", "AVGO", "MU"}  # Net losers in concurrent backtest (2026-05-30)

# Portfolio
PORTFOLIO_START = 23_000
MAX_CONCURRENT = 8       # Matches docker-compose.yml for all bots
MAX_POSITION_PCT = 0.15
MAX_RISK_PCT = 0.75
GFV_BUFFER_PCT = 15.0
DAILY_LOSS_CB_PCT = 10.0   # Prod parity: DAILY_LOSS_LIMIT_PCT=10 in .env
WEEKLY_LOSS_HALT_PCT = 20.0  # Prod parity: WEEKLY_LOSS_LIMIT_PCT=20 in .env
MAX_SAME_DIRECTION = 8   # Matched to MAX_CONCURRENT — production has no same-direction cap
PREMIUM_CAP = 6.0
SPREAD_GATE_PCT = 15.0
# V2: Per-ticker dollar OTM thresholds via get_max_otm_distance() — matches production.
# Old static MAX_OTM_DISTANCE_PCT removed; per-ticker logic in exit_v5/config.py.

MIN_PREMIUM_FLOOR = 0.20      # Reject penny premiums (lottery tickets)
MIN_SCORE = 75                # Prod parity: ScoreGate MIN_SCORE=75 (.env) / vinny floor 75
ENTRY_SLIPPAGE_PCT = 0.5      # Prod fills at ask+5% limit; realized ~ask + 50bps

# Consecutive loser circuit breaker (prod ConsecutiveLoserGate)
CONSECUTIVE_LOSER_MAX = 2     # Pause after N consecutive losses
CONSECUTIVE_LOSER_PAUSE_MIN = 15  # Minutes to pause after consecutive losses

# Production gate settings (match pipeline.py / prod .env — all ON in prod).
# Parity fix 2026-06-10: these were disabled after ablation testing, but prod
# runs them, so the backtest must too.
ENABLE_ANTI_CHASE = True          # Prod: ON
ENABLE_MOMENTUM_CONFIRM = True    # Prod: ON
ENABLE_DIRECTIONAL_REGIME = True  # Prod: ON
ENABLE_CONSECUTIVE_LOSER = True   # Prod: ON
ENABLE_CORRELATION_CAP = True     # Prod: ON
ENABLE_AFTERNOON_DANGER = True    # Keep: hard time block (1:30-3:00 PM loses money)
ENABLE_HARD_CUTOFF = True         # Keep: no entries after 3:55 PM
ENABLE_DIP_CONFIRM = True         # Match production: wait for premium dip before entering
ENABLE_PUT_BEARISH_CONFIRM = True # Prod: ON (VWAP breakdown + bearish candles + RSI<45, 2 of 3)
PUT_DIRECTION_TRIGGER_PCT = -0.15  # Prod bot_runner: PUT only when underlying < -0.15% from open

# DipConfirm settings (match production defaults)
DIP_CONFIRM_FADE_PCT = 1.0       # Minimum fade % to trigger dip-wait (below this, enter immediately)
DIP_CONFIRM_MAX_POLLS = 6        # Max minutes to wait for uptick after fade detected
DIP_CONFIRM_ALWAYS_ENTER = True   # ML behavior: enter at decision-time price if no uptick (True = ML, False = Discord)

# V6 DCA settings (prod parity: settings.py V6_DCA_* + docker-compose MAX_DCA_POSITION_PCT=10)
DCA_TICKERS = {"IWM", "SPY", "QQQ", "AMZN", "NVDA"}  # prod V6_DCA_TICKERS whitelist
DCA_MIN_MINUTES = 8.0
DCA_MAX_MINUTES = 20.0
DCA_MIN_DIP_PCT = 15.0
DCA_MAX_DIP_PCT = 35.0
DCA_MAX_UNDERLYING_AGAINST_PCT = 0.5
MAX_DCA_POSITION_PCT = 0.10       # DCA add capped at 10% of balance (MAX_DCA_POSITION_PCT=10)

# PUT trading settings (match production PutMarketDirectionGate)
ENABLE_PUTS = False               # Disabled by default; enable via --puts or --puts-only
PUTS_ONLY = False                 # Only trade PUTs (for comparison)
PUT_SPY_BEAR_THRESHOLD = -0.5     # SPY down >= 0.5% from open = bear mode (PUTs allowed)
PUT_EXCLUDED_TICKERS = {"PLTR", "AMD", "MSTR", "AVGO", "AMZN", "GOOGL"}  # Match production settings.py default
PUT_MAX_CONCURRENT = 2            # Max simultaneous PUT positions
PUT_BEAR_MAX_CONCURRENT = 4       # Max PUT positions in bear mode

# PUT config overrides for sweep testing (applied on top of PUT_SCALP_CONFIG)
PUT_CONFIG_OVERRIDES: dict = {}   # Set by --put-sweep; keys are V5Config field names

# Grace period override (None = use per-ticker defaults from V5Config)
GRACE_OVERRIDE = None

# ---------------------------------------------------------------------------
# EXIT-PARAM TUNING OVERRIDES (TASK 1 — disciplined exit sweep, 2026-06-10)
# All default to None = use the REAL V5Config / _V6_SETTINGS values unchanged,
# so the baseline is byte-for-byte identical to relaxed_C with no flags set.
# These are read at FSM-build time and applied via dataclasses.replace on the
# per-ticker V5Config (and a copy of _V6_SETTINGS for the V6 trigger params).
# We do NOT edit the live options_owl config — overrides are runtime-only.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# SIZING-SCHEME OVERRIDES (position-sizing experiment, 2026-06-11)
# Default SIZING_MODE="current" reproduces production score_to_contracts()
# exactly (incl. the non-monotonic 0.80-0.90→60% confidence tier). All other
# modes are evaluated ONLY in the offline backtest; live code is untouched.
# MULTI_DAY_CAP=2 reproduces the production paper_trader.py multi-day cap that
# the backtest previously did NOT model. Set to None / large to disable.
# ---------------------------------------------------------------------------
SIZING_MODE = "current"           # current|flat|conf_linear|conf_step|score_linear
CONF_BUDGET_MIN = 0.60            # budget multiplier at conf == CONF_REF_MIN (linear modes)
CONF_BUDGET_MAX = 1.20            # budget multiplier at conf == CONF_REF_MAX (linear modes)
CONF_REF_MIN = 0.74              # confidence that maps to CONF_BUDGET_MIN
CONF_REF_MAX = 0.95              # confidence that maps to CONF_BUDGET_MAX
SCORE_REF_MIN = 75               # score that maps to CONF_BUDGET_MIN (score_linear)
SCORE_REF_MAX = 100              # score that maps to CONF_BUDGET_MAX (score_linear)
# NOTE: production paper_trader.py applies a multi-day cap of 2 + late-0DTE cap,
# but the prior gold-standard backtest baseline did NOT model either, so to keep
# the documented relaxed_C headline reproducible the DEFAULTS here are OFF.
# The sweep turns the multi-day cap ON (2/4/6) to study production behavior.
MULTI_DAY_CAP = None              # prod runs 2; default None reproduces prior baseline
MULTI_DAY_EXPENSIVE_THRESHOLD = 5.0  # prod: premium > $5 multi-day → cap at 1 (only when cap active)
LATE_0DTE_CAP = False             # prod runs this; default off to match prior baseline

SCALP_THRESH_OVERRIDE = None      # scalp_peak_threshold_pct (default ~20)
SOFT_KEEP_OVERRIDE = None         # soft_trail_keep_pct (default ~0.60)
ADAPTIVE_MULT_OVERRIDE = None     # multiplier on adaptive trail widths (1.0 = unchanged)
THETA_MIN_OVERRIDE = None         # theta_bleed_min minutes (0DTE, default 120)
BREAKEVEN_TRIGGER_OVERRIDE = None  # V6_BREAKEVEN_TRIGGER_PCT (default 20)
SCALEOUT_TRIGGER_OVERRIDE = None  # V6_SCALEOUT_GAIN_PCT (default 20)


# ---------------------------------------------------------------------------
# Sizing-scheme dispatcher (position-sizing experiment)
# ---------------------------------------------------------------------------
def _conf_to_budget_linear(conf: float, ref_min: float, ref_max: float,
                           b_min: float, b_max: float) -> float:
    """Monotonic linear map of a quality metric to a budget multiplier, clamped."""
    if ref_max <= ref_min:
        return b_min
    frac = (conf - ref_min) / (ref_max - ref_min)
    frac = max(0.0, min(1.0, frac))
    return b_min + frac * (b_max - b_min)


def size_position(score: int, cost_per: float, balance: float, pattern_conf: float,
                  is_put: bool, dte: int, minute: int) -> int:
    """Compute contracts under the active SIZING_MODE, then apply multi-day +
    late-0DTE caps (prod paper_trader.py parity). Returns 0 if rejected.

    SIZING_MODE="current" delegates to the real production score_to_contracts()
    so the control arm is byte-for-byte production sizing. The other modes
    replicate the same budget math but swap the confidence→multiplier curve for
    a monotonic one (or flat), letting us isolate the curve's effect.
    """
    put_budget_multiplier = 0.50 if is_put else 1.0

    if SIZING_MODE == "current":
        contracts = score_to_contracts(
            score, cost_per_contract=cost_per, balance=balance,
            max_position_pct=MAX_POSITION_PCT * 100, max_concurrent=MAX_CONCURRENT,
            max_portfolio_risk_pct=MAX_RISK_PCT * 100,
            ml_confidence=float(pattern_conf), is_put=is_put,
            put_budget_multiplier=put_budget_multiplier,
        )
    else:
        # Replicate score_to_contracts budget math with a chosen multiplier.
        from options_owl.risk.vinny_strategy import (
            _MIN_ML_CONFIDENCE, _MIN_ML_CONFIDENCE_PUT, _SCORE_FLOOR,
        )
        if score < _SCORE_FLOOR:
            return 0
        min_conf = _MIN_ML_CONFIDENCE_PUT if is_put else _MIN_ML_CONFIDENCE
        if pattern_conf < min_conf:
            return 0
        if SIZING_MODE == "flat":
            score_mult = 0.85
        elif SIZING_MODE == "conf_linear":
            score_mult = _conf_to_budget_linear(
                pattern_conf, CONF_REF_MIN, CONF_REF_MAX, CONF_BUDGET_MIN, CONF_BUDGET_MAX)
        elif SIZING_MODE == "score_linear":
            score_mult = _conf_to_budget_linear(
                float(score), float(SCORE_REF_MIN), float(SCORE_REF_MAX),
                CONF_BUDGET_MIN, CONF_BUDGET_MAX)
        elif SIZING_MODE == "conf_step":
            # Monotonic 3-step: low/mid/high. Uses CONF_BUDGET_MIN/MAX as anchors.
            mid = (CONF_BUDGET_MIN + CONF_BUDGET_MAX) / 2
            if pattern_conf >= 0.90:
                score_mult = CONF_BUDGET_MAX
            elif pattern_conf >= 0.82:
                score_mult = mid
            else:
                score_mult = CONF_BUDGET_MIN
        else:
            raise ValueError(f"unknown SIZING_MODE={SIZING_MODE}")

        if cost_per <= 0 or balance <= 0:
            return 0
        total_deployable = balance * MAX_RISK_PCT
        target_per_trade = total_deployable / max(1, MAX_CONCURRENT)
        scaled_target = target_per_trade * score_mult * put_budget_multiplier
        raw_contracts = int(scaled_target / cost_per)
        max_spend = balance * MAX_POSITION_PCT
        max_by_position = int(max_spend / cost_per)
        if max_by_position == 0:
            return 0
        contracts = max(1, min(raw_contracts, max_by_position))

    if contracts <= 0:
        return 0

    # ── Multi-day cap (prod paper_trader.py parity) ──
    # Only active when MULTI_DAY_CAP is set (None = disabled = prior baseline).
    if dte > 0 and MULTI_DAY_CAP is not None and MULTI_DAY_CAP > 0:
        premium = cost_per / 100.0
        if premium > MULTI_DAY_EXPENSIVE_THRESHOLD:
            contracts = min(contracts, 1)
        else:
            contracts = min(contracts, MULTI_DAY_CAP)

    # ── Late-session 0DTE cap (prod: after 2PM ET → 1 contract) ──
    # minute is minutes after 9:30 ET open. 2:00 PM ET = 270 min after open.
    if LATE_0DTE_CAP and dte == 0 and contracts > 1 and minute >= 270:
        contracts = 1

    return contracts


def _apply_exit_overrides(tcfg, settings):
    """Apply CLI exit-param overrides onto a per-ticker V5Config + V6 settings.

    Returns (tcfg, settings). When no overrides are set, returns the inputs
    unchanged (baseline parity). Always called at the 3 FSM-build sites so the
    ExitFSM actually sees the overridden values.
    """
    from dataclasses import replace as _replace

    cfg_changes = {}
    if SCALP_THRESH_OVERRIDE is not None:
        cfg_changes["scalp_peak_threshold_pct"] = SCALP_THRESH_OVERRIDE
    if SOFT_KEEP_OVERRIDE is not None:
        cfg_changes["soft_trail_keep_pct"] = SOFT_KEEP_OVERRIDE
    if THETA_MIN_OVERRIDE is not None:
        cfg_changes["theta_bleed_min"] = THETA_MIN_OVERRIDE
    if ADAPTIVE_MULT_OVERRIDE is not None and ADAPTIVE_MULT_OVERRIDE != 1.0:
        m = ADAPTIVE_MULT_OVERRIDE

        def _scale(tiers):
            # Scale trail_width by the multiplier (<1 tightens, >1 widens),
            # clamp to a sane 5-90% band so we never produce degenerate trails.
            return tuple(
                AdaptiveTier(t.min_peak_gain, max(5.0, min(90.0, t.trail_width * m)))
                for t in tiers
            )

        cfg_changes["adaptive_highvol_tiers"] = _scale(tcfg.adaptive_highvol_tiers)
        cfg_changes["adaptive_index_tiers"] = _scale(tcfg.adaptive_index_tiers)
        cfg_changes["adaptive_standard_tiers"] = _scale(tcfg.adaptive_standard_tiers)
    if cfg_changes:
        tcfg = _replace(tcfg, **cfg_changes)

    if BREAKEVEN_TRIGGER_OVERRIDE is not None or SCALEOUT_TRIGGER_OVERRIDE is not None:
        # _V6_SETTINGS is a SimpleNamespace; copy so per-trade overrides don't
        # mutate the shared module global.
        from copy import copy as _copy

        settings = _copy(settings)
        if BREAKEVEN_TRIGGER_OVERRIDE is not None:
            settings.V6_BREAKEVEN_TRIGGER_PCT = BREAKEVEN_TRIGGER_OVERRIDE
        if SCALEOUT_TRIGGER_OVERRIDE is not None:
            settings.V6_SCALEOUT_GAIN_PCT = SCALEOUT_TRIGGER_OVERRIDE
    return tcfg, settings
# Price gates: premium_cap + otm_distance DISABLED in prod (2026-06-10).
# min_premium + spread_gate still active (principled microstructure filters).
# Delta gate replaces the old static gates.
ENABLE_PRICE_GATES = False     # Old static gates OFF (matches prod)

# Delta entry gate: replaces premium_cap + otm_distance (deployed 2026-06-10).
# Backtested: delta 0.15-0.70 = $+491K, $742/trade, 75% WR over 126 days.
ENABLE_DELTA_GATE = True       # ON in prod
DELTA_MIN = 0.15               # Reject far OTM (lottery tickets, low delta)
DELTA_MAX = 0.70               # Reject deep ITM (overpaying for intrinsic value)

ANTI_CHASE_MAX_MOVE_PCT = 0.3     # Max underlying move from 5min ago
AFTERNOON_DANGER_START = 240       # 1:30 PM = 240 min after open
AFTERNOON_DANGER_END = 330         # 3:00 PM = 330 min after open
HARD_CUTOFF_MIN = 385              # 3:55 PM = 385 min after open
# TimeOfDayGate parity (prod TimeOfDayGate + CB_OPENING_BUFFER_MINUTES=10):
OPENING_BUFFER_MIN = 10            # Block first 10 min after open (CB_OPENING_BUFFER_MINUTES)
TOD_EARLY_CUTOFF_MIN = 15          # Before 9:45 AM (15 min after open)
TOD_EARLY_MIN_SCORE = 85           # Score >= 85 (conf >= 0.85) required before 9:45 ET

# Correlation groups (match pipeline.py CorrelationCapGate)
CORRELATION_GROUPS = {
    "index_megacap": {"SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NFLX"},
    "semis": {"NVDA", "AMD", "AVGO", "MU", "SMCI"},
    "tech_runners": {"TSLA", "MSTR", "PLTR", "COIN"},
    "tradfi": {"JPM", "BA"},
}
CORRELATION_CAP_MAX_PER_GROUP = 3

# Scanning
SCAN_START_MIN = 5
SCAN_END_MIN = 90
PUT_SCAN_END_MIN = 360   # PUTs scan all day (bearish moves happen anytime)
SCAN_INTERVAL = 1

# V6 settings
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
    # FSM never reads this — set False to match prod signal-level behavior and
    # avoid future drift (the unconditional $6 cap lives at the signal level).
    ENABLE_V6_PREMIUM_CAP=False,
    V6_PREMIUM_CAP=PREMIUM_CAP,
    V6_PREMIUM_CAP_MID=7.0,
    V6_PREMIUM_CAP_HIGH=9.0,
    ENABLE_V6_SPREAD_GATE=True,
    V6_MAX_SPREAD_PCT=SPREAD_GATE_PCT,
    ENABLE_V6_EARLY_POP_GATE=True,
    ENABLE_V6_SIDEWAYS_SCALP=False,  # Only owlet-yank has this enabled
    ENABLE_SCALP_TARGET=True,
    SCALP_TARGET_PCT=35.0,           # Match production default (was 25.0)
    SCALP_RUNNER_CONFIRM_PCT=40.0,
)


# ── Model Loading ──────────────────────────────────────────────────────────


def load_models(use_entry_filter: bool, use_regime: bool = True):
    """Load pattern_entry and optionally entry_timing + stop_calibration + regime models."""
    # Pattern entry (sourcing)
    pattern_path = MODEL_DIR / "pattern_entry.txt"
    pattern_meta_path = MODEL_DIR / "pattern_entry_meta.json"
    if not pattern_path.exists():
        print(f"ERROR: No pattern model at {pattern_path}")
        sys.exit(1)
    pattern_model = lgb.Booster(model_file=str(pattern_path))
    with open(pattern_meta_path) as f:
        pattern_meta = json.load(f)
    print(f"  Pattern entry: AUC={pattern_meta['auc']:.4f}")

    # Entry timing (filter)
    entry_model = None
    entry_features = None
    if use_entry_filter:
        entry_path = MODEL_DIR / "entry_timing.txt"
        if entry_path.exists():
            entry_model = lgb.Booster(model_file=str(entry_path))
            entry_features = entry_model.feature_name()
            print(f"  Entry timing: {len(entry_features)} features (quality gate)")
        else:
            print(f"  WARNING: No entry_timing model at {entry_path}, skipping filter")

    # Stop calibration (dynamic stop width)
    stop_model = None
    stop_path = MODEL_DIR / "stop_calibration.txt"
    if stop_path.exists():
        stop_model = lgb.Booster(model_file=str(stop_path))
        print(f"  Stop calibration: {stop_model.num_feature()} features (dynamic stops)")

    # Regime classifier (daily pre-filter)
    regime_model = None
    if use_regime:
        regime_path = MODEL_DIR / "regime_classifier.txt"
        if regime_path.exists():
            regime_model = lgb.Booster(model_file=str(regime_path))
            print(f"  Regime classifier: {regime_model.num_feature()} features (daily pre-filter)")
        else:
            print(f"  WARNING: No regime model at {regime_path}, skipping daily filter")

    # Signal quality (ranking model — picks best signals when multiple fire)
    signal_model = None
    signal_path = MODEL_DIR / "signal_quality.txt"
    if signal_path.exists():
        signal_model = lgb.Booster(model_file=str(signal_path))
        print(f"  Signal quality: {signal_model.num_feature()} features (ranking)")

    # PUT pattern model (dedicated model trained on PUT chain data)
    put_pattern_model = None
    put_pattern_meta = None
    put_pattern_path = MODEL_DIR / "put_pattern_v1.lgb"
    put_pattern_meta_path = MODEL_DIR / "put_pattern_v1_meta.json"
    if put_pattern_path.exists() and put_pattern_meta_path.exists():
        put_pattern_model = lgb.Booster(model_file=str(put_pattern_path))
        with open(put_pattern_meta_path) as f:
            put_pattern_meta = json.load(f)
        print(f"  PUT pattern: AUC={put_pattern_meta['auc']:.4f}, "
              f"threshold={put_pattern_meta.get('best_threshold', 0.40)}, "
              f"{len(put_pattern_meta['features'])} features")
    else:
        print("  PUT pattern: NOT FOUND (will use CALL model for PUTs)")

    # PUT entry timing model (dedicated model trained on PUT chain data)
    # Prod parity: PUT_ENTRY_TIMING_THRESHOLD=0.0 — the gate is DISABLED in prod
    # (model is net destructive for PUTs). Threshold 0.0 means it never blocks.
    put_entry_model = None
    put_entry_features = []
    put_entry_threshold = 0.0
    put_entry_path = MODEL_DIR / "put_entry_timing.txt"
    put_entry_meta_path = MODEL_DIR / "put_entry_timing_meta.json"
    if put_entry_path.exists() and put_entry_meta_path.exists():
        put_entry_model = lgb.Booster(model_file=str(put_entry_path))
        with open(put_entry_meta_path) as f:
            put_entry_meta = json.load(f)
        put_entry_features = put_entry_meta.get("features", [])
        print(f"  PUT entry timing: AUC={put_entry_meta['auc']:.4f}, "
              f"threshold={put_entry_threshold}, "
              f"{len(put_entry_features)} features")
    else:
        print("  PUT entry timing: NOT FOUND (PUTs will skip entry timing)")

    return (pattern_model, pattern_meta, entry_model, entry_features,
            stop_model, regime_model, signal_model, put_pattern_model, put_pattern_meta,
            put_entry_model, put_entry_features, put_entry_threshold)


# ── Pattern Entry Features (must match train_pattern_entry.py) ─────────────


def compute_pattern_features(closes, volumes, ivs, deltas, thetas, underlyings,
                              bids, asks, idx, opening_price):
    """Compute trailing features for pattern model at position idx."""
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

    f = {}
    f["prem_slope_5"] = (valid5[-1] / valid5[0] - 1) * 100
    f["prem_slope_10"] = (valid10[-1] / valid10[0] - 1) * 100 if len(valid10) >= 5 and valid10[0] > 0 else f["prem_slope_5"]

    if len(valid5) >= 4:
        mid = len(valid5) // 2
        first_rate = (valid5[mid] / valid5[0] - 1) * 100 if valid5[0] > 0 else 0
        second_rate = (valid5[-1] / valid5[mid] - 1) * 100 if valid5[mid] > 0 else 0
        f["prem_accel"] = second_rate - first_rate
    else:
        f["prem_accel"] = 0

    last3 = valid5[-3:] if len(valid5) >= 3 else valid5
    f["prem_stabilizing"] = (max(last3) - min(last3)) / max(last3) * 100 if max(last3) > 0 else 0

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


# ── PUT Pattern Features (must match ml_pipeline.py compute_put_pattern_features) ──


def compute_put_pattern_features(closes, volumes, ivs, deltas, thetas, underlyings,
                                  bids, asks, idx, opening_price,
                                  vegas=None, bid_sizes=None, ask_sizes=None,
                                  call_ivs=None, call_volumes=None):
    """Compute 27 features for PUT pattern model at position idx."""
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

    f = {}

    # Premium trajectory
    f["prem_slope_5"] = (valid5[-1] / valid5[0] - 1) * 100
    f["prem_slope_10"] = (valid10[-1] / valid10[0] - 1) * 100 if len(valid10) >= 5 and valid10[0] > 0 else f["prem_slope_5"]

    if len(valid5) >= 4:
        mid = len(valid5) // 2
        first_rate = (valid5[mid] / valid5[0] - 1) * 100 if valid5[0] > 0 else 0
        second_rate = (valid5[-1] / valid5[mid] - 1) * 100 if valid5[mid] > 0 else 0
        f["prem_accel"] = second_rate - first_rate
    else:
        f["prem_accel"] = 0

    last3 = valid5[-3:] if len(valid5) >= 3 else valid5
    f["prem_stabilizing"] = (max(last3) - min(last3)) / max(last3) * 100 if max(last3) > 0 else 0

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
    f["und_slope_10"] = (valid10_u[-1] / valid10_u[0] - 1) * 100 if len(valid10_u) >= 5 and valid10_u[0] > 0 else f["und_slope_5"]
    f["und_slope_15"] = (valid15_u[-1] / valid15_u[0] - 1) * 100 if len(valid15_u) >= 5 and valid15_u[0] > 0 else f["und_slope_10"]

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

    # Candle range (intrabar volatility — approximated from bid-ask range)
    f["candle_range_pct"] = (ask - bid) / max(current, 0.01) * 100 if bid >= 0 else 0.0

    # Bid/ask size imbalance
    if bid_sizes is not None and ask_sizes is not None and idx < len(bid_sizes):
        bs = bid_sizes[idx] if not np.isnan(bid_sizes[idx]) else 0
        as_ = ask_sizes[idx] if not np.isnan(ask_sizes[idx]) else 0
        total = bs + as_
        f["bid_size_ratio"] = bs / max(total, 1)
    else:
        f["bid_size_ratio"] = 0.5

    # IV skew (PUT IV / CALL IV) — cross-chain feature
    if call_ivs is not None and idx < len(call_ivs):
        put_iv = valid5_iv[-1] if len(valid5_iv) > 0 else 0
        call_iv = call_ivs[idx] if not np.isnan(call_ivs[idx]) else 0
        f["iv_skew"] = put_iv / call_iv if call_iv > 0 else 1.0
    else:
        f["iv_skew"] = 1.0

    # PUT volume / CALL volume — cross-chain feature
    if call_volumes is not None and idx < len(call_volumes):
        call_vol = call_volumes[idx] if not np.isnan(call_volumes[idx]) else 0
        put_vol = volumes[idx] if not np.isnan(volumes[idx]) else 0
        f["put_call_volume_ratio"] = put_vol / max(call_vol, 1)
    else:
        f["put_call_volume_ratio"] = 1.0

    return f


# ── Entry Timing Features (must match train_ml_models_v3.py) ──────────────


def compute_entry_timing_features(closes, volumes, bids_arr, asks_arr, bid_sizes,
                                   ask_sizes, ivs, deltas, thetas, vegas,
                                   underlyings, stock_closes, stock_highs,
                                   stock_lows, idx, entry_features):
    """Compute entry_timing model features at position idx."""
    lookback = 15
    if idx < lookback + 1:
        return None

    entry_price = closes[idx]
    if np.isnan(entry_price) or entry_price <= 0:
        return None

    f = {}

    # Time
    f["minutes_since_open"] = idx
    f["hour_bucket"] = idx // 60
    f["is_first_30min"] = 1 if idx <= 30 else 0

    # Premium action
    prices = closes[max(0, idx - lookback):idx + 1]
    valid_prices = prices[~np.isnan(prices) & (prices > 0)]
    if len(valid_prices) < 3:
        return None

    f["premium"] = float(entry_price)
    f["premium_change_5m"] = float((valid_prices[-1] / valid_prices[max(-6, -len(valid_prices))] - 1) * 100) if valid_prices[max(-6, -len(valid_prices))] > 0 else 0
    f["premium_change_10m"] = float((valid_prices[-1] / valid_prices[max(-11, -len(valid_prices))] - 1) * 100) if valid_prices[max(-11, -len(valid_prices))] > 0 else 0
    f["premium_change_15m"] = float((valid_prices[-1] / valid_prices[0] - 1) * 100) if valid_prices[0] > 0 else 0

    if len(valid_prices) > 2 and all(valid_prices[:-1] > 0):
        returns = np.diff(valid_prices) / valid_prices[:-1]
        f["premium_volatility"] = float(np.std(returns) * 100)
    else:
        f["premium_volatility"] = 0

    # Volume
    vols = volumes[max(0, idx - lookback):idx + 1]
    valid_vols = vols[~np.isnan(vols)]
    f["current_volume"] = float(volumes[idx]) if not np.isnan(volumes[idx]) else 0
    avg_vol = float(np.mean(valid_vols[:-1])) if len(valid_vols) > 1 else 1
    f["volume_ratio"] = float(f["current_volume"] / max(avg_vol, 1))
    if len(valid_vols) > 5 and np.std(valid_vols[:-1]) > 0:
        f["volume_zscore"] = float((valid_vols[-1] - np.mean(valid_vols[:-1])) / np.std(valid_vols[:-1]))
    else:
        f["volume_zscore"] = 0

    # Bid/ask
    bid = float(bids_arr[idx]) if not np.isnan(bids_arr[idx]) else 0
    ask = float(asks_arr[idx]) if not np.isnan(asks_arr[idx]) else 0
    mid = (bid + ask) / 2 if (bid + ask) > 0 else entry_price
    f["spread"] = float(ask - bid) if ask > bid else 0
    f["spread_pct"] = float(f["spread"] / mid * 100) if mid > 0 else 0
    f["bid_size"] = float(bid_sizes[idx]) if idx < len(bid_sizes) and not np.isnan(bid_sizes[idx]) else 0
    f["ask_size"] = float(ask_sizes[idx]) if idx < len(ask_sizes) and not np.isnan(ask_sizes[idx]) else 0
    f["size_imbalance"] = float((f["bid_size"] - f["ask_size"]) / max(f["bid_size"] + f["ask_size"], 1))

    # Greeks
    f["iv"] = float(ivs[idx]) if not np.isnan(ivs[idx]) else 0
    f["delta"] = float(abs(deltas[idx])) if not np.isnan(deltas[idx]) else 0
    f["theta"] = float(thetas[idx]) if not np.isnan(thetas[idx]) else 0
    f["vega"] = float(vegas[idx]) if idx < len(vegas) and not np.isnan(vegas[idx]) else 0

    iv_window = ivs[max(0, idx - lookback):idx + 1]
    valid_iv = iv_window[~np.isnan(iv_window)]
    f["iv_change_15m"] = float(valid_iv[-1] - valid_iv[0]) if len(valid_iv) > 3 else 0

    f["underlying_price"] = float(underlyings[idx]) if not np.isnan(underlyings[idx]) else 0

    # Underlying price action (from stock data)
    s_idx = min(idx, len(stock_closes) - 1)
    if s_idx > 5 and len(stock_closes) > 5:
        s_window = stock_closes[max(0, s_idx - lookback):s_idx + 1]
        s_valid = s_window[~np.isnan(s_window) & (s_window > 0)]
        if len(s_valid) > 1:
            f["underlying_change_5m"] = float((s_valid[-1] / s_valid[max(-6, -len(s_valid))] - 1) * 100)
            f["underlying_change_15m"] = float((s_valid[-1] / s_valid[0] - 1) * 100)
            if len(s_valid) > 2 and all(s_valid[:-1] > 0):
                f["underlying_volatility"] = float(np.std(np.diff(s_valid) / s_valid[:-1]) * 100)
            else:
                f["underlying_volatility"] = 0
        else:
            f["underlying_change_5m"] = 0
            f["underlying_change_15m"] = 0
            f["underlying_volatility"] = 0

        # Daily trend
        s_all = stock_closes[:s_idx + 1]
        s_all_valid = s_all[~np.isnan(s_all) & (s_all > 0)]
        if len(s_all_valid) > 10 and s_all_valid[0] > 0:
            f["daily_trend_pct"] = float((s_all_valid[-1] / s_all_valid[0] - 1) * 100)
        else:
            f["daily_trend_pct"] = 0

        if len(s_all_valid) > 1:
            day_lo = s_all_valid.min()
            day_hi = s_all_valid.max()
            f["daily_range_position"] = float((s_all_valid[-1] - day_lo) / (day_hi - day_lo)) if day_hi > day_lo else 0.5
        else:
            f["daily_range_position"] = 0.5

        # ATR
        if s_idx > 14 and len(stock_highs) > 14:
            h_window = stock_highs[max(0, s_idx - 14):s_idx]
            l_window = stock_lows[max(0, s_idx - 14):s_idx]
            h_valid = h_window[~np.isnan(h_window)]
            l_valid = l_window[~np.isnan(l_window)]
            if len(h_valid) >= 14 and len(l_valid) >= 14 and s_all_valid[-1] > 0:
                f["atr_pct"] = float(np.mean(h_valid[-14:] - l_valid[-14:]) / s_all_valid[-1] * 100)
            else:
                f["atr_pct"] = 0
        else:
            f["atr_pct"] = 0
    else:
        for k in ["underlying_change_5m", "underlying_change_15m", "underlying_volatility",
                   "daily_trend_pct", "daily_range_position", "atr_pct"]:
            f[k] = 0

    # Premium drop from recent peak (top feature)
    recent = closes[max(0, idx - 10):idx + 1]
    valid_recent = recent[~np.isnan(recent) & (recent > 0)]
    if len(valid_recent) > 0:
        f["prem_drop_from_recent_peak"] = float((closes[idx] / np.max(valid_recent) - 1) * 100)
    else:
        f["prem_drop_from_recent_peak"] = 0

    # Decline deceleration
    if len(valid_recent) >= 3:
        first_half = valid_recent[:len(valid_recent) // 2]
        second_half = valid_recent[len(valid_recent) // 2:]
        if len(first_half) > 0 and len(second_half) > 0 and first_half[0] > 0 and second_half[0] > 0:
            first_change = (first_half[-1] / first_half[0] - 1) * 100
            second_change = (second_half[-1] / second_half[0] - 1) * 100
            f["decline_deceleration"] = float(second_change - first_change)
        else:
            f["decline_deceleration"] = 0
    else:
        f["decline_deceleration"] = 0

    # Return only features the model expects
    return {k: f.get(k, 0) for k in entry_features}


# ── DipConfirm Simulation ─────────────────────────────────────────────────


def simulate_dip_confirm(asks, bids, closes, entry_minute, n_rows):
    """Simulate production DipConfirm logic using minute-level data.

    Production flow (ML signals):
    1. At signal minute, t0 = ask price
    2. Next minute, t1 = ask price — if fade < 1%, enter at min(t0, t1)
    3. If fading >= 1%, poll up to DIP_CONFIRM_MAX_POLLS minutes for uptick
    4. On uptick (price > prev), enter at that price
    5. No uptick after polls → enter at lowest price seen (ML behavior)

    Returns (entry_premium, delay_minutes, savings_pct, outcome).
    - outcome: 'stable' | 'vwap_enter' | 'uptick' | 'timeout_enter' | 'timeout_skip'
    """
    t0_ask = float(asks[entry_minute]) if asks[entry_minute] > 0 else float(closes[entry_minute])
    if t0_ask <= 0 or np.isnan(t0_ask):
        return t0_ask, 0, 0.0, "no_data"

    # Step 1: check next minute for fade
    t1_minute = entry_minute + 1
    if t1_minute >= n_rows:
        return t0_ask, 0, 0.0, "no_data"

    t1 = float(asks[t1_minute]) if asks[t1_minute] > 0 else float(closes[t1_minute])
    if t1 <= 0 or np.isnan(t1):
        return t0_ask, 0, 0.0, "no_data"

    fade = (t0_ask - t1) / t0_ask * 100

    if fade < DIP_CONFIRM_FADE_PCT:
        # Premium stable/rising — enter at the price at DECISION time (t1).
        # (Bug fix 2026-06-10: previously filled at min(t0, t1), a retroactive
        # fill at a price already gone by decision time.)
        entry_price = t1
        savings = (t0_ask - entry_price) / t0_ask * 100
        return entry_price, 1, savings, "stable"

    # Step 2: premium IS fading — poll for uptick
    prev = t1
    last_poll_minute = t1_minute
    for poll in range(DIP_CONFIRM_MAX_POLLS):
        poll_minute = t1_minute + 1 + poll
        if poll_minute >= n_rows:
            break
        last_poll_minute = poll_minute

        current_ask = float(asks[poll_minute]) if asks[poll_minute] > 0 else 0
        current = current_ask if current_ask > 0 else float(closes[poll_minute])
        if current <= 0 or np.isnan(current):
            continue

        if current > prev:
            # Uptick detected — enter here
            savings = (t0_ask - current) / t0_ask * 100
            delay = poll_minute - entry_minute
            return current, delay, savings, "uptick"
        prev = current

    # No uptick after all polls
    delay = last_poll_minute - entry_minute
    if DIP_CONFIRM_ALWAYS_ENTER:
        # ML behavior: enter at the ask at the TIMEOUT minute (decision time).
        # (Bug fix 2026-06-10: previously filled at low_water, a price already gone.)
        timeout_ask = float(asks[last_poll_minute]) if asks[last_poll_minute] > 0 else float(closes[last_poll_minute])
        if timeout_ask <= 0 or np.isnan(timeout_ask):
            timeout_ask = prev if (prev > 0 and not np.isnan(prev)) else t0_ask
        savings = (t0_ask - timeout_ask) / t0_ask * 100
        return timeout_ask, delay, savings, "timeout_enter"
    else:
        # Discord behavior: skip the trade
        return 0, delay, 0, "timeout_skip"


# ── FSM Exit Simulation ───────────────────────────────────────────────────


def simulate_exit(closes, bids, asks, underlyings, entry_idx,
                  entry_premium, contracts, ticker, dte, expiry_date,
                  ml_stop_pct=None, grace_override=None):
    """Run V5 FSM on remaining candles after entry.

    If ml_stop_pct is provided, override the FSM's hard stop with ML-predicted value.
    """
    if entry_premium <= 0:
        return {"pnl": 0, "reason": "no_data", "hold_min": 0, "peak_gain": 0, "exit_prem": 0}

    tcfg = get_ticker_config(ticker, use_per_ticker=True)
    from dataclasses import replace
    # Override grace period if specified
    if grace_override is not None:
        tcfg = replace(tcfg, grace_period_min=grace_override)
    # Override stop width with ML prediction if available
    if ml_stop_pct is not None:
        # Clamp to reasonable range (15-55%)
        clamped = max(15.0, min(55.0, ml_stop_pct))
        tcfg = replace(tcfg,
                       tight_stop_0dte_pct=clamped,
                       backstop_0dte_pct=min(clamped + 20, 65.0))
    _tcfg, _settings = _apply_exit_overrides(tcfg, _V6_SETTINGS)
    fsm = ExitFSM(_tcfg, settings=_settings)

    entry_ts = datetime(2026, 1, 1, 9, 30) + timedelta(minutes=entry_idx)

    underlying_0 = 0
    for i in range(entry_idx, min(entry_idx + 5, len(underlyings))):
        u = underlyings[i]
        if not np.isnan(u) and u > 0:
            underlying_0 = float(u)
            break

    state = TradeState(
        trade_id=1, ticker=ticker, option_type="call",
        entry_premium=entry_premium, entry_time=entry_ts,
        contracts=contracts, peak_premium=entry_premium,
        entry_underlying_price=underlying_0,
        dte=dte, expiry_date=expiry_date or "",
    )

    locked_pnl = 0.0
    remaining = contracts

    for idx in range(entry_idx + 1, len(closes)):
        prem = closes[idx]
        if np.isnan(prem) or prem <= 0:
            continue

        bid = float(bids[idx]) if idx < len(bids) and not np.isnan(bids[idx]) else prem
        ask = float(asks[idx]) if idx < len(asks) and not np.isnan(asks[idx]) else prem
        underlying = float(underlyings[idx]) if idx < len(underlyings) and not np.isnan(underlyings[idx]) else 0

        now = entry_ts + timedelta(minutes=(idx - entry_idx))
        minutes_to_close = max(0, (16 * 60) - (now.hour * 60 + now.minute))

        action = fsm.evaluate(
            state, prem, bid, ask, now,
            current_underlying=underlying,
            minutes_to_close=minutes_to_close,
            candle_data={},
        )

        if action.should_exit:
            # Use bid for exit (realistic slippage — we're selling)
            exit_price = bid if bid > 0 else prem

            if action.contracts_to_close > 0 and action.contracts_to_close < remaining:
                locked_pnl += (exit_price - entry_premium) * action.contracts_to_close * 100
                remaining -= action.contracts_to_close
                state.contracts = remaining
                continue

            elapsed = idx - entry_idx
            peak_gain = (state.peak_premium - entry_premium) / entry_premium * 100
            pnl = locked_pnl + (exit_price - entry_premium) * remaining * 100
            return {
                "pnl": pnl, "reason": action.reason.value,
                "hold_min": elapsed, "peak_gain": peak_gain,
                "exit_prem": exit_price,
            }

    # EOD — fill at BID (we're selling; close/mid is a leak)
    last_valid = entry_premium
    for i in range(len(closes) - 1, entry_idx, -1):
        b = bids[i] if i < len(bids) else np.nan
        if not np.isnan(b) and b > 0:
            last_valid = float(b)
            break
        if not np.isnan(closes[i]) and closes[i] > 0:
            last_valid = closes[i]
            break
    elapsed = len(closes) - entry_idx
    peak_gain = (state.peak_premium - entry_premium) / entry_premium * 100
    pnl = locked_pnl + (last_valid - entry_premium) * remaining * 100
    return {
        "pnl": pnl, "reason": "eod_data_end", "hold_min": elapsed,
        "peak_gain": peak_gain, "exit_prem": last_valid,
    }


# ── Backtest Runner ────────────────────────────────────────────────────────


def _regime_rth_bars(conn, ticker: str, cache: dict) -> dict:
    """Grouped RTH 1-min bars {date: [bar,...]} for one ticker (cached per run).

    Mirrors the trainer's _load_rth_bars_by_date (scripts/train_ml_models_v3.py)
    so the backtest feeds the SHARED regime feature module identical inputs.
    One query per ticker for the whole backtest window.
    """
    key = f"__regime_rth__{ticker}"
    if key not in cache:
        rows = conn.execute(
            f"SELECT substr(timestamp, 1, 10) AS d, substr(timestamp, 12, 5) AS tm, "
            f"open, high, low, close, volume FROM stock_ohlc "
            f"WHERE ticker=? AND {RTH_FILTER_SQL} ORDER BY timestamp",
            (ticker,),
        ).fetchall()
        cache[key] = rth_bars_by_date_from_rows([
            {"d": r[0], "tm": r[1], "open": r[2], "high": r[3],
             "low": r[4], "close": r[5], "volume": r[6]}
            for r in rows
        ])
    return cache[key]


def _regime_gex_row(uw_conn, ticker: str, date_str: str, cache: dict,
                    max_staleness_days: int = 5) -> dict:
    """Raw UW greek_exposure legs for ticker/date (trainer parity, cached).

    Most recent row at/before date_str — OI is fixed at the open, so a same-day
    row is serve-time-safe (matches the trainer's _gex_features). Missing or
    stale (>5 days) -> {}; the shared module zero-fills GEX deterministically.
    """
    key = f"__regime_gex__{ticker}__{date_str}"
    if key in cache:
        return cache[key]
    gex: dict = {}
    if uw_conn is not None:
        row = uw_conn.execute(
            "SELECT date, call_gamma, put_gamma, call_delta, put_delta, "
            "call_charm, put_charm, call_vanna, put_vanna "
            "FROM greek_exposure WHERE ticker=? AND date<=? "
            "ORDER BY date DESC LIMIT 1",
            (ticker, date_str),
        ).fetchone()
        if row:
            staleness = (
                datetime.strptime(date_str, "%Y-%m-%d")
                - datetime.strptime(row[0], "%Y-%m-%d")
            ).days
            if staleness <= max_staleness_days:
                gex = {
                    "call_gamma": float(row[1] or 0),
                    "put_gamma": float(row[2] or 0),
                    "call_delta": float(row[3] or 0),
                    "put_delta": float(row[4] or 0),
                    "call_charm": float(row[5] or 0),
                    "put_charm": float(row[6] or 0),
                    "call_vanna": float(row[7] or 0),
                    "put_vanna": float(row[8] or 0),
                }
    cache[key] = gex
    return gex


def compute_regime_score(regime_model, ticker, date_str, conn, uw_conn,
                         stock_data_cache: dict, prev_days_cache: dict) -> float:
    """Compute regime model prediction for a ticker-day using morning data.

    Builds the FULL 40-feature vector via the SHARED feature module
    (options_owl.sourcing.features.regime_features) — the exact path the
    trainer uses — instead of the old inline ~18-feature dict that zero-filled
    22 of 40 model features. Every model feature MUST be produced by the
    shared module; a missing feature raises (no silent .get(feat, 0)).

    Uses only data available at 9:45 AM ET: 09:30-09:44 RTH bars, prior-day
    lags, SPY/QQQ market context, OI-based GEX legs.
    Returns prediction (0-1). Higher = more likely a trending day.
    """
    features = list(regime_model.feature_name())

    # Own-ticker grouped RTH bars (one query per ticker, cached for the run)
    by_date = _regime_rth_bars(conn, ticker, stock_data_cache)

    # Early-return guard (unchanged semantics): need a usable morning window.
    morning_bars = [
        b for b in by_date.get(date_str, []) if b["tm"] < REGIME_EARLY_END
    ]
    if sum(1 for b in morning_bars if b["close"] > 0 and b["open"] > 0) < 5:
        return 0.0

    # SPY/QQQ cross-market context bars (same cache, same loader)
    market_by_date = {
        mkt: _regime_rth_bars(conn, mkt, stock_data_cache)
        for mkt in ("SPY", "QQQ")
    }

    raw_inputs = load_training_inputs(
        ticker, date_str,
        by_date=by_date,
        market_by_date=market_by_date,
        gex_row=_regime_gex_row(uw_conn, ticker, date_str, prev_days_cache),
    )
    vec = compute_regime_feature_vector(raw_inputs)

    # Zero-fill regression guard: the shared module must cover EVERY model
    # feature. A miss here means train/backtest feature skew — fail loudly.
    missing = [feat for feat in features if feat not in vec]
    assert not missing, (
        f"REGIME FEATURE SKEW: model expects features the shared module did "
        f"not produce: {missing} (model={len(features)} feats, "
        f"produced={len(vec)})"
    )
    if not getattr(compute_regime_score, "_parity_logged", False):
        print(f"  Regime features: shared module covers all {len(features)} "
              f"model features (no zero-fill)")
        compute_regime_score._parity_logged = True

    X = np.array([[vec[feat] for feat in features]], dtype=np.float32)
    return float(regime_model.predict(X)[0])


def compute_signal_quality_features(closes, volumes, bids_arr, asks_arr,
                                     bid_sizes, ask_sizes, ivs, deltas, thetas, vegas,
                                     underlyings, stock_closes, stock_highs, stock_lows,
                                     idx, signal_features, is_call=True):
    """Compute signal_quality model features at position idx for ranking."""
    lookback = 15
    if idx < lookback + 1:
        return None

    entry_price = closes[idx]
    if np.isnan(entry_price) or entry_price <= 0:
        return None

    f = {}
    f["minutes_since_open"] = idx
    f["hour_bucket"] = (idx + 30) // 60  # 0-indexed minutes since 9:30
    f["is_first_30min"] = 1 if idx <= 30 else 0
    f["premium"] = float(entry_price)

    # Premium changes
    for offset, key in [(5, "premium_change_5m"), (10, "premium_change_10m"), (15, "premium_change_15m")]:
        prev_idx = max(0, idx - offset)
        prev_val = closes[prev_idx]
        if not np.isnan(prev_val) and prev_val > 0:
            f[key] = (entry_price / prev_val - 1) * 100
        else:
            f[key] = 0

    # Premium volatility
    window = closes[max(0, idx - 15):idx + 1]
    valid = window[~np.isnan(window) & (window > 0)]
    if len(valid) >= 3:
        returns = np.diff(valid) / valid[:-1]
        f["premium_volatility"] = float(np.std(returns) * 100)
    else:
        f["premium_volatility"] = 0

    # Volume
    f["current_volume"] = float(volumes[idx]) if idx < len(volumes) else 0
    vol_window = volumes[max(0, idx - 20):idx]
    vol_valid = vol_window[~np.isnan(vol_window)]
    avg_vol = float(np.mean(vol_valid)) if len(vol_valid) > 0 else 1
    f["volume_ratio"] = f["current_volume"] / max(avg_vol, 1)
    f["volume_zscore"] = (f["current_volume"] - avg_vol) / max(float(np.std(vol_valid)), 1) if len(vol_valid) > 1 else 0

    # Spread
    bid = float(bids_arr[idx]) if idx < len(bids_arr) else 0
    ask = float(asks_arr[idx]) if idx < len(asks_arr) else entry_price
    f["spread"] = ask - bid
    f["spread_pct"] = (ask - bid) / ask * 100 if ask > 0 else 0
    f["bid_size"] = float(bid_sizes[idx]) if idx < len(bid_sizes) else 0
    f["ask_size"] = float(ask_sizes[idx]) if idx < len(ask_sizes) else 0
    total_size = f["bid_size"] + f["ask_size"]
    f["size_imbalance"] = (f["bid_size"] - f["ask_size"]) / total_size if total_size > 0 else 0

    # Greeks
    f["iv"] = float(ivs[idx]) if idx < len(ivs) and not np.isnan(ivs[idx]) else 0
    f["delta"] = float(deltas[idx]) if idx < len(deltas) and not np.isnan(deltas[idx]) else 0
    f["theta"] = float(thetas[idx]) if idx < len(thetas) and not np.isnan(thetas[idx]) else 0
    f["vega"] = float(vegas[idx]) if idx < len(vegas) and not np.isnan(vegas[idx]) else 0

    # IV change
    iv_prev = ivs[max(0, idx - 15)]
    if not np.isnan(iv_prev) and not np.isnan(ivs[idx]):
        f["iv_change_15m"] = float(ivs[idx] - iv_prev)
    else:
        f["iv_change_15m"] = 0

    # Underlying
    und = float(underlyings[idx]) if idx < len(underlyings) and not np.isnan(underlyings[idx]) else 0
    f["underlying_price"] = und

    for offset, key in [(5, "underlying_change_5m"), (15, "underlying_change_15m")]:
        prev_idx = max(0, idx - offset)
        prev_u = underlyings[prev_idx] if prev_idx < len(underlyings) and not np.isnan(underlyings[prev_idx]) else 0
        if prev_u > 0 and und > 0:
            f[key] = (und / prev_u - 1) * 100
        else:
            f[key] = 0

    # Underlying volatility
    und_window = underlyings[max(0, idx - 15):idx + 1]
    und_valid = und_window[~np.isnan(und_window) & (und_window > 0)]
    if len(und_valid) >= 3:
        und_returns = np.diff(und_valid) / und_valid[:-1]
        f["underlying_volatility"] = float(np.std(und_returns) * 100)
    else:
        f["underlying_volatility"] = 0

    # Stock data features
    if len(stock_closes) > idx and idx > 0:
        open_price = stock_closes[0] if len(stock_closes) > 0 else und
        if open_price > 0:
            f["daily_trend_pct"] = (und / open_price - 1) * 100
        else:
            f["daily_trend_pct"] = 0
        day_high = float(np.max(stock_highs[:idx + 1])) if len(stock_highs) > idx else und
        day_low = float(np.min(stock_lows[:idx + 1])) if len(stock_lows) > idx else und
        if day_high > day_low:
            f["daily_range_position"] = (und - day_low) / (day_high - day_low)
        else:
            f["daily_range_position"] = 0.5
        if len(stock_highs) > 14 and len(stock_lows) > 14:
            ranges = stock_highs[:idx] - stock_lows[:idx]
            valid_ranges = ranges[~np.isnan(ranges) & (ranges > 0)]
            f["atr_pct"] = float(np.mean(valid_ranges[-14:]) / und * 100) if len(valid_ranges) > 0 and und > 0 else 0
        else:
            f["atr_pct"] = 0
    else:
        f["daily_trend_pct"] = 0
        f["daily_range_position"] = 0.5
        f["atr_pct"] = 0

    f["is_call"] = 1 if is_call else 0

    return {k: f.get(k, 0) for k in signal_features}


# ── RTH Stock/Option Alignment Helpers (B1 fix 2026-06-10) ─────────────────
#
# stock_ohlc includes premarket/extended-hours bars (04:00-19:59 ET on most
# days: 700-948 bars vs 391 RTH option bars), but the backtest indexes stock
# arrays POSITIONALLY by option-minute. Without an RTH filter, every consumer
# (check_spy_put_gate, compute_entry_timing_features, compute_regime_score,
# signal_quality) reads premarket data. Fix: restrict stock queries to regular
# trading hours and align each stock bar to the option series' session minute.
# DB timestamps are stored ET-LOCALIZED (e.g. '2026-04-10 09:30:00-04:00'),
# so the wall-time component is Eastern regardless of DST — alignment via the
# wall-time minutes is DST-robust.

RTH_FILTER_SQL = "substr(timestamp, 12, 8) >= '09:30:00' AND substr(timestamp, 12, 8) <= '16:00:00'"
SESSION_MINUTES = 391  # 09:30-16:00 inclusive


def _ts_session_minute(ts: str) -> int:
    """Minutes since 09:30 ET from an ET-localized 'YYYY-MM-DD HH:MM:SS±HH:MM' string."""
    return int(ts[11:13]) * 60 + int(ts[14:16]) - 570


def load_stock_by_minute(conn, ticker: str, date_str: str) -> dict[int, tuple]:
    """Load RTH stock bars keyed by session minute (0..390)."""
    rows = conn.execute(f"""
        SELECT timestamp, open, close, high, low, volume FROM stock_ohlc
        WHERE ticker=? AND date(timestamp)=? AND {RTH_FILTER_SQL}
        ORDER BY timestamp
    """, (ticker, date_str)).fetchall()
    by_minute: dict[int, tuple] = {}
    for r in rows:
        m = _ts_session_minute(r[0])
        if 0 <= m < SESSION_MINUTES:
            by_minute[m] = r
    return by_minute


def align_stock_arrays(by_minute: dict[int, tuple], option_minutes: list[int]):
    """Build stock arrays aligned 1:1 with the option series rows.

    Returns (opens, closes, highs, lows, volumes) of len(option_minutes);
    minutes with no stock bar are NaN.
    """
    n = len(option_minutes)
    opens = np.full(n, np.nan)
    closes = np.full(n, np.nan)
    highs = np.full(n, np.nan)
    lows = np.full(n, np.nan)
    volumes = np.full(n, np.nan)
    for i, m in enumerate(option_minutes):
        r = by_minute.get(m)
        if r is None:
            continue
        opens[i] = float(r[1]) if r[1] else np.nan
        closes[i] = float(r[2]) if r[2] else np.nan
        highs[i] = float(r[3]) if r[3] else np.nan
        lows[i] = float(r[4]) if r[4] else np.nan
        volumes[i] = float(r[5]) if r[5] is not None else np.nan
    return opens, closes, highs, lows, volumes


def build_minute_indexed_closes(by_minute: dict[int, tuple]) -> np.ndarray:
    """Build a SESSION_MINUTES-length close array indexed by session minute (for SPY gate)."""
    arr = np.full(SESSION_MINUTES, np.nan)
    for m, r in by_minute.items():
        if r[2]:
            arr[m] = float(r[2])
    # Forward-fill gaps so positional reads always see the latest known price
    last = np.nan
    for i in range(SESSION_MINUTES):
        if np.isnan(arr[i]):
            arr[i] = last
        else:
            last = arr[i]
    return arr


# ── Production Gate Helpers ────────────────────────────────────────────────


def compute_rsi(prices: np.ndarray, period: int = 14) -> float | None:
    """Compute RSI from a price array. Returns None if insufficient data."""
    if len(prices) < period + 1:
        return None
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def compute_ema(prices: np.ndarray, period: int) -> float | None:
    """Compute EMA of last N prices. Returns None if insufficient data."""
    if len(prices) < period:
        return None
    multiplier = 2.0 / (period + 1)
    ema = float(prices[0])
    for p in prices[1:]:
        ema = (p - ema) * multiplier + ema
    return ema


def _group_for_ticker(ticker: str) -> str | None:
    """Get correlation group for ticker."""
    for name, members in CORRELATION_GROUPS.items():
        if ticker.upper() in members:
            return name
    return None


def check_momentum_confirm(stock_closes: np.ndarray, stock_highs: np.ndarray,
                            stock_lows: np.ndarray, idx: int,
                            is_call: bool = True) -> tuple[bool, str]:
    """Simulate MomentumConfirmGate using stock data.

    Returns (passed, reason). Blocks when 2+ negative signals detected.
    Uses 5m candle direction + RSI from stock_closes.
    """
    s_idx = min(idx, len(stock_closes) - 1)
    if s_idx < 15:
        return True, "insufficient_data"

    reasons = []

    # Compute 5m RSI (use last 20 1-min bars for ~14-period RSI)
    rsi_window = stock_closes[max(0, s_idx - 20):s_idx + 1]
    valid = rsi_window[~np.isnan(rsi_window) & (rsi_window > 0)]
    rsi_5m = compute_rsi(valid) if len(valid) >= 15 else None

    # 15m RSI (use last 30 bars)
    rsi_15m_window = stock_closes[max(0, s_idx - 30):s_idx + 1]
    valid_15m = rsi_15m_window[~np.isnan(rsi_15m_window) & (rsi_15m_window > 0)]
    rsi_15m = compute_rsi(valid_15m) if len(valid_15m) >= 15 else None

    # RSI extreme against direction on BOTH timeframes
    if rsi_5m is not None and rsi_15m is not None:
        if is_call and rsi_5m < 35 and rsi_15m < 40:
            reasons.append(f"RSI bearish (5m={rsi_5m:.0f}, 15m={rsi_15m:.0f})")
        elif not is_call and rsi_5m > 65 and rsi_15m > 60:
            reasons.append(f"RSI bullish (5m={rsi_5m:.0f}, 15m={rsi_15m:.0f})")

    # Check last 3 5-min candles (approximate: use 5-bar groups from 1-min data)
    against_count = 0
    if s_idx >= 15:
        for candle_start in range(s_idx - 15, s_idx, 5):
            candle_end = min(candle_start + 5, s_idx + 1)
            if candle_end <= candle_start:
                continue
            c_open = stock_closes[candle_start] if not np.isnan(stock_closes[candle_start]) else 0
            c_close = stock_closes[candle_end - 1] if not np.isnan(stock_closes[candle_end - 1]) else 0
            if c_open > 0 and c_close > 0:
                if is_call and c_close < c_open:
                    against_count += 1
                elif not is_call and c_close > c_open:
                    against_count += 1

    if against_count >= 3:
        reasons.append("Last 3 5m candles all against direction")

    # Need 2+ negative signals to reject
    if len(reasons) >= 2:
        return False, f"momentum_reject: {'; '.join(reasons)}"

    return True, "momentum_ok"


def check_directional_regime(stock_closes: np.ndarray, idx: int,
                              is_call: bool = True) -> tuple[bool, str]:
    """Simulate DirectionalRegimeGate using stock data.

    Computes regime score from RSI, candle bars, underlying momentum, EMA trend.
    Returns (passed, reason).
    """
    s_idx = min(idx, len(stock_closes) - 1)
    if s_idx < 20:
        return True, "insufficient_data"

    # RSI (5m proxy from 1-min data)
    rsi_window = stock_closes[max(0, s_idx - 20):s_idx + 1]
    valid = rsi_window[~np.isnan(rsi_window) & (rsi_window > 0)]
    rsi_5m = compute_rsi(valid) if len(valid) >= 15 else None

    # RSI (15m proxy)
    rsi_15m_window = stock_closes[max(0, s_idx - 30):s_idx + 1]
    valid_15m = rsi_15m_window[~np.isnan(rsi_15m_window) & (rsi_15m_window > 0)]
    rsi_15m = compute_rsi(valid_15m) if len(valid_15m) >= 15 else None

    # Count bearish vs bullish 5m candles (last 6 = ~30 min)
    bearish_bars = 0
    bullish_bars = 0
    lookback_bars = min(6, s_idx // 5)
    for i in range(lookback_bars):
        c_start = s_idx - (lookback_bars - i) * 5
        c_end = c_start + 5
        if c_start < 0 or c_end > len(stock_closes):
            continue
        c_open = stock_closes[c_start] if not np.isnan(stock_closes[c_start]) else 0
        c_close = stock_closes[min(c_end - 1, len(stock_closes) - 1)]
        if np.isnan(c_close):
            c_close = 0
        if c_open > 0 and c_close > 0:
            if c_close < c_open:
                bearish_bars += 1
            elif c_close > c_open:
                bullish_bars += 1

    # Underlying momentum (% change over lookback period)
    lookback_min = min(30, s_idx)
    first_price = stock_closes[s_idx - lookback_min] if not np.isnan(stock_closes[s_idx - lookback_min]) else 0
    last_price = stock_closes[s_idx] if not np.isnan(stock_closes[s_idx]) else 0
    underlying_change_pct = 0.0
    if first_price > 0 and last_price > 0:
        underlying_change_pct = (last_price - first_price) / first_price * 100

    # EMA9 vs EMA21
    ema_window = stock_closes[max(0, s_idx - 30):s_idx + 1]
    ema_valid = ema_window[~np.isnan(ema_window) & (ema_window > 0)]
    ema9 = compute_ema(ema_valid, 9) if len(ema_valid) >= 9 else None
    ema21 = compute_ema(ema_valid, 21) if len(ema_valid) >= 21 else None
    ema_bearish = ema9 is not None and ema21 is not None and ema9 < ema21
    ema_bullish = ema9 is not None and ema21 is not None and ema9 > ema21

    # Score the directional evidence (positive = bullish, negative = bearish)
    regime_score = 0.0

    # RSI contribution (-2.5 to +2.5)
    if rsi_5m is not None:
        if rsi_5m < 40:
            regime_score -= 1.5
        elif rsi_5m < 50:
            regime_score -= 0.5
        elif rsi_5m > 60:
            regime_score += 1.5
        elif rsi_5m > 50:
            regime_score += 0.5

    if rsi_15m is not None:
        if rsi_15m < 40:
            regime_score -= 1.0
        elif rsi_15m > 60:
            regime_score += 1.0

    # Candle count contribution (-1.5 to +1.5)
    total_bars = bearish_bars + bullish_bars
    if total_bars > 0:
        regime_score += (bullish_bars - bearish_bars) / total_bars * 1.5

    # Underlying momentum contribution (-2 to +2)
    regime_score += max(-2.0, min(2.0, underlying_change_pct * 4))

    # EMA trend contribution (-1 to +1)
    if ema_bearish:
        regime_score -= 1.0
    elif ema_bullish:
        regime_score += 1.0

    # Decision: CALLs blocked if score < -1.0, PUTs blocked if score > +1.0
    if is_call and regime_score < -1.0:
        return False, f"CALL blocked — bearish regime (score={regime_score:+.1f})"
    if not is_call and regime_score > 1.0:
        return False, f"PUT blocked — bullish regime (score={regime_score:+.1f})"

    return True, f"regime_ok (score={regime_score:+.1f})"


def check_put_bearish_confirm(stock_opens: np.ndarray, stock_closes: np.ndarray,
                               stock_highs: np.ndarray, stock_lows: np.ndarray,
                               stock_volumes: np.ndarray, idx: int) -> tuple[bool, str]:
    """Simulate production PutBearishConfirmGate (pipeline.py) — prod default ON.

    Requires 2 of 3 confirmations before allowing a PUT:
    1. Price below VWAP (selling pressure)
    2. RSI < 45 on 5m timeframe
    3. 4+ of the last 6 5m candles bearish

    Fail-closed like prod (no candle data → block PUT). 5m candles are
    approximated by grouping aligned 1-min RTH stock bars.
    """
    s_idx = min(idx, len(stock_closes) - 1)
    if s_idx < 1 or len(stock_closes) == 0:
        return False, "no_stock_data (fail-closed)"

    confirmations = 0

    # Check 1: price below session VWAP
    last_close = stock_closes[s_idx] if not np.isnan(stock_closes[s_idx]) else 0
    tp_num = 0.0
    tp_den = 0.0
    for i in range(s_idx + 1):
        h, lo, c, v = stock_highs[i], stock_lows[i], stock_closes[i], stock_volumes[i]
        if np.isnan(c) or np.isnan(v) or v <= 0:
            continue
        h = c if np.isnan(h) else h
        lo = c if np.isnan(lo) else lo
        tp_num += (h + lo + c) / 3 * v
        tp_den += v
    vwap = tp_num / tp_den if tp_den > 0 else 0
    if vwap > 0 and last_close > 0 and last_close < vwap:
        confirmations += 1

    # Build 5m candles from 1-min bars (complete buckets only)
    candles = []  # (open, close)
    for start in range(0, s_idx + 1 - 5, 5):
        seg_o = seg_c = np.nan
        for i in range(start, start + 5):
            o, c = stock_opens[i], stock_closes[i]
            if np.isnan(seg_o) and not np.isnan(o) and o > 0:
                seg_o = o
            if not np.isnan(c) and c > 0:
                seg_c = c
        if not np.isnan(seg_o) and not np.isnan(seg_c):
            candles.append((seg_o, seg_c))

    # Check 2: RSI < 45 on 5m closes
    closes_5m = np.array([c for _, c in candles])
    rsi_5m = compute_rsi(closes_5m) if len(closes_5m) >= 15 else None
    if rsi_5m is not None and rsi_5m < 45:
        confirmations += 1

    # Check 3: 4+ of last 6 5m candles bearish
    if len(candles) >= 6:
        bearish = sum(1 for o, c in candles[-6:] if c < o)
        if bearish >= 4:
            confirmations += 1

    if confirmations >= 2:
        return True, f"bearish_confirmed ({confirmations}/3)"
    return False, f"insufficient_bearish_confirmation ({confirmations}/2 needed)"


def check_spy_put_gate(spy_stock_closes: np.ndarray, minute: int) -> tuple[bool, bool]:
    """Check PutMarketDirectionGate using SPY stock data.

    Matches production pipeline.py logic:
    - SPY change >= 0% from open → allow PUTs (green day reversal play)
    - SPY change <= -0.5% from open → allow PUTs (bear mode)
    - Between 0% and -0.5% → block PUTs (mild red, unclear direction)
    - No data → block PUTs (fail-closed)

    Returns (allowed, is_bear_mode).
    """
    if len(spy_stock_closes) < 2 or minute >= len(spy_stock_closes):
        return False, False

    spy_open = spy_stock_closes[0]
    if spy_open <= 0 or np.isnan(spy_open):
        return False, False

    spy_now = spy_stock_closes[min(minute, len(spy_stock_closes) - 1)]
    if spy_now <= 0 or np.isnan(spy_now):
        return False, False

    spy_change = (spy_now - spy_open) / spy_open * 100

    if spy_change >= 0:
        return True, False  # Green day — PUTs allowed
    elif spy_change <= PUT_SPY_BEAR_THRESHOLD:
        return True, True   # Bear mode — PUTs allowed with expanded slots
    else:
        return False, False  # Mild red — block PUTs


def run_backtest(pattern_model, pattern_meta, entry_model, entry_features,
                 pattern_threshold: float, entry_threshold: float,
                 tickers: list[str], start_date: str, end_date: str,
                 stop_model=None, regime_model=None, regime_threshold: float = 0.0,
                 signal_model=None,
                 put_pattern_model=None, put_pattern_meta=None,
                 put_entry_model=None, put_entry_features=None, put_entry_threshold=0.80,
                 put_crash_mode: str = "none"):
    """Run the gold standard backtest."""
    p_features = pattern_meta["features"]
    # PUT model features and threshold (fall back to CALL model if no PUT model)
    put_features = put_pattern_meta["features"] if put_pattern_meta else p_features
    put_threshold = put_pattern_meta.get("best_threshold", 0.40) if put_pattern_meta else pattern_threshold
    put_model = put_pattern_model if put_pattern_model else pattern_model
    use_put_model = put_pattern_model is not None

    conn = sqlite3.connect(THETADATA_DB)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")

    dates = [r[0] for r in conn.execute("""
        SELECT DISTINCT substr(timestamp, 1, 10) FROM option_ohlc
        WHERE ticker = 'SPY' AND substr(timestamp, 1, 10) >= ? AND substr(timestamp, 1, 10) <= ?
        ORDER BY 1
    """, (start_date, end_date)).fetchall()]

    print(f"\n  Period: {dates[0]} to {dates[-1]} ({len(dates)} trading days)")
    print(f"  CALL pattern threshold: {pattern_threshold}")
    print(f"  PUT pattern: {'dedicated model (t=' + str(put_threshold) + ', ' + str(len(put_features)) + ' features)' if use_put_model else 'using CALL model (no dedicated PUT model)'}")
    print(f"  CALL scan: {SCAN_START_MIN}-{SCAN_END_MIN}min | PUT scan: {SCAN_START_MIN}-{PUT_SCAN_END_MIN}min")
    print(f"  Entry filter: {'ON (t=' + str(entry_threshold) + ')' if entry_model else 'OFF'}")
    if regime_model:
        print(f"  Regime filter: ON (threshold={regime_threshold})")
    print(f"  Tickers: {', '.join(tickers)}")
    print(f"  Portfolio: ${PORTFOLIO_START:,}")

    # Connect to UW DB for regime features
    uw_conn = None
    UW_DB = str(PROJECT_DIR / "journal" / "uw_historical.db")
    if regime_model and Path(UW_DB).exists():
        uw_conn = sqlite3.connect(UW_DB)

    stock_data_cache = {}
    prev_days_cache = {}
    regime_skipped_days = 0

    portfolio = PORTFOLIO_START
    peak_portfolio = portfolio
    max_dd = 0.0
    trades = []
    daily_pnls = {}
    per_ticker = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0, "total_contracts": 0})
    exit_reasons = defaultdict(int)
    signals_sourced = 0
    signals_pattern_pass = 0
    signals_entry_blocked = 0
    signals_gate_blocked = 0
    # Per-gate block counts for diagnostics
    gate_blocks = defaultdict(int)
    # DipConfirm tracking
    dip_confirm_stats = defaultdict(int)  # outcome → count
    dip_confirm_total_savings = 0.0
    dip_confirm_count = 0
    weekly_pnls = defaultdict(float)
    equity_curve = [(dates[0], PORTFOLIO_START)]

    for day_idx, date_str in enumerate(dates):
        day_spent = 0.0
        day_realized = 0.0
        day_cb = False
        sod_balance = portfolio
        consecutive_losses = 0          # Track consecutive losses for ConsecutiveLoserGate
        last_loss_minute = -999         # When last consecutive loser pause started
        # Track tickers that have entered today (for one-per-ticker-per-day limit)
        day_entered_tickers: set[str] = set()
        # Per-day RTH stock bar cache: ticker -> {session_minute: row}
        stock_by_minute_cache: dict[str, dict] = {}

        # Weekly loss halt (prod WEEKLY_LOSS_LIMIT_PCT=20): if this week's
        # realized P&L is already down >= 20% of the portfolio, skip the day.
        try:
            _dt = datetime.strptime(date_str, "%Y-%m-%d")
            _week_key = f"{_dt.isocalendar()[0]}-W{_dt.isocalendar()[1]:02d}"
        except ValueError:
            _week_key = None
        if _week_key is not None and weekly_pnls.get(_week_key, 0.0) < 0:
            week_loss_pct = abs(weekly_pnls[_week_key]) / max(sod_balance, 1) * 100
            if week_loss_pct >= WEEKLY_LOSS_HALT_PCT:
                gate_blocks["weekly_loss_halt_days"] += 1
                equity_curve.append((date_str, round(portfolio, 2)))
                continue

        # Regime pre-filter: skip entire day if regime model says "bad day"
        if regime_model and regime_threshold > 0:
            # Compute regime score for SPY (market-level indicator)
            regime_score = compute_regime_score(
                regime_model, "SPY", date_str, conn, uw_conn,
                stock_data_cache, prev_days_cache,
            )
            if regime_score < regime_threshold:
                regime_skipped_days += 1
                equity_curve.append((date_str, round(portfolio, 2)))
                continue

        if (day_idx + 1) % 10 == 0 or day_idx == 0:
            print(f"  [{day_idx+1}/{len(dates)}] {date_str}  portfolio=${portfolio:,.0f}  "
                  f"trades={len(trades)}  dd={max_dd:.1f}%", flush=True)

        # ── Pre-load ALL ticker data for the day ──────────────────────────
        ticker_data: dict[str, dict] = {}
        for ticker in tickers:
            atm = conn.execute("""
                SELECT oohlc.strike FROM option_ohlc oohlc
                JOIN option_greeks og ON oohlc.ticker=og.ticker AND oohlc.expiration=og.expiration
                    AND oohlc.strike=og.strike AND oohlc.right=og.right AND oohlc.timestamp=og.timestamp
                WHERE oohlc.ticker=? AND date(oohlc.timestamp)=? AND oohlc.right='CALL'
                    AND og.underlying_price > 0
                GROUP BY oohlc.strike ORDER BY MIN(ABS(og.underlying_price - oohlc.strike)) LIMIT 1
            """, (ticker, date_str)).fetchone()
            if not atm:
                continue
            strike = atm[0]

            rows = conn.execute("""
                SELECT oohlc.close, COALESCE(og.underlying_price, 0),
                       COALESCE(og.implied_vol, 0), COALESCE(oq.bid, 0), COALESCE(oq.ask, 0),
                       COALESCE(og.delta, 0), COALESCE(og.theta, 0),
                       oohlc.volume, oohlc.expiration,
                       COALESCE(og.vega, 0),
                       COALESCE(oq.bid_size, 0), COALESCE(oq.ask_size, 0),
                       oohlc.timestamp
                FROM option_ohlc oohlc
                LEFT JOIN option_quotes oq ON oohlc.ticker=oq.ticker AND oohlc.expiration=oq.expiration
                    AND oohlc.strike=oq.strike AND oohlc.right=oq.right AND oohlc.timestamp=oq.timestamp
                LEFT JOIN option_greeks og ON oohlc.ticker=og.ticker AND oohlc.expiration=og.expiration
                    AND oohlc.strike=og.strike AND oohlc.right=og.right AND oohlc.timestamp=og.timestamp
                WHERE oohlc.ticker=? AND date(oohlc.timestamp)=? AND oohlc.right='CALL' AND oohlc.strike=?
                ORDER BY oohlc.timestamp
            """, (ticker, date_str, strike)).fetchall()

            if len(rows) < 30:
                continue

            closes = np.array([float(r[0]) if r[0] else np.nan for r in rows])
            underlyings = np.array([float(r[1]) if r[1] else np.nan for r in rows])
            ivs = np.array([float(r[2]) if r[2] else np.nan for r in rows])
            bids_arr = np.array([float(r[3]) if r[3] else 0 for r in rows])
            asks_arr = np.array([float(r[4]) if r[4] else 0 for r in rows])
            deltas_arr = np.array([float(r[5]) if r[5] else np.nan for r in rows])
            thetas_arr = np.array([float(r[6]) if r[6] else np.nan for r in rows])
            volumes_arr = np.array([float(r[7]) if r[7] else 0 for r in rows])
            expiry_date = rows[0][8] if rows else date_str
            vegas_arr = np.array([float(r[9]) if r[9] else np.nan for r in rows])
            bid_sizes = np.array([float(r[10]) if r[10] else 0 for r in rows])
            ask_sizes = np.array([float(r[11]) if r[11] else 0 for r in rows])
            option_minutes = [_ts_session_minute(r[12]) for r in rows]

            # Stock data for entry_timing features — RTH only, aligned 1:1 with
            # the option series (B1 fix: previously included premarket bars and
            # was misaligned positionally).
            if ticker not in stock_by_minute_cache:
                stock_by_minute_cache[ticker] = load_stock_by_minute(conn, ticker, date_str)
            sbm = stock_by_minute_cache[ticker]
            stock_opens, stock_closes, stock_highs, stock_lows, stock_volumes = \
                align_stock_arrays(sbm, option_minutes)

            opening_price = 0
            for c in closes[:5]:
                if not np.isnan(c) and c > 0:
                    opening_price = c
                    break
            if opening_price <= 0:
                continue

            # DTE
            try:
                exp_dt = datetime.strptime(expiry_date, "%Y-%m-%d").date()
                day_dt = datetime.strptime(date_str, "%Y-%m-%d").date()
                dte = max(0, (exp_dt - day_dt).days)
            except (ValueError, TypeError):
                dte = 0

            ticker_data[ticker] = {
                "closes": closes, "underlyings": underlyings, "ivs": ivs,
                "bids": bids_arr, "asks": asks_arr, "deltas": deltas_arr,
                "thetas": thetas_arr, "volumes": volumes_arr, "vegas": vegas_arr,
                "bid_sizes": bid_sizes, "ask_sizes": ask_sizes,
                "expiry_date": expiry_date, "opening_price": opening_price,
                "dte": dte, "stock_closes": stock_closes,
                "stock_highs": stock_highs, "stock_lows": stock_lows,
                "stock_opens": stock_opens, "stock_volumes": stock_volumes,
                "n_rows": len(rows), "strike": strike,
            }

        # ── Pre-load PUT data if enabled ──────────────────────────────
        put_ticker_data: dict[str, dict] = {}
        # Multi-strike PUT data: put_all_strikes[ticker][strike] = {data dict}
        put_all_strikes: dict[str, dict[float, dict]] = {}
        if ENABLE_PUTS or PUTS_ONLY:
            # SPY stock closes for the direction gate — RTH only, indexed by
            # session minute (B1 fix: previously included premarket bars and
            # read positionally, so spy_stock_closes[0] was a 4:00 AM bar).
            if "SPY" not in stock_by_minute_cache:
                stock_by_minute_cache["SPY"] = load_stock_by_minute(conn, "SPY", date_str)
            spy_stock_closes = build_minute_indexed_closes(stock_by_minute_cache["SPY"])
            if not stock_by_minute_cache["SPY"]:
                spy_stock_closes = np.array([])

            for ticker in tickers:
                if ticker in PUT_EXCLUDED_TICKERS:
                    continue
                # RTH stock bars for this ticker (aligned per-strike below)
                if ticker not in stock_by_minute_cache:
                    stock_by_minute_cache[ticker] = load_stock_by_minute(conn, ticker, date_str)
                put_sbm = stock_by_minute_cache[ticker]

                # Load ALL available PUT strikes for this ticker (multi-strike scanning)
                all_put_strikes_rows = conn.execute("""
                    SELECT DISTINCT strike FROM option_ohlc
                    WHERE ticker=? AND date(timestamp)=? AND right='PUT'
                """, (ticker, date_str)).fetchall()
                if not all_put_strikes_rows:
                    continue
                available_strikes = sorted([r[0] for r in all_put_strikes_rows])

                # Load data for each strike
                ticker_strikes_data: dict[float, dict] = {}
                for put_strike in available_strikes:
                    put_rows = conn.execute("""
                        SELECT oohlc.close, COALESCE(og.underlying_price, 0),
                               COALESCE(og.implied_vol, 0), COALESCE(oq.bid, 0), COALESCE(oq.ask, 0),
                               COALESCE(og.delta, 0), COALESCE(og.theta, 0),
                               oohlc.volume, oohlc.expiration,
                               COALESCE(og.vega, 0),
                               COALESCE(oq.bid_size, 0), COALESCE(oq.ask_size, 0),
                               oohlc.timestamp
                        FROM option_ohlc oohlc
                        LEFT JOIN option_quotes oq ON oohlc.ticker=oq.ticker AND oohlc.expiration=oq.expiration
                            AND oohlc.strike=oq.strike AND oohlc.right=oq.right AND oohlc.timestamp=oq.timestamp
                        LEFT JOIN option_greeks og ON oohlc.ticker=og.ticker AND oohlc.expiration=og.expiration
                            AND oohlc.strike=og.strike AND oohlc.right=og.right AND oohlc.timestamp=og.timestamp
                        WHERE oohlc.ticker=? AND date(oohlc.timestamp)=? AND oohlc.right='PUT' AND oohlc.strike=?
                        ORDER BY oohlc.timestamp
                    """, (ticker, date_str, put_strike)).fetchall()

                    if len(put_rows) < 30:
                        continue

                    p_closes = np.array([float(r[0]) if r[0] else np.nan for r in put_rows])
                    p_underlyings = np.array([float(r[1]) if r[1] else np.nan for r in put_rows])
                    p_ivs = np.array([float(r[2]) if r[2] else np.nan for r in put_rows])
                    p_bids = np.array([float(r[3]) if r[3] else 0 for r in put_rows])
                    p_asks = np.array([float(r[4]) if r[4] else 0 for r in put_rows])
                    p_deltas = np.array([float(r[5]) if r[5] else np.nan for r in put_rows])
                    p_thetas = np.array([float(r[6]) if r[6] else np.nan for r in put_rows])
                    p_volumes = np.array([float(r[7]) if r[7] else 0 for r in put_rows])
                    p_expiry = put_rows[0][8] if put_rows else date_str
                    p_vegas = np.array([float(r[9]) if r[9] else np.nan for r in put_rows])
                    p_bid_sizes = np.array([float(r[10]) if r[10] else 0 for r in put_rows])
                    p_ask_sizes = np.array([float(r[11]) if r[11] else 0 for r in put_rows])
                    p_option_minutes = [_ts_session_minute(r[12]) for r in put_rows]
                    # RTH stock arrays aligned 1:1 with THIS strike's option series
                    p_stock_opens, p_stock_closes, p_stock_highs, p_stock_lows, p_stock_volumes = \
                        align_stock_arrays(put_sbm, p_option_minutes)

                    p_opening = 0
                    for c in p_closes[:5]:
                        if not np.isnan(c) and c > 0:
                            p_opening = c
                            break
                    if p_opening <= 0:
                        continue

                    try:
                        p_exp_dt = datetime.strptime(p_expiry, "%Y-%m-%d").date()
                        p_day_dt = datetime.strptime(date_str, "%Y-%m-%d").date()
                        p_dte = max(0, (p_exp_dt - p_day_dt).days)
                    except (ValueError, TypeError):
                        p_dte = 0

                    ticker_strikes_data[put_strike] = {
                        "closes": p_closes, "underlyings": p_underlyings, "ivs": p_ivs,
                        "bids": p_bids, "asks": p_asks, "deltas": p_deltas,
                        "thetas": p_thetas, "volumes": p_volumes, "vegas": p_vegas,
                        "bid_sizes": p_bid_sizes, "ask_sizes": p_ask_sizes,
                        "expiry_date": p_expiry, "opening_price": p_opening,
                        "dte": p_dte, "stock_closes": p_stock_closes,
                        "stock_highs": p_stock_highs, "stock_lows": p_stock_lows,
                        "stock_opens": p_stock_opens, "stock_volumes": p_stock_volumes,
                        "n_rows": len(put_rows), "strike": put_strike,
                        "spy_stock_closes": spy_stock_closes,
                    }

                if ticker_strikes_data:
                    put_all_strikes[ticker] = ticker_strikes_data
                    # Default: use the strike closest to open for backwards compat
                    open_und = next(iter(ticker_strikes_data.values()))["underlyings"]
                    open_price = 0
                    for u in open_und[:5]:
                        if not np.isnan(u) and u > 0:
                            open_price = u
                            break
                    if open_price > 0:
                        default_strike = min(ticker_strikes_data.keys(), key=lambda s: abs(s - open_price))
                    else:
                        default_strike = min(ticker_strikes_data.keys())
                    put_ticker_data[ticker] = ticker_strikes_data[default_strike]
        else:
            spy_stock_closes = np.array([])

        # ── Open positions: concurrent FSM tracking ──────────────────────
        # Each entry is a dict with FSM state, data refs, and accounting info.
        open_positions: list[dict] = []

        # Determine the scan range — use max data length across tickers
        all_data = list(ticker_data.values()) + list(put_ticker_data.values())
        max_data_len = max((td["n_rows"] for td in all_data), default=0) if all_data else 0
        # Scan from SCAN_START_MIN through end of day data (for exit processing)
        day_end_minute = max_data_len

        for minute in range(SCAN_START_MIN, day_end_minute, SCAN_INTERVAL):
            if day_cb:
                break

            # ── Phase 1: Step all open position FSMs forward ─────────
            closed_this_tick: list[dict] = []
            for pos in open_positions:
                td = pos["ticker_data"]
                idx = minute
                if idx >= td["n_rows"]:
                    continue  # past this ticker's data

                prem = td["closes"][idx]
                if np.isnan(prem) or prem <= 0:
                    continue

                bid = float(td["bids"][idx]) if not np.isnan(td["bids"][idx]) else prem
                ask = float(td["asks"][idx]) if not np.isnan(td["asks"][idx]) else prem
                underlying = float(td["underlyings"][idx]) if not np.isnan(td["underlyings"][idx]) else 0

                entry_ts = pos["entry_ts"]
                now = entry_ts + timedelta(minutes=(idx - pos["entry_minute"]))
                minutes_to_close = max(0, (16 * 60) - (now.hour * 60 + now.minute))

                action = pos["fsm"].evaluate(
                    pos["state"], prem, bid, ask, now,
                    current_underlying=underlying,
                    minutes_to_close=minutes_to_close,
                    candle_data={},
                )

                if action.should_exit:
                    exit_price = bid if bid > 0 else prem

                    if action.contracts_to_close > 0 and action.contracts_to_close < pos["remaining"]:
                        # Partial close (scaleout)
                        pos["locked_pnl"] += (exit_price - pos["effective_entry"]) * action.contracts_to_close * 100
                        pos["remaining"] -= action.contracts_to_close
                        pos["state"].contracts = pos["remaining"]
                        continue

                    # Full close
                    elapsed = idx - pos["entry_minute"]
                    peak_gain = (pos["state"].peak_premium - pos["effective_entry"]) / pos["effective_entry"] * 100
                    trade_pnl = pos["locked_pnl"] + (exit_price - pos["effective_entry"]) * pos["remaining"] * 100
                    pos["result"] = {
                        "pnl": trade_pnl, "reason": action.reason.value,
                        "hold_min": elapsed, "peak_gain": peak_gain,
                        "exit_prem": exit_price,
                    }
                    closed_this_tick.append(pos)
                    continue

                # ── V6 DCA: auto-add when premium dips 15-35% from entry ──
                # Simulated INSIDE the stepping loop with decision-time data
                # only. (Bug fix 2026-06-10: the old block scanned 8-20 min
                # into the future at entry time and blended from minute 0.)
                # Prod parity: whitelist {IWM,SPY,QQQ,AMZN,NVDA}, window
                # 8-20 min, dip 15-35%, underlying-against <= 0.5%, add capped
                # at 10% of balance (MAX_DCA_POSITION_PCT=10). CALLs only.
                if (
                    pos["direction"] == "call"
                    and not pos["dca_triggered"]
                    and pos["ticker"] in DCA_TICKERS
                ):
                    elapsed_min = idx - pos["entry_minute"]
                    if DCA_MIN_MINUTES <= elapsed_min <= DCA_MAX_MINUTES:
                        dip_pct = (pos["entry_premium"] - prem) / pos["entry_premium"] * 100
                        dca_ask = ask if ask > 0 else prem
                        if DCA_MIN_DIP_PCT <= dip_pct <= DCA_MAX_DIP_PCT and dca_ask > 0:
                            und_entry = pos["state"].entry_underlying_price
                            und_ok = True
                            if und_entry > 0 and underlying > 0:
                                und_change = abs(underlying / und_entry - 1) * 100
                                und_ok = und_change <= DCA_MAX_UNDERLYING_AGAINST_PCT
                            if und_ok:
                                # Fill at THIS minute's ask (+ slippage)
                                dca_fill = dca_ask * (1 + ENTRY_SLIPPAGE_PCT / 100)
                                dca_cost_per = dca_fill * 100
                                dca_budget = portfolio * MAX_DCA_POSITION_PCT
                                dca_ct = min(pos["remaining"], int(dca_budget / dca_cost_per)) if dca_cost_per > 0 else 0
                                gfv_limit = sod_balance * (1 - GFV_BUFFER_PCT / 100)
                                gfv_ct = int(max(0.0, gfv_limit - day_spent) / dca_cost_per) if dca_cost_per > 0 else 0
                                dca_ct = min(dca_ct, gfv_ct)
                                if dca_ct >= 1:
                                    day_spent += dca_ct * dca_cost_per
                                    old_ct = pos["remaining"]
                                    new_ct = old_ct + dca_ct
                                    blended = (pos["effective_entry"] * old_ct + dca_fill * dca_ct) / new_ct
                                    pos["effective_entry"] = blended
                                    pos["dca_contracts"] = dca_ct
                                    pos["effective_contracts"] += dca_ct
                                    pos["remaining"] = new_ct
                                    pos["dca_triggered"] = True
                                    # Update the live FSM state (blended entry)
                                    pos["state"].contracts = new_ct
                                    pos["state"].entry_premium = blended

            # Also check for EOD close on positions past their data
            for pos in open_positions:
                if pos in closed_this_tick:
                    continue
                td = pos["ticker_data"]
                if minute >= td["n_rows"] - 1 and "result" not in pos:
                    # EOD close — fill at BID (we're selling; close/mid is a leak)
                    last_valid = pos["effective_entry"]
                    for i in range(td["n_rows"] - 1, pos["entry_minute"], -1):
                        b = td["bids"][i] if i < len(td["bids"]) else np.nan
                        if not np.isnan(b) and b > 0:
                            last_valid = float(b)
                            break
                        if not np.isnan(td["closes"][i]) and td["closes"][i] > 0:
                            last_valid = td["closes"][i]
                            break
                    elapsed = td["n_rows"] - pos["entry_minute"]
                    peak_gain = (pos["state"].peak_premium - pos["effective_entry"]) / pos["effective_entry"] * 100
                    trade_pnl = pos["locked_pnl"] + (last_valid - pos["effective_entry"]) * pos["remaining"] * 100
                    pos["result"] = {
                        "pnl": trade_pnl, "reason": "eod_data_end",
                        "hold_min": elapsed, "peak_gain": peak_gain,
                        "exit_prem": last_valid,
                    }
                    closed_this_tick.append(pos)

            # Process closed positions — update portfolio, stats
            for pos in closed_this_tick:
                result = pos["result"]
                trade_pnl = result["pnl"]
                portfolio += trade_pnl
                is_win = trade_pnl > 0
                tk = pos["ticker"]

                per_ticker[tk]["trades"] += 1
                if is_win:
                    per_ticker[tk]["wins"] += 1
                per_ticker[tk]["pnl"] += trade_pnl
                per_ticker[tk]["total_contracts"] += pos["effective_contracts"]
                exit_reasons[result["reason"]] += 1

                trades.append({
                    "day": date_str, "ticker": tk, "minute": pos["entry_minute"],
                    "direction": pos["direction"],
                    "signal_minute": pos.get("signal_minute", pos["entry_minute"]),
                    "entry": pos["entry_premium"], "effective_entry": round(pos["effective_entry"], 2),
                    "contracts": pos["contracts"], "dca_contracts": pos.get("dca_contracts", 0),
                    "effective_contracts": pos["effective_contracts"],
                    "pnl": round(trade_pnl, 2), "reason": result["reason"],
                    "hold_min": result["hold_min"],
                    "peak_gain": round(result.get("peak_gain", 0), 1),
                    "pattern_conf": round(pos["pattern_conf"], 3),
                    "dte": int(pos["ticker_data"].get("dte", 0)),
                    "exit_prem": round(result.get("exit_prem", 0), 2),
                    "dca": pos.get("dca_triggered", False),
                    "signal_quality": round(pos["signal_quality"], 1) if pos.get("signal_quality") is not None else None,
                    "dip_confirm": pos.get("dip_confirm", "off"),
                    "dip_savings": round(pos.get("dip_savings", 0), 2),
                    "is_bear_mode": pos.get("is_bear_mode", False),
                })

                if date_str not in daily_pnls:
                    daily_pnls[date_str] = 0
                daily_pnls[date_str] += trade_pnl

                try:
                    dt = datetime.strptime(date_str, "%Y-%m-%d")
                    week_key = f"{dt.isocalendar()[0]}-W{dt.isocalendar()[1]:02d}"
                    weekly_pnls[week_key] += trade_pnl
                except ValueError:
                    pass

                # Circuit breaker
                day_realized += trade_pnl
                if day_realized < 0:
                    loss_pct = abs(day_realized) / sod_balance * 100
                    if loss_pct >= DAILY_LOSS_CB_PCT:
                        day_cb = True

                # Consecutive-loss tracking (feeds ConsecutiveLoserGate — prod parity).
                # The old backtest-only "bad-day threshold raise to 0.90" mechanic
                # was removed 2026-06-10: prod uses this circuit breaker instead.
                if trade_pnl <= 0:
                    consecutive_losses += 1
                    last_loss_minute = minute
                else:
                    consecutive_losses = 0

                # Drawdown
                if portfolio > peak_portfolio:
                    peak_portfolio = portfolio
                dd = (peak_portfolio - portfolio) / peak_portfolio * 100
                if dd > max_dd:
                    max_dd = dd

            # Remove closed positions
            open_positions = [p for p in open_positions if p not in closed_this_tick]

            # ── Phase 2: Scan tickers for new entries ────────────────
            # Opening buffer (prod CB_OPENING_BUFFER_MINUTES=10): no entries
            # during the first 10 minutes after open.
            call_scan_open = OPENING_BUFFER_MIN <= minute <= SCAN_END_MIN
            put_scan_open = OPENING_BUFFER_MIN <= minute <= PUT_SCAN_END_MIN

            # Build current open set from live positions
            current_open_tickers = {p["ticker"] for p in open_positions}
            current_open_dirs = [p["direction"] for p in open_positions]

            # ── Phase 2a: CALL entries (skip if --puts-only or past CALL scan window) ──
            for ticker in (tickers if not PUTS_ONLY and call_scan_open else []):
                if day_cb:
                    break
                if len(open_positions) >= MAX_CONCURRENT:
                    break
                # Skip if already have an open position on this ticker
                if ticker in current_open_tickers:
                    continue
                # Skip if already entered this ticker today (one entry per ticker per day)
                if ticker in day_entered_tickers:
                    continue
                if ticker not in ticker_data:
                    continue
                # (Index correlation guard MAX_INDEX_CONCURRENT removed 2026-06-10:
                #  prod uses CorrelationCapGate instead, enabled below.)

                td = ticker_data[ticker]
                if minute >= td["n_rows"]:
                    continue

                signals_sourced += 1

                # Step 1: Pattern model (sourcing)
                feat = compute_pattern_features(
                    td["closes"], td["volumes"], td["ivs"], td["deltas"], td["thetas"],
                    td["underlyings"], td["bids"], td["asks"], minute, td["opening_price"],
                )
                if feat is None:
                    continue

                X_pattern = np.array([[feat.get(f, 0) for f in p_features]], dtype=np.float32)
                pattern_conf = pattern_model.predict(X_pattern)[0]

                if pattern_conf < pattern_threshold:
                    continue

                # Score floor (prod ScoreGate MIN_SCORE=75 / vinny floor 75)
                score = int(pattern_conf * 100)
                if score < MIN_SCORE:
                    signals_gate_blocked += 1
                    gate_blocks["score_floor"] += 1
                    continue

                # TimeOfDayGate parity: before 9:45 ET (minute < 15) require score >= 85
                if minute < TOD_EARLY_CUTOFF_MIN and score < TOD_EARLY_MIN_SCORE:
                    signals_gate_blocked += 1
                    gate_blocks["tod_early"] += 1
                    continue

                signals_pattern_pass += 1

                # Step 2: Entry timing model (quality gate)
                et_feat = None
                if entry_model and entry_features:
                    et_feat = compute_entry_timing_features(
                        td["closes"], td["volumes"], td["bids"], td["asks"], td["bid_sizes"],
                        td["ask_sizes"], td["ivs"], td["deltas"], td["thetas"], td["vegas"],
                        td["underlyings"], td["stock_closes"], td["stock_highs"], td["stock_lows"],
                        minute, entry_features,
                    )
                    if et_feat is not None:
                        X_entry = np.array([[et_feat.get(f, 0) for f in entry_features]], dtype=np.float32)
                        entry_conf = entry_model.predict(X_entry)[0]
                        if entry_conf < entry_threshold:
                            signals_entry_blocked += 1
                            continue

                # Step 2.5: Signal quality ranking score
                signal_quality_score = None
                if signal_model is not None:
                    sq_features = signal_model.feature_name()
                    sq_feat = compute_signal_quality_features(
                        td["closes"], td["volumes"], td["bids"], td["asks"], td["bid_sizes"],
                        td["ask_sizes"], td["ivs"], td["deltas"], td["thetas"], td["vegas"],
                        td["underlyings"], td["stock_closes"], td["stock_highs"], td["stock_lows"],
                        minute, sq_features,
                    )
                    if sq_feat is not None:
                        X_sq = np.array([[sq_feat.get(f, 0) for f in sq_features]], dtype=np.float32)
                        signal_quality_score = float(signal_model.predict(X_sq)[0])

                # Step 3: Entry gates
                entry_premium = float(td["asks"][minute]) if td["asks"][minute] > 0 else float(td["closes"][minute])
                if entry_premium <= 0 or np.isnan(entry_premium):
                    continue

                # Principled gates (always active in prod): min_premium + cap + spread
                if entry_premium < MIN_PREMIUM_FLOOR:
                    signals_gate_blocked += 1
                    gate_blocks["min_premium"] += 1
                    continue

                # Unconditional $6 premium cap — prod ml_pipeline rejects
                # premium > $6.0 at the SIGNAL level regardless of
                # ENABLE_V6_PREMIUM_CAP (bot_runner.py:990).
                if entry_premium > PREMIUM_CAP:
                    signals_gate_blocked += 1
                    gate_blocks["premium_cap_signal"] += 1
                    continue

                bid_val = float(td["bids"][minute]) if td["bids"][minute] > 0 else 0
                if bid_val > 0 and entry_premium > 0:
                    spread = (entry_premium - bid_val) / entry_premium * 100
                    if spread > SPREAD_GATE_PCT:
                        signals_gate_blocked += 1
                        gate_blocks["spread_gate"] += 1
                        continue

                # Static gates (disabled in prod, overridden by delta gate)
                if ENABLE_PRICE_GATES:
                    if entry_premium > PREMIUM_CAP:
                        signals_gate_blocked += 1
                        gate_blocks["premium_cap"] += 1
                        continue

                    # OTM distance gate: per-ticker dollar threshold (V2)
                    und_at_entry = float(td["underlyings"][minute]) if not np.isnan(td["underlyings"][minute]) else 0
                    if und_at_entry > 0 and td["strike"] > 0:
                        call_otm_dollars = td["strike"] - und_at_entry
                        max_otm_dollars = get_max_otm_distance(ticker)
                        if call_otm_dollars > max_otm_dollars:
                            signals_gate_blocked += 1
                            gate_blocks["otm_distance"] += 1
                            continue

                # Delta gate (dynamic alternative to premium_cap + otm_distance)
                if ENABLE_DELTA_GATE and "deltas" in td:
                    delta_val = abs(float(td["deltas"][minute])) if not np.isnan(td["deltas"][minute]) else 0
                    if delta_val > 0:
                        if delta_val < DELTA_MIN:
                            signals_gate_blocked += 1
                            gate_blocks["delta_too_low"] += 1
                            continue
                        if delta_val > DELTA_MAX:
                            signals_gate_blocked += 1
                            gate_blocks["delta_too_high"] += 1
                            continue

                direction = "call"
                same_dir = sum(1 for d in current_open_dirs if d == direction)
                if same_dir >= MAX_SAME_DIRECTION:
                    signals_gate_blocked += 1
                    gate_blocks["same_direction"] += 1
                    continue

                # ── Production gates ─────────────────────────────────
                if ENABLE_AFTERNOON_DANGER and AFTERNOON_DANGER_START <= minute <= AFTERNOON_DANGER_END:
                    signals_gate_blocked += 1
                    gate_blocks["afternoon_danger"] += 1
                    continue

                if ENABLE_HARD_CUTOFF and minute >= HARD_CUTOFF_MIN:
                    signals_gate_blocked += 1
                    gate_blocks["hard_cutoff"] += 1
                    continue

                if ENABLE_CONSECUTIVE_LOSER and consecutive_losses >= CONSECUTIVE_LOSER_MAX:
                    if minute - last_loss_minute < CONSECUTIVE_LOSER_PAUSE_MIN:
                        signals_gate_blocked += 1
                        gate_blocks["consecutive_loser"] += 1
                        continue
                    else:
                        consecutive_losses = 0

                if ENABLE_CORRELATION_CAP:
                    group = _group_for_ticker(ticker)
                    if group is not None:
                        # Prod CorrelationCapGate counts OPEN positions in the
                        # same group + direction (not day-cumulative entries)
                        same_group_open = sum(
                            1 for p in open_positions
                            if p["direction"] == direction and _group_for_ticker(p["ticker"]) == group
                        )
                        if same_group_open >= CORRELATION_CAP_MAX_PER_GROUP:
                            signals_gate_blocked += 1
                            gate_blocks["correlation_cap"] += 1
                            continue

                if ENABLE_ANTI_CHASE and minute >= 5 and len(td["underlyings"]) > minute:
                    und_now = td["underlyings"][minute]
                    und_5ago = td["underlyings"][minute - 5]
                    if not np.isnan(und_now) and not np.isnan(und_5ago) and und_5ago > 0:
                        und_move = abs(und_now / und_5ago - 1) * 100
                        if und_move > ANTI_CHASE_MAX_MOVE_PCT:
                            signals_gate_blocked += 1
                            gate_blocks["anti_chase"] += 1
                            continue

                if ENABLE_MOMENTUM_CONFIRM and len(td["stock_closes"]) > 15:
                    mom_passed, _mom_reason = check_momentum_confirm(
                        td["stock_closes"], td["stock_highs"], td["stock_lows"], minute, is_call=True,
                    )
                    if not mom_passed:
                        signals_gate_blocked += 1
                        gate_blocks["momentum_confirm"] += 1
                        continue

                if ENABLE_DIRECTIONAL_REGIME and len(td["stock_closes"]) > 20:
                    regime_passed, _regime_reason = check_directional_regime(
                        td["stock_closes"], minute, is_call=True,
                    )
                    if not regime_passed:
                        signals_gate_blocked += 1
                        gate_blocks["directional_regime"] += 1
                        continue
                # ── End production gates ──────────────────────────────

                # ── DipConfirm: simulate waiting for a cheaper entry ──
                dip_entry_minute = minute  # effective entry minute (may be delayed)
                dip_savings = 0.0
                dip_outcome = "off"
                if ENABLE_DIP_CONFIRM:
                    dc_prem, dc_delay, dc_savings, dc_outcome = simulate_dip_confirm(
                        td["asks"], td["bids"], td["closes"], minute, td["n_rows"],
                    )
                    dip_outcome = dc_outcome
                    if dc_outcome == "timeout_skip":
                        signals_gate_blocked += 1
                        gate_blocks["dip_confirm_skip"] += 1
                        continue
                    if dc_prem > 0 and not np.isnan(dc_prem):
                        entry_premium = dc_prem
                        dip_entry_minute = minute + dc_delay
                        dip_savings = dc_savings
                        dip_confirm_stats[dc_outcome] += 1
                        dip_confirm_total_savings += dc_savings
                        dip_confirm_count += 1
                # ── End DipConfirm ────────────────────────────────────

                # Entry slippage (B4): prod fills via ask+5% limit; realized
                # fill averages ~ask + 50bps. Apply to the recorded entry.
                entry_premium = entry_premium * (1 + ENTRY_SLIPPAGE_PCT / 100)
                cost_per = entry_premium * 100

                gfv_limit = sod_balance * (1 - GFV_BUFFER_PCT / 100)
                gfv_remaining = gfv_limit - day_spent
                if gfv_remaining < cost_per:
                    signals_gate_blocked += 1
                    gate_blocks["gfv_limit"] += 1
                    continue

                # Position sizing: dispatched via size_position() so the sweep
                # can swap the confidence→budget curve and the multi-day cap.
                # SIZING_MODE="current" reproduces prod score_to_contracts().
                contracts = size_position(
                    score, cost_per, portfolio, float(pattern_conf),
                    is_put=False, dte=int(td.get("dte", 0)), minute=dip_entry_minute,
                )
                if contracts <= 0:
                    signals_gate_blocked += 1
                    gate_blocks["sizing_rejected"] += 1
                    continue
                # GFV cap (cash-account settled-funds simulation — kept)
                gfv_ct = int(gfv_remaining / cost_per) if cost_per > 0 else 1
                contracts = max(1, min(contracts, gfv_ct))

                trade_cost = contracts * cost_per
                day_spent += trade_cost

                # DCA fields — DCA is now simulated inside the position-stepping
                # loop with data available at decision time. (Bug fix 2026-06-10:
                # the old block scanned 8-20 min into the FUTURE at entry time.)
                dca_contracts = 0
                effective_entry = entry_premium
                effective_contracts = contracts
                dca_triggered = False

                # Create FSM for this position
                ml_stop = None
                if stop_model is not None and entry_model and entry_features and et_feat is not None:
                    stop_features = stop_model.feature_name()
                    X_stop = np.array([[et_feat.get(f, 0) for f in stop_features]], dtype=np.float32)
                    ml_stop = float(stop_model.predict(X_stop)[0])

                tcfg = get_ticker_config(ticker, use_per_ticker=True)
                from dataclasses import replace as dc_replace
                if GRACE_OVERRIDE is not None:
                    tcfg = dc_replace(tcfg, grace_period_min=GRACE_OVERRIDE)
                if ml_stop is not None:
                    clamped = max(15.0, min(55.0, ml_stop))
                    tcfg = dc_replace(tcfg,
                                      tight_stop_0dte_pct=clamped,
                                      backstop_0dte_pct=min(clamped + 20, 65.0))
                tcfg, _settings = _apply_exit_overrides(tcfg, _V6_SETTINGS)
                fsm = ExitFSM(tcfg, settings=_settings)

                entry_ts = datetime(2026, 1, 1, 9, 30) + timedelta(minutes=dip_entry_minute)

                underlying_0 = 0
                for i in range(dip_entry_minute, min(dip_entry_minute + 5, len(td["underlyings"]))):
                    u = td["underlyings"][i]
                    if not np.isnan(u) and u > 0:
                        underlying_0 = float(u)
                        break

                state = TradeState(
                    trade_id=len(trades) + 1, ticker=ticker, option_type="call",
                    entry_premium=effective_entry, entry_time=entry_ts,
                    contracts=effective_contracts, peak_premium=effective_entry,
                    entry_underlying_price=underlying_0,
                    dte=td["dte"], expiry_date=td["expiry_date"] or "",
                )

                open_positions.append({
                    "ticker": ticker, "direction": direction,
                    "fsm": fsm, "state": state,
                    "entry_minute": dip_entry_minute, "signal_minute": minute,
                    "entry_ts": entry_ts,
                    "entry_premium": entry_premium,
                    "effective_entry": effective_entry,
                    "contracts": contracts, "dca_contracts": dca_contracts,
                    "effective_contracts": effective_contracts,
                    "dca_triggered": dca_triggered,
                    "locked_pnl": 0.0, "remaining": effective_contracts,
                    "ticker_data": td,
                    "pattern_conf": pattern_conf,
                    "signal_quality": signal_quality_score,
                    "dip_confirm": dip_outcome,
                    "dip_savings": dip_savings,
                })

                day_entered_tickers.add(ticker)
                current_open_tickers.add(ticker)
                current_open_dirs.append(direction)

            # ── Phase 2b: Scan for PUT entries (all day, not just morning) ──
            if (ENABLE_PUTS or PUTS_ONLY) and put_scan_open and put_ticker_data and len(spy_stock_closes) > 0:
                # Check SPY direction gate for this minute
                put_allowed, is_bear = check_spy_put_gate(spy_stock_closes, minute)
                if put_allowed:
                    put_max = PUT_BEAR_MAX_CONCURRENT if is_bear else PUT_MAX_CONCURRENT
                    current_put_count = sum(1 for p in open_positions if p["direction"] == "put")

                    for ticker in tickers:
                        if day_cb:
                            break
                        if len(open_positions) >= MAX_CONCURRENT:
                            break
                        if current_put_count >= put_max:
                            break
                        if ticker in PUT_EXCLUDED_TICKERS:
                            continue
                        if ticker in current_open_tickers:
                            continue
                        # Allow re-entry as PUT on a DIFFERENT strike (multi-strike).
                        # Block re-entry on the same strike same day.
                        if ticker not in put_ticker_data:
                            continue
                        # Allow re-entry as PUT if entered as CALL today, but not if entered as PUT
                        put_day_key = f"{ticker}_put"
                        if put_day_key in day_entered_tickers:
                            continue

                        ptd = put_ticker_data[ticker]
                        if minute >= ptd["n_rows"]:
                            continue

                        # PUT direction trigger (prod bot_runner.py:485):
                        # only consider PUT when the ticker's underlying is
                        # down more than 0.15% from its open.
                        put_und_open = 0.0
                        for u in ptd["underlyings"][:5]:
                            if not np.isnan(u) and u > 0:
                                put_und_open = float(u)
                                break
                        put_und_now = float(ptd["underlyings"][minute]) if not np.isnan(ptd["underlyings"][minute]) else 0.0
                        if put_und_open <= 0 or put_und_now <= 0:
                            continue
                        und_move_pct = (put_und_now - put_und_open) / put_und_open * 100
                        if und_move_pct >= PUT_DIRECTION_TRIGGER_PCT:
                            continue

                        signals_sourced += 1

                        # Step 1: PUT pattern model (dedicated or fallback to CALL)
                        if use_put_model:
                            # Cross-chain data from CALL side for iv_skew, put_call_volume_ratio
                            call_td = ticker_data.get(ticker)
                            call_ivs_arr = call_td["ivs"] if call_td else None
                            call_vols_arr = call_td["volumes"] if call_td else None
                            feat = compute_put_pattern_features(
                                ptd["closes"], ptd["volumes"], ptd["ivs"], ptd["deltas"], ptd["thetas"],
                                ptd["underlyings"], ptd["bids"], ptd["asks"], minute, ptd["opening_price"],
                                vegas=ptd["vegas"], bid_sizes=ptd["bid_sizes"], ask_sizes=ptd["ask_sizes"],
                                call_ivs=call_ivs_arr, call_volumes=call_vols_arr,
                            )
                        else:
                            feat = compute_pattern_features(
                                ptd["closes"], ptd["volumes"], ptd["ivs"], ptd["deltas"], ptd["thetas"],
                                ptd["underlyings"], ptd["bids"], ptd["asks"], minute, ptd["opening_price"],
                            )
                        if feat is None:
                            continue

                        X_pattern = np.array([[feat.get(f, 0) for f in put_features]], dtype=np.float32)
                        pattern_conf = put_model.predict(X_pattern)[0]

                        if pattern_conf < put_threshold:
                            continue

                        # Map PUT confidence to score like prod (bot_runner):
                        # threshold → 78, conf 1.0 → 100 (always >= MIN_SCORE)
                        if use_put_model and put_threshold < 1.0:
                            put_score = int(78 + (pattern_conf - put_threshold) / (1.0 - put_threshold) * 22)
                            put_score = max(0, min(100, put_score))
                        else:
                            put_score = int(pattern_conf * 100)
                        if put_score < MIN_SCORE:
                            signals_gate_blocked += 1
                            gate_blocks["score_floor"] += 1
                            continue

                        # TimeOfDayGate parity: before 9:45 ET require score >= 85
                        if minute < TOD_EARLY_CUTOFF_MIN and put_score < TOD_EARLY_MIN_SCORE:
                            signals_gate_blocked += 1
                            gate_blocks["tod_early"] += 1
                            continue

                        signals_pattern_pass += 1

                        # Step 2: Entry timing model for PUTs.
                        # Use dedicated PUT entry timing model if available,
                        # otherwise skip entry timing for PUTs (CALL model doesn't fit).
                        #
                        # Crash mode strategies can bypass or loosen this gate:
                        #   "none"     — standard threshold
                        #   "crash"    — skip entry timing when SPY drops >0.5% in last 30min
                        #   "dynamic"  — scale threshold down proportional to SPY drop speed
                        #   "regime"   — skip entry timing when regime classifier says BEARISH
                        skip_put_entry_timing = False
                        effective_put_entry_threshold = put_entry_threshold

                        if put_crash_mode == "crash" and len(spy_stock_closes) > 0:
                            # Crash override: skip entry timing if SPY dropped >0.5% in last 30min
                            lookback = min(30, minute)
                            if lookback > 0 and minute < len(spy_stock_closes):
                                spy_now = spy_stock_closes[min(minute, len(spy_stock_closes) - 1)]
                                spy_back = spy_stock_closes[max(0, minute - lookback)]
                                if spy_back > 0 and not np.isnan(spy_back) and not np.isnan(spy_now):
                                    spy_30m_change = (spy_now - spy_back) / spy_back * 100
                                    if spy_30m_change <= -0.5:
                                        skip_put_entry_timing = True

                        elif put_crash_mode == "dynamic" and len(spy_stock_closes) > 0:
                            # Dynamic threshold: lower threshold proportional to SPY drop
                            lookback = min(30, minute)
                            if lookback > 0 and minute < len(spy_stock_closes):
                                spy_now = spy_stock_closes[min(minute, len(spy_stock_closes) - 1)]
                                spy_back = spy_stock_closes[max(0, minute - lookback)]
                                if spy_back > 0 and not np.isnan(spy_back) and not np.isnan(spy_now):
                                    spy_30m_change = (spy_now - spy_back) / spy_back * 100
                                    # Scale: at -0.5% SPY drop → threshold drops to 0.70
                                    #         at -1.0% SPY drop → threshold drops to 0.55
                                    #         at -1.5%+ → threshold drops to 0.40
                                    if spy_30m_change <= -0.3:
                                        drop_factor = min(abs(spy_30m_change), 1.5) / 1.5
                                        effective_put_entry_threshold = max(0.40, put_entry_threshold - drop_factor * 0.45)

                        elif put_crash_mode == "regime":
                            # Regime-aware: skip entry timing on bearish days
                            if is_bear:
                                skip_put_entry_timing = True

                        if not skip_put_entry_timing and put_entry_model and put_entry_features:
                            et_feat = compute_entry_timing_features(
                                ptd["closes"], ptd["volumes"], ptd["bids"], ptd["asks"], ptd["bid_sizes"],
                                ptd["ask_sizes"], ptd["ivs"], ptd["deltas"], ptd["thetas"], ptd["vegas"],
                                ptd["underlyings"], ptd["stock_closes"], ptd["stock_highs"], ptd["stock_lows"],
                                minute, put_entry_features,
                            )
                            if et_feat is not None:
                                X_entry = np.array([[et_feat.get(f, 0) for f in put_entry_features]], dtype=np.float32)
                                entry_conf = put_entry_model.predict(X_entry)[0]
                                if entry_conf < effective_put_entry_threshold:
                                    signals_entry_blocked += 1
                                    continue

                        # Step 3: Entry gates (same as CALLs)
                        entry_premium = float(ptd["asks"][minute]) if ptd["asks"][minute] > 0 else float(ptd["closes"][minute])
                        if entry_premium <= 0 or np.isnan(entry_premium):
                            continue

                        # Principled gates (always active in prod): min_premium + cap + spread
                        if entry_premium < MIN_PREMIUM_FLOOR:
                            signals_gate_blocked += 1
                            gate_blocks["min_premium"] += 1
                            continue

                        # Unconditional $6 premium cap — prod ml_pipeline rejects
                        # premium > $6.0 at the SIGNAL level for all directions
                        # regardless of ENABLE_V6_PREMIUM_CAP (bot_runner.py:990).
                        if entry_premium > PREMIUM_CAP:
                            signals_gate_blocked += 1
                            gate_blocks["premium_cap_signal"] += 1
                            continue

                        bid_val = float(ptd["bids"][minute]) if ptd["bids"][minute] > 0 else 0
                        if bid_val > 0 and entry_premium > 0:
                            spread = (entry_premium - bid_val) / entry_premium * 100
                            if spread > SPREAD_GATE_PCT:
                                signals_gate_blocked += 1
                                gate_blocks["spread_gate"] += 1
                                continue

                        # Static gates (disabled in prod, overridden by delta gate)
                        if ENABLE_PRICE_GATES:
                            # PUT premium cap: raise in bear mode (crash days have inflated IV)
                            put_premium_cap = PREMIUM_CAP * 2.0 if is_bear else PREMIUM_CAP
                            if entry_premium > put_premium_cap:
                                signals_gate_blocked += 1
                                gate_blocks["premium_cap"] += 1
                                continue

                            # OTM distance gate for PUTs (V2: per-ticker dollar threshold)
                            # In bear mode, widen by 2x — underlying dropping fast, OTM becomes ATM quickly
                            put_und_at_entry = float(ptd["underlyings"][minute]) if not np.isnan(ptd["underlyings"][minute]) else 0
                            if put_und_at_entry > 0 and ptd["strike"] > 0:
                                put_otm_dollars = put_und_at_entry - ptd["strike"]
                                put_max_otm = get_max_otm_distance(ticker)
                                if is_bear:
                                    put_max_otm *= 2.0
                                if put_otm_dollars > put_max_otm:
                                    signals_gate_blocked += 1
                                    gate_blocks["otm_distance"] += 1
                                    continue

                        # Delta gate for PUTs
                        if ENABLE_DELTA_GATE and "deltas" in ptd:
                            delta_val = abs(float(ptd["deltas"][minute])) if not np.isnan(ptd["deltas"][minute]) else 0
                            if delta_val > 0:
                                if delta_val < DELTA_MIN:
                                    signals_gate_blocked += 1
                                    gate_blocks["delta_too_low"] += 1
                                    continue
                                if delta_val > DELTA_MAX:
                                    signals_gate_blocked += 1
                                    gate_blocks["delta_too_high"] += 1
                                    continue

                        # Afternoon danger: apply to PUTs during normal mode,
                        # but skip during bear mode (crash days need afternoon PUT entries)
                        if ENABLE_AFTERNOON_DANGER and AFTERNOON_DANGER_START <= minute <= AFTERNOON_DANGER_END:
                            if not is_bear:
                                signals_gate_blocked += 1
                                gate_blocks["afternoon_danger"] += 1
                                continue

                        if ENABLE_HARD_CUTOFF and minute >= HARD_CUTOFF_MIN:
                            signals_gate_blocked += 1
                            gate_blocks["hard_cutoff"] += 1
                            continue

                        # PutBearishConfirm (prod default ON): require 2 of 3 —
                        # VWAP breakdown, RSI<45 (5m), 4+/6 bearish 5m candles.
                        # Fail-closed like prod when stock data is missing.
                        if ENABLE_PUT_BEARISH_CONFIRM:
                            if "stock_opens" not in ptd or len(ptd.get("stock_closes", [])) == 0:
                                print(f"  WARNING: {ticker} {date_str} PUT bearish confirm "
                                      f"has no aligned stock data — blocking (fail-closed)")
                                signals_gate_blocked += 1
                                gate_blocks["put_bearish_confirm"] += 1
                                continue
                            pbc_ok, _pbc_reason = check_put_bearish_confirm(
                                ptd["stock_opens"], ptd["stock_closes"],
                                ptd["stock_highs"], ptd["stock_lows"],
                                ptd["stock_volumes"], minute,
                            )
                            if not pbc_ok:
                                signals_gate_blocked += 1
                                gate_blocks["put_bearish_confirm"] += 1
                                continue

                        # DipConfirm for PUTs
                        dip_entry_minute = minute
                        dip_savings = 0.0
                        dip_outcome = "off"
                        if ENABLE_DIP_CONFIRM:
                            dc_prem, dc_delay, dc_savings, dc_outcome = simulate_dip_confirm(
                                ptd["asks"], ptd["bids"], ptd["closes"], minute, ptd["n_rows"],
                            )
                            dip_outcome = dc_outcome
                            if dc_outcome == "timeout_skip":
                                signals_gate_blocked += 1
                                gate_blocks["dip_confirm_skip"] += 1
                                continue
                            if dc_prem > 0 and not np.isnan(dc_prem):
                                entry_premium = dc_prem
                                dip_entry_minute = minute + dc_delay
                                dip_savings = dc_savings
                                dip_confirm_stats[dc_outcome] += 1
                                dip_confirm_total_savings += dc_savings
                                dip_confirm_count += 1

                        # Entry slippage (B4): prod fills via ask+5% limit;
                        # realized fill averages ~ask + 50bps.
                        entry_premium = entry_premium * (1 + ENTRY_SLIPPAGE_PCT / 100)
                        cost_per = entry_premium * 100

                        gfv_limit = sod_balance * (1 - GFV_BUFFER_PCT / 100)
                        gfv_remaining = gfv_limit - day_spent
                        if gfv_remaining < cost_per:
                            signals_gate_blocked += 1
                            gate_blocks["gfv_limit"] += 1
                            continue

                        # Position sizing via size_position() (PUT half-size
                        # budget baked into the dispatcher). Mode-aware + caps.
                        contracts = size_position(
                            put_score, cost_per, portfolio, float(pattern_conf),
                            is_put=True, dte=int(ptd.get("dte", 0)), minute=dip_entry_minute,
                        )
                        if contracts <= 0:
                            signals_gate_blocked += 1
                            gate_blocks["sizing_rejected"] += 1
                            continue
                        gfv_ct = int(gfv_remaining / cost_per) if cost_per > 0 else 1
                        contracts = max(1, min(contracts, gfv_ct))

                        trade_cost = contracts * cost_per
                        day_spent += trade_cost

                        # NO DCA for PUTs (production rule)
                        effective_entry = entry_premium
                        effective_contracts = contracts

                        # Create FSM with PUT_SCALP_CONFIG
                        tcfg = get_ticker_config(ticker, use_per_ticker=True, option_type="put")
                        from dataclasses import replace as dc_replace
                        if GRACE_OVERRIDE is not None:
                            tcfg = dc_replace(tcfg, grace_period_min=GRACE_OVERRIDE)
                        if PUT_CONFIG_OVERRIDES:
                            tcfg = dc_replace(tcfg, **PUT_CONFIG_OVERRIDES)
                        tcfg, _settings = _apply_exit_overrides(tcfg, _V6_SETTINGS)
                        fsm = ExitFSM(tcfg, settings=_settings)

                        entry_ts = datetime(2026, 1, 1, 9, 30) + timedelta(minutes=dip_entry_minute)

                        underlying_0 = 0
                        for i in range(dip_entry_minute, min(dip_entry_minute + 5, len(ptd["underlyings"]))):
                            u = ptd["underlyings"][i]
                            if not np.isnan(u) and u > 0:
                                underlying_0 = float(u)
                                break

                        state = TradeState(
                            trade_id=len(trades) + 1, ticker=ticker, option_type="put",
                            entry_premium=effective_entry, entry_time=entry_ts,
                            contracts=effective_contracts, peak_premium=effective_entry,
                            entry_underlying_price=underlying_0,
                            dte=ptd["dte"], expiry_date=ptd["expiry_date"] or "",
                        )

                        open_positions.append({
                            "ticker": ticker, "direction": "put",
                            "fsm": fsm, "state": state,
                            "entry_minute": dip_entry_minute, "signal_minute": minute,
                            "entry_ts": entry_ts,
                            "entry_premium": entry_premium,
                            "effective_entry": effective_entry,
                            "contracts": contracts, "dca_contracts": 0,
                            "effective_contracts": effective_contracts,
                            "dca_triggered": False,
                            "locked_pnl": 0.0, "remaining": effective_contracts,
                            "ticker_data": ptd,
                            "pattern_conf": pattern_conf,
                            "signal_quality": None,
                            "dip_confirm": dip_outcome,
                            "dip_savings": dip_savings,
                            "is_bear_mode": is_bear,
                        })

                        day_entered_tickers.add(put_day_key)
                        current_open_tickers.add(ticker)
                        current_open_dirs.append("put")
                        current_put_count += 1
            # ── End Phase 2b (PUT entries) ────────────────────────────

        # Force-close any positions still open at end of day
        # (fill at BID — we're selling; close/mid is a leak)
        for pos in open_positions:
            td = pos["ticker_data"]
            last_valid = pos["effective_entry"]
            for i in range(td["n_rows"] - 1, pos["entry_minute"], -1):
                b = td["bids"][i] if i < len(td["bids"]) else np.nan
                if not np.isnan(b) and b > 0:
                    last_valid = float(b)
                    break
                if not np.isnan(td["closes"][i]) and td["closes"][i] > 0:
                    last_valid = td["closes"][i]
                    break
            elapsed = td["n_rows"] - pos["entry_minute"]
            peak_gain = (pos["state"].peak_premium - pos["effective_entry"]) / pos["effective_entry"] * 100
            trade_pnl = pos["locked_pnl"] + (last_valid - pos["effective_entry"]) * pos["remaining"] * 100
            portfolio += trade_pnl

            tk = pos["ticker"]
            per_ticker[tk]["trades"] += 1
            if trade_pnl > 0:
                per_ticker[tk]["wins"] += 1
            per_ticker[tk]["pnl"] += trade_pnl
            per_ticker[tk]["total_contracts"] += pos["effective_contracts"]
            exit_reasons["eod_data_end"] += 1

            trades.append({
                "day": date_str, "ticker": tk, "minute": pos["entry_minute"],
                "direction": pos["direction"],
                "signal_minute": pos.get("signal_minute", pos["entry_minute"]),
                "entry": pos["entry_premium"], "effective_entry": round(pos["effective_entry"], 2),
                "contracts": pos["contracts"], "dca_contracts": pos.get("dca_contracts", 0),
                "effective_contracts": pos["effective_contracts"],
                "pnl": round(trade_pnl, 2), "reason": "eod_data_end",
                "hold_min": elapsed,
                "peak_gain": round(peak_gain, 1),
                "pattern_conf": round(pos["pattern_conf"], 3),
                "dte": int(pos["ticker_data"].get("dte", 0)),
                "exit_prem": round(last_valid, 2),
                "dca": pos.get("dca_triggered", False),
                "signal_quality": round(pos["signal_quality"], 1) if pos.get("signal_quality") is not None else None,
                "dip_confirm": pos.get("dip_confirm", "off"),
                "dip_savings": round(pos.get("dip_savings", 0), 2),
                "is_bear_mode": pos.get("is_bear_mode", False),
            })

            if date_str not in daily_pnls:
                daily_pnls[date_str] = 0
            daily_pnls[date_str] += trade_pnl

            if portfolio > peak_portfolio:
                peak_portfolio = portfolio
            dd = (peak_portfolio - portfolio) / peak_portfolio * 100
            if dd > max_dd:
                max_dd = dd

        equity_curve.append((date_str, round(portfolio, 2)))

    conn.close()
    if uw_conn:
        uw_conn.close()

    # Compute results
    total_pnl = portfolio - PORTFOLIO_START
    n_trades = len(trades)
    wins = sum(1 for t in trades if t["pnl"] > 0)
    losses = sum(1 for t in trades if t["pnl"] <= 0)
    win_rate = wins / n_trades * 100 if n_trades > 0 else 0

    pnl_list = [t["pnl"] for t in trades]
    wins_list = [p for p in pnl_list if p > 0]
    losses_list = [p for p in pnl_list if p <= 0]
    avg_win = np.mean(wins_list) if wins_list else 0
    avg_loss = np.mean(losses_list) if losses_list else 0
    gross_profit = sum(wins_list)
    gross_loss = abs(sum(losses_list))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    daily_returns = list(daily_pnls.values())
    sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 1 and np.std(daily_returns) > 0 else 0

    minutes = [t["minute"] for t in trades]
    avg_minute = np.mean(minutes) if minutes else 0

    # Win/loss streaks
    max_win_streak = max_loss_streak = cur_win = cur_loss = 0
    for t in trades:
        if t["pnl"] > 0:
            cur_win += 1
            cur_loss = 0
        else:
            cur_loss += 1
            cur_win = 0
        max_win_streak = max(max_win_streak, cur_win)
        max_loss_streak = max(max_loss_streak, cur_loss)

    # Best/worst trades
    best_trade = max(trades, key=lambda t: t["pnl"]) if trades else None
    worst_trade = min(trades, key=lambda t: t["pnl"]) if trades else None

    # ── Runner-capture metric ──────────────────────────────────────────
    # Realized return % per trade on the cost basis (effective_entry × eff
    # contracts × 100). This accounts for DCA blend + scaleout via pnl.
    def _trade_return_pct(t: dict) -> float:
        basis = t.get("effective_entry", t["entry"]) * t.get(
            "effective_contracts", t["contracts"]
        ) * 100
        return (t["pnl"] / basis * 100) if basis > 0 else 0.0

    for t in trades:
        t["pnl_pct"] = round(_trade_return_pct(t), 1)

    runners = [t for t in trades if t["pnl_pct"] >= 100.0]
    big_runners = [t for t in trades if t["pnl_pct"] >= 200.0]
    runner_pnl = sum(t["pnl"] for t in runners)
    largest_winner = best_trade  # by $; pnl_pct attached above
    largest_winner_pct = largest_winner["pnl_pct"] if largest_winner else 0.0
    largest_winner_dollars = largest_winner["pnl"] if largest_winner else 0.0

    return {
        "period": f"{dates[0]} to {dates[-1]}",
        "trading_days": len(dates),
        "pattern_threshold": pattern_threshold,
        "entry_threshold": entry_threshold if entry_model else None,
        "trades": n_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 1),
        "total_pnl": round(total_pnl, 2),
        "final_portfolio": round(portfolio, 2),
        "return_pct": round(total_pnl / PORTFOLIO_START * 100, 1),
        "profit_factor": round(pf, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd, 1),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "avg_pnl_per_trade": round(total_pnl / n_trades, 2) if n_trades > 0 else 0,
        "trades_per_day": round(n_trades / len(dates), 2) if dates else 0,
        # Runner-capture metrics
        "runners_100": len(runners),
        "big_runners_200": len(big_runners),
        "runner_pnl": round(runner_pnl, 2),
        "largest_winner_pct": round(largest_winner_pct, 1),
        "largest_winner_dollars": round(largest_winner_dollars, 2),
        "avg_winner": round(avg_win, 2),
        "avg_entry_minute": round(avg_minute, 1),
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
        "signals_sourced": signals_sourced,
        "signals_pattern_pass": signals_pattern_pass,
        "signals_entry_blocked": signals_entry_blocked,
        "signals_gate_blocked": signals_gate_blocked,
        "gate_blocks": dict(gate_blocks),
        "regime_skipped_days": regime_skipped_days,
        "per_ticker": dict(per_ticker),
        "exit_reasons": dict(exit_reasons),
        "daily_pnls": daily_pnls,
        "weekly_pnls": dict(weekly_pnls),
        "equity_curve": equity_curve,
        "trade_details": trades,
        "best_trade": best_trade,
        "worst_trade": worst_trade,
        "dip_confirm_stats": dict(dip_confirm_stats),
        "dip_confirm_avg_savings": round(dip_confirm_total_savings / dip_confirm_count, 2) if dip_confirm_count > 0 else 0,
        "dip_confirm_count": dip_confirm_count,
    }


# ── Report Generation ─────────────────────────────────────────────────────


def generate_report(r: dict, output_path: Path):
    """Generate a comprehensive markdown report."""
    lines = []
    lines.append("# Gold Standard Backtest Report")
    lines.append(f"Generated: {datetime.now(ET).strftime('%Y-%m-%d %H:%M ET')}\n")

    lines.append("## Configuration\n")
    lines.append("| Parameter | Value |")
    lines.append("|---|---|")
    lines.append(f"| Period | {r['period']} ({r['trading_days']} trading days) |")
    lines.append(f"| Starting Portfolio | ${PORTFOLIO_START:,} |")
    lines.append(f"| Pattern Model Threshold | {r['pattern_threshold']} |")
    if r["entry_threshold"] is not None:
        lines.append(f"| Entry Timing Filter | ON (threshold={r['entry_threshold']}) |")
    else:
        lines.append(f"| Entry Timing Filter | OFF |")
    lines.append(f"| Tickers | {', '.join(t for t in r['per_ticker'].keys())} |")
    lines.append(f"| Excluded | {', '.join(EXCLUDED_TICKERS)} |")
    lines.append(f"| Max Concurrent | {MAX_CONCURRENT} |")
    lines.append(f"| Position Size Cap | {MAX_POSITION_PCT*100:.0f}% |")
    lines.append(f"| Premium Cap | ${PREMIUM_CAP} |")
    lines.append(f"| Exit Engine | V5 FSM + V6 enhancements |")
    lines.append(f"| CALL Scan Window | {SCAN_START_MIN}-{SCAN_END_MIN} min after open |")
    lines.append(f"| PUT Scan Window | {SCAN_START_MIN}-{PUT_SCAN_END_MIN} min after open (all day) |")

    lines.append("\n## Performance Summary\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| **Total P&L** | **${r['total_pnl']:+,.0f}** |")
    lines.append(f"| **Final Portfolio** | **${r['final_portfolio']:,.0f}** |")
    lines.append(f"| **Return** | **{r['return_pct']:+.1f}%** |")
    lines.append(f"| Total Trades | {r['trades']} |")
    lines.append(f"| Trades / Day | {r.get('trades_per_day', 0)} |")
    lines.append(f"| Wins / Losses | {r['wins']} / {r['losses']} |")
    lines.append(f"| **Win Rate** | **{r['win_rate']}%** |")
    lines.append(f"| **Profit Factor** | **{r['profit_factor']}** |")
    lines.append(f"| **Sharpe Ratio** | **{r['sharpe']}** |")
    lines.append(f"| **Max Drawdown** | **{r['max_drawdown_pct']}%** |")
    lines.append(f"| Avg Win | ${r['avg_win']:+,.0f} |")
    lines.append(f"| Avg Loss | ${r['avg_loss']:+,.0f} |")
    lines.append(f"| **Runners (>=+100%)** | **{r.get('runners_100', 0)}** |")
    lines.append(f"| **Big Runners (>=+200%)** | **{r.get('big_runners_200', 0)}** |")
    lines.append(f"| Runner P&L | ${r.get('runner_pnl', 0):+,.0f} |")
    lines.append(f"| Largest Winner | {r.get('largest_winner_pct', 0):+.0f}% (${r.get('largest_winner_dollars', 0):+,.0f}) |")
    lines.append(f"| Avg P&L/Trade | ${r['avg_pnl_per_trade']:+,.0f} |")
    lines.append(f"| Avg Entry Minute | {r['avg_entry_minute']:.0f} min after open |")
    lines.append(f"| Max Win Streak | {r['max_win_streak']} |")
    lines.append(f"| Max Loss Streak | {r['max_loss_streak']} |")

    lines.append("\n## Signal Funnel\n")
    lines.append("| Stage | Count | Pass Rate |")
    lines.append("|---|---|---|")
    lines.append(f"| Minutes scanned | {r['signals_sourced']:,} | — |")
    ppass = r['signals_pattern_pass'] / r['signals_sourced'] * 100 if r['signals_sourced'] > 0 else 0
    lines.append(f"| Pattern model pass | {r['signals_pattern_pass']:,} | {ppass:.2f}% |")
    if r['entry_threshold'] is not None:
        epass = r['signals_pattern_pass'] - r['signals_entry_blocked']
        erate = epass / r['signals_pattern_pass'] * 100 if r['signals_pattern_pass'] > 0 else 0
        lines.append(f"| Entry timing pass | {epass:,} | {erate:.1f}% of pattern pass |")
        lines.append(f"| Entry timing blocked | {r['signals_entry_blocked']:,} | — |")
    lines.append(f"| Gate blocked (total) | {r['signals_gate_blocked']:,} | — |")
    if r.get("gate_blocks"):
        for gate_name, count in sorted(r["gate_blocks"].items(), key=lambda x: -x[1]):
            lines.append(f"|   ↳ {gate_name} | {count:,} | — |")
    lines.append(f"| **Trades executed** | **{r['trades']}** | — |")

    if r.get("dip_confirm_count", 0) > 0:
        lines.append("\n## DipConfirm Analysis\n")
        lines.append("| Outcome | Count | % |")
        lines.append("|---|---|---|")
        dc_total = r["dip_confirm_count"]
        for outcome, count in sorted(r["dip_confirm_stats"].items(), key=lambda x: -x[1]):
            pct = count / dc_total * 100
            lines.append(f"| {outcome} | {count} | {pct:.1f}% |")
        lines.append(f"| **Total** | **{dc_total}** | — |")
        lines.append(f"| Avg Entry Savings | **{r['dip_confirm_avg_savings']:+.2f}%** | — |")
        # Calculate total dollar savings from dip confirm
        dc_trades = [t for t in r["trade_details"] if t.get("dip_savings", 0) != 0]
        if dc_trades:
            total_dollar_saved = sum(
                t["dip_savings"] / 100 * t["entry"] * t.get("effective_contracts", t["contracts"]) * 100
                for t in dc_trades
            )
            lines.append(f"| Est. Dollar Savings | **${total_dollar_saved:+,.0f}** | — |")

    if r["best_trade"]:
        lines.append("\n## Best & Worst Trades\n")
        bt = r["best_trade"]
        wt = r["worst_trade"]
        lines.append(f"- **Best**: {bt['ticker']} on {bt['day']} — ${bt['pnl']:+,.0f} "
                      f"(entry ${bt['entry']:.2f}, {bt['contracts']} contracts, "
                      f"peak +{bt['peak_gain']:.0f}%, {bt['reason']})")
        lines.append(f"- **Worst**: {wt['ticker']} on {wt['day']} — ${wt['pnl']:+,.0f} "
                      f"(entry ${wt['entry']:.2f}, {wt['contracts']} contracts, "
                      f"peak +{wt['peak_gain']:.0f}%, {wt['reason']})")

    lines.append("\n## Per-Ticker Breakdown\n")
    lines.append("| Ticker | Trades | WR% | P&L | Avg P&L | Contracts |")
    lines.append("|---|---|---|---|---|---|")
    for ticker in sorted(r["per_ticker"].keys(), key=lambda t: r["per_ticker"][t]["pnl"], reverse=True):
        t = r["per_ticker"][ticker]
        wr = t["wins"] / t["trades"] * 100 if t["trades"] > 0 else 0
        avg = t["pnl"] / t["trades"] if t["trades"] > 0 else 0
        profitable = "+" if t["pnl"] > 0 else ""
        lines.append(f"| {ticker} | {t['trades']} | {wr:.0f}% | ${t['pnl']:+,.0f} | ${avg:+,.0f} | {t['total_contracts']} |")

    lines.append("\n## Exit Reason Distribution\n")
    lines.append("| Exit Reason | Count | % |")
    lines.append("|---|---|---|")
    for reason, count in sorted(r["exit_reasons"].items(), key=lambda x: -x[1]):
        pct = count / r["trades"] * 100 if r["trades"] > 0 else 0
        lines.append(f"| {reason} | {count} | {pct:.1f}% |")

    lines.append("\n## Weekly P&L\n")
    lines.append("| Week | P&L | Cumulative |")
    lines.append("|---|---|---|")
    cum = 0
    for week in sorted(r["weekly_pnls"].keys()):
        wpnl = r["weekly_pnls"][week]
        cum += wpnl
        lines.append(f"| {week} | ${wpnl:+,.0f} | ${cum:+,.0f} |")

    lines.append("\n## Entry Minute Distribution\n")
    lines.append("| Window | Trades | P&L | WR% | Avg P&L |")
    lines.append("|---|---|---|---|---|")
    for label, lo, hi in [("5-15min", 5, 15), ("15-30min", 15, 30), ("30-45min", 30, 45),
                           ("45-60min", 45, 60), ("60-75min", 60, 75), ("75-90min", 75, 90)]:
        bucket_trades = [t for t in r["trade_details"] if lo <= t["minute"] < hi]
        if bucket_trades:
            n = len(bucket_trades)
            bpnl = sum(t["pnl"] for t in bucket_trades)
            bwr = sum(1 for t in bucket_trades if t["pnl"] > 0) / n * 100
            bavg = bpnl / n
            lines.append(f"| {label} | {n} | ${bpnl:+,.0f} | {bwr:.0f}% | ${bavg:+,.0f} |")

    lines.append("\n## Daily P&L (last 20 days)\n")
    lines.append("| Date | P&L | Portfolio |")
    lines.append("|---|---|---|")
    eq_dict = {d: v for d, v in r["equity_curve"]}
    sorted_days = sorted(r["daily_pnls"].keys())
    for day in sorted_days[-20:]:
        dpnl = r["daily_pnls"][day]
        port = eq_dict.get(day, "—")
        port_str = f"${port:,.0f}" if isinstance(port, (int, float)) else port
        lines.append(f"| {day} | ${dpnl:+,.0f} | {port_str} |")

    # DCA stats
    dca_trades = [t for t in r["trade_details"] if t.get("dca")]
    if dca_trades:
        lines.append("\n## DCA Stats\n")
        dca_wins = sum(1 for t in dca_trades if t["pnl"] > 0)
        dca_pnl = sum(t["pnl"] for t in dca_trades)
        lines.append(f"- DCA triggered: {len(dca_trades)} trades ({len(dca_trades)/len(r['trade_details'])*100:.0f}%)")
        lines.append(f"- DCA win rate: {dca_wins/len(dca_trades)*100:.0f}%")
        lines.append(f"- DCA total P&L: ${dca_pnl:+,.0f}")

    lines.append("\n## Trade Log (all trades)\n")
    lines.append("| # | Date | Ticker | Dir | Min | Entry | Eff.Entry | Ct | DCA | P&L | Reason | Peak | Conf |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, t in enumerate(r["trade_details"], 1):
        dca_flag = "Y" if t.get("dca") else ""
        eff_entry = t.get("effective_entry", t["entry"])
        eff_ct = t.get("effective_contracts", t["contracts"])
        direction = t.get("direction", "call").upper()[0]  # C or P
        lines.append(f"| {i} | {t['day']} | {t['ticker']} | {direction} | {t['minute']} | "
                      f"${t['entry']:.2f} | ${eff_entry:.2f} | {eff_ct} | {dca_flag} | ${t['pnl']:+,.0f} | "
                      f"{t['reason']} | +{t['peak_gain']:.0f}% | {t['pattern_conf']:.2f} |")

    report = "\n".join(lines) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    return report


# ── Main ───────────────────────────────────────────────────────────────────


def main():
    # All module-level constants mutated by CLI flags must be declared global at
    # the top of main() — argparse defaults read them before assignment below.
    global ENABLE_PUTS, PUTS_ONLY, ENABLE_DIP_CONFIRM, GRACE_OVERRIDE, ENABLE_PRICE_GATES
    global ENABLE_DELTA_GATE, DELTA_MIN, DELTA_MAX
    global ENABLE_ANTI_CHASE, ENABLE_MOMENTUM_CONFIRM, ENABLE_CONSECUTIVE_LOSER
    global ENABLE_CORRELATION_CAP, ENABLE_DIRECTIONAL_REGIME, ENABLE_PUT_BEARISH_CONFIRM
    global MIN_PREMIUM_FLOOR, MIN_SCORE, OPENING_BUFFER_MIN, TOD_EARLY_MIN_SCORE
    global SCALP_THRESH_OVERRIDE, SOFT_KEEP_OVERRIDE, ADAPTIVE_MULT_OVERRIDE
    global THETA_MIN_OVERRIDE, BREAKEVEN_TRIGGER_OVERRIDE, SCALEOUT_TRIGGER_OVERRIDE
    global SIZING_MODE, CONF_BUDGET_MIN, CONF_BUDGET_MAX, CONF_REF_MIN, CONF_REF_MAX
    global MULTI_DAY_CAP, LATE_0DTE_CAP

    parser = argparse.ArgumentParser(description="Gold Standard E2E Backtest")
    parser.add_argument("--days", type=int, default=60, help="Last N trading days (default: 60)")
    parser.add_argument("--pattern-threshold", type=float, default=0.74, help="Pattern model threshold")
    parser.add_argument("--entry-threshold", type=float, default=0.80, help="Entry timing threshold")
    parser.add_argument("--no-entry-filter", action="store_true", help="Disable entry timing filter")
    parser.add_argument("--no-regime", action="store_true", help="Disable regime daily filter")
    parser.add_argument("--regime-threshold", type=float, default=0.19, help="Regime filter threshold (default: 0.19, prod DEFAULT_REGIME_THRESHOLD)")
    parser.add_argument("--include-losers", action="store_true", help="Include excluded tickers")
    parser.add_argument("--sweep", action="store_true", help="Sweep entry thresholds")
    parser.add_argument("--start", type=str, help="Override start date")
    parser.add_argument("--end", type=str, help="Override end date")
    parser.add_argument("--no-dip-confirm", action="store_true", help="Disable DipConfirm simulation")
    parser.add_argument("--grace", type=float, default=None, help="Override grace period (minutes) for all tickers")
    parser.add_argument("--grace-sweep", action="store_true", help="Sweep grace periods: 0, 1, 2, 3, 5 min")
    parser.add_argument("--puts", action="store_true", help="Enable PUT trading alongside CALLs (SPY direction gate)")
    parser.add_argument("--puts-only", action="store_true", help="Only trade PUTs (no CALLs) for comparison")
    parser.add_argument("--put-sweep", action="store_true", help="Sweep PUT exit parameters to find profitable config")
    parser.add_argument("--put-entry-threshold", type=float, default=None, help="Override PUT entry timing threshold")
    parser.add_argument("--no-put-entry-timing", action="store_true", help="Disable PUT entry timing model entirely")
    parser.add_argument("--put-entry-sweep", action="store_true", help="Sweep PUT entry timing thresholds: 0.85, 0.75, 0.70, disabled")
    parser.add_argument("--no-price-gates", action="store_true", help="Disable premium_cap, otm_distance, min_premium, spread_gate")
    parser.add_argument("--delta-gate", action="store_true", help="Replace price gates with delta-based filtering (0.15-0.70)")
    parser.add_argument("--delta-min", type=float, default=0.15, help="Min delta (reject far OTM)")
    parser.add_argument("--delta-max", type=float, default=0.70, help="Max delta (reject deep ITM)")

    # ── Gate ablation toggles (default = current prod behavior = ON) ──
    # Consistent on/off convention: pass "off" to disable a gate.
    parser.add_argument("--gate-anti-chase", choices=["on", "off"], default="on",
                        help="AntiChase gate (block entry if underlying moved >0.3%% in 5min). Default: on")
    parser.add_argument("--gate-momentum", choices=["on", "off"], default="on",
                        help="MomentumConfirm gate. Default: on")
    parser.add_argument("--gate-consecutive-loser", choices=["on", "off"], default="on",
                        help="ConsecutiveLoser circuit breaker. Default: on")
    parser.add_argument("--gate-correlation-cap", choices=["on", "off"], default="on",
                        help="CorrelationCap gate (max 3 same-group same-dir open). Default: on")
    parser.add_argument("--gate-directional-regime", choices=["on", "off"], default="on",
                        help="Rule-based DirectionalRegime gate. Default: on")
    parser.add_argument("--gate-put-bearish", choices=["on", "off"], default="on",
                        help="PutBearishConfirm gate (VWAP/RSI/candle confirm). Default: on")

    # ── Gate range sweeps (defaults = current hardcoded values) ──
    parser.add_argument("--min-premium", type=float, default=MIN_PREMIUM_FLOOR,
                        help=f"Min premium floor (default: {MIN_PREMIUM_FLOOR})")
    parser.add_argument("--score-floor", type=int, default=MIN_SCORE,
                        help=f"Min score floor (default: {MIN_SCORE})")
    parser.add_argument("--delta-floor", type=float, default=DELTA_MIN,
                        help=f"Delta floor / min (default: {DELTA_MIN})")
    parser.add_argument("--delta-ceiling", type=float, default=DELTA_MAX,
                        help=f"Delta ceiling / max (default: {DELTA_MAX})")
    parser.add_argument("--tod-buffer-min", type=int, default=OPENING_BUFFER_MIN,
                        help=f"Opening-minutes block / opening buffer (default: {OPENING_BUFFER_MIN})")
    parser.add_argument("--tod-early-score", type=int, default=TOD_EARLY_MIN_SCORE,
                        help=f"Min score required before 9:45 ET / early cutoff (default: {TOD_EARLY_MIN_SCORE})")

    # ---- Position-sizing experiment (default = production behavior) ----
    parser.add_argument("--sizing-mode",
                        choices=["current", "flat", "conf_linear", "conf_step", "score_linear"],
                        default="current",
                        help="Sizing scheme. 'current' = prod score_to_contracts (default)")
    parser.add_argument("--conf-budget-min", type=float, default=None,
                        help="Budget multiplier at low confidence/score (linear/step modes)")
    parser.add_argument("--conf-budget-max", type=float, default=None,
                        help="Budget multiplier at high confidence/score (linear/step modes)")
    parser.add_argument("--conf-ref-min", type=float, default=None,
                        help="Confidence mapped to conf-budget-min (default 0.74)")
    parser.add_argument("--conf-ref-max", type=float, default=None,
                        help="Confidence mapped to conf-budget-max (default 0.95)")
    parser.add_argument("--multiday-cap", type=str, default=None,
                        help="Override multi-day (DTE>0) contract cap. Integer, or 'none'/'off' to disable. Default 2 (prod).")
    parser.add_argument("--no-late-0dte-cap", action="store_true",
                        help="Disable the prod after-2PM-ET 0DTE 1-contract cap")

    # ---- TASK 1: EXIT-PARAM tuning overrides (default None = V5Config/V6 unchanged) ----
    parser.add_argument("--grace-min", type=float, default=None,
                        help="Override grace period (minutes) for all tickers (default: per-config ~5)")
    parser.add_argument("--scalp-thresh", type=float, default=None,
                        help="scalp_peak_threshold_pct: peak-gain %% to arm scalp trail (default ~20)")
    parser.add_argument("--soft-keep", type=float, default=None,
                        help="soft_trail_keep_pct: fraction of (peak-entry) gain to keep (default ~0.60)")
    parser.add_argument("--adaptive-mult", type=float, default=None,
                        help="Multiplier on adaptive trail widths (1.0=unchanged, <1 tightens, >1 widens)")
    parser.add_argument("--theta-min", type=float, default=None,
                        help="theta_bleed_min minutes (0DTE hold limit, default 120)")
    parser.add_argument("--breakeven-trigger", type=float, default=None,
                        help="V6 breakeven ratchet arm %% (default 20)")
    parser.add_argument("--scaleout-trigger", type=float, default=None,
                        help="V6 scaleout trigger %% (default 20)")
    args = parser.parse_args()

    print("=" * 70)
    print("GOLD STANDARD END-TO-END BACKTEST")
    print("Pattern Entry (sourcing) + Entry Timing (filter) + V5 FSM (exits)")
    print("=" * 70)

    # Determine date range
    conn = sqlite3.connect(THETADATA_DB)
    all_dates = [r[0] for r in conn.execute("""
        SELECT DISTINCT substr(timestamp, 1, 10) FROM option_ohlc
        WHERE ticker = 'SPY' ORDER BY 1 DESC
    """).fetchall()]
    conn.close()

    if args.start and args.end:
        start_date = args.start
        end_date = args.end
    elif args.end:
        end_date = args.end
        target_dates = [d for d in all_dates if d <= end_date]
        start_date = target_dates[min(args.days - 1, len(target_dates) - 1)]
    else:
        end_date = all_dates[0]
        start_date = all_dates[min(args.days - 1, len(all_dates) - 1)]

    if args.include_losers:
        tickers = TICKERS
    else:
        tickers = [t for t in TICKERS if t not in EXCLUDED_TICKERS]

    # Apply PUT flags
    if args.puts:
        ENABLE_PUTS = True
        print("PUT Trading: ENABLED (CALL + PUT with SPY direction gate)")
    if args.puts_only:
        PUTS_ONLY = True
        ENABLE_PUTS = True
        print("PUT Trading: PUTS ONLY (no CALLs)")
    if not args.puts and not args.puts_only:
        print("PUT Trading: DISABLED (CALL only)")

    # Apply DipConfirm flag
    if args.no_dip_confirm:
        ENABLE_DIP_CONFIRM = False
        print("DipConfirm: DISABLED")
    else:
        print(f"DipConfirm: ENABLED (fade={DIP_CONFIRM_FADE_PCT}%, polls={DIP_CONFIRM_MAX_POLLS}, always_enter={DIP_CONFIRM_ALWAYS_ENTER})")

    # Apply price gates flag
    # Default matches prod: premium_cap + otm_distance OFF, delta gate ON,
    # min_premium + spread_gate always ON (principled microstructure filters).
    # ── Apply gate ablation toggles + range sweeps ──
    ENABLE_ANTI_CHASE = args.gate_anti_chase == "on"
    ENABLE_MOMENTUM_CONFIRM = args.gate_momentum == "on"
    ENABLE_CONSECUTIVE_LOSER = args.gate_consecutive_loser == "on"
    ENABLE_CORRELATION_CAP = args.gate_correlation_cap == "on"
    ENABLE_DIRECTIONAL_REGIME = args.gate_directional_regime == "on"
    ENABLE_PUT_BEARISH_CONFIRM = args.gate_put_bearish == "on"
    MIN_PREMIUM_FLOOR = args.min_premium
    MIN_SCORE = args.score_floor
    OPENING_BUFFER_MIN = args.tod_buffer_min
    TOD_EARLY_MIN_SCORE = args.tod_early_score

    print(
        "Gate toggles: "
        f"anti_chase={'on' if ENABLE_ANTI_CHASE else 'off'}, "
        f"momentum={'on' if ENABLE_MOMENTUM_CONFIRM else 'off'}, "
        f"consecutive_loser={'on' if ENABLE_CONSECUTIVE_LOSER else 'off'}, "
        f"correlation_cap={'on' if ENABLE_CORRELATION_CAP else 'off'}, "
        f"directional_regime={'on' if ENABLE_DIRECTIONAL_REGIME else 'off'}, "
        f"put_bearish={'on' if ENABLE_PUT_BEARISH_CONFIRM else 'off'}"
    )
    print(
        "Gate ranges: "
        f"min_premium={MIN_PREMIUM_FLOOR}, score_floor={MIN_SCORE}, "
        f"tod_buffer={OPENING_BUFFER_MIN}, tod_early_score={TOD_EARLY_MIN_SCORE}"
    )

    # New range-sweep flags --delta-floor / --delta-ceiling (defaults = current).
    # These thread into DELTA_MIN/MAX. The legacy --delta-min/--delta-max still
    # win when --delta-gate is explicitly passed (back-compat).
    DELTA_MIN = args.delta_floor
    DELTA_MAX = args.delta_ceiling
    if args.no_price_gates:
        ENABLE_PRICE_GATES = False
        ENABLE_DELTA_GATE = False
        print("Price Gates: ALL DISABLED (premium_cap, otm_distance, delta all OFF; min_premium + spread still active)")
    elif args.delta_gate or ENABLE_DELTA_GATE:
        ENABLE_PRICE_GATES = False  # disable old static gates
        ENABLE_DELTA_GATE = True
        if args.delta_gate:
            DELTA_MIN = args.delta_min
            DELTA_MAX = args.delta_max
        print(f"Price Gates: delta gate ON ({DELTA_MIN:.2f}-{DELTA_MAX:.2f}), premium_cap + otm_distance OFF (matches prod)")
    else:
        ENABLE_PRICE_GATES = True
        print(f"Price Gates: OLD mode (cap=${PREMIUM_CAP}, min=${MIN_PREMIUM_FLOOR}, spread={SPREAD_GATE_PCT}%, otm=per-ticker)")

    # Apply grace override (--grace-min is the canonical flag; --grace kept for back-compat)
    if args.grace_min is not None:
        GRACE_OVERRIDE = args.grace_min
        print(f"Grace Override: {GRACE_OVERRIDE} min (all tickers, via --grace-min)")
    elif args.grace is not None:
        GRACE_OVERRIDE = args.grace
        print(f"Grace Override: {GRACE_OVERRIDE} min (all tickers)")
    else:
        print("Grace: per-ticker defaults (5min standard, 8min TSLA/QQQ)")

    # Apply exit-param tuning overrides (TASK 1)
    SCALP_THRESH_OVERRIDE = args.scalp_thresh
    SOFT_KEEP_OVERRIDE = args.soft_keep
    ADAPTIVE_MULT_OVERRIDE = args.adaptive_mult
    THETA_MIN_OVERRIDE = args.theta_min
    BREAKEVEN_TRIGGER_OVERRIDE = args.breakeven_trigger
    SCALEOUT_TRIGGER_OVERRIDE = args.scaleout_trigger

    # ---- Position-sizing experiment wiring ----
    SIZING_MODE = args.sizing_mode
    if args.conf_budget_min is not None:
        CONF_BUDGET_MIN = args.conf_budget_min
    if args.conf_budget_max is not None:
        CONF_BUDGET_MAX = args.conf_budget_max
    if args.conf_ref_min is not None:
        CONF_REF_MIN = args.conf_ref_min
    if args.conf_ref_max is not None:
        CONF_REF_MAX = args.conf_ref_max
    if args.multiday_cap is not None:
        _mc = args.multiday_cap.strip().lower()
        MULTI_DAY_CAP = None if _mc in ("none", "off", "disable", "0", "-1") else int(_mc)
    if args.no_late_0dte_cap:
        LATE_0DTE_CAP = False
    print(f"[SIZING] mode={SIZING_MODE} conf_budget=[{CONF_BUDGET_MIN},{CONF_BUDGET_MAX}] "
          f"conf_ref=[{CONF_REF_MIN},{CONF_REF_MAX}] multiday_cap={MULTI_DAY_CAP} "
          f"late_0dte_cap={LATE_0DTE_CAP}", flush=True)

    _exit_ovr = {
        "scalp_thresh": SCALP_THRESH_OVERRIDE, "soft_keep": SOFT_KEEP_OVERRIDE,
        "adaptive_mult": ADAPTIVE_MULT_OVERRIDE, "theta_min": THETA_MIN_OVERRIDE,
        "breakeven_trigger": BREAKEVEN_TRIGGER_OVERRIDE,
        "scaleout_trigger": SCALEOUT_TRIGGER_OVERRIDE,
    }
    _active_ovr = {k: v for k, v in _exit_ovr.items() if v is not None}
    if _active_ovr:
        print(f"Exit-Param Overrides: {_active_ovr}")
    else:
        print("Exit-Param Overrides: none (V5Config/V6 defaults)")

    # Load models
    print("\nLoading models...")
    use_entry_filter = not args.no_entry_filter
    use_regime = not args.no_regime
    (pattern_model, pattern_meta, entry_model, entry_features,
     stop_model, regime_model, signal_model, put_pattern_model, put_pattern_meta,
     put_entry_model, put_entry_features, put_entry_threshold) = load_models(
        use_entry_filter, use_regime=use_regime
    )

    # Apply PUT entry timing overrides
    if args.no_put_entry_timing:
        put_entry_model = None
        put_entry_features = []
        print("PUT Entry Timing: DISABLED (pattern model only)")
    elif args.put_entry_threshold is not None:
        put_entry_threshold = args.put_entry_threshold
        print(f"PUT Entry Timing: threshold overridden to {put_entry_threshold:.2f}")

    if args.put_entry_sweep:
        # Force PUTs-only mode for sweep
        PUTS_ONLY = True
        ENABLE_PUTS = True
        print("\n" + "=" * 100)
        print("PUT ENTRY TIMING THRESHOLD SWEEP (PUTs only)")
        print("=" * 100)
        print(f"{'Config':<24} {'Trades':<7} {'WR%':<6} {'P&L':>11} {'PF':>6} "
              f"{'Sharpe':>7} {'MaxDD':>7} {'AvgWin':>9} {'AvgLoss':>9}")
        print("-" * 100)

        # Save original model refs
        orig_put_entry_model = put_entry_model
        orig_put_entry_features = put_entry_features
        orig_put_entry_threshold = put_entry_threshold

        # Part 1: Static threshold sweep
        print("\n── Static Threshold Sweep ──")
        sweep_thresholds = [
            ("0.85 (current)", orig_put_entry_model, orig_put_entry_features, 0.85, "none"),
            ("0.75", orig_put_entry_model, orig_put_entry_features, 0.75, "none"),
            ("0.70", orig_put_entry_model, orig_put_entry_features, 0.70, "none"),
            ("DISABLED", None, [], 0.0, "none"),
        ]

        results = []
        for label, pe_model, pe_feats, pe_thresh, crash_mode in sweep_thresholds:
            r = run_backtest(pattern_model, pattern_meta, entry_model, entry_features,
                             args.pattern_threshold, args.entry_threshold,
                             tickers, start_date, end_date, stop_model,
                             regime_model, args.regime_threshold, signal_model,
                             put_pattern_model, put_pattern_meta,
                             pe_model, pe_feats, pe_thresh, crash_mode)
            results.append((label, r))
            pnl_str = f"${r['total_pnl']:+,.0f}"
            print(f"{label:<24} {r['trades']:<7} {r['win_rate']:<6.1f} "
                  f"{pnl_str:>11} {r['profit_factor']:>6.2f} {r['sharpe']:>7.2f} "
                  f"{r['max_drawdown_pct']:>6.1f}% {r['avg_win']:>+9,.0f} "
                  f"{r['avg_loss']:>+9,.0f}")

        # Part 2: Smart crash mode strategies (all use 0.85 base threshold)
        print(f"\n── Smart Crash Mode Strategies (base threshold=0.85) ──")
        print(f"{'Config':<24} {'Trades':<7} {'WR%':<6} {'P&L':>11} {'PF':>6} "
              f"{'Sharpe':>7} {'MaxDD':>7} {'AvgWin':>9} {'AvgLoss':>9}")
        print("-" * 100)

        crash_modes = [
            ("CRASH (SPY -0.5%/30m)", orig_put_entry_model, orig_put_entry_features, 0.85, "crash"),
            ("DYNAMIC (scale w/SPY)", orig_put_entry_model, orig_put_entry_features, 0.85, "dynamic"),
            ("REGIME (bear=skip)", orig_put_entry_model, orig_put_entry_features, 0.85, "regime"),
        ]

        for label, pe_model, pe_feats, pe_thresh, crash_mode in crash_modes:
            r = run_backtest(pattern_model, pattern_meta, entry_model, entry_features,
                             args.pattern_threshold, args.entry_threshold,
                             tickers, start_date, end_date, stop_model,
                             regime_model, args.regime_threshold, signal_model,
                             put_pattern_model, put_pattern_meta,
                             pe_model, pe_feats, pe_thresh, crash_mode)
            results.append((label, r))
            pnl_str = f"${r['total_pnl']:+,.0f}"
            print(f"{label:<24} {r['trades']:<7} {r['win_rate']:<6.1f} "
                  f"{pnl_str:>11} {r['profit_factor']:>6.2f} {r['sharpe']:>7.2f} "
                  f"{r['max_drawdown_pct']:>6.1f}% {r['avg_win']:>+9,.0f} "
                  f"{r['avg_loss']:>+9,.0f}")

        # Summary table
        print(f"\n{'=' * 100}")
        print("SUMMARY — PUT Entry Timing Strategies")
        print(f"{'=' * 100}")
        print(f"{'Config':<24} {'Trades':<7} {'WR%':<6} {'P&L':>11} {'PF':>6} "
              f"{'Sharpe':>7} {'MaxDD':>7}")
        print("-" * 80)
        for label, r in results:
            pnl_str = f"${r['total_pnl']:+,.0f}"
            best = " ◀" if r['total_pnl'] == max(x[1]['total_pnl'] for x in results) else ""
            print(f"{label:<24} {r['trades']:<7} {r['win_rate']:<6.1f} "
                  f"{pnl_str:>11} {r['profit_factor']:>6.2f} {r['sharpe']:>7.2f} "
                  f"{r['max_drawdown_pct']:>6.1f}%{best}")

        sys.exit(0)

    if args.put_sweep:
        # Force PUTs-only mode for sweep
        PUTS_ONLY = True
        ENABLE_PUTS = True
        print("\n" + "=" * 100)
        print("PUT EXIT PARAMETER SWEEP (PUTs only, SPY direction gate)")
        print("=" * 100)

        # Define sweep configurations: (label, overrides_dict)
        sweep_configs = [
            # Baseline (current PUT_SCALP_CONFIG)
            ("BASELINE (30/50/60m)", {}),

            # Profit target sweeps
            ("Target 15%", {"profit_target_general_pct": 15.0, "profit_target_index_0dte_pct": 15.0}),
            ("Target 20%", {"profit_target_general_pct": 20.0, "profit_target_index_0dte_pct": 20.0}),
            ("Target 40%", {"profit_target_general_pct": 40.0, "profit_target_index_0dte_pct": 40.0}),
            ("Target 50%", {"profit_target_general_pct": 50.0, "profit_target_index_0dte_pct": 50.0}),

            # Stop loss sweeps
            ("Stop 30%", {"tight_stop_0dte_pct": 30.0, "backstop_0dte_pct": 30.0,
                          "tight_stop_multiday_pct": 30.0, "backstop_multiday_pct": 30.0}),
            ("Stop 40%", {"tight_stop_0dte_pct": 40.0, "backstop_0dte_pct": 40.0,
                          "tight_stop_multiday_pct": 40.0, "backstop_multiday_pct": 40.0}),
            ("Stop 60%", {"tight_stop_0dte_pct": 60.0, "backstop_0dte_pct": 60.0,
                          "tight_stop_multiday_pct": 60.0, "backstop_multiday_pct": 60.0}),

            # Max hold time sweeps
            ("Hold 30m", {"theta_bleed_min": 30.0, "theta_timer_minutes": 30.0}),
            ("Hold 45m", {"theta_bleed_min": 45.0, "theta_timer_minutes": 45.0}),
            ("Hold 90m", {"theta_bleed_min": 90.0, "theta_timer_minutes": 90.0}),
            ("Hold 120m", {"theta_bleed_min": 120.0, "theta_timer_minutes": 120.0}),

            # Grace period sweeps
            ("Grace 1m", {"grace_period_min": 1.0}),
            ("Grace 5m", {"grace_period_min": 5.0}),
            ("Grace 8m", {"grace_period_min": 8.0}),

            # Combined: tight scalp (quick in, quick out)
            ("Scalp 15/30/30m", {"profit_target_general_pct": 15.0, "profit_target_index_0dte_pct": 15.0,
                                  "tight_stop_0dte_pct": 30.0, "backstop_0dte_pct": 30.0,
                                  "tight_stop_multiday_pct": 30.0, "backstop_multiday_pct": 30.0,
                                  "theta_bleed_min": 30.0, "theta_timer_minutes": 30.0}),
            ("Scalp 20/40/45m", {"profit_target_general_pct": 20.0, "profit_target_index_0dte_pct": 20.0,
                                  "tight_stop_0dte_pct": 40.0, "backstop_0dte_pct": 40.0,
                                  "tight_stop_multiday_pct": 40.0, "backstop_multiday_pct": 40.0,
                                  "theta_bleed_min": 45.0, "theta_timer_minutes": 45.0}),

            # Combined: patient (wider targets and holds)
            ("Patient 50/60/120m", {"profit_target_general_pct": 50.0, "profit_target_index_0dte_pct": 50.0,
                                     "tight_stop_0dte_pct": 60.0, "backstop_0dte_pct": 60.0,
                                     "tight_stop_multiday_pct": 60.0, "backstop_multiday_pct": 60.0,
                                     "theta_bleed_min": 120.0, "theta_timer_minutes": 120.0}),

            # Adaptive trail tweaks (tighter/wider)
            ("Adaptive 15/30", {"adaptive_highvol_tiers": (AdaptiveTier(15, 30),),
                                 "adaptive_index_tiers": (AdaptiveTier(15, 25),),
                                 "adaptive_standard_tiers": (AdaptiveTier(15, 25),)}),
            ("Adaptive 30/50", {"adaptive_highvol_tiers": (AdaptiveTier(30, 50),),
                                 "adaptive_index_tiers": (AdaptiveTier(30, 45),),
                                 "adaptive_standard_tiers": (AdaptiveTier(30, 45),)}),

            # Soft trail keep % tweaks
            ("SoftKeep 40%", {"soft_trail_keep_pct": 0.4}),
            ("SoftKeep 80%", {"soft_trail_keep_pct": 0.8}),

            # Best combo candidates
            ("Best1: 20/40/45m/g5", {"profit_target_general_pct": 20.0, "profit_target_index_0dte_pct": 20.0,
                                      "tight_stop_0dte_pct": 40.0, "backstop_0dte_pct": 40.0,
                                      "tight_stop_multiday_pct": 40.0, "backstop_multiday_pct": 40.0,
                                      "theta_bleed_min": 45.0, "theta_timer_minutes": 45.0,
                                      "grace_period_min": 5.0}),
            ("Best2: 15/30/30m/g1", {"profit_target_general_pct": 15.0, "profit_target_index_0dte_pct": 15.0,
                                      "tight_stop_0dte_pct": 30.0, "backstop_0dte_pct": 30.0,
                                      "tight_stop_multiday_pct": 30.0, "backstop_multiday_pct": 30.0,
                                      "theta_bleed_min": 30.0, "theta_timer_minutes": 30.0,
                                      "grace_period_min": 1.0}),
            ("Best3: 40/50/90m/g5", {"profit_target_general_pct": 40.0, "profit_target_index_0dte_pct": 40.0,
                                      "tight_stop_0dte_pct": 50.0, "backstop_0dte_pct": 50.0,
                                      "tight_stop_multiday_pct": 50.0, "backstop_multiday_pct": 50.0,
                                      "theta_bleed_min": 90.0, "theta_timer_minutes": 90.0,
                                      "grace_period_min": 5.0}),
        ]

        print(f"\n{'Config':<24} {'Trades':<7} {'WR%':<6} {'P&L':>11} "
              f"{'PF':>6} {'Sharpe':>7} {'MaxDD':>7} {'AvgWin':>9} {'AvgLoss':>9}")
        print("-" * 95)

        for label, overrides in sweep_configs:
            PUT_CONFIG_OVERRIDES.clear()
            PUT_CONFIG_OVERRIDES.update(overrides)
            r = run_backtest(pattern_model, pattern_meta, entry_model, entry_features,
                             args.pattern_threshold, args.entry_threshold,
                             tickers, start_date, end_date, stop_model,
                             regime_model, args.regime_threshold, signal_model,
                             put_pattern_model, put_pattern_meta,
                             put_entry_model, put_entry_features, put_entry_threshold)
            pnl_str = f"${r['total_pnl']:+,.0f}"
            marker = " ***" if r['total_pnl'] > 0 else ""
            print(f"{label:<24} {r['trades']:<7} {r['win_rate']:<6.1f} "
                  f"{pnl_str:>11} {r['profit_factor']:>6.2f} {r['sharpe']:>7.2f} "
                  f"{r['max_drawdown_pct']:>6.1f}% {r['avg_win']:>+9,.0f} "
                  f"{r['avg_loss']:>+9,.0f}{marker}")

        PUT_CONFIG_OVERRIDES.clear()
        PUTS_ONLY = False
        sys.exit(0)

    if args.grace_sweep:
        print("\n" + "=" * 70)
        print("GRACE PERIOD SWEEP")
        print("=" * 70)
        print(f"\n{'Grace':<7} {'Trades':<7} {'WR%':<6} {'P&L':>11} "
              f"{'PF':>6} {'Sharpe':>7} {'MaxDD':>7} {'AvgWin':>9} {'AvgLoss':>9} {'AvgHold':>8}")
        print("-" * 85)

        for gp in [0, 1, 2, 3, 5]:
            GRACE_OVERRIDE = float(gp)
            r = run_backtest(pattern_model, pattern_meta, entry_model, entry_features,
                             args.pattern_threshold, args.entry_threshold,
                             tickers, start_date, end_date, stop_model,
                             regime_model, args.regime_threshold, signal_model,
                             put_pattern_model, put_pattern_meta,
                             put_entry_model, put_entry_features, put_entry_threshold)
            pnl_str = f"${r['total_pnl']:+,.0f}"
            label = f"{gp}min" if gp > 0 else "0min"
            current = " ← current" if gp == 5 else ""
            print(f"{label:<7} {r['trades']:<7} {r['win_rate']:<6.1f} "
                  f"{pnl_str:>11} {r['profit_factor']:>6.2f} {r['sharpe']:>7.2f} "
                  f"{r['max_drawdown_pct']:>6.1f}% {r['avg_win']:>+9,.0f} "
                  f"{r['avg_loss']:>+9,.0f} {r['avg_entry_minute']:>7.1f}m{current}")

        GRACE_OVERRIDE = None  # reset
        sys.exit(0)

    if args.sweep:
        print("\n" + "=" * 70)
        print("THRESHOLD SWEEP")
        print("=" * 70)
        print(f"\n{'PatTh':<7} {'EntTh':<7} {'Trades':<7} {'WR%':<6} {'P&L':>11} "
              f"{'PF':>6} {'Sharpe':>7} {'MaxDD':>7} {'AvgMin':>7}")
        print("-" * 70)

        for pt in [0.75, 0.80, 0.85]:
            for et in [0.50, 0.60, 0.70, 0.80]:
                r = run_backtest(pattern_model, pattern_meta, entry_model, entry_features,
                                 pt, et, tickers, start_date, end_date, stop_model,
                                 regime_model, args.regime_threshold, signal_model,
                                 put_pattern_model, put_pattern_meta)
                pnl_str = f"${r['total_pnl']:+,.0f}"
                print(f"{pt:<7.2f} {et:<7.2f} {r['trades']:<7} {r['win_rate']:<6.1f} "
                      f"{pnl_str:>11} {r['profit_factor']:>6.2f} {r['sharpe']:>7.2f} "
                      f"{r['max_drawdown_pct']:>6.1f}% {r['avg_entry_minute']:>7.1f}")

        # Also run without entry filter for comparison
        print(f"\n{'No filter comparison:'}")
        for pt in [0.75, 0.80, 0.85]:
            r = run_backtest(pattern_model, pattern_meta, None, None,
                             pt, 0, tickers, start_date, end_date, stop_model,
                             regime_model, args.regime_threshold, signal_model,
                             put_pattern_model, put_pattern_meta,
                             put_entry_model, put_entry_features, put_entry_threshold)
            pnl_str = f"${r['total_pnl']:+,.0f}"
            print(f"{pt:<7.2f} {'OFF':<7} {r['trades']:<7} {r['win_rate']:<6.1f} "
                  f"{pnl_str:>11} {r['profit_factor']:>6.2f} {r['sharpe']:>7.2f} "
                  f"{r['max_drawdown_pct']:>6.1f}% {r['avg_entry_minute']:>7.1f}")
    else:
        # Single run with full report
        t0 = time.time()
        r = run_backtest(pattern_model, pattern_meta, entry_model, entry_features,
                         args.pattern_threshold, args.entry_threshold,
                         tickers, start_date, end_date, stop_model,
                         regime_model, args.regime_threshold, signal_model,
                         put_pattern_model, put_pattern_meta,
                         put_entry_model, put_entry_features, put_entry_threshold)
        elapsed = time.time() - t0

        # Print summary
        print(f"\n{'=' * 70}")
        print(f"RESULTS — Gold Standard ({r['period']})")
        print(f"{'=' * 70}")
        print(f"  Total P&L:       ${r['total_pnl']:+,.0f}")
        print(f"  Final Portfolio:  ${r['final_portfolio']:,.0f}  ({r['return_pct']:+.1f}%)")
        print(f"  Trades:          {r['trades']} ({r['wins']}W / {r['losses']}L)")
        print(f"  Win Rate:        {r['win_rate']}%")
        print(f"  Profit Factor:   {r['profit_factor']}")
        print(f"  Sharpe Ratio:    {r['sharpe']}")
        print(f"  Max Drawdown:    {r['max_drawdown_pct']}%")
        print(f"  Avg Win:         ${r['avg_win']:+,.0f}")
        print(f"  Avg Loss:        ${r['avg_loss']:+,.0f}")
        print(f"  Avg Entry:       {r['avg_entry_minute']:.0f} min after open")
        print(f"  Runtime:         {elapsed:.0f}s")

        # Machine-parseable metric block (sweep_gates.py parses these lines).
        print("\n  === SWEEP_METRICS ===")
        print(f"  METRIC trades {r['trades']}")
        print(f"  METRIC trades_per_day {r.get('trades_per_day', 0)}")
        print(f"  METRIC trading_days {r['trading_days']}")
        print(f"  METRIC win_rate {r['win_rate']}")
        print(f"  METRIC profit_factor {r['profit_factor']}")
        print(f"  METRIC total_pnl {r['total_pnl']}")
        print(f"  METRIC max_drawdown_pct {r['max_drawdown_pct']}")
        print(f"  METRIC sharpe {r['sharpe']}")
        print(f"  METRIC avg_win {r['avg_win']}")
        print(f"  METRIC avg_loss {r['avg_loss']}")
        print(f"  METRIC runners_100 {r.get('runners_100', 0)}")
        print(f"  METRIC big_runners_200 {r.get('big_runners_200', 0)}")
        print(f"  METRIC runner_pnl {r.get('runner_pnl', 0)}")
        print(f"  METRIC largest_winner_pct {r.get('largest_winner_pct', 0)}")
        print(f"  METRIC largest_winner_dollars {r.get('largest_winner_dollars', 0)}")
        print("  === END_SWEEP_METRICS ===")

        print(f"\n  Signal Funnel:")
        print(f"    Scanned:           {r['signals_sourced']:,} minutes")
        print(f"    Pattern pass:      {r['signals_pattern_pass']:,}")
        if r["entry_threshold"] is not None:
            print(f"    Entry blocked:     {r['signals_entry_blocked']:,}")
        print(f"    Gate blocked:      {r['signals_gate_blocked']:,}")
        if r.get("regime_skipped_days", 0) > 0:
            print(f"    Regime skipped:    {r['regime_skipped_days']} days")
        print(f"    Executed:          {r['trades']}")

        print(f"\n  Per-Ticker:")
        print(f"  {'Ticker':<8} {'Trades':<7} {'WR%':<6} {'P&L':>10} {'Avg':>8}")
        print(f"  {'-'*42}")
        for ticker in sorted(r["per_ticker"].keys(), key=lambda t: r["per_ticker"][t]["pnl"], reverse=True):
            t = r["per_ticker"][ticker]
            wr = t["wins"] / t["trades"] * 100 if t["trades"] > 0 else 0
            avg = t["pnl"] / t["trades"] if t["trades"] > 0 else 0
            print(f"  {ticker:<8} {t['trades']:<7} {wr:<6.0f} ${t['pnl']:>+9,.0f} ${avg:>+7,.0f}")

        # PUT vs CALL breakdown (if PUTs were enabled)
        if ENABLE_PUTS or PUTS_ONLY:
            call_trades = [t for t in r["trade_details"] if t.get("direction", "call") == "call"]
            put_trades = [t for t in r["trade_details"] if t.get("direction") == "put"]
            bear_puts = [t for t in put_trades if t.get("is_bear_mode")]
            green_puts = [t for t in put_trades if not t.get("is_bear_mode")]

            if call_trades:
                c_wins = sum(1 for t in call_trades if t["pnl"] > 0)
                c_pnl = sum(t["pnl"] for t in call_trades)
                c_wr = c_wins / len(call_trades) * 100
                print(f"\n  CALL Summary:  {len(call_trades)} trades, {c_wr:.0f}% WR, ${c_pnl:+,.0f}")

            if put_trades:
                p_wins = sum(1 for t in put_trades if t["pnl"] > 0)
                p_pnl = sum(t["pnl"] for t in put_trades)
                p_wr = p_wins / len(put_trades) * 100
                print(f"  PUT Summary:   {len(put_trades)} trades, {p_wr:.0f}% WR, ${p_pnl:+,.0f}")

                if bear_puts:
                    bp_wins = sum(1 for t in bear_puts if t["pnl"] > 0)
                    bp_pnl = sum(t["pnl"] for t in bear_puts)
                    bp_wr = bp_wins / len(bear_puts) * 100
                    print(f"    Bear mode:   {len(bear_puts)} trades, {bp_wr:.0f}% WR, ${bp_pnl:+,.0f}")

                if green_puts:
                    gp_wins = sum(1 for t in green_puts if t["pnl"] > 0)
                    gp_pnl = sum(t["pnl"] for t in green_puts)
                    gp_wr = gp_wins / len(green_puts) * 100
                    print(f"    Green mode:  {len(green_puts)} trades, {gp_wr:.0f}% WR, ${gp_pnl:+,.0f}")

        if r.get("dip_confirm_count", 0) > 0:
            print(f"\n  DipConfirm:")
            print(f"    Trades with DC:    {r['dip_confirm_count']}")
            print(f"    Avg Savings:       {r['dip_confirm_avg_savings']:+.2f}%")
            for outcome, count in sorted(r["dip_confirm_stats"].items(), key=lambda x: -x[1]):
                print(f"    {outcome:<20} {count:>4}")

        print(f"\n  Exit Reasons:")
        for reason, count in sorted(r["exit_reasons"].items(), key=lambda x: -x[1])[:10]:
            pct = count / r["trades"] * 100
            print(f"    {reason:<25} {count:>4} ({pct:.1f}%)")

        # Generate report
        report_path = REPORT_DIR / "gold_standard_report.md"
        report = generate_report(r, report_path)
        print(f"\n  Report: {report_path}")

        # Save raw results
        raw_path = REPORT_DIR / "gold_standard_raw.json"
        with open(raw_path, "w") as f:
            json.dump(r, f, indent=2, default=str)
        print(f"  Raw data: {raw_path}")

        # Methodology bias check (B5): EXCLUDED_TICKERS is an IN-SAMPLE
        # exclusion (net losers identified on the same data, 2026-05-30).
        # Report the --include-losers number alongside the headline so the
        # selection bias is visible.
        if not args.include_losers and EXCLUDED_TICKERS:
            print(f"\n{'=' * 70}")
            print(f"BIAS CHECK — rerun WITH excluded tickers ({', '.join(sorted(EXCLUDED_TICKERS))})")
            print(f"{'=' * 70}")
            r_all = run_backtest(pattern_model, pattern_meta, entry_model, entry_features,
                                 args.pattern_threshold, args.entry_threshold,
                                 TICKERS, start_date, end_date, stop_model,
                                 regime_model, args.regime_threshold, signal_model,
                                 put_pattern_model, put_pattern_meta,
                                 put_entry_model, put_entry_features, put_entry_threshold)
            print(f"\n  Headline (excl. losers): P&L ${r['total_pnl']:+,.0f} | "
                  f"PF {r['profit_factor']} | WR {r['win_rate']}% | {r['trades']} trades")
            print(f"  Runners (excl. losers):  R100 {r.get('runners_100', 0)} | "
                  f"R200 {r.get('big_runners_200', 0)} | runnerPnL ${r.get('runner_pnl', 0):+,.0f} | "
                  f"largestWin {r.get('largest_winner_pct', 0):+.0f}% (${r.get('largest_winner_dollars', 0):+,.0f}) | "
                  f"maxDD {r.get('max_drawdown_pct', 0)}%")
            print(f"  Include-losers:          P&L ${r_all['total_pnl']:+,.0f} | "
                  f"PF {r_all['profit_factor']} | WR {r_all['win_rate']}% | {r_all['trades']} trades")
            print("  NOTE: EXCLUDED_TICKERS were selected from in-sample results — "
                  "the headline number carries selection bias.")


if __name__ == "__main__":
    main()
