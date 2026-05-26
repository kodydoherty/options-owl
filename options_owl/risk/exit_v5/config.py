"""V5 exit engine configuration — category-aware, DTE-aware.

Category-aware thresholds from backtest_category_sweep.py:
  - High-vol (MSTR, AMD, TSLA, NVDA, etc): wider adaptive trails
  - Index (SPY, QQQ, IWM): tighter trails + profit target at 30%
  - Standard (everything else): moderate trails

DTE-aware thresholds:
  - 0DTE: tight stops, checkpoint, theta bleed
  - Multi-day: wider stops, theta timer at 180min, no checkpoint
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from options_owl.config.settings import Settings


class TickerCategory(Enum):
    HIGH_VOL = "high_vol"
    INDEX = "index"
    STANDARD = "standard"


HIGH_VOL_TICKERS = frozenset({
    "MSTR", "AMD", "TSLA", "NVDA", "AVGO", "META", "COIN", "SMCI", "PLTR",
})

INDEX_TICKERS = frozenset({
    "SPY", "QQQ", "IWM", "DIA", "XLF", "XLK",
})


def categorize_ticker(ticker: str) -> TickerCategory:
    """Classify a ticker into its volatility category."""
    if ticker in HIGH_VOL_TICKERS:
        return TickerCategory.HIGH_VOL
    if ticker in INDEX_TICKERS:
        return TickerCategory.INDEX
    return TickerCategory.STANDARD


@dataclass(frozen=True)
class DefensiveConfig:
    """Bid disappearance detection settings."""
    bid_zero_timeout_sec: float = 30.0


@dataclass(frozen=True)
class AdaptiveTier:
    """One stage of the adaptive trailing stop."""
    min_peak_gain: float   # peak gain % to activate this tier
    trail_width: float     # % drop from peak to trigger exit


@dataclass(frozen=True)
class V5Config:
    """Complete v5 exit engine configuration.

    Category-aware strategy from backtest_category_sweep.py.
    Key insight: high-vol tickers need wider room, indexes benefit from
    profit targets, and soft trail should keep 60% (not 50%) of gains.
    """

    # Gate 1: EOD cutoff — 0DTE only, 15min before close
    eod_cutoff_minutes_before_close: float = 15.0

    # Gate 2: Bid disappearance
    defensive: DefensiveConfig = field(default_factory=DefensiveConfig)

    # Grace period — skip all exits in first 5 minutes
    grace_period_min: float = 5.0

    # Gate 3: Profit target — index 0DTE only (take gains at 30%)
    profit_target_index_0dte_pct: float = 30.0

    # Gate 3 alt: General profit target — fires for ALL trades (0 = disabled)
    # Used by PUT scalp strategy: take gains at 50%
    profit_target_general_pct: float = 0.0

    # Gate 4: Scalp trail — peaked +20%, faded to <60% of peak
    scalp_peak_threshold_pct: float = 20.0
    scalp_fade_ratio: float = 0.6
    scalp_confirm_threshold: float = 0.2

    # Gate 5: Checkpoint cut — 0DTE only (uses underlying_against_threshold from gate 6)
    # Backtested 2026-05-22: 15% checkpoint optimal with scalp+runner combo
    checkpoint_drop_pct: float = 15.0

    # Gate 6: Graduated stop — underlying-based, DTE-aware
    # Backtested 2026-05-22: 15/30 tight/backstop optimal (was 35/65)
    # Rationale: ML entries target precise timing, cut wrong trades fast
    tight_stop_0dte_pct: float = 15.0
    backstop_0dte_pct: float = 30.0
    tight_stop_multiday_pct: float = 30.0
    backstop_multiday_pct: float = 50.0
    underlying_against_threshold: float = 0.5

    # Gate 7: Soft trail — 15-50% peak band, keep 60% of gains
    # Backtested: band_low=15 avoids premature exits on small pops (+$4K improvement)
    soft_trail_band_low_pct: float = 15.0
    soft_trail_band_high_pct: float = 50.0
    soft_trail_keep_pct: float = 0.60

    # Gate 8: Adaptive trail — per-category tiers (drop from peak)
    # High-vol: wider trails — let wild swings breathe
    adaptive_highvol_tiers: tuple[AdaptiveTier, ...] = (
        AdaptiveTier(400, 35),   # moonshot: 35% drop (was 30)
        AdaptiveTier(150, 55),   # runner: 55% drop (was 45)
        AdaptiveTier(40, 50),    # active: 50% drop (was 40)
    )
    # Index: tighter trails — more predictable moves
    adaptive_index_tiers: tuple[AdaptiveTier, ...] = (
        AdaptiveTier(300, 25),   # moonshot: 25% drop
        AdaptiveTier(100, 40),   # runner: 40% drop
        AdaptiveTier(30, 35),    # active: 35% drop (was 40)
    )
    # Standard: same as index
    adaptive_standard_tiers: tuple[AdaptiveTier, ...] = (
        AdaptiveTier(300, 25),
        AdaptiveTier(100, 40),
        AdaptiveTier(30, 35),
    )

    # Gate 9: Theta bleed (0DTE) — 120min+ and down 30%+
    theta_bleed_min: float = 120.0
    theta_bleed_drop_pct: float = 30.0

    # Gate 9 alt: Theta timer (multi-day) — 180min+ and down 15%+
    # Cuts stale multi-day losers instead of holding forever
    theta_timer_minutes: float = 180.0
    theta_timer_loss_pct: float = 15.0

    # Sideways scalp — detect choppy/range-bound trades and take small profits
    # Requires premium history (accumulated on TradeState each cycle).
    # Only fires when: trade is profitable >= take_profit_pct AND peak gain < peak_cap_pct
    # AND at least signals_needed of 4 indicators agree the trade is sideways.
    sideways_take_profit_pct: float = 10.0      # minimum gain % to scalp
    sideways_peak_cap_pct: float = 30.0         # skip if trade already trended > this %
    sideways_signals_needed: int = 2            # how many of 4 indicators must agree
    sideways_lookback: int = 20                 # ticks to look back for range calculation
    sideways_range_pct: float = 10.0            # premium range / entry < this % = range-bound
    sideways_no_new_high_min: float = 8.0       # minutes since last premium peak
    sideways_underlying_flat_pct: float = 0.15  # underlying moved < this % from entry
    sideways_cross_count: int = 3               # premium crossed entry N+ times = choppy
    sideways_min_ticks: int = 10                # minimum history length before firing

    # Early-pop gate: tighten backstop for trades that peaked early then faded.
    # Backtested: +$1,737 at zero cost — catches crashers without hurting runners.
    # Detection: peak must occur within first N min, then fade M% from peak by check time.
    # Action: override backstop_0dte/multiday with tighter values for graduated_stop.
    early_pop_peak_window_min: float = 12.0     # peak must occur within first N minutes
    early_pop_fade_pct: float = 10.0            # premium must fade this % from peak
    early_pop_check_after_min: float = 12.0     # evaluate the pattern at this elapsed time
    early_pop_min_peak_gain_pct: float = 3.0    # peak must be at least +N% above entry
    early_pop_backstop_0dte_pct: float = 25.0   # tighter backstop when pattern detected (0DTE)
    early_pop_backstop_multiday_pct: float = 40.0  # tighter backstop (multi-day)

    def get_adaptive_tiers(self, category: TickerCategory) -> tuple[AdaptiveTier, ...]:
        """Get adaptive trail tiers for a ticker category."""
        if category == TickerCategory.HIGH_VOL:
            return self.adaptive_highvol_tiers
        if category == TickerCategory.INDEX:
            return self.adaptive_index_tiers
        return self.adaptive_standard_tiers

    @classmethod
    def from_settings(cls, settings: Settings) -> V5Config:
        """Create config from Settings object. V5 uses backtested defaults."""
        return cls()


# ── V6: Per-ticker optimal configs (from backtest_per_ticker_tuning.py) ────
#
# Each ticker was tested against 12 FSM config variations. The optimal config
# was selected based on total P&L improvement over default V5Config.
# Tickers not listed here use default V5Config.

_default = V5Config

TICKER_CONFIGS: dict[str, V5Config] = {
    # NVDA: EARLY_PROFIT — take gains at 20%, keep 70% of peak in soft trail
    "NVDA": V5Config(
        profit_target_general_pct=20.0,
        soft_trail_keep_pct=0.70,
    ),
    # GOOGL: WIDE_STOP — wider than default to avoid premature stop-outs
    "GOOGL": V5Config(
        tight_stop_0dte_pct=20.0,
        backstop_0dte_pct=40.0,
        checkpoint_drop_pct=20.0,
    ),
    # TSLA: LONG_GRACE — 8 min grace to let momentum develop
    "TSLA": V5Config(grace_period_min=8.0),
    # IWM: WIDE_STOP — wider room for ETF swings
    "IWM": V5Config(
        tight_stop_0dte_pct=20.0,
        backstop_0dte_pct=40.0,
        checkpoint_drop_pct=20.0,
    ),
    # QQQ: LONG_GRACE — 8 min grace
    "QQQ": V5Config(grace_period_min=8.0),
    # META: DEFENSIVE — tighter adaptive trails, faster theta bleed
    "META": V5Config(
        adaptive_highvol_tiers=(
            AdaptiveTier(400, 25), AdaptiveTier(150, 40), AdaptiveTier(40, 35),
        ),
        theta_bleed_min=90.0,
        theta_bleed_drop_pct=20.0,
    ),
    # AAPL: DEFENSIVE — tighter adaptive trails, faster theta bleed
    "AAPL": V5Config(
        adaptive_standard_tiers=(
            AdaptiveTier(400, 25), AdaptiveTier(150, 40), AdaptiveTier(40, 35),
        ),
        theta_bleed_min=90.0,
        theta_bleed_drop_pct=20.0,
    ),
    # AMZN: TIGHT_TRAIL — tighter adaptive trail tiers (AMZN is STANDARD category)
    "AMZN": V5Config(
        adaptive_standard_tiers=(
            AdaptiveTier(300, 20), AdaptiveTier(100, 30), AdaptiveTier(30, 25),
        ),
    ),
    # AVGO: EARLY_PROFIT — same as NVDA
    "AVGO": V5Config(
        profit_target_general_pct=20.0,
        soft_trail_keep_pct=0.70,
    ),
    # MSFT: EARLY_PROFIT — same as NVDA
    "MSFT": V5Config(
        profit_target_general_pct=20.0,
        soft_trail_keep_pct=0.70,
    ),
    # MSTR: TIGHT+QUICK — tight trail + quick scalp + higher keep
    "MSTR": V5Config(
        adaptive_highvol_tiers=(
            AdaptiveTier(400, 25), AdaptiveTier(150, 40), AdaptiveTier(40, 35),
        ),
        scalp_peak_threshold_pct=15.0,
        scalp_fade_ratio=0.50,
        soft_trail_keep_pct=0.70,
    ),
}


def get_ticker_config(
    ticker: str,
    use_per_ticker: bool = False,
    option_type: str = "call",
) -> V5Config:
    """Return per-ticker V5Config if enabled, else default.

    When use_per_ticker is True, looks up the ticker in TICKER_CONFIGS.
    Unknown tickers fall through to default V5Config.

    PUT trades use PUT_SCALP_CONFIG — simple fixed target/stop/time exits
    optimized for cheap 0DTE PUT scalps (backtested over 3+ years).
    """
    if option_type.lower() == "put":
        return PUT_SCALP_CONFIG
    if use_per_ticker and ticker in TICKER_CONFIGS:
        return TICKER_CONFIGS[ticker]
    return V5Config()


# ── PUT Scalp Config ─────────────────────────────────────────────────────────
# Backtested (3+ years): +50% target, -60% stop, 60min max hold.
# Cheap premiums ($0.05-$0.50) in afternoon (1:00-2:30 PM) slots.
# Simple exits — no adaptive trails or soft trails needed.

PUT_SCALP_CONFIG = V5Config(
    # Grace period — shorter for PUTs (premium moves fast)
    grace_period_min=3.0,

    # Gate 3: General profit target at 50% (fires for all PUTs)
    profit_target_general_pct=50.0,
    profit_target_index_0dte_pct=50.0,

    # Gate 6: Hard stop at 60% (both tight and backstop same = simple stop)
    tight_stop_0dte_pct=60.0,
    backstop_0dte_pct=60.0,
    tight_stop_multiday_pct=60.0,
    backstop_multiday_pct=60.0,

    # Gate 9: Max hold time = 60 minutes, exit regardless of P&L
    theta_bleed_min=60.0,
    theta_bleed_drop_pct=-100.0,  # always true (any P&L triggers at 60min)
    theta_timer_minutes=60.0,
    theta_timer_loss_pct=-100.0,

    # Disable complex trails — PUTs use simple target/stop
    scalp_peak_threshold_pct=999.0,    # effectively disabled
    soft_trail_band_low_pct=999.0,     # effectively disabled
    adaptive_highvol_tiers=(AdaptiveTier(9999, 99),),
    adaptive_index_tiers=(AdaptiveTier(9999, 99),),
    adaptive_standard_tiers=(AdaptiveTier(9999, 99),),
)


# Backward compat alias
V4Config = V5Config
