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
    "MSTR", "AMD", "TSLA", "NVDA", "META", "SMCI", "PLTR",
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


# ---------------------------------------------------------------------------
# Strike grid configuration — per-ticker option strike intervals
# ---------------------------------------------------------------------------
# From ThetaData research (6.7M rows, 14 tickers, 124 trading days).
# Strike spacing varies 7.6x (SPY $1/0.14% vs AMZN $2.50/1.07%).

STRIKE_INTERVALS: dict[str, float] = {
    # Fine grid — small strikes relative to price
    "SPY": 1.0, "QQQ": 1.0, "IWM": 0.5, "NVDA": 0.5, "MSTR": 0.5,
    # Standard grid
    "META": 2.5, "MSFT": 2.5, "TSLA": 2.5, "PLTR": 1.0, "AVGO": 2.5,
    # Wide grid — large strikes relative to price
    "GOOGL": 2.5, "AAPL": 2.5, "AMD": 2.5, "AMZN": 2.5,
}

DEFAULT_STRIKE_INTERVAL: float = 2.5

# Tickers where the strike grid is fine enough to allow 3 strikes OTM
_FINE_GRID_TICKERS = frozenset({"SPY", "QQQ", "IWM", "NVDA", "MSTR"})
# Tickers where the strike grid is wide — only 1 strike OTM allowed
_WIDE_GRID_TICKERS = frozenset({"AAPL", "AMD", "AMZN", "GOOGL"})


def get_max_otm_distance(ticker: str) -> float:
    """Return the maximum allowed OTM distance in dollars for a ticker.

    Fine-grid tickers (SPY, QQQ, IWM, NVDA, MSTR): 3 strikes OTM
    Standard tickers (META, TSLA, PLTR, etc.): 2 strikes OTM
    Wide-grid tickers (AAPL, AMD, AMZN, GOOGL): 1 strike OTM
    """
    interval = STRIKE_INTERVALS.get(ticker, DEFAULT_STRIKE_INTERVAL)

    if ticker in _FINE_GRID_TICKERS:
        return interval * 3
    elif ticker in _WIDE_GRID_TICKERS:
        return interval * 1
    else:
        return interval * 2


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
# Aggressive scalp: take profits early, trail tightly, cut losers fast.
# PUTs are momentum plays — lock gains quickly before theta + reversal kill them.

PUT_SCALP_CONFIG = V5Config(
    # Grace period — shorter for PUTs (premium moves fast)
    grace_period_min=3.0,

    # Gate 3: No fixed profit target — let trail system lock in gains (backtest: +$63K vs +$20K)
    # Breakeven ratchet at +20% guarantees no loss, soft/adaptive trail lock in profits progressively
    profit_target_general_pct=0.0,
    profit_target_index_0dte_pct=0.0,

    # Gate 3.5: Breakeven ratchet still works (once +20%, floor = entry)

    # Gate 3.7: Scaleout still works (sell 1/3 at +20% if 3+ contracts)

    # Gate 4: Scalp trail — peaked +15%, faded below 60% of peak → exit
    scalp_peak_threshold_pct=15.0,

    # Gate 6: Hard stop at 50% loss
    tight_stop_0dte_pct=50.0,
    backstop_0dte_pct=50.0,
    tight_stop_multiday_pct=50.0,
    backstop_multiday_pct=50.0,

    # Gate 7: Soft trail — once peaked +15%, keep 60% of gains
    soft_trail_band_low_pct=15.0,
    soft_trail_band_high_pct=50.0,
    soft_trail_keep_pct=0.6,

    # Gate 8: Adaptive trail — active at +20%, trail at 40% drop from peak
    adaptive_highvol_tiers=(AdaptiveTier(20, 40),),
    adaptive_index_tiers=(AdaptiveTier(20, 35),),
    adaptive_standard_tiers=(AdaptiveTier(20, 35),),

    # Gate 9: No hold time limit — let trail system handle exits
    # Backtest: no limit = $63K PF 1.57 vs 60m limit = $52K PF 1.46
    theta_bleed_min=999.0,
    theta_bleed_drop_pct=-100.0,
    theta_timer_minutes=999.0,
    theta_timer_loss_pct=-100.0,
)


# Backward compat alias
V4Config = V5Config


def apply_v7_wide_trail_exits(cfg: V5Config, is_put: bool = False) -> V5Config:
    """Apply the V7 convex EXIT transform to a V5Config.

    Validated in the exit-only ablation (both OOS windows beat V6 exits at identical
    win rate: OOS +72%, OOS2 +27% P&L, higher PF).

    - no profit ceiling (let winners run) — both CALL and PUT
    - widening adaptive trail: moonshot x1.5, runner x1.3, active x1.1 (clamp 5-90%)
    - CALLs ONLY: faster stall-cut on dead losers (theta 60min / 25%).
      PUTs KEEP their no-hold-limit (theta 999) — they ride slow-building crashes;
      a time-cut would clip exactly the down-day moves we want to capture
      (separately validated: PUT no-limit $63K vs 60min-limit $52K).
    The scaleout-off / 2PM-tighten-off / breakeven-ratchet-on parts are handled via
    settings in monitor_bridge (they are flag-driven, not V5Config fields).
    """
    from dataclasses import replace as _replace

    def _widen(tiers: tuple[AdaptiveTier, ...]) -> tuple[AdaptiveTier, ...]:
        out = []
        for t in tiers:
            if t.min_peak_gain >= 300:
                w = t.trail_width * 1.5
            elif t.min_peak_gain >= 100:
                w = t.trail_width * 1.3
            else:
                w = t.trail_width * 1.1
            out.append(AdaptiveTier(t.min_peak_gain, max(5.0, min(90.0, w))))
        return tuple(out)

    changes: dict = dict(
        profit_target_index_0dte_pct=0.0,
        profit_target_general_pct=0.0,
        adaptive_highvol_tiers=_widen(cfg.adaptive_highvol_tiers),
        adaptive_index_tiers=_widen(cfg.adaptive_index_tiers),
        adaptive_standard_tiers=_widen(cfg.adaptive_standard_tiers),
    )
    if not is_put:
        changes["theta_bleed_min"] = 60.0
        changes["theta_bleed_drop_pct"] = 25.0
    return _replace(cfg, **changes)
