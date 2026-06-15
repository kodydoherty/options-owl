from pydantic import computed_field
from pydantic_settings import BaseSettings


def _parse_int_list(val: str) -> list[int]:
    val = val.strip()
    if not val:
        return []
    return [int(x.strip()) for x in val.split(",") if x.strip()]


class Settings(BaseSettings):
    DISCORD_TOKEN: str = ""
    DISCORD_GUILD_IDS: str = "1469404711613497591"
    DISCORD_CHANNEL_IDS: str = ""
    WEBULL_APP_KEY: str = ""
    WEBULL_APP_SECRET: str = ""
    WEBULL_ACCOUNT_ID: str = ""  # optional — auto-detects from API based on MARGIN_ACCOUNT setting
    WEBULL_KILL_SWITCH: bool = False  # emergency halt all orders
    WEBULL_ENTRY_AGGRESS_PCT: float = 5.0  # bump BUY limit price above ask to cross spread (0DTE spreads are wide)
    MAX_ENTRY_RETRIES: int = 3  # retry entry up to N times with fresh pricing (10s per attempt)
    MAX_ENTRY_CHASE_PCT: float = 15.0  # max % above signal premium we'll chase on retries
    GFV_BUFFER_PCT: float = 15.0  # safety buffer on GFV limit (only allow 85% of start-of-day balance)
    MARGIN_ACCOUNT: bool = False  # margin accounts skip GFV protection (no unsettled fund concerns)
    PAPER_TRADE: bool = True

    # Emergency alerts — DM these Discord user IDs on critical events
    DISCORD_ALERT_USER_IDS: str = ""  # comma-separated Discord user IDs to DM on alerts
    # Expiry safety: force-close all open trades N minutes before market close
    EXPIRY_SAFETY_MINUTES: int = 10  # close everything 10 min before 4 PM ET
    PORTFOLIO_SIZE: float = 2000.0
    # Auto-adaptive sizing: set to 0 to auto-compute from portfolio size
    # Small portfolios (<$8K) → 2 concurrent, 20% position (concentrate capital)
    # Large portfolios (>=$8K) → 4 concurrent, 15% position (diversify)
    MAX_POSITION_PCT: float = 0.0  # 0 = auto-adapt from portfolio size
    MAX_DCA_POSITION_PCT: float = 0.0  # 0 = auto-adapt (half of MAX_POSITION_PCT)
    MAX_CONCURRENT: int = 0  # 0 = auto-adapt from portfolio size
    MIN_SCORE: int = 78  # only trade signals with score >= this (v2.1: raised from 75)
    # ML-sourced signals carry score = int(model_confidence * 100), a DIFFERENT
    # scale than Discord scores (which run well past 100). The model's own
    # probability threshold is the real entry gate; this floor is only a sanity
    # check applied by ScoreGate and signal_consumer to ML_SOURCING signals so
    # they don't fall into the MIN_SCORE dead band (e.g. conf 0.74-0.77 → score
    # 74-77 < MIN_SCORE 78 would otherwise always be rejected).
    ML_MIN_SCORE: int = 60
    DAILY_LOSS_LIMIT_PCT: float = 10.0
    DB_PATH: str = "journal/raw_messages.db"

    # Feature 1: Greeks calculation
    ENABLE_GREEKS: bool = False

    # Feature 2: IV Rank / IV Percentile filtering
    ENABLE_IV_FILTER: bool = False
    IV_RANK_MIN: float = 0.0  # minimum IV rank to accept a trade (0 = no filter)
    IV_RANK_MAX: float = 100.0  # maximum IV rank

    # Feature 3: VIX regime detection
    ENABLE_VIX_FILTER: bool = False
    VIX_MAX: float = 35.0  # pause trading above this VIX level
    VIX_HIGH_THRESHOLD: float = 25.0  # reduce position size above this
    VIX_POSITION_REDUCTION_PCT: float = 50.0  # reduce position size by this %

    # Feature 4: Theta decay exit rules
    ENABLE_THETA_DECAY_EXIT: bool = False
    THETA_EXIT_DTE_THRESHOLD: int = 1  # aggressive exits when DTE <= this
    THETA_EXIT_LOSS_PCT: float = 50.0  # exit if down more than this %
    THETA_EXIT_TIME_MINUTES: int = 150  # 0DTE: exit if held longer than 2.5hrs with no profit

    # Feature 5: Analyst / bot performance filtering
    ENABLE_ANALYST_FILTER: bool = False
    ANALYST_MIN_WIN_RATE: float = 0.0  # minimum historical win rate (0 = no filter)
    ANALYST_MIN_TRADES: int = 10  # min trades before filtering kicks in

    # Data feed configuration
    DATA_FEED_PROVIDER: str = "yfinance"  # "yfinance", "polygon", "webull"
    DATA_FEED_POLL_INTERVAL: int = 3  # seconds — Redis WS data is primary, fast exit decisions
    POLYGON_API_KEY: str = ""
    WEBULL_PRIMARY_QUOTES: bool = False  # Use Webull DataClient as primary option quote source
    # Polygon WS uses 1 concurrent connection per API key. When running multiple
    # owlets against a single API key, flip this off on the non-primary owlets
    # so only one bot holds the WS and the others fall back to REST polling.
    ENABLE_POLYGON_WS: bool = True

    # Shared candle DB from harvester (mounted read-only into agent containers)
    SHARED_CANDLE_DB: str = ""

    # Feature 7: Dynamic Kelly position sizing
    ENABLE_KELLY_SIZING: bool = False
    KELLY_FRACTION: float = 0.25  # quarter Kelly (conservative)
    KELLY_MIN_PCT: float = 5.0  # minimum position size %
    KELLY_MAX_PCT: float = 25.0  # maximum position size %
    KELLY_MIN_TRADES: int = 20  # min trades before Kelly kicks in
    KELLY_DRAWDOWN_HALVE_PCT: float = 10.0  # halve sizing if drawdown exceeds this

    # Feature 10: Graduated scale-out (partial profit taking at each target)
    ENABLE_PARTIAL_PROFITS: bool = False
    PARTIAL_CLOSE_PCT: float = 50.0  # legacy: close this % of contracts at T1 (used if scale-out disabled)
    ENABLE_SCALE_OUT: bool = True  # graduated exits across T1-T5
    SCALE_OUT_T1_PCT: float = 20.0  # sell 20% of remaining at T1
    SCALE_OUT_T2_PCT: float = 25.0  # sell 25% of remaining at T2
    SCALE_OUT_T3_PCT: float = 33.0  # sell 33% of remaining at T3
    SCALE_OUT_T4_PCT: float = 50.0  # sell 50% of remaining at T4
    # T5: always closes all remaining contracts

    # Feature 8: Circuit breakers
    ENABLE_CIRCUIT_BREAKERS: bool = False
    CB_MAX_CONSECUTIVE_LOSSES: int = 3  # halt after N consecutive losses
    CB_MAX_DRAWDOWN_PCT: float = 15.0  # halt if portfolio down this % from peak
    CB_OPENING_BUFFER_MINUTES: int = 10  # avoid first N minutes after open
    CB_CLOSING_BUFFER_MINUTES: int = 15  # avoid last N minutes before close
    CB_INTRADAY_LOSS_HALT_PCT: float = 5.0  # emergency close all if daily loss hits this

    # Feature 9: Spread strategies
    ENABLE_SPREADS: bool = False
    SPREAD_DEFAULT_WIDTH: float = 5.0  # strike width for vertical spreads
    SPREAD_PROFIT_TARGET_PCT: float = 50.0  # close at this % of max profit
    SPREAD_STRATEGY: str = "single_leg"  # "single_leg", "vertical_spread", "iron_condor"

    # Feature 11: Liquidity filter
    ENABLE_LIQUIDITY_FILTER: bool = False
    MIN_OPEN_INTEREST: int = 100
    MIN_VOLUME: int = 50
    MAX_BID_ASK_SPREAD_PCT: float = 15.0  # max spread as % of midpoint

    # Slippage simulation
    SIMULATED_ENTRY_SLIPPAGE_BPS: float = 50.0  # basis points (50 bps = 0.5%)
    SIMULATED_EXIT_SLIPPAGE_BPS: float = 50.0

    # DCA (Dollar Cost Averaging) — split entry into tranches
    ENABLE_DCA: bool = True
    DCA_TRANCHES: int = 3  # split into 3 buys
    DCA_FIRST_PCT: float = 40.0  # first buy is 40% of total contracts
    DCA_DIP_PCT: float = 15.0  # buy next tranche when premium dips 15% from entry
    DCA_TIME_LIMIT_MINUTES: int = 30  # stop trying to DCA after 30 min

    # ENRG: Early Negative Thesis Revalidation Gate
    # v3: DISABLED — grace period + hard stop removed, thesis_cut replaces this
    ENABLE_ENRG: bool = False
    ENRG_WIDEN_STOP_PCT: float = 15.0  # only used if ENABLE_ENRG=True

    # Momentum confirmation: reject entry if 5m/15m candles show underlying fading
    # RE-ENABLED (2026-06-05): Was disabled but ML alone missed candle-level signals.
    # Both directional_regime and momentum_confirm were off, leaving zero candle checks.
    ENABLE_MOMENTUM_CONFIRM: bool = True

    # Smart dip-confirm: when premium is fading at entry, check underlying vs
    # support/VWAP before deciding to wait or skip.
    # - Above VWAP + fading → enter (premium decay, not trend break)
    # - Below VWAP + fading → poll for uptick (bounce off support)
    # - No uptick after max_polls → skip (breakdown, no floor)
    # Backtested: +$2,525 improvement, skips ~20% of trades that would lose.
    ENABLE_DIP_CONFIRM: bool = False  # off by default — enable per-owlet
    DIP_CONFIRM_MAX_POLLS: int = 6  # max polls waiting for uptick (total ~30s after initial 5s check)
    DIP_CONFIRM_POLL_SEC: float = 5.0  # seconds between premium checks
    DIP_CONFIRM_FADE_PCT: float = 1.0  # premium must have faded >= this % to trigger wait

    # Stop loss: v2.2 §3 — hard stop at -30% from entry (always enabled)
    PREMIUM_STOP_ENABLED: bool = True
    PREMIUM_STOP_PCT: float = 30.0  # -30% from entry = full exit
    PEAK_STOP_DROP_PCT: float = 40.0  # legacy
    STOP_GRACE_PERIOD_MINUTES: int = 20  # max grace period (safety cap)
    # Smart grace: end grace early when underlying confirms trade direction
    # Backtested: +$689 over fixed 20min grace on 67 signals
    ENABLE_SMART_GRACE: bool = True
    ENABLE_CATASTROPHIC_STOP: bool = False
    CATASTROPHIC_STOP_PCT: float = 45.0
    ENABLE_UNDERLYING_STOP: bool = False
    MIN_UNDERLYING_STOP_PCT: float = 0.5

    # v2.2 §4: BE clamp — once peak gain reaches +15%, floor = entry (never go red after green)
    ENABLE_BE_CLAMP: bool = True
    BE_CLAMP_ACTIVATION_PCT: float = 15.0  # activate once peak gain reaches this %

    # v2.2 §11: Soft trail (15-35% band) — floor = entry + 50% of peak gain
    # Fills the gap where adaptive trail is dormant. More protective than profit_retrace.
    ENABLE_SOFT_TRAIL: bool = True
    SOFT_TRAIL_MIN_PCT: float = 15.0   # activate at this peak gain %
    SOFT_TRAIL_MAX_PCT: float = 35.0   # hand off to adaptive trail above this
    SOFT_TRAIL_FLOOR_PCT: float = 50.0 # keep this % of peak gain (floor = entry + gain * 50%)

    # Trailing premium stop: once in profit, trail the stop up from peak
    ENABLE_TRAILING_STOP: bool = True
    TRAILING_STOP_ACTIVATION_PCT: float = 15.0  # activate when premium up 15% from entry (v2.1: earlier BE — half of contracts peaked +85% within 40min)
    TRAILING_STOP_DROP_PCT: float = 40.0  # exit if premium drops 40% from its peak

    # No-momentum exit: DISABLED in v4.1 — never fired in production across 67 signals.
    # When it did fire historically, ALL 6 triggers were losses (-$517 total).
    ENABLE_NO_MOMENTUM_EXIT: bool = False
    NO_MOMENTUM_MINUTES: int = 45
    NO_MOMENTUM_MIN_GAIN_PCT: float = 5.0

    # --- ML sell model (trained on 2yr of 1-minute options data) ---
    ENABLE_ML_EXIT: bool = False  # ML disabled — actively harmful with 10-day training data (backtested: every ML scenario lost money)
    ML_OVERRIDE_TARGETS: bool = True  # when ML says hold, skip target scale-outs
    ML_OVERRIDE_TRAILS: bool = True  # when ML says hold, suppress dollar/adaptive trail exits
    ML_OVERRIDE_MIN_FUTURE_PNL: float = 5.0  # only override if ML expects >= this % future gain

    # --- Vinny's strategy (phase-based trailing stops, VIX-adjusted) ---
    ENABLE_VINNY_STRATEGY: bool = True  # master switch for Vinny's strategy

    # Exit engine version: "v3" (current production) or "v5" (scalp_and_hold)
    # v5: 45min grace, scalp trail, graduated stop w/ momentum confirm, checkpoint, wider trails
    EXIT_ENGINE: str = "v5"  # V5 FSM is the active production engine for all owlets

    # --- V6 enhancements (all default OFF — enable per-owlet via docker-compose) ---
    # V6 builds on V5 FSM with per-ticker configs, entry filters, and exit improvements.
    # Backtested: $7,020 → $16,514 (+135%) on 133 signals (Apr 10 – May 1, 2026).

    # Per-ticker FSM configs: each ticker gets its backtested-optimal V5Config
    # (e.g., NVDA → EARLY_PROFIT, GOOGL → WIDE_STOP, META → DEFENSIVE)
    ENABLE_V6_PER_TICKER_CONFIG: bool = False

    # Break-even ratchet: once gain hits trigger %, stop floor moves to entry price
    ENABLE_V6_BREAKEVEN_RATCHET: bool = False
    V6_BREAKEVEN_TRIGGER_PCT: float = 20.0

    # 2PM trail tightening: tighten adaptive trail widths + raise soft trail keep after 2PM ET
    # Accounts for gamma acceleration in last 2 hours of 0DTE ("gamma death zone")
    ENABLE_V6_2PM_TIGHTEN: bool = False
    V6_2PM_TRAIL_TIGHTEN_FACTOR: float = 0.7   # multiply adaptive trail widths by this
    V6_2PM_SOFT_TRAIL_BOOST: float = 0.15       # add this to soft_trail_keep_pct

    # Premium cap: reject non-index entries where premium > cap (blocks META $25.35 disasters)
    # Tiered: base=$6, score 120+=$7, score 150+=$9. Index tickers exempt.
    # Backtested: $6/$7/$9 adds +$195 vs $5/$6/$8 (lets profitable GOOGL/TSLA through)
    ENABLE_V6_PREMIUM_CAP: bool = False
    V6_PREMIUM_CAP: float = 6.0
    V6_PREMIUM_CAP_MID: float = 7.0    # score 120+
    V6_PREMIUM_CAP_HIGH: float = 9.0   # score 150+

    # Spread-cost gate: reject entries where bid-ask spread > threshold % of premium
    ENABLE_V6_SPREAD_GATE: bool = False
    V6_MAX_SPREAD_PCT: float = 15.0

    # OTM distance gate: reject strikes too far out-of-the-money.
    # Disabled after backtest showed it blocks $170K+ in profitable trades.
    # The ML pattern model already captures moneyness — this gate double-filters.
    ENABLE_OTM_DISTANCE_GATE: bool = True
    MAX_OTM_DISTANCE_PCT: float = 1.5

    # Delta entry gate: reject options outside the delta sweet spot.
    # Replaces static premium_cap + otm_distance with a market-derived measure
    # that adapts to ticker price, IV, and DTE automatically.
    # Backtested: delta 0.15-0.70 = $+491K, $742/trade, 75% WR over 126 days.
    ENABLE_DELTA_GATE: bool = False
    DELTA_ENTRY_MIN: float = 0.15  # reject far OTM (< 15% chance ITM)
    DELTA_ENTRY_MAX: float = 0.70  # reject deep ITM (overpaying for intrinsic)

    # Sideways scalp: detect choppy/range-bound trades and take small profits early.
    # Backtested: +$1,509 improvement over baseline (30 sideways exits, 76% WR).
    # Only fires when trade is profitable AND hasn't trended significantly (peak < 30%).
    ENABLE_V6_SIDEWAYS_SCALP: bool = False

    # Early-pop gate: tighten backstop for trades that peaked early then faded.
    # Backtested: +$1,737 at zero cost across 192 trades — catches crashers without hurting runners.
    # Detection: premium peaked in first 12min, faded 10%+ by minute 12, peak was at least +3%.
    # Action: tighten backstop from 65% to 35% (0DTE) for these trades only.
    ENABLE_V6_EARLY_POP_GATE: bool = False

    # Scale-out at +20%: sell fraction of contracts at first +20% gain to lock partial profits
    # Backtested: 46 fires, $4,224 locked. Single biggest P&L improvement mechanism.
    ENABLE_V6_SCALEOUT: bool = False
    V6_SCALEOUT_GAIN_PCT: float = 20.0
    V6_SCALEOUT_FRACTION: float = 0.333         # 1/3 of contracts
    V6_SCALEOUT_MIN_CONTRACTS: int = 3           # need at least this many to scale out

    # Mid-trade DCA: add contracts when premium dips during the developing phase.
    # Backtested: +$4,120 improvement (23 fires across 6 tickers).
    # Only fires for tickers where backtesting showed positive DCA impact.
    ENABLE_V6_DCA: bool = False
    V6_DCA_TICKERS: str = "IWM,SPY,QQQ,AMZN,NVDA"  # comma-separated whitelist
    V6_DCA_MIN_MINUTES: float = 8.0              # earliest DCA can fire (minutes after entry)
    V6_DCA_MAX_MINUTES: float = 20.0             # latest DCA can fire
    V6_DCA_MIN_DIP_PCT: float = 15.0             # minimum dip from entry to trigger
    V6_DCA_MAX_DIP_PCT: float = 35.0             # maximum dip (beyond this, thesis is broken)
    V6_DCA_UNDERLYING_THRESHOLD: float = 0.5     # block DCA if underlying moved against > this %

    # --- v5 dynamic exit gates (signal-driven, no fixed time windows) ---

    # Dynamic scalp: take profit when premium peaked AND fading AND underlying NOT confirming
    # No time window — fires at 5min or 50min, whenever the signals say so
    ENABLE_SCALP_TRAIL: bool = True
    SCALP_TRAIL_PEAK_PCT: float = 20.0      # peak gain threshold to consider scalping
    SCALP_TRAIL_FADE_PCT: float = 60.0      # exit if current gain < this % of peak gain
    SCALP_TRAIL_UNDERLYING_CONFIRM_PCT: float = 0.2  # if underlying moved > this % in trade dir, HOLD not scalp

    # Dynamic checkpoint: cut when BOTH premium AND underlying are against us
    # No fixed time — fires as soon as both signals agree the trade is dead
    ENABLE_CHECKPOINT: bool = True
    CHECKPOINT_PREMIUM_DROP_PCT: float = 30.0   # premium down this % from entry (v5b: was 15)
    CHECKPOINT_UNDERLYING_AGAINST_PCT: float = 0.3  # underlying moved this % against trade
    CHECKPOINT_MIN_ELAPSED_MINUTES: float = 5.0  # minimum time before checkpoint can fire (avoid noise)

    # Graduated stop: tightens based on BOTH elapsed time AND underlying movement
    # Wide when underlying hasn't confirmed against; tight when it has
    ENABLE_GRADUATED_STOP: bool = True
    GRADUATED_STOP_WIDE_PCT: float = 50.0     # stop when underlying hasn't confirmed against (v5b: was 40)
    GRADUATED_STOP_TIGHT_PCT: float = 35.0    # stop when underlying confirms against (v5b: was 25)
    GRADUATED_STOP_GRACE_MINUTES: float = 5.0 # minimum hold before any stop (avoid open noise)
    # Momentum confirmation: only stop if underlying moved against trade direction
    ENABLE_MOMENTUM_CONFIRMED_STOP: bool = True
    STOP_UNDERLYING_CONFIRM_PCT: float = 0.4  # underlying must move 0.4%+ against for tight stop
    STOP_BACKSTOP_EXTRA_PCT: float = 15.0     # absolute backstop = wide stop% + this = 65% (always fires)

    # Ticker blocklist — never trade these (comma-separated, empty = none blocked)
    # MSFT: 22% WR, -$2,641 across 9 trades. COIN: 55% WR, -$8,933 across 20 trades.
    # AVGO: 71% WR but -$3,601 (big avg loss). MU: flat, too few trades.
    # Backtested 2026-05-30 with concurrent position architecture (60 days).
    BLOCKED_TICKERS: str = "MSFT,COIN,AVGO,MU"

    # Master PUT kill switch — blocks ALL PUT entries across all agents.
    # Enabled 2026-06-05: dual-chain scanning feeds PUT chain data to ML (not CALL data).
    # Safety: PutBearishConfirmGate, PUT_BUDGET_MULTIPLIER=0.50, no DCA, PUT_SCALP_CONFIG.
    ENABLE_PUT_TRADING: bool = True

    # PUT-excluded tickers — these are allowed for CALLs but blocked for PUTs
    # Backtested (60 days, 2026-03-11 to 2026-06-09):
    #   AMZN -$10K, GOOGL -$7K, PLTR -$48K, AMD -$9K on PUTs
    # High-potential PUT tickers: SPY, QQQ, IWM, TSLA, META, AAPL, NVDA
    PUT_EXCLUDED_TICKERS: str = "PLTR,AMD,MSTR,AVGO,AMZN,GOOGL"

    # PUT market direction gate — only enter PUTs when SPY is green (market up)
    # Rationale: cheap PUTs on green days catch intraday reversals; on red days
    # PUT premiums are already inflated. When market drops (bear mode), PUTs get
    # expanded ticker list and more slots.
    ENABLE_PUT_MARKET_DIRECTION_GATE: bool = True
    PUT_MARKET_UP_MIN_PCT: float = 0.0  # SPY must be >= this % from open to allow PUTs
    PUT_BEAR_MODE_THRESHOLD: float = -0.5  # SPY down this % = bear mode (expand PUT tickers)
    PUT_BEAR_EXPANDED_TICKERS: str = "SPY,QQQ,NVDA,TSLA,META,AAPL,AMZN,GOOGL,AMD,MSTR,PLTR,AVGO,IWM"

    # PUT bearish confirmation gate — requires candle confirmation before PUT entry
    # Checks VWAP breakdown + bearish candle trend + RSI to confirm downtrend
    ENABLE_PUT_BEARISH_CONFIRM: bool = True

    # PUT position sizing — reduce position size for PUTs (structurally worse odds)
    PUT_BUDGET_MULTIPLIER: float = 0.50  # 50% of normal CALL budget for PUTs

    # Directional regime gate — uses candle data to confirm signal direction
    # RE-ENABLED (2026-06-05): Was disabled but candle cache didn't compute EMA9/EMA21,
    # so the gate was handicapped. Now EMAs are computed, giving proper trend confirmation.
    # Also: with zero candle gates active, PUTs entered on green days with no confirmation.
    ENABLE_DIRECTIONAL_REGIME: bool = True

    # Calls-only tickers — fallback when candle data unavailable
    CALLS_ONLY_TICKERS: str = "SPY,QQQ,TSLA,AAPL,GOOGL,IWM,AMZN,META"

    # Minimum option premium — reject deep OTM trades that bleed to theta
    MIN_OPTION_PREMIUM: float = 0.30  # reject options under $0.30 (backtested: $0.30 floor filters lottery tickets)

    # Anti-chase: reject if underlying moved too far from alert price
    ANTI_CHASE_MAX_MOVE_PCT: float = 0.3

    # Smart entry: verify live option premium before placing orders
    ENABLE_SMART_ENTRY: bool = True  # fetch live quote and compare to signal
    SMART_ENTRY_MAX_DEVIATION_PCT: float = 75.0  # max % deviation from signal premium (0DTE premiums move fast)
    SMART_ENTRY_PREFER_LIVE: bool = True  # use live quote as limit price (not stale signal)
    SMART_ENTRY_MIN_PREMIUM: float = 0.10  # reject if live premium below this

    # Score-based position sizing (overrides MAX_POSITION_PCT when enabled)
    ENABLE_SCORE_SIZING: bool = True  # 5/3/1 contracts by score tier

    # Adaptive 3-stage trailing stop (v2.1) — replaces phase trail when enabled
    # Backtested: 2x P&L improvement ($17K vs $8.3K) at same 80% win rate
    ENABLE_ADAPTIVE_TRAIL: bool = True
    ADAPTIVE_TRAIL_ACTIVATION_PCT: float = 35.0   # DORMANT below this (v2.1: earlier trail = more captured peak)
    ADAPTIVE_TRAIL_ACTIVE_WIDTH: float = 35.0      # ACTIVE stage trail width
    ADAPTIVE_TRAIL_RUNNER_THRESHOLD: float = 150.0  # enter RUNNER stage at this peak gain %
    ADAPTIVE_TRAIL_RUNNER_WIDTH: float = 45.0       # RUNNER stage trail width (wider, let it run)
    ADAPTIVE_TRAIL_MOONSHOT_THRESHOLD: float = 400.0  # enter MOONSHOT at this peak gain %
    ADAPTIVE_TRAIL_MOONSHOT_WIDTH: float = 30.0     # MOONSHOT trail width (tighter, lock in gains)

    # Underlying-anchored trail (v2.1 §5): trail on underlying price instead of premium
    # Tighter than premium trail since underlying moves less; catches reversals faster.
    # Backtested: +$112 AMD, +$25.50 AMZN on real trades.
    ENABLE_UNDERLYING_TRAIL: bool = True
    UNDERLYING_TRAIL_TIERS: str = "100:0.50,50:0.40,15:0.30,0:0.20"  # gain%:trail% pairs

    # Volume-peak modifier (v2.1 §6): tighten trail when underlying diverges from premium
    # Detects exhaustion by comparing recent underlying momentum.
    # Backtested: +$464 on real signals.
    ENABLE_VOLUME_PEAK: bool = True
    VOLUME_PEAK_TIGHTEN_FACTOR: float = 0.7  # multiply trail width by this when triggered
    VOLUME_PEAK_MIN_GAIN_PCT: float = 35.0  # only check after this % gain

    # Tranche scale-out (v2.1 §4): lock 1/3 of contracts at +25% gain
    # Backtested: +$375 over baseline on 37 real trades (9 improved, 4 worse).
    ENABLE_TRANCHE_SCALEOUT: bool = True
    TRANCHE_LOCK_GAIN_PCT: float = 25.0  # lock T1 tranche at this % gain
    TRANCHE_MIN_CONTRACTS: int = 3  # need at least this many contracts to tranche

    # Time decay zone: DISABLED in v4.1.
    # Full-fidelity backtest: fired 1x, lost $398. The 10min stale window is too
    # tight for 0DTE — options naturally have lulls. Dollar trail + adaptive trail
    # handle profitable exits; stop_loss handles losers.
    ENABLE_TIME_DECAY_ZONE: bool = False
    TIME_DECAY_HOLD_MINUTES: float = 45.0
    TIME_DECAY_AFTERNOON_HOUR: int = 15
    TIME_DECAY_AFTERNOON_MINUTE: int = 30
    TIME_DECAY_STALE_MINUTES: float = 10.0

    # Theta bleed: exit if held too long and losing
    THETA_BLEED_HOLD_MINUTES: float = 45.0
    THETA_BLEED_MAX_LOSS_PCT: float = 30.0

    # --- Backtested safeguards (velocity, profit lock, adaptive tighten) ---

    # Velocity exit: REPLACED by dollar-based stair-step trailing stop
    ENABLE_VELOCITY_EXIT: bool = False  # legacy — kept for backward compat
    VELOCITY_DROP_PCT: float = 12.0  # legacy
    VELOCITY_WINDOW_MINUTES: int = 4  # legacy

    # Dollar-based stair-step trailing stop (replaces velocity exit)
    # Steps scale with entry cost so cheap and expensive options are treated equally.
    # Default: 10% activation, 10% small steps, 25% threshold, 5% large steps.
    # On a $2.00 option ($200/contract): activation=$20, steps=$20/$10, threshold=$50.
    # On a $0.50 option ($50/contract): activation=$5, steps=$5/$2.50, threshold=$12.50.
    ENABLE_DOLLAR_TRAIL: bool = True
    DOLLAR_TRAIL_ACTIVATION_PCT: float = 40.0  # activate at this % profit from entry cost (backtested: 40% lets trades breathe to 1-2.5hr peak)
    DOLLAR_TRAIL_SMALL_STEP_PCT: float = 20.0  # step size as % of entry cost (below threshold) — wider to avoid premature exits
    DOLLAR_TRAIL_STEP_THRESHOLD_PCT: float = 25.0  # switch to tighter steps at this % of cost
    DOLLAR_TRAIL_LARGE_STEP_PCT: float = 10.0  # tighter step size as % of entry cost — wider to let winners run

    # Profit-based retracement: exit when X% of profit is given back (covers the
    # dormant zone below adaptive trail activation where no trail protects gains).
    # Example: entry $1.00, peak $1.50 (+50% gain = $0.50 profit), retrace 35% of
    # profit → exit at $1.325 (locks in $0.325). Only activates between
    # PROFIT_RETRACE_MIN_GAIN_PCT and ADAPTIVE_TRAIL_ACTIVATION_PCT.
    # Profit retrace: SUPERSEDED by profit_floor (v3) which activates earlier and keeps more.
    # Kept as fallback if profit_floor is disabled.
    ENABLE_PROFIT_RETRACE: bool = False
    PROFIT_RETRACE_PCT: float = 50.0        # exit when this % of profit is given back
    PROFIT_RETRACE_MIN_GAIN_PCT: float = 25.0  # only activate after peak gain >= this %

    # Premium deceleration exit: DISABLED in v4.1.
    # Full-fidelity backtest: decel fires 37% of all exits (25/67 trades) with 92% WR,
    # but it cuts runners at +$43 that could be +$2,061. It preempts soft_trail,
    # dollar_trail, and adaptive_trail — making them dead code.
    # Removing decel: +$504 P&L improvement, soft_trail/dollar_trail take over.
    ENABLE_DECEL_EXIT: bool = False
    DECEL_SHORT_WINDOW: int = 5       # short-term velocity window (bars/seconds)
    DECEL_LONG_WINDOW: int = 15       # long-term velocity window (bars/seconds)
    DECEL_THRESHOLD: float = -3.0     # exit when short_vel - long_vel < this
    DECEL_MIN_GAIN_PCT: float = 5.0   # only after trade was up at least this %
    DECEL_MIN_HOLD_SECONDS: int = 480 # minimum hold time before decel can fire (8 min = past grace)

    # --- Exit v3: Ratcheting profit floor + Bounce-fade (backtested: +$120 P&L, +4% WR) ---

    # Ratcheting profit floor: activates at +15%, locks in 60% of peak gain.
    # Replaces the gap between profit_retrace and adaptive trail.
    # Floor only goes UP, never down. Tightens with time urgency near expiry.
    ENABLE_PROFIT_FLOOR: bool = True
    PROFIT_FLOOR_ACTIVATION_PCT: float = 15.0   # floor activates at this % gain
    PROFIT_FLOOR_RATCHET_PCT: float = 60.0       # keep this % of peak gain (floor = entry + peak_gain * 60%)

    # Bounce-and-fade: v3 DISABLED — thesis_cut handles loss-cutting better.
    # Bounce-fade's time-critical mode was selling at bad prices on deep dips.
    ENABLE_BOUNCE_FADE: bool = False
    BOUNCE_FADE_WATCH_PCT: float = 50.0
    BOUNCE_FADE_MIN_RECOVERY_PCT: float = 10.0
    BOUNCE_FADE_PCT: float = 15.0

    # Continuous thesis cut: when down -40%+, check if trend is dead (making new lows)
    # or finding support (decelerating, bouncing). Replaces hard stop + grace period.
    # Backtested: +$324 P&L improvement, 70.1% WR (vs 63.6% current)
    ENABLE_THESIS_CUT: bool = True
    THESIS_CUT_THRESHOLD_PCT: float = 40.0      # start checking when down this % from entry
    THESIS_CUT_LOOKBACK_TICKS: int = 8           # window of recent ticks to analyze
    THESIS_CUT_NEW_LOW_EXIT: int = 3             # exit if N+ of lookback ticks made new lows
    THESIS_CUT_BOUNCE_HOLD_PCT: float = 5.0      # hold if bounced this % from recent low
    THESIS_CUT_MIN_TICKS: int = 4                # minimum ticks in danger zone before cutting
    THESIS_CUT_TIME_URGENCY_MIN: float = 30.0    # tighten criteria with < N min to expiry
    THESIS_CUT_TIME_CUT_DROP_PCT: float = 40.0   # with < 30min left, cut if still down this %

    # Profit lock ratchet: SUPERSEDED by profit_floor (v3) which ratchets continuously.
    # Kept as fallback if profit_floor is disabled.
    ENABLE_PROFIT_LOCK: bool = False
    PROFIT_LOCK_TIERS: str = "80:25,150:70,250:150"  # backtested: lock gains at higher thresholds to avoid premature exit

    # Adaptive time tightening: DISABLED in v4.1.
    # Full-fidelity backtest: never fired across 67 signals. Conflicts with adaptive
    # trail by narrowing width after 60min, fighting the RUNNER stage that intentionally widens.
    ENABLE_TIME_TIGHTEN: bool = False
    TIME_TIGHTEN_AFTER_MINUTES: float = 60.0
    TIME_TIGHTEN_FACTOR: float = 0.7

    # Consecutive loser pause
    CONSECUTIVE_LOSER_MAX: int = 2  # pause after N losses in a row
    CONSECUTIVE_LOSER_PAUSE_MINUTES: float = 15.0

    # Time-of-day score thresholds (reject below threshold at given time)
    # Format: list of (hour, minute, min_score) — signals after this time need higher scores
    TOD_EARLY_CUTOFF_HOUR: int = 9
    TOD_EARLY_CUTOFF_MINUTE: int = 45  # before 9:45 AM: need score >= 85
    TOD_EARLY_MIN_SCORE: int = 85
    TOD_LATE_CUTOFF_HOUR: int = 14  # after 2 PM: need score >= 85
    TOD_LATE_CUTOFF_MINUTE: int = 0
    TOD_LATE_MIN_SCORE: int = 85
    ENTRY_HARD_CUTOFF_HOUR: int = 15  # no new entries after 3:55 PM ET regardless of score
    ENTRY_HARD_CUTOFF_MINUTE: int = 55  # (theta crush in last 5 min makes even correct alerts lose)

    # Morning cutoff: block ALL entries after this time (backtest: only 9:30-11:00 AM ET is profitable)
    ENABLE_MORNING_CUTOFF: bool = True
    ENTRY_MORNING_CUTOFF_HOUR: int = 11  # no new entries after 11:00 AM ET
    ENTRY_MORNING_CUTOFF_MINUTE: int = 0

    # Scalp target gate: take profit at +35% unless confirmed runner
    # H9 backtest: 35% scalp → 92.6% WR, +$90K, PF=11.70, MaxDD=5.3% (vs 25% at +$55K)
    ENABLE_SCALP_TARGET: bool = True
    SCALP_TARGET_PCT: float = 35.0  # take profit at this % gain
    SCALP_RUNNER_CONFIRM_PCT: float = 40.0  # skip scalp if peak gain exceeds this (confirmed runner)

    # Correlation cap: max same-direction positions per correlated group
    ENABLE_CORRELATION_CAP: bool = True
    CORRELATION_CAP_MAX_PER_GROUP: int = 3  # max 3 same-direction positions per group

    # Daily portfolio sync from Webull (auto-update PORTFOLIO_SIZE from live balance)
    ENABLE_PORTFOLIO_SYNC: bool = True

    # Category-aware exit strategy (backtested: +$13,809 / +173% on $8K, 8/11 days won)
    # Multi-day contract cap: expensive multi-day options cause outsized losses.
    # Cap at 2 contracts for multi-day, 1 for premiums > $5.
    MULTI_DAY_MAX_CONTRACTS: int = 2  # max contracts for DTE > 0 trades
    MULTI_DAY_EXPENSIVE_THRESHOLD: float = 5.0  # premiums above this get capped to 1 contract
    # Daily circuit breaker: stop opening new trades after losing this % of portfolio in a day.
    # Only counts today's closed + today's open unrealized (multi-day holds excluded).
    # Raised from 12.5% → 25% because one bad trade was shutting down small accounts entirely.
    MAX_TRADE_LOSS_EXIT_PCT: float = 8.0  # force-exit any trade losing > this % of total portfolio (0 = disabled)
    DAILY_LOSS_CIRCUIT_BREAKER_PCT: float = 25.0  # 0 = disabled
    # Index profit target: SPY/QQQ/IWM take profits at +30% (100% WR in backtest).
    INDEX_PROFIT_TARGET_PCT: float = 30.0  # 0 = disabled
    INDEX_TICKERS: str = "SPY,QQQ,IWM,DIA,XLF,XLK"

    # Feature 6: Unified risk manager
    ENABLE_RISK_MANAGER: bool = False
    MAX_PORTFOLIO_RISK_PCT: float = 75.0  # max % of portfolio deployable (backtested: 75% optimal)
    MAX_LOSS_PER_TRADE_PCT: float = 25.0  # max % of portfolio per single trade
    WEEKLY_LOSS_LIMIT_PCT: float = 100.0  # effectively disabled (was 20% — blocked all trades on Apr 24)

    # --- Redis (cross-agent coordination: dedup, regime, rate limiting) ---
    ENABLE_REDIS: bool = False
    REDIS_URL: str = "redis://redis:6379/0"

    # --- Shared PostgreSQL (cross-agent trades, signals, analytics) ---
    ENABLE_POSTGRES: bool = False  # Phase 1: dual-write to SQLite + Postgres
    DATABASE_URL: str = "postgresql://owl:owl_dev_2026@postgres:5432/options_owl"

    # --- Intraday Regime Detector (spec 06) ---
    # Rule-based market direction classification using SPY 5m candles.
    # Gates trade direction, sizing, and stop tightening.
    ENABLE_REGIME_DETECTOR: bool = True
    REGIME_CHECK_INTERVAL_MIN: int = 5
    REGIME_HYSTERESIS_CHECKS: int = 2   # consecutive readings to confirm flip
    REGIME_MIN_HOLD_MIN: int = 15       # min time before re-evaluation
    REGIME_HARD_REVERSAL_PCT: float = 0.5  # SPY drop % for immediate BEARISH
    REGIME_CHOPPY_SIZE_MULT: float = 0.6   # size reduction in CHOPPY regime

    # --- Extended Scan Window (spec 07) ---
    # Expand ML scanning beyond 9:35-11:00 to midday and afternoon.
    # Requires ENABLE_REGIME_DETECTOR for midday gating.
    ENABLE_EXTENDED_SCAN: bool = False
    MIDDAY_SCALP_TARGET_PCT: float = 25.0
    LATE_BACKSTOP_PCT: float = 40.0

    # --- Regime-Triggered Stop Tightening (spec 08) ---
    # Auto-tighten exits when regime flips against open positions.
    ENABLE_REGIME_STOP_TIGHTEN: bool = False
    REGIME_TIGHTEN_FACTOR: float = 0.60        # 40% tighter on reversal
    REGIME_CHOPPY_TIGHTEN_FACTOR: float = 0.80  # 20% tighter in chop
    REGIME_EMERGENCY_TRAIL_PCT: float = 25.0
    REGIME_EMERGENCY_BACKSTOP_PCT: float = 40.0

    # --- Conviction-Based Sizing (spec 09) ---
    # Scale position size by ML confidence, regime alignment, and time of day.
    ENABLE_CONVICTION_SIZING: bool = False

    # --- Dynamic PUT Expansion (spec 10) ---
    # Automatically expand PUT trading when regime is BEARISH.
    ENABLE_DYNAMIC_PUTS: bool = True
    PUT_SLOTS_BULLISH: int = 2
    PUT_SLOTS_BEARISH: int = 6
    PUT_SLOTS_CHOPPY: int = 3

    # --- Signal source control ---
    # When False, Discord signals are still logged/parsed but NOT traded.
    # Bots only trade signals from the ML sourcing pipeline (via PostgreSQL).
    ENABLE_DISCORD_SIGNALS: bool = True

    AGENT_ID: str = ""  # unique ID per bot for signal consumer + PG tracking

    @computed_field  # type: ignore[prop-decorator]
    @property
    def effective_max_concurrent(self) -> int:
        """Auto-adapt concurrent slots based on portfolio size.

        Small portfolios (<$8K): 2 slots → concentrate capital into fewer, larger positions
        Large portfolios (>=$8K): 4 slots → diversify across more positions
        Override: set MAX_CONCURRENT > 0 to use a fixed value.
        """
        if self.MAX_CONCURRENT > 0:
            return self.MAX_CONCURRENT
        return 2 if self.PORTFOLIO_SIZE < 8000 else 4

    @computed_field  # type: ignore[prop-decorator]
    @property
    def effective_max_position_pct(self) -> float:
        """Auto-adapt position sizing based on portfolio size.

        Small portfolios (<$8K): 20% → enough contracts for scaleout/runners
        Large portfolios (>=$8K): 15% → standard diversification
        Override: set MAX_POSITION_PCT > 0 to use a fixed value.
        """
        if self.MAX_POSITION_PCT > 0:
            return self.MAX_POSITION_PCT
        return 20.0 if self.PORTFOLIO_SIZE < 8000 else 15.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def effective_max_dca_position_pct(self) -> float:
        """Auto-adapt DCA position cap (half of effective position %)."""
        if self.MAX_DCA_POSITION_PCT > 0:
            return self.MAX_DCA_POSITION_PCT
        return self.effective_max_position_pct / 2.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def guild_ids(self) -> list[int]:
        return _parse_int_list(self.DISCORD_GUILD_IDS)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def channel_ids(self) -> list[int]:
        return _parse_int_list(self.DISCORD_CHANNEL_IDS)

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}
