"""Trade pipeline state machine — formalizes every gate a signal must pass.

Replaces scattered if-chains with an explicit, ordered pipeline of named gates.
Each gate is a callable that returns (passed, reason). If any gate fails, the
signal is rejected and all failure reasons are collected.

Entry pipeline:  Signal → [gate1, gate2, ...] → Approved / Rejected
Exit pipeline:   OpenTrade → [exit_check1, exit_check2, ...] → Hold / Close(reason)

Every gate logs its result, so the full decision path is auditable.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import timezone
from enum import Enum
from typing import Any

from loguru import logger

try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")
except ImportError:
    from datetime import timedelta as _td
    _ET = timezone(_td(hours=-5))


def _now_et():
    """Current time in Eastern Time (DST-aware)."""
    from datetime import datetime
    return datetime.now(tz=_ET)


# ---------------------------------------------------------------------------
# DTE helper — determines days to expiry from trade context
# ---------------------------------------------------------------------------


def _get_dte(trade: dict, now_et=None) -> int:
    """Return days-to-expiry for a trade. 0 = expires today."""
    from datetime import datetime, date

    expiry_date = trade.get("expiry_date")
    if not expiry_date:
        return 0  # assume 0DTE if unknown

    try:
        exp = datetime.strptime(expiry_date, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return 0

    if now_et:
        today = now_et.date() if hasattr(now_et, "date") else now_et
    else:
        today = date.today()

    return max(0, (exp - today).days)


# ---------------------------------------------------------------------------
# Smart grace: replaces fixed timer with underlying-based confirmation
# ---------------------------------------------------------------------------


def is_grace_active(ctx: dict[str, Any]) -> tuple[bool, str]:
    """Check if grace period is still active for this trade.

    Smart grace (und_confirm): grace ends when EITHER:
      (a) underlying price confirms trade direction (crosses entry price), OR
      (b) STOP_GRACE_PERIOD_MINUTES elapsed (safety cap, default 20min)

    Returns (is_active, reason_str).
    """
    trade = ctx.get("trade", {})
    settings = ctx.get("settings")
    grace_minutes = getattr(settings, "STOP_GRACE_PERIOD_MINUTES", 20) if settings else 20

    if grace_minutes <= 0:
        return False, "grace disabled"

    opened_at = trade.get("opened_at")
    if not opened_at:
        return False, "no opened_at"

    try:
        from datetime import datetime as _dt
        now = ctx.get("now_et") or _dt.now(tz=timezone.utc)
        opened_dt = _dt.fromisoformat(opened_at)
        if opened_dt.tzinfo is None:
            opened_dt = opened_dt.replace(tzinfo=now.tzinfo)
        elapsed = (now - opened_dt).total_seconds() / 60
    except (ValueError, TypeError):
        return False, "parse error"

    # Time cap: always end grace after max minutes
    if elapsed >= grace_minutes:
        return False, f"elapsed {elapsed:.0f}m >= {grace_minutes}m cap"

    # Smart grace: check underlying confirmation
    # Requires underlying to move >0.1% past entry in the trade's direction.
    # A tiny blip above entry doesn't count — need a real move.
    smart_grace = getattr(settings, "ENABLE_SMART_GRACE", False) if settings else False
    if smart_grace:
        current_price = ctx.get("current_price")
        entry_price = trade.get("entry_price")
        option_type = trade.get("option_type", "").lower()
        confirm_pct = 0.1  # underlying must move 0.1% past entry to confirm

        if current_price and entry_price and entry_price > 0:
            move_pct = (current_price - entry_price) / entry_price * 100
            if option_type in ("call", "bullish", "long"):
                # For calls: underlying must be >0.1% above entry
                if move_pct > confirm_pct:
                    return False, (
                        f"smart grace: underlying ${current_price:.2f} "
                        f"+{move_pct:.2f}% > entry ${entry_price:.2f} "
                        f"(confirmed at {elapsed:.0f}m)"
                    )
            else:
                # For puts: underlying must be >0.1% below entry
                if move_pct < -confirm_pct:
                    return False, (
                        f"smart grace: underlying ${current_price:.2f} "
                        f"{move_pct:.2f}% < entry ${entry_price:.2f} "
                        f"(confirmed at {elapsed:.0f}m)"
                    )

    return True, f"grace: {elapsed:.0f}m / {grace_minutes}m"


# ---------------------------------------------------------------------------
# Pipeline state
# ---------------------------------------------------------------------------


class GateResult(str, Enum):
    """Outcome of a single gate evaluation."""

    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"  # gate disabled or not applicable


@dataclass
class GateOutcome:
    """Result from evaluating one gate."""

    gate_name: str
    result: GateResult
    reason: str = ""


@dataclass
class PipelineResult:
    """Aggregate result of running all gates in a pipeline."""

    approved: bool
    outcomes: list[GateOutcome] = field(default_factory=list)

    @property
    def failures(self) -> list[GateOutcome]:
        return [o for o in self.outcomes if o.result == GateResult.FAIL]

    @property
    def failure_reasons(self) -> list[str]:
        return [o.reason for o in self.failures]

    def summary(self) -> str:
        parts = []
        for o in self.outcomes:
            icon = {"pass": "+", "fail": "X", "skip": "-"}[o.result.value]
            parts.append(f"  [{icon}] {o.gate_name}: {o.reason or 'ok'}")
        status = "APPROVED" if self.approved else "REJECTED"
        return f"Pipeline {status}\n" + "\n".join(parts)


# ---------------------------------------------------------------------------
# Entry pipeline — signal admission gates
# ---------------------------------------------------------------------------


class EntryGate:
    """Base class for an entry gate. Subclass and implement ``evaluate``."""

    name: str = "unnamed_gate"

    async def evaluate(self, ctx: dict[str, Any]) -> GateOutcome:
        raise NotImplementedError


class BlockedTickerGate(EntryGate):
    """Gate 0: Reject tickers on the blocklist (historically unprofitable)."""

    name = "blocked_ticker"

    async def evaluate(self, ctx: dict[str, Any]) -> GateOutcome:
        signal = ctx["signal"]
        settings = ctx["settings"]
        blocked_str = getattr(settings, "BLOCKED_TICKERS", "")
        blocked = {t.strip().upper() for t in blocked_str.split(",") if t.strip()}
        ticker = getattr(signal, "ticker", "").upper()
        if ticker in blocked:
            return GateOutcome(self.name, GateResult.FAIL,
                               f"{ticker} is on the blocklist")
        return GateOutcome(self.name, GateResult.PASS,
                           f"{ticker} not blocked")


class PutTickerExclusionGate(EntryGate):
    """Gate 0a: Block PUTs on historically unprofitable PUT tickers.

    Backtested (3+ years): PLTR -$48K, AMD -$9K on PUTs.
    These tickers are fine for CALLs but lose money on PUT scalps.
    In bear mode (SPY down 0.5%+), all tickers are allowed for PUTs.
    """

    name = "put_ticker_exclusion"

    async def evaluate(self, ctx: dict[str, Any]) -> GateOutcome:
        signal = ctx["signal"]
        settings = ctx["settings"]
        from options_owl.models.signals import Direction

        direction = getattr(signal, "direction", None)
        if direction != Direction.PUT:
            return GateOutcome(self.name, GateResult.PASS, "Not a PUT trade")

        excluded_str = getattr(settings, "PUT_EXCLUDED_TICKERS", "")
        excluded = {t.strip().upper() for t in excluded_str.split(",") if t.strip()}
        ticker = getattr(signal, "ticker", "").upper()

        if ticker not in excluded:
            return GateOutcome(self.name, GateResult.PASS,
                               f"{ticker} allowed for PUTs")

        # Check bear mode — if SPY is down enough, allow all tickers
        bear_threshold = getattr(settings, "PUT_BEAR_MODE_THRESHOLD", -0.5)
        spy_change = ctx.get("spy_change_from_open")
        if spy_change is not None and spy_change <= bear_threshold:
            return GateOutcome(self.name, GateResult.PASS,
                               f"Bear mode (SPY {spy_change:+.2f}%) — {ticker} PUT allowed")

        return GateOutcome(self.name, GateResult.FAIL,
                           f"{ticker} excluded from PUTs (backtest loser)")


class PutMarketDirectionGate(EntryGate):
    """Gate 0c: Only enter PUTs when market (SPY) is green.

    Rationale: cheap PUTs on green days catch intraday reversals.
    On red days, PUT premiums are already inflated (expensive entry).
    When SPY dips into bear mode, the PutTickerExclusionGate expands
    the ticker list — this gate is about entry timing, not bear mode.
    """

    name = "put_market_direction"

    async def evaluate(self, ctx: dict[str, Any]) -> GateOutcome:
        signal = ctx["signal"]
        settings = ctx["settings"]
        from options_owl.models.signals import Direction

        direction = getattr(signal, "direction", None)
        if direction != Direction.PUT:
            return GateOutcome(self.name, GateResult.PASS, "Not a PUT trade")

        if not getattr(settings, "ENABLE_PUT_MARKET_DIRECTION_GATE", True):
            return GateOutcome(self.name, GateResult.SKIP, "PUT market gate disabled")

        spy_change = ctx.get("spy_change_from_open")
        if spy_change is None:
            # No SPY data — allow (fail-open)
            return GateOutcome(self.name, GateResult.PASS,
                               "No SPY data available — allowing PUT")

        min_pct = getattr(settings, "PUT_MARKET_UP_MIN_PCT", 0.0)
        if spy_change >= min_pct:
            return GateOutcome(self.name, GateResult.PASS,
                               f"SPY {spy_change:+.2f}% (green) — PUT allowed")

        # Bear mode override — if SPY is deeply red, PUTs should still fire
        bear_threshold = getattr(settings, "PUT_BEAR_MODE_THRESHOLD", -0.5)
        if spy_change <= bear_threshold:
            return GateOutcome(self.name, GateResult.PASS,
                               f"Bear mode (SPY {spy_change:+.2f}%) — PUT allowed")

        return GateOutcome(self.name, GateResult.FAIL,
                           f"SPY {spy_change:+.2f}% (red but not bear mode) — PUTs blocked")


class DirectionalRegimeGate(EntryGate):
    """Gate 0b: Confirm signal direction matches market regime using candle data.

    Replaces the old blunt CallsOnlyGate with a dynamic check:
    - PUTs require bearish candle confirmation (underlying dropping, RSI < 45)
    - CALLs require bullish confirmation (underlying rising, RSI > 55)
    - On regime mismatch, blocks the trade with a clear reason

    This allows PUTs to trade during actual selloffs while blocking PUTs in bull markets,
    and vice versa for CALLs.
    """

    name = "directional_regime"

    async def evaluate(self, ctx: dict[str, Any]) -> GateOutcome:
        settings = ctx["settings"]
        if not getattr(settings, "ENABLE_DIRECTIONAL_REGIME", True):
            return GateOutcome(self.name, GateResult.SKIP, "Directional regime disabled")

        signal = ctx["signal"]
        direction = getattr(signal, "direction", None)
        if direction is None:
            return GateOutcome(self.name, GateResult.SKIP, "No direction on signal")

        from options_owl.models.signals import Direction
        is_put = direction == Direction.PUT

        # Try candle data for direction confirmation
        candle_cache = ctx.get("candle_cache")
        if candle_cache is None:
            # No candle data — fall back to old CallsOnlyGate behavior
            return self._fallback_calls_only(signal, settings, is_put)

        try:
            data = await asyncio.wait_for(
                candle_cache.get_candle_data(signal.ticker), timeout=15,
            )
        except (asyncio.TimeoutError, Exception):
            return self._fallback_calls_only(signal, settings, is_put)

        indicators = data.get("indicators", {})
        tf_5m = indicators.get("5m", {})
        tf_15m = indicators.get("15m", {})
        bars_5m = data.get("5m", [])

        rsi_5m = tf_5m.get("rsi")
        rsi_15m = tf_15m.get("rsi")

        # Count bearish vs bullish 5m candles (last 6 = 30 min of trading)
        bearish_bars = 0
        bullish_bars = 0
        if len(bars_5m) >= 3:
            lookback = min(6, len(bars_5m))
            for i in range(-lookback, 0):
                bar = bars_5m[i]
                if bar.close < bar.open:
                    bearish_bars += 1
                elif bar.close > bar.open:
                    bullish_bars += 1

        # Calculate underlying momentum from bars (% change over lookback)
        underlying_change_pct = 0.0
        if len(bars_5m) >= 3:
            lookback = min(6, len(bars_5m))
            first_open = bars_5m[-lookback].open
            last_close = bars_5m[-1].close
            if first_open > 0:
                underlying_change_pct = (last_close - first_open) / first_open * 100

        # EMA trend from indicators
        ema9 = tf_5m.get("ema9")
        ema21 = tf_5m.get("ema21")
        ema_bearish = ema9 is not None and ema21 is not None and ema9 < ema21
        ema_bullish = ema9 is not None and ema21 is not None and ema9 > ema21

        # Score the directional evidence (positive = bullish, negative = bearish)
        regime_score = 0.0

        # RSI contribution (-2 to +2)
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

        # Decision: regime_score ranges roughly -7.5 to +7.5
        # PUTs need bearish regime (score < -1), CALLs need bullish (score > +1)
        detail = (
            f"regime={regime_score:+.1f} RSI5m={rsi_5m or 0:.0f} RSI15m={rsi_15m or 0:.0f} "
            f"bars={bullish_bars}B/{bearish_bars}b chg={underlying_change_pct:+.2f}% "
            f"EMA={'bear' if ema_bearish else 'bull' if ema_bullish else 'flat'}"
        )

        if is_put:
            if regime_score > 1.0:
                return GateOutcome(
                    self.name, GateResult.FAIL,
                    f"PUT blocked — bullish regime ({detail})"
                )
            return GateOutcome(self.name, GateResult.PASS, f"PUT confirmed bearish ({detail})")
        else:
            if regime_score < -1.0:
                return GateOutcome(
                    self.name, GateResult.FAIL,
                    f"CALL blocked — bearish regime ({detail})"
                )
            return GateOutcome(self.name, GateResult.PASS, f"CALL confirmed bullish ({detail})")

    def _fallback_calls_only(self, signal, settings, is_put: bool) -> GateOutcome:
        """Fallback to static calls-only list when no candle data is available."""
        if not is_put:
            return GateOutcome(self.name, GateResult.PASS, "CALL — no regime check needed")
        calls_only_str = getattr(settings, "CALLS_ONLY_TICKERS", "")
        calls_only = {t.strip().upper() for t in calls_only_str.split(",") if t.strip()}
        ticker = getattr(signal, "ticker", "").upper()
        if ticker in calls_only:
            return GateOutcome(
                self.name, GateResult.FAIL,
                f"{ticker} PUT blocked (no candle data, fallback to blocklist)"
            )
        return GateOutcome(self.name, GateResult.PASS, "PUT allowed (no candle data, not on blocklist)")


class ScoreGate(EntryGate):
    """Gate 1: Minimum signal score."""

    name = "score"

    async def evaluate(self, ctx: dict[str, Any]) -> GateOutcome:
        signal = ctx["signal"]
        settings = ctx["settings"]
        min_score = settings.MIN_SCORE
        if signal.score < min_score:
            return GateOutcome(self.name, GateResult.FAIL,
                               f"Score {signal.score} < min {min_score}")
        return GateOutcome(self.name, GateResult.PASS,
                           f"Score {signal.score} >= {min_score}")


class PremiumGate(EntryGate):
    """Gate 2: Signal must have a valid ATM premium above minimum threshold.

    Rejects deep OTM options with tiny premiums (e.g., $0.09) that have
    almost no chance of profiting and just bleed to zero via theta.
    """

    name = "premium"

    async def evaluate(self, ctx: dict[str, Any]) -> GateOutcome:
        signal = ctx["signal"]
        settings = ctx["settings"]
        if not signal.atm_premium or signal.atm_premium <= 0:
            return GateOutcome(self.name, GateResult.FAIL, "No ATM premium available")
        min_premium = getattr(settings, "MIN_OPTION_PREMIUM", 0.20)
        if signal.atm_premium < min_premium:
            return GateOutcome(
                self.name, GateResult.FAIL,
                f"Premium ${signal.atm_premium:.2f} < min ${min_premium:.2f} "
                f"(too deep OTM, high theta bleed risk)"
            )
        return GateOutcome(self.name, GateResult.PASS,
                           f"Premium ${signal.atm_premium:.2f} >= ${min_premium:.2f}")


class PremiumCapGate(EntryGate):
    """V6 Gate: Reject non-index entries with premium > tiered cap.

    High-premium single-stock options (e.g., META $25.35) carry outsized risk
    with wide bid-ask spreads and poor fill quality. Index tickers (SPY, QQQ,
    IWM) are exempt because their options are highly liquid even at high premiums.

    Tiered caps (backtested $6/$7/$9 = +$195 vs $5/$6/$8):
      - Base:       $6.00 (V6_PREMIUM_CAP)
      - Score 120+: $7.00 (V6_PREMIUM_CAP_MID)
      - Score 150+: $9.00 (V6_PREMIUM_CAP_HIGH)
      - Index:      exempt
    """

    name = "v6_premium_cap"

    async def evaluate(self, ctx: dict[str, Any]) -> GateOutcome:
        settings = ctx["settings"]
        if not getattr(settings, "ENABLE_V6_PREMIUM_CAP", False):
            return GateOutcome(self.name, GateResult.SKIP, "V6 premium cap disabled")

        signal = ctx["signal"]
        premium = signal.atm_premium or 0
        score = getattr(signal, "score", 0) or 0
        base_cap = getattr(settings, "V6_PREMIUM_CAP", 6.0)
        mid_cap = getattr(settings, "V6_PREMIUM_CAP_MID", 7.0)
        high_cap = getattr(settings, "V6_PREMIUM_CAP_HIGH", 9.0)

        # Tiered cap based on signal score
        if score >= 150:
            cap = high_cap
        elif score >= 120:
            cap = mid_cap
        else:
            cap = base_cap

        # Index tickers are exempt — liquid even at high premiums
        from options_owl.risk.exit_v5.config import INDEX_TICKERS
        ticker = signal.ticker or ""
        if ticker in INDEX_TICKERS:
            return GateOutcome(self.name, GateResult.PASS,
                               f"Index ticker {ticker} exempt from premium cap")

        if premium > cap:
            return GateOutcome(
                self.name, GateResult.FAIL,
                f"Premium ${premium:.2f} > ${cap:.2f} cap (non-index {ticker}, score {score})"
            )
        return GateOutcome(self.name, GateResult.PASS,
                           f"Premium ${premium:.2f} <= ${cap:.2f} cap (score {score})")


class SpreadCostGate(EntryGate):
    """V6 Gate: Reject entries where bid-ask spread > threshold % of premium.

    Wide spreads indicate illiquid options where you pay a large hidden cost
    on entry and exit. A $2.00 option with 20% spread means you pay $2.20
    and sell at $1.80 — you need a 22% move just to break even.

    Backtested: blocked 2 trades with wide spreads.
    """

    name = "v6_spread_gate"

    async def evaluate(self, ctx: dict[str, Any]) -> GateOutcome:
        settings = ctx["settings"]
        if not getattr(settings, "ENABLE_V6_SPREAD_GATE", False):
            return GateOutcome(self.name, GateResult.SKIP, "V6 spread gate disabled")

        signal = ctx["signal"]
        max_spread_pct = getattr(settings, "V6_MAX_SPREAD_PCT", 15.0)

        # Bid/ask come from smart entry (live premium lookup)
        bid = ctx.get("bid", 0.0) or 0.0
        ask = ctx.get("ask", 0.0) or 0.0
        premium = signal.atm_premium or 0

        if bid <= 0 or ask <= 0 or premium <= 0:
            return GateOutcome(self.name, GateResult.SKIP,
                               "No bid/ask data — skipping spread check")

        spread = ask - bid
        spread_pct = (spread / premium) * 100 if premium > 0 else 0

        if spread_pct > max_spread_pct:
            return GateOutcome(
                self.name, GateResult.FAIL,
                f"Spread ${spread:.2f} = {spread_pct:.1f}% of ${premium:.2f} "
                f"> {max_spread_pct}% max"
            )
        return GateOutcome(self.name, GateResult.PASS,
                           f"Spread {spread_pct:.1f}% <= {max_spread_pct}% max")


class StopPriceGate(EntryGate):
    """Gate 3: Signal must have a stop price defined."""

    name = "stop_price"

    async def evaluate(self, ctx: dict[str, Any]) -> GateOutcome:
        signal = ctx["signal"]
        if not signal.stop_price:
            return GateOutcome(self.name, GateResult.FAIL, "No stop price defined")
        return GateOutcome(self.name, GateResult.PASS,
                           f"Stop ${signal.stop_price:.2f}")


class DailyLossGate(EntryGate):
    """Gate 4: Daily loss limit check.

    Fetches live Webull balance and compares against start-of-day baseline
    (starting_balance set by portfolio sync at startup).  This gives the true
    daily P&L regardless of paper DB premium discrepancies.

    Falls back to paper_portfolio.daily_pnl for paper-only mode or when
    webull_executor is unavailable.
    """

    name = "daily_loss_limit"

    async def evaluate(self, ctx: dict[str, Any]) -> GateOutcome:
        import asyncio

        settings = ctx["settings"]
        portfolio = ctx.get("portfolio")
        if not portfolio:
            return GateOutcome(self.name, GateResult.SKIP, "No portfolio data")

        today = _now_et().strftime("%Y-%m-%d")
        limit = settings.PORTFOLIO_SIZE * (settings.DAILY_LOSS_LIMIT_PCT / 100)

        # Best source: live Webull balance vs start-of-day baseline
        webull_executor = ctx.get("webull_executor")
        starting = portfolio.get("starting_balance", 0)

        if webull_executor and starting and starting > 0 and not settings.PAPER_TRADE:
            try:
                live_balance = await webull_executor.get_account_balance()
                if live_balance and live_balance > 0:
                    daily_pnl = live_balance - starting
                    if daily_pnl <= -limit:
                        return GateOutcome(self.name, GateResult.FAIL,
                                           f"Daily loss ${daily_pnl:.2f} exceeds "
                                           f"-${limit:.2f} (live: ${live_balance:.2f} "
                                           f"vs start: ${starting:.2f})")
                    return GateOutcome(self.name, GateResult.PASS,
                                       f"Daily P&L ${daily_pnl:.2f} within limit "
                                       f"(live: ${live_balance:.2f} vs start: ${starting:.2f})")
            except Exception as exc:
                logger.warning(f"DailyLossGate: Webull balance fetch failed: {exc}")
                # fall through to paper fallback

        # Fallback: paper portfolio daily_pnl
        if portfolio.get("last_trade_date") != today:
            return GateOutcome(self.name, GateResult.PASS, "New trading day")

        daily_pnl = portfolio.get("daily_pnl", 0)
        if daily_pnl <= -limit:
            return GateOutcome(self.name, GateResult.FAIL,
                               f"Daily loss ${daily_pnl:.2f} exceeds -${limit:.2f} (paper)")
        return GateOutcome(self.name, GateResult.PASS,
                           f"Daily P&L ${daily_pnl:.2f} within limit (paper)")


class ConcurrentPositionsGate(EntryGate):
    """Gate 5: Maximum concurrent open positions."""

    name = "concurrent_positions"

    async def evaluate(self, ctx: dict[str, Any]) -> GateOutcome:
        settings = ctx["settings"]
        open_count = ctx.get("open_count", 0)
        max_concurrent = settings.MAX_CONCURRENT
        if max_concurrent <= 0:
            # Auto-adapt: use portfolio size to determine tier
            portfolio = getattr(settings, "PORTFOLIO_SIZE", 10000)
            max_concurrent = 2 if portfolio < 8000 else 4
        if open_count >= max_concurrent:
            return GateOutcome(self.name, GateResult.FAIL,
                               f"{open_count} open >= max {max_concurrent}")
        return GateOutcome(self.name, GateResult.PASS,
                           f"{open_count} open < max {max_concurrent}")


class DuplicateTickerGate(EntryGate):
    """Gate 6: No duplicate open positions on the same ticker.

    Same-direction duplicate: FAIL (already have same trade).
    Opposite-direction (signal flip): PASS but flag for auto-close of old position.
    """

    name = "duplicate_ticker"

    async def evaluate(self, ctx: dict[str, Any]) -> GateOutcome:
        signal = ctx["signal"]
        open_tickers = ctx.get("open_tickers", set())
        if signal.ticker not in open_tickers:
            return GateOutcome(self.name, GateResult.PASS, "No duplicate")

        # Check if same direction or opposite (signal flip)
        open_positions = ctx.get("open_positions", [])
        new_dir = signal.direction.value.lower()  # "call" or "put"
        for ticker, opt_type in open_positions:
            if ticker == signal.ticker:
                existing_dir = (opt_type or "").lower()
                if existing_dir == new_dir:
                    return GateOutcome(self.name, GateResult.FAIL,
                                       f"Already have open {signal.ticker} {existing_dir}")
                else:
                    # Opposite direction — flag as signal flip so caller can close old
                    ctx["signal_flip_ticker"] = signal.ticker
                    ctx["signal_flip_old_direction"] = existing_dir
                    return GateOutcome(
                        self.name, GateResult.PASS,
                        f"Signal flip: {signal.ticker} {existing_dir}→{new_dir}, will close old"
                    )

        return GateOutcome(self.name, GateResult.PASS, "No duplicate")


class CorrelationCapGate(EntryGate):
    """Gate: Prevent 3+ correlated same-direction positions from running simultaneously."""

    name = "correlation_cap"

    # Ticker groups that tend to move together
    CORRELATION_GROUPS = {
        "index_megacap": {"SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "META"},
        "semis": {"NVDA", "AMD", "AVGO"},
        "tech_runners": {"TSLA", "MSTR", "PLTR"},
    }

    @classmethod
    def _group_for(cls, ticker: str) -> str | None:
        for name, members in cls.CORRELATION_GROUPS.items():
            if ticker.upper() in members:
                return name
        return None

    async def evaluate(self, ctx: dict[str, Any]) -> GateOutcome:
        settings = ctx["settings"]
        if not getattr(settings, "ENABLE_CORRELATION_CAP", False):
            return GateOutcome(self.name, GateResult.SKIP, "Correlation cap disabled")

        signal = ctx["signal"]
        group = self._group_for(signal.ticker)
        if group is None:
            return GateOutcome(self.name, GateResult.PASS, "No correlation group")

        max_per_group = getattr(settings, "CORRELATION_CAP_MAX_PER_GROUP", 3)
        signal_direction = "put" if signal.direction.value == "put" else "call"

        # Count open positions in the same group with the same direction
        open_positions = ctx.get("open_positions", [])
        same_group_same_dir = [
            t for t, d in open_positions
            if self._group_for(t) == group and d == signal_direction
        ]

        if len(same_group_same_dir) >= max_per_group:
            return GateOutcome(
                self.name, GateResult.FAIL,
                f"Correlation cap: {group}/{signal_direction} = "
                f"{len(same_group_same_dir)}/{max_per_group} "
                f"({', '.join(same_group_same_dir)})",
            )
        return GateOutcome(
            self.name, GateResult.PASS,
            f"{group}/{signal_direction}: {len(same_group_same_dir)}/{max_per_group}",
        )


class PortfolioRiskGate(EntryGate):
    """Gate 7: Total portfolio risk exposure limit."""

    name = "portfolio_risk"

    async def evaluate(self, ctx: dict[str, Any]) -> GateOutcome:
        settings = ctx["settings"]
        if not getattr(settings, "ENABLE_RISK_MANAGER", False):
            return GateOutcome(self.name, GateResult.SKIP, "Risk manager disabled")

        signal = ctx["signal"]
        db_path = ctx["db_path"]

        from options_owl.journal.db import connect as _connect_db
        portfolio_size = settings.PORTFOLIO_SIZE
        max_pct = settings.MAX_PORTFOLIO_RISK_PCT

        try:
            async with _connect_db(db_path) as conn:
                cursor = await conn.execute(
                    "SELECT COALESCE(SUM(total_cost), 0) FROM paper_trades WHERE status = 'open'"
                )
                row = await cursor.fetchone()
                open_cost = float(row[0]) if row else 0.0
        except Exception as exc:
            logger.warning(f"Portfolio risk gate DB error: {exc}")
            open_cost = 0.0

        premium = signal.atm_premium or 0.0
        est_cost = premium * 100.0
        total_risk = open_cost + est_cost
        risk_pct = (total_risk / portfolio_size * 100.0) if portfolio_size > 0 else 0.0

        if risk_pct > max_pct:
            return GateOutcome(self.name, GateResult.FAIL,
                               f"Portfolio risk {risk_pct:.1f}% > {max_pct:.0f}% limit "
                               f"(open=${open_cost:.0f} + new=${est_cost:.0f})")
        return GateOutcome(self.name, GateResult.PASS,
                           f"Portfolio risk {risk_pct:.1f}% within {max_pct:.0f}% limit")


class PerTradeRiskGate(EntryGate):
    """Gate 8: Single trade cost vs portfolio limit."""

    name = "per_trade_risk"

    async def evaluate(self, ctx: dict[str, Any]) -> GateOutcome:
        settings = ctx["settings"]
        if not getattr(settings, "ENABLE_RISK_MANAGER", False):
            return GateOutcome(self.name, GateResult.SKIP, "Risk manager disabled")

        signal = ctx["signal"]
        portfolio_size = settings.PORTFOLIO_SIZE
        max_pct = settings.MAX_LOSS_PER_TRADE_PCT

        premium = signal.atm_premium or 0.0
        est_cost = premium * 100.0
        trade_pct = (est_cost / portfolio_size * 100.0) if portfolio_size > 0 else 0.0

        if trade_pct > max_pct:
            return GateOutcome(self.name, GateResult.FAIL,
                               f"Trade {trade_pct:.1f}% > {max_pct:.1f}% limit")
        return GateOutcome(self.name, GateResult.PASS,
                           f"Trade {trade_pct:.1f}% within {max_pct:.1f}% limit")


class WeeklyLossGate(EntryGate):
    """Gate 9: Weekly cumulative loss limit."""

    name = "weekly_loss"

    async def evaluate(self, ctx: dict[str, Any]) -> GateOutcome:
        settings = ctx["settings"]
        if not getattr(settings, "ENABLE_RISK_MANAGER", False):
            return GateOutcome(self.name, GateResult.SKIP, "Risk manager disabled")

        db_path = ctx["db_path"]
        portfolio_size = settings.PORTFOLIO_SIZE
        max_pct = settings.WEEKLY_LOSS_LIMIT_PCT

        from datetime import timedelta
        from options_owl.journal.db import connect as _connect_db

        try:
            now = _now_et()
            week_start = (now - timedelta(days=now.weekday())).strftime("%Y-%m-%d")
            async with _connect_db(db_path) as conn:
                cursor = await conn.execute(
                    "SELECT COALESCE(SUM(pnl_dollars), 0) FROM paper_trades "
                    "WHERE status = 'closed' AND pnl_dollars < 0 AND closed_at >= ?",
                    (week_start,),
                )
                row = await cursor.fetchone()
                weekly_loss = float(row[0]) if row else 0.0
        except Exception as exc:
            logger.warning(f"Weekly loss gate DB error: {exc}")
            weekly_loss = 0.0

        loss_pct = (abs(weekly_loss) / portfolio_size * 100.0) if portfolio_size > 0 else 0.0
        if loss_pct >= max_pct:
            return GateOutcome(self.name, GateResult.FAIL,
                               f"Weekly loss {loss_pct:.1f}% >= {max_pct:.0f}% limit")
        return GateOutcome(self.name, GateResult.PASS,
                           f"Weekly loss {loss_pct:.1f}% within {max_pct:.0f}% limit")


class LiquidityGate(EntryGate):
    """Gate 10: Open interest / volume / bid-ask spread filter."""

    name = "liquidity"

    async def evaluate(self, ctx: dict[str, Any]) -> GateOutcome:
        settings = ctx["settings"]
        if not getattr(settings, "ENABLE_LIQUIDITY_FILTER", False):
            return GateOutcome(self.name, GateResult.SKIP, "Liquidity filter disabled")

        try:
            from options_owl.risk.liquidity_filter import (
                check_liquidity,
                fetch_option_liquidity,
            )

            signal = ctx["signal"]
            ticker = signal.ticker
            strike = signal.strike
            expiry = getattr(signal, "expiry", None) or ""
            option_type = "put" if signal.direction.value == "put" else "call"

            liquidity = await fetch_option_liquidity(
                ticker, strike, expiry, option_type, settings,
            )
            passes, reason = check_liquidity(liquidity, settings)
            if not passes:
                return GateOutcome(self.name, GateResult.FAIL, reason)
            return GateOutcome(self.name, GateResult.PASS, reason)
        except Exception as exc:
            return GateOutcome(self.name, GateResult.SKIP, f"Liquidity check error: {exc}")


class IVFilterGate(EntryGate):
    """Gate 11: IV Rank/Percentile filter."""

    name = "iv_filter"

    async def evaluate(self, ctx: dict[str, Any]) -> GateOutcome:
        settings = ctx["settings"]
        if not getattr(settings, "ENABLE_IV_FILTER", False):
            return GateOutcome(self.name, GateResult.SKIP, "IV filter disabled")

        try:
            import asyncio
            from options_owl.signals.iv_filter import check_iv_filter
            signal = ctx["signal"]
            passes, reason = await asyncio.to_thread(check_iv_filter, signal.ticker, settings)
            if not passes:
                return GateOutcome(self.name, GateResult.FAIL, reason)
            return GateOutcome(self.name, GateResult.PASS, reason)
        except Exception as exc:
            return GateOutcome(self.name, GateResult.SKIP, f"IV filter error: {exc}")


class VIXRegimeGate(EntryGate):
    """Gate 11: VIX regime check."""

    name = "vix_regime"

    async def evaluate(self, ctx: dict[str, Any]) -> GateOutcome:
        settings = ctx["settings"]
        if not getattr(settings, "ENABLE_VIX_FILTER", False):
            return GateOutcome(self.name, GateResult.SKIP, "VIX filter disabled")

        try:
            from options_owl.risk.vix_regime import check_vix_regime
            regime = check_vix_regime(settings)
            if not regime.can_trade:
                return GateOutcome(self.name, GateResult.FAIL, regime.reason)
            return GateOutcome(self.name, GateResult.PASS, regime.reason)
        except Exception as exc:
            return GateOutcome(self.name, GateResult.SKIP, f"VIX check error: {exc}")


class AnalystFilterGate(EntryGate):
    """Gate 12: Bot/analyst performance filter."""

    name = "analyst_filter"

    async def evaluate(self, ctx: dict[str, Any]) -> GateOutcome:
        settings = ctx["settings"]
        if not getattr(settings, "ENABLE_ANALYST_FILTER", False):
            return GateOutcome(self.name, GateResult.SKIP, "Analyst filter disabled")

        try:
            from options_owl.signals.analyst_tracker import check_analyst_filter
            signal = ctx["signal"]
            db_path = ctx["db_path"]
            passes, reason, _stats = await check_analyst_filter(
                db_path, signal.bot_source.value, settings,
            )
            if not passes:
                return GateOutcome(self.name, GateResult.FAIL, reason)
            return GateOutcome(self.name, GateResult.PASS, reason)
        except Exception as exc:
            return GateOutcome(self.name, GateResult.SKIP, f"Analyst check error: {exc}")


class CircuitBreakerGate(EntryGate):
    """Gate 13: Circuit breaker checks (consecutive losses, drawdown, time buffers)."""

    name = "circuit_breaker"

    async def evaluate(self, ctx: dict[str, Any]) -> GateOutcome:
        settings = ctx["settings"]
        if not getattr(settings, "ENABLE_CIRCUIT_BREAKERS", False):
            return GateOutcome(self.name, GateResult.SKIP, "Circuit breakers disabled")

        try:
            from options_owl.risk.circuit_breaker import CircuitBreaker
            db_path = ctx["db_path"]
            approved, reasons = await CircuitBreaker.check_all(db_path, settings)
            if not approved:
                combined = "; ".join(reasons)
                return GateOutcome(self.name, GateResult.FAIL, combined)
            return GateOutcome(self.name, GateResult.PASS, "All circuit breakers clear")
        except Exception as exc:
            return GateOutcome(self.name, GateResult.SKIP, f"Circuit breaker error: {exc}")


class BalanceGate(EntryGate):
    """Gate 14: Sufficient balance to open the trade."""

    name = "balance"

    async def evaluate(self, ctx: dict[str, Any]) -> GateOutcome:
        signal = ctx["signal"]
        portfolio = ctx.get("portfolio")
        if not portfolio:
            return GateOutcome(self.name, GateResult.SKIP, "No portfolio data")

        premium = signal.atm_premium or 0.0
        cost = premium * 100.0  # minimum 1 contract
        balance = portfolio["current_balance"]

        if cost > balance:
            return GateOutcome(self.name, GateResult.FAIL,
                               f"Cost ${cost:.2f} > balance ${balance:.2f}")
        return GateOutcome(self.name, GateResult.PASS,
                           f"Balance ${balance:.2f} sufficient")


class AntiChaseGate(EntryGate):
    """Gate: Reject if underlying has moved too far from alert price (chasing)."""

    name = "anti_chase"

    async def evaluate(self, ctx: dict[str, Any]) -> GateOutcome:
        settings = ctx["settings"]
        if not getattr(settings, "ENABLE_VINNY_STRATEGY", False):
            return GateOutcome(self.name, GateResult.SKIP, "Vinny strategy disabled")

        signal = ctx["signal"]
        current_price = ctx.get("current_price")
        if current_price is None:
            return GateOutcome(self.name, GateResult.SKIP, "No current price available")

        # Tiered anti-chase: high-score signals get more room
        score = getattr(signal, "score", 0) or 0
        base_move = settings.ANTI_CHASE_MAX_MOVE_PCT
        if score >= 150:
            max_move = max(base_move, 0.75)
        elif score >= 120:
            max_move = max(base_move, 0.5)
        else:
            max_move = base_move

        from options_owl.risk.vinny_strategy import check_anti_chase
        passed, reason = check_anti_chase(signal.entry_price, current_price, max_move)
        if not passed:
            return GateOutcome(self.name, GateResult.FAIL,
                               f"{reason} (score {score}, max {max_move}%)")
        return GateOutcome(self.name, GateResult.PASS,
                           f"{reason} (score {score}, max {max_move}%)")


class MomentumConfirmGate(EntryGate):
    """Gate: Use 5m/15m candle data to confirm underlying momentum before entry.

    Rejects trades where the underlying is moving AGAINST the signal direction.
    Uses RSI + recent candle trend to detect fading momentum.

    Backtested insight: the biggest losers (GOOGL -$172, NVDA -$120, AMZN -$180)
    all entered when the underlying was flat or drifting against the thesis.
    """

    name = "momentum_confirm"

    async def evaluate(self, ctx: dict[str, Any]) -> GateOutcome:
        settings = ctx["settings"]
        if not getattr(settings, "ENABLE_MOMENTUM_CONFIRM", True):
            return GateOutcome(self.name, GateResult.SKIP, "Momentum confirm disabled")

        signal = ctx["signal"]
        is_call = signal.direction.value.lower() in ("call", "bullish")

        # Try to get candle data from the shared candle cache
        candle_cache = ctx.get("candle_cache")
        if candle_cache is None:
            return GateOutcome(self.name, GateResult.SKIP, "No candle cache available")

        try:
            data = await asyncio.wait_for(
                candle_cache.get_candle_data(signal.ticker), timeout=15,
            )
        except asyncio.TimeoutError:
            return GateOutcome(self.name, GateResult.SKIP, "Candle fetch timed out (15s)")
        except Exception as exc:
            return GateOutcome(self.name, GateResult.SKIP, f"Candle fetch failed: {exc}")

        indicators = data.get("indicators", {})
        tf_5m = indicators.get("5m", {})
        tf_15m = indicators.get("15m", {})

        rsi_5m = tf_5m.get("rsi")
        rsi_15m = tf_15m.get("rsi")

        # Check 5m bars for recent price direction
        bars_5m = data.get("5m", [])
        against_count = 0
        if len(bars_5m) >= 3:
            # Check last 3 bars: are closes trending against our direction?
            for i in range(-3, 0):
                bar = bars_5m[i]
                if is_call and bar.close < bar.open:
                    against_count += 1
                elif not is_call and bar.close > bar.open:
                    against_count += 1

        reasons = []

        # Strong rejection: RSI extreme against direction on BOTH timeframes
        if rsi_5m is not None and rsi_15m is not None:
            if is_call and rsi_5m < 35 and rsi_15m < 40:
                reasons.append(f"RSI bearish (5m={rsi_5m:.0f}, 15m={rsi_15m:.0f})")
            elif not is_call and rsi_5m > 65 and rsi_15m > 60:
                reasons.append(f"RSI bullish (5m={rsi_5m:.0f}, 15m={rsi_15m:.0f})")

        # 3 of last 3 candles against direction = fading
        if against_count >= 3:
            reasons.append(f"Last 3 5m candles all against direction")

        # Bearish candle pattern on 5m
        pattern_5m = tf_5m.get("pattern")
        if pattern_5m:
            if is_call and pattern_5m in ("shooting_star", "engulfing_bearish"):
                reasons.append(f"5m bearish pattern: {pattern_5m}")
            elif not is_call and pattern_5m in ("hammer", "engulfing_bullish"):
                reasons.append(f"5m bullish pattern: {pattern_5m}")

        if len(reasons) >= 2:
            # Need 2+ negative signals to reject — single signal could be noise
            return GateOutcome(
                self.name, GateResult.FAIL,
                f"Momentum against thesis: {'; '.join(reasons)}"
            )

        # Build pass reason with what we saw
        pass_parts = []
        if rsi_5m is not None:
            pass_parts.append(f"RSI5m={rsi_5m:.0f}")
        if rsi_15m is not None:
            pass_parts.append(f"RSI15m={rsi_15m:.0f}")
        if against_count > 0:
            pass_parts.append(f"{against_count}/3 candles against")
        if not pass_parts:
            pass_parts.append("no candle data")
        return GateOutcome(self.name, GateResult.PASS, ", ".join(pass_parts))


class TimeOfDayGate(EntryGate):
    """Gate: Time-of-day score thresholds — require higher scores at market open/close."""

    name = "time_of_day"

    async def evaluate(self, ctx: dict[str, Any]) -> GateOutcome:
        settings = ctx["settings"]
        if not getattr(settings, "ENABLE_VINNY_STRATEGY", False):
            return GateOutcome(self.name, GateResult.SKIP, "Vinny strategy disabled")

        signal = ctx["signal"]
        from datetime import datetime, timedelta

        try:
            from zoneinfo import ZoneInfo
            et = ZoneInfo("America/New_York")
        except ImportError:
            from datetime import timezone
            et = timezone(timedelta(hours=-5))

        now = datetime.now(tz=et)

        # Hard cutoff: no new entries after ENTRY_HARD_CUTOFF (default 3:55 PM ET)
        hard_cutoff_h = getattr(settings, "ENTRY_HARD_CUTOFF_HOUR", 15)
        hard_cutoff_m = getattr(settings, "ENTRY_HARD_CUTOFF_MINUTE", 55)
        hard_cutoff = now.replace(hour=hard_cutoff_h, minute=hard_cutoff_m,
                                  second=0, microsecond=0)
        if now >= hard_cutoff:
            return GateOutcome(
                self.name, GateResult.FAIL,
                f"No new entries after {hard_cutoff_h}:{hard_cutoff_m:02d} ET "
                f"(theta crush makes late entries unprofitable)",
            )

        # Morning cutoff: block ALL entries after 11:00 AM ET
        # Backtest: 9:30-10:30 AM ET = +$62K, after 1:30 PM = -$231K
        if getattr(settings, "ENABLE_MORNING_CUTOFF", False):
            morning_h = getattr(settings, "ENTRY_MORNING_CUTOFF_HOUR", 11)
            morning_m = getattr(settings, "ENTRY_MORNING_CUTOFF_MINUTE", 0)
            morning_cutoff = now.replace(hour=morning_h, minute=morning_m,
                                         second=0, microsecond=0)
            if now >= morning_cutoff:
                return GateOutcome(
                    self.name, GateResult.FAIL,
                    f"Morning cutoff: no new entries after {morning_h}:{morning_m:02d} ET "
                    f"(backtest shows only pre-{morning_h}:{morning_m:02d} ET entries are profitable)",
                )

        # Early morning: before TOD_EARLY_CUTOFF, need higher score
        early_cutoff = now.replace(
            hour=settings.TOD_EARLY_CUTOFF_HOUR,
            minute=settings.TOD_EARLY_CUTOFF_MINUTE,
            second=0, microsecond=0,
        )
        if now < early_cutoff and signal.score < settings.TOD_EARLY_MIN_SCORE:
            return GateOutcome(
                self.name, GateResult.FAIL,
                f"Before {settings.TOD_EARLY_CUTOFF_HOUR}:{settings.TOD_EARLY_CUTOFF_MINUTE:02d} — "
                f"score {signal.score} < {settings.TOD_EARLY_MIN_SCORE} required",
            )

        # Late afternoon: after TOD_LATE_CUTOFF, need higher score
        late_cutoff = now.replace(
            hour=settings.TOD_LATE_CUTOFF_HOUR,
            minute=settings.TOD_LATE_CUTOFF_MINUTE,
            second=0, microsecond=0,
        )
        if now >= late_cutoff and signal.score < settings.TOD_LATE_MIN_SCORE:
            return GateOutcome(
                self.name, GateResult.FAIL,
                f"After {settings.TOD_LATE_CUTOFF_HOUR}:{settings.TOD_LATE_CUTOFF_MINUTE:02d} — "
                f"score {signal.score} < {settings.TOD_LATE_MIN_SCORE} required",
            )

        return GateOutcome(self.name, GateResult.PASS,
                           f"Time-of-day OK (score {signal.score})")


class ConsecutiveLoserGate(EntryGate):
    """Gate: Pause trading after consecutive losses."""

    name = "consecutive_loser"

    async def evaluate(self, ctx: dict[str, Any]) -> GateOutcome:
        settings = ctx["settings"]
        if not getattr(settings, "ENABLE_VINNY_STRATEGY", False):
            return GateOutcome(self.name, GateResult.SKIP, "Vinny strategy disabled")

        db_path = ctx.get("db_path")
        if not db_path:
            return GateOutcome(self.name, GateResult.SKIP, "No db_path")

        from options_owl.journal.db import connect as _connect_db
        try:
            async with _connect_db(db_path) as conn:
                # Get last N closed trades ordered by close time
                max_consec = settings.CONSECUTIVE_LOSER_MAX
                # Only count today's losses — consecutive loser resets daily.
                # closed_at is stored in UTC; convert today's ET boundaries to UTC.
                from datetime import datetime, timedelta
                from zoneinfo import ZoneInfo
                now_et = datetime.now(tz=ZoneInfo("America/New_York"))
                today_start_et = now_et.replace(hour=0, minute=0, second=0, microsecond=0)
                today_end_et = today_start_et + timedelta(days=1)
                utc = ZoneInfo("UTC")
                start_utc = today_start_et.astimezone(utc).strftime("%Y-%m-%d %H:%M:%S")
                end_utc = today_end_et.astimezone(utc).strftime("%Y-%m-%d %H:%M:%S")
                cursor = await conn.execute(
                    "SELECT pnl_dollars, closed_at FROM paper_trades "
                    "WHERE status = 'closed' AND parent_trade_id IS NULL "
                    "AND (exit_reason IS NULL OR exit_reason != 'reconcile_phantom') "
                    "AND closed_at >= ? AND closed_at < ? "
                    "ORDER BY closed_at DESC LIMIT ?",
                    (start_utc, end_utc, max_consec),
                )
                rows = await cursor.fetchall()

                if len(rows) < max_consec:
                    return GateOutcome(self.name, GateResult.PASS,
                                       f"Only {len(rows)} recent trades")

                # Check if all recent trades are losses
                all_losses = all(r[0] is not None and r[0] < 0 for r in rows)
                if not all_losses:
                    return GateOutcome(self.name, GateResult.PASS,
                                       "Recent trades include wins")

                last_loss_at = rows[0][1]  # most recent close time

                from options_owl.risk.vinny_strategy import check_consecutive_loser_pause
                can_trade, reason = check_consecutive_loser_pause(
                    consecutive_losses=max_consec,
                    last_loss_at=last_loss_at,
                    max_consecutive=max_consec,
                    pause_minutes=settings.CONSECUTIVE_LOSER_PAUSE_MINUTES,
                )
                if not can_trade:
                    return GateOutcome(self.name, GateResult.FAIL, reason)
                return GateOutcome(self.name, GateResult.PASS, reason)

        except Exception as exc:
            return GateOutcome(self.name, GateResult.SKIP,
                               f"Consecutive loser check error: {exc}")


class CandleConfirmationGate(EntryGate):
    """Gate: Multi-TF candle confirmation before entry.

    Uses the same ENRG voting logic (RSI, OBV, patterns across 5m/15m/30m/1h/4h)
    to confirm the signal direction BEFORE entering. If the candles disagree
    with the signal, the trade is blocked.

    This prevents entering trades where the signal says "bullish" but the
    actual price action across multiple timeframes is bearish.
    """

    name = "candle_confirmation"

    async def evaluate(self, ctx: dict[str, Any]) -> GateOutcome:
        settings = ctx["settings"]
        if not getattr(settings, "ENABLE_CANDLE_CONFIRMATION", True):
            return GateOutcome(self.name, GateResult.SKIP, "Candle confirmation disabled")

        signal = ctx["signal"]
        direction = getattr(signal, "direction", None)
        if direction is None:
            return GateOutcome(self.name, GateResult.SKIP, "No direction on signal")

        direction_str = direction.value.lower() if hasattr(direction, "value") else str(direction).lower()

        # Get candle cache from context (injected by paper_trader)
        candle_cache = ctx.get("candle_cache")
        if candle_cache is None:
            return GateOutcome(self.name, GateResult.SKIP, "No candle cache available")

        try:
            candle_data = await asyncio.wait_for(
                candle_cache.get_candle_data(signal.ticker), timeout=15,
            )
        except asyncio.TimeoutError:
            return GateOutcome(self.name, GateResult.SKIP, "Candle fetch timed out (15s)")
        except Exception as exc:
            return GateOutcome(self.name, GateResult.SKIP, f"Candle fetch error: {exc}")

        from options_owl.collectors.candle_cache import evaluate_enrg

        action, reason = evaluate_enrg(candle_data, direction_str)

        if action == "IMMEDIATE_EXIT":
            # Candles strongly disagree — block the trade
            return GateOutcome(
                self.name, GateResult.FAIL,
                f"Candles DISAGREE with {direction_str}: {reason}",
            )

        if action == "HOLD":
            # Candles confirm the direction
            return GateOutcome(
                self.name, GateResult.PASS,
                f"Candles CONFIRM {direction_str}: {reason}",
            )

        # PROCEED = inconclusive, allow the trade
        return GateOutcome(
            self.name, GateResult.PASS,
            f"Candles inconclusive: {reason}",
        )


# ---------------------------------------------------------------------------
# Exit pipeline — position exit gates
# ---------------------------------------------------------------------------


class ExitGate:
    """Base class for an exit gate. Returns (should_exit, reason, exit_code)."""

    name: str = "unnamed_exit"

    async def evaluate(self, ctx: dict[str, Any]) -> GateOutcome:
        raise NotImplementedError


class StopLossExitGate(ExitGate):
    """Exit 1: Premium-based stop loss.

    Uses option premium drop from entry as the stop mechanism.
    Underlying price stops are disabled by default (too tight for 0DTE,
    causes whipsaw — data shows 54% of trades stopped before T1 hit).

    Grace period: no stop checks for the first N minutes to let the trade breathe.
    """

    name = "stop_loss"

    async def evaluate(self, ctx: dict[str, Any]) -> GateOutcome:
        trade = ctx["trade"]
        price = ctx["current_price"]
        exit_premium = ctx.get("exit_premium")
        settings = ctx.get("settings")
        option_type = trade["option_type"]
        entry_premium = trade["premium_per_contract"]

        # --- Catastrophic stop: fires REGARDLESS of grace period ---
        # SUPERSEDED by bounce_fade gate (v3) when ENABLE_BOUNCE_FADE=true.
        # Bounce-fade waits for a recovery instead of panic selling at the bottom.
        enable_catastrophic = getattr(settings, "ENABLE_CATASTROPHIC_STOP", False) if settings else False
        catastrophic_pct = getattr(settings, "CATASTROPHIC_STOP_PCT", 45.0) if settings else 45.0
        if (
            enable_catastrophic
            and exit_premium is not None
            and entry_premium > 0
            and catastrophic_pct > 0
        ):
            drop_pct = (entry_premium - exit_premium) / entry_premium * 100
            if drop_pct >= catastrophic_pct:
                return GateOutcome(
                    self.name, GateResult.FAIL,
                    f"CATASTROPHIC STOP: ${exit_premium:.2f} is -{drop_pct:.1f}% from "
                    f"entry ${entry_premium:.2f} (threshold -{catastrophic_pct:.0f}%, "
                    f"bypasses grace period)"
                )

        # --- Grace period: centralized smart grace check ---
        grace_active, grace_reason = is_grace_active(ctx)
        if grace_active:
            return GateOutcome(self.name, GateResult.PASS, grace_reason)

        # --- Premium-based stop (primary) ---
        if (
            settings
            and getattr(settings, "PREMIUM_STOP_ENABLED", False)
            and exit_premium is not None
            and entry_premium > 0
        ):
            drop_pct = (entry_premium - exit_premium) / entry_premium * 100
            threshold = settings.PREMIUM_STOP_PCT
            # ENRG widening: if ENRG said HOLD, widen the hard stop
            enrg_widen = ctx.get("enrg_widen_stop_pct", 0.0)
            if enrg_widen > 0:
                threshold = threshold * (1 + enrg_widen / 100)
            if drop_pct >= threshold:
                return GateOutcome(
                    self.name, GateResult.FAIL,
                    f"Premium stop: ${exit_premium:.2f} is -{drop_pct:.1f}% from "
                    f"entry ${entry_premium:.2f} (threshold -{threshold:.0f}%)"
                )
            return GateOutcome(
                self.name, GateResult.PASS,
                f"Premium ${exit_premium:.2f} down {drop_pct:.1f}% "
                f"(threshold -{threshold:.0f}%)"
            )

        # --- Underlying price-based stop (only if explicitly enabled) ---
        if settings and not getattr(settings, "ENABLE_UNDERLYING_STOP", False):
            return GateOutcome(self.name, GateResult.SKIP, "Underlying stop disabled")

        stop = trade.get("stop_price")
        if stop is None:
            return GateOutcome(self.name, GateResult.SKIP, "No stop price")

        entry_price = trade.get("entry_price", 0)
        if settings and entry_price > 0:
            min_pct = getattr(settings, "MIN_UNDERLYING_STOP_PCT", 0.5)
            min_distance = entry_price * min_pct / 100
            if option_type == "call":
                max_stop = entry_price - min_distance
                if stop > max_stop:
                    stop = max_stop
            else:
                min_stop = entry_price + min_distance
                if stop < min_stop:
                    stop = min_stop

        hit = (
            (option_type == "call" and price <= stop)
            or (option_type == "put" and price >= stop)
        )
        if hit:
            return GateOutcome(self.name, GateResult.FAIL,
                               f"Stop hit @ ${price:.2f} (stop=${stop:.2f})")
        return GateOutcome(self.name, GateResult.PASS,
                           f"Price ${price:.2f} clear of stop ${stop:.2f}")


class ENRGExitGate(ExitGate):
    """Exit: Early Negative Thesis Revalidation Gate (ENRG).

    Fires during the grace period when the position is negative. Uses
    multi-timeframe candle voting (5m/15m/30m/1h/4h with weights 1/1/1/2/2)
    to decide:
      HOLD           — thesis intact, widen hard stop by +15%
      IMMEDIATE_EXIT — extreme reversal pattern on higher TF, exit now
      PROCEED        — inconclusive, let normal hard stop handle it

    One-shot per position: once evaluated, the result is stored in
    ctx['enrg_result'] and persisted via trade_events so it doesn't re-fire.
    """

    name = "enrg"

    async def evaluate(self, ctx: dict[str, Any]) -> GateOutcome:
        settings = ctx.get("settings")
        if not settings or not getattr(settings, "ENABLE_ENRG", False):
            return GateOutcome(self.name, GateResult.SKIP, "ENRG disabled")

        trade = ctx["trade"]

        # One-shot: skip if already evaluated for this trade
        if trade.get("enrg_result"):
            return GateOutcome(self.name, GateResult.SKIP,
                               f"ENRG already fired: {trade['enrg_result']}")

        # Only fires during grace period
        grace_minutes = getattr(settings, "STOP_GRACE_PERIOD_MINUTES", 20)
        opened_at = trade.get("opened_at")
        if not opened_at:
            return GateOutcome(self.name, GateResult.SKIP, "No opened_at")

        try:
            from datetime import datetime
            opened_dt = datetime.fromisoformat(opened_at)
            now = ctx.get("now_et") or _now_et()
            if now.tzinfo and opened_dt.tzinfo is None:
                opened_dt = opened_dt.replace(tzinfo=now.tzinfo)
            elapsed = (now - opened_dt).total_seconds() / 60
        except (ValueError, TypeError):
            return GateOutcome(self.name, GateResult.SKIP, "Cannot parse opened_at")

        if elapsed >= grace_minutes:
            return GateOutcome(self.name, GateResult.SKIP,
                               f"Past grace period ({elapsed:.0f}m >= {grace_minutes}m)")

        # Only fires when position is negative
        exit_premium = ctx.get("exit_premium")
        entry_premium = trade["premium_per_contract"]
        if exit_premium is None or entry_premium <= 0:
            return GateOutcome(self.name, GateResult.SKIP, "Missing premium data")
        if exit_premium >= entry_premium:
            return GateOutcome(self.name, GateResult.PASS,
                               f"Position positive (${exit_premium:.2f} >= ${entry_premium:.2f})")

        # Need candle data
        candle_data = ctx.get("candle_data", {})
        if not candle_data or not candle_data.get("indicators"):
            return GateOutcome(self.name, GateResult.SKIP, "No candle data for ENRG")

        direction = trade.get("option_type", "call")

        from options_owl.collectors.candle_cache import evaluate_enrg
        action, reason = evaluate_enrg(candle_data, direction)

        # Store result for one-shot persistence
        ctx["enrg_result"] = action
        ctx["enrg_reason"] = reason

        if action == "IMMEDIATE_EXIT":
            return GateOutcome(self.name, GateResult.FAIL, reason)

        if action == "HOLD":
            # Widen hard stop — store the widened factor in ctx for StopLossExitGate
            widen_pct = getattr(settings, "ENRG_WIDEN_STOP_PCT", 15.0)
            ctx["enrg_widen_stop_pct"] = widen_pct
            return GateOutcome(self.name, GateResult.PASS,
                               f"{reason} — widening stop by +{widen_pct:.0f}%")

        # PROCEED — let normal stop handle it
        return GateOutcome(self.name, GateResult.PASS, reason)


class BEClampExitGate(ExitGate):
    """Exit: Breakeven clamp (v2.2 §4).

    Once peak gain reaches +15%, the floor = entry premium. The trade should
    never go from green back to red. This is a defensive gate — zero downside.

    Only fires when:
      1. Peak gain ever reached BE_CLAMP_ACTIVATION_PCT (+15%)
      2. Current premium has dropped back to or below entry

    Does NOT fire during grace period (first 5 minutes).
    """

    name = "be_clamp"

    async def evaluate(self, ctx: dict[str, Any]) -> GateOutcome:
        settings = ctx.get("settings")
        if not settings or not getattr(settings, "ENABLE_BE_CLAMP", False):
            return GateOutcome(self.name, GateResult.SKIP, "BE clamp disabled")

        trade = ctx["trade"]
        exit_premium = ctx.get("exit_premium")
        entry_premium = trade["premium_per_contract"]
        mfe_premium = trade.get("mfe_premium")

        if exit_premium is None or entry_premium <= 0 or mfe_premium is None:
            return GateOutcome(self.name, GateResult.SKIP, "Missing premium data")

        # Check grace period (centralized smart grace)
        grace_active, grace_reason = is_grace_active(ctx)
        if grace_active:
            return GateOutcome(self.name, GateResult.PASS, grace_reason)

        activation = getattr(settings, "BE_CLAMP_ACTIVATION_PCT", 15.0)
        peak_gain_pct = (mfe_premium - entry_premium) / entry_premium * 100

        if peak_gain_pct < activation:
            return GateOutcome(
                self.name, GateResult.PASS,
                f"Peak +{peak_gain_pct:.1f}% < activation +{activation:.0f}%"
            )

        # Peak was above activation — check if current is back at or below entry
        if exit_premium <= entry_premium:
            return GateOutcome(
                self.name, GateResult.FAIL,
                f"BE clamp: peaked at +{peak_gain_pct:.0f}% (${mfe_premium:.2f}), "
                f"now ${exit_premium:.2f} <= entry ${entry_premium:.2f} — "
                f"locking breakeven"
            )

        current_gain = (exit_premium - entry_premium) / entry_premium * 100
        return GateOutcome(
            self.name, GateResult.PASS,
            f"BE clamp OK: +{current_gain:.1f}% (peak +{peak_gain_pct:.0f}%)"
        )


class SoftTrailExitGate(ExitGate):
    """Exit: Soft trail in the 15-35% gain band (v2.2 §11).

    When peak gain is between SOFT_TRAIL_MIN_PCT and SOFT_TRAIL_MAX_PCT,
    the floor = entry + (peak_gain * SOFT_TRAIL_FLOOR_PCT). Keeps 50% of
    the move instead of giving it all back.

    This fills the gap where adaptive trail is dormant (below +35%) and
    no other trail protects gains. Hands off to adaptive trail above +35%.

    Does NOT fire during grace period.
    """

    name = "soft_trail"

    async def evaluate(self, ctx: dict[str, Any]) -> GateOutcome:
        settings = ctx.get("settings")
        if not settings or not getattr(settings, "ENABLE_SOFT_TRAIL", False):
            return GateOutcome(self.name, GateResult.SKIP, "Soft trail disabled")

        trade = ctx["trade"]
        exit_premium = ctx.get("exit_premium")
        entry_premium = trade["premium_per_contract"]
        mfe_premium = trade.get("mfe_premium")

        if exit_premium is None or entry_premium <= 0 or mfe_premium is None:
            return GateOutcome(self.name, GateResult.SKIP, "Missing premium data")

        if mfe_premium <= entry_premium:
            return GateOutcome(self.name, GateResult.SKIP, "No profit to protect")

        # Check grace period (centralized smart grace)
        grace_active, grace_reason = is_grace_active(ctx)
        if grace_active:
            return GateOutcome(self.name, GateResult.PASS, grace_reason)

        min_pct = getattr(settings, "SOFT_TRAIL_MIN_PCT", 15.0)
        max_pct = getattr(settings, "SOFT_TRAIL_MAX_PCT", 35.0)
        floor_pct = getattr(settings, "SOFT_TRAIL_FLOOR_PCT", 50.0)

        peak_gain_pct = (mfe_premium - entry_premium) / entry_premium * 100

        # Only active in the soft trail band (15-35%)
        if peak_gain_pct < min_pct:
            return GateOutcome(
                self.name, GateResult.PASS,
                f"Peak +{peak_gain_pct:.1f}% < min +{min_pct:.0f}%"
            )

        if peak_gain_pct >= max_pct:
            # Hand off to adaptive trail
            return GateOutcome(
                self.name, GateResult.SKIP,
                f"Peak +{peak_gain_pct:.1f}% >= max +{max_pct:.0f}% (adaptive trail takes over)"
            )

        # Calculate floor: entry + (peak_gain * floor_pct)
        peak_gain_dollars = mfe_premium - entry_premium
        floor = entry_premium + peak_gain_dollars * (floor_pct / 100)

        if exit_premium <= floor:
            current_gain = (exit_premium - entry_premium) / entry_premium * 100
            return GateOutcome(
                self.name, GateResult.FAIL,
                f"Soft trail: prem ${exit_premium:.2f} <= floor ${floor:.2f} "
                f"(peak +{peak_gain_pct:.0f}%, keeping {floor_pct:.0f}% = "
                f"+{(floor - entry_premium) / entry_premium * 100:.0f}%)"
            )

        current_gain = (exit_premium - entry_premium) / entry_premium * 100
        floor_gain = (floor - entry_premium) / entry_premium * 100
        return GateOutcome(
            self.name, GateResult.PASS,
            f"Soft trail: +{current_gain:.1f}% above floor +{floor_gain:.0f}% "
            f"(peak +{peak_gain_pct:.0f}%)"
        )


class TrailingStopExitGate(ExitGate):
    """Exit: Trailing premium stop.

    Once premium rises above TRAILING_STOP_ACTIVATION_PCT from entry, this gate
    tracks the peak premium (via MFE). If current premium drops more than
    TRAILING_STOP_DROP_PCT from that peak, trigger an exit — locking in profit
    instead of riding back to a loss.
    """

    name = "trailing_stop"

    async def evaluate(self, ctx: dict[str, Any]) -> GateOutcome:
        settings = ctx.get("settings")
        if not settings or not getattr(settings, "ENABLE_TRAILING_STOP", False):
            return GateOutcome(self.name, GateResult.SKIP, "Trailing stop disabled")

        # Skip when adaptive trail is enabled — it's the primary mechanism
        # and this legacy gate conflicts (tighter activation, cuts winners short)
        if getattr(settings, "ENABLE_ADAPTIVE_TRAIL", False):
            return GateOutcome(self.name, GateResult.SKIP,
                               "Skipped: adaptive trail is primary")

        trade = ctx["trade"]
        exit_premium = ctx.get("exit_premium")
        entry_premium = trade["premium_per_contract"]
        mfe_premium = trade.get("mfe_premium")

        if exit_premium is None or entry_premium <= 0 or mfe_premium is None:
            return GateOutcome(self.name, GateResult.SKIP, "Missing premium data")

        # Check if trailing stop has been activated (premium rose enough from entry)
        activation_pct = settings.TRAILING_STOP_ACTIVATION_PCT
        peak_gain_pct = (mfe_premium - entry_premium) / entry_premium * 100

        if peak_gain_pct < activation_pct:
            return GateOutcome(
                self.name, GateResult.PASS,
                f"Peak gain {peak_gain_pct:.1f}% < activation {activation_pct:.0f}%"
            )

        # Trailing stop is active — check if premium dropped too far from peak
        drop_from_peak_pct = (mfe_premium - exit_premium) / mfe_premium * 100
        drop_threshold = settings.TRAILING_STOP_DROP_PCT

        if drop_from_peak_pct >= drop_threshold:
            return GateOutcome(
                self.name, GateResult.FAIL,
                f"Trailing stop: prem ${exit_premium:.2f} dropped {drop_from_peak_pct:.1f}% "
                f"from peak ${mfe_premium:.2f} (threshold {drop_threshold:.0f}%)"
            )

        return GateOutcome(
            self.name, GateResult.PASS,
            f"Trailing: prem ${exit_premium:.2f}, peak ${mfe_premium:.2f}, "
            f"drop {drop_from_peak_pct:.1f}% < {drop_threshold:.0f}%"
        )


class Target2ExitGate(ExitGate):
    """Exit 2: Target 2 hit."""

    name = "target_2"

    async def evaluate(self, ctx: dict[str, Any]) -> GateOutcome:
        trade = ctx["trade"]
        price = ctx["current_price"]
        t2 = trade.get("target_2")
        option_type = trade["option_type"]
        last_hit = trade.get("last_target_hit") or 0

        if t2 is None:
            return GateOutcome(self.name, GateResult.SKIP, "No T2 set")
        if last_hit >= 2:
            return GateOutcome(self.name, GateResult.SKIP, "T2 already hit")

        hit = (
            (option_type == "call" and price >= t2)
            or (option_type == "put" and price <= t2)
        )
        if hit:
            return GateOutcome(self.name, GateResult.FAIL,
                               f"T2 hit @ ${price:.2f} (T2=${t2:.2f})")
        return GateOutcome(self.name, GateResult.PASS,
                           f"Price ${price:.2f} below T2 ${t2:.2f}")


class Target1ExitGate(ExitGate):
    """Exit 3: Target 1 (partial profit target) hit."""

    name = "target_1"

    async def evaluate(self, ctx: dict[str, Any]) -> GateOutcome:
        trade = ctx["trade"]
        price = ctx["current_price"]
        t1 = trade.get("target_1")
        option_type = trade["option_type"]
        last_hit = trade.get("last_target_hit") or 0

        if t1 is None:
            return GateOutcome(self.name, GateResult.SKIP, "No T1 set")
        if last_hit >= 1:
            return GateOutcome(self.name, GateResult.SKIP, "T1 already hit")

        hit = (
            (option_type == "call" and price >= t1)
            or (option_type == "put" and price <= t1)
        )
        if hit:
            return GateOutcome(self.name, GateResult.FAIL,
                               f"T1 hit @ ${price:.2f} (T1=${t1:.2f})")
        return GateOutcome(self.name, GateResult.PASS,
                           f"Price ${price:.2f} below T1 ${t1:.2f}")


class Target3ExitGate(ExitGate):
    """Exit: Target 3 hit."""

    name = "target_3"

    async def evaluate(self, ctx: dict[str, Any]) -> GateOutcome:
        trade = ctx["trade"]
        price = ctx["current_price"]
        t3 = trade.get("target_3")
        option_type = trade["option_type"]
        last_hit = trade.get("last_target_hit") or 0

        if t3 is None:
            return GateOutcome(self.name, GateResult.SKIP, "No T3 set")
        if last_hit >= 3:
            return GateOutcome(self.name, GateResult.SKIP, "T3 already hit")

        hit = (
            (option_type == "call" and price >= t3)
            or (option_type == "put" and price <= t3)
        )
        if hit:
            return GateOutcome(self.name, GateResult.FAIL,
                               f"T3 hit @ ${price:.2f} (T3=${t3:.2f})")
        return GateOutcome(self.name, GateResult.PASS,
                           f"Price ${price:.2f} below T3 ${t3:.2f}")


class Target4ExitGate(ExitGate):
    """Exit: Target 4 hit."""

    name = "target_4"

    async def evaluate(self, ctx: dict[str, Any]) -> GateOutcome:
        trade = ctx["trade"]
        price = ctx["current_price"]
        t4 = trade.get("target_4")
        option_type = trade["option_type"]
        last_hit = trade.get("last_target_hit") or 0

        if t4 is None:
            return GateOutcome(self.name, GateResult.SKIP, "No T4 set")
        if last_hit >= 4:
            return GateOutcome(self.name, GateResult.SKIP, "T4 already hit")

        hit = (
            (option_type == "call" and price >= t4)
            or (option_type == "put" and price <= t4)
        )
        if hit:
            return GateOutcome(self.name, GateResult.FAIL,
                               f"T4 hit @ ${price:.2f} (T4=${t4:.2f})")
        return GateOutcome(self.name, GateResult.PASS,
                           f"Price ${price:.2f} below T4 ${t4:.2f}")


class Target5ExitGate(ExitGate):
    """Exit: Target 5 (full profit) hit."""

    name = "target_5"

    async def evaluate(self, ctx: dict[str, Any]) -> GateOutcome:
        trade = ctx["trade"]
        price = ctx["current_price"]
        t5 = trade.get("target_5")
        option_type = trade["option_type"]

        if t5 is None:
            return GateOutcome(self.name, GateResult.SKIP, "No T5 set")

        hit = (
            (option_type == "call" and price >= t5)
            or (option_type == "put" and price <= t5)
        )
        if hit:
            return GateOutcome(self.name, GateResult.FAIL,
                               f"T5 hit @ ${price:.2f} (T5=${t5:.2f})")
        return GateOutcome(self.name, GateResult.PASS,
                           f"Price ${price:.2f} below T5 ${t5:.2f}")


class TimeExpiryExitGate(ExitGate):
    """Exit 4: Signal's exit-by time reached."""

    name = "time_expiry"

    async def evaluate(self, ctx: dict[str, Any]) -> GateOutcome:
        trade = ctx["trade"]
        now = ctx["now_et"]
        exit_by = trade.get("exit_by")

        if not exit_by:
            return GateOutcome(self.name, GateResult.SKIP, "No exit_by time")

        try:
            exit_h, exit_m = (int(x) for x in exit_by.split(":"))
            exit_time = now.replace(hour=exit_h, minute=exit_m, second=0, microsecond=0)
            if now >= exit_time:
                return GateOutcome(self.name, GateResult.FAIL,
                                   f"Exit-by time {exit_by} ET reached")
            return GateOutcome(self.name, GateResult.PASS,
                               f"Before exit time {exit_by} ET")
        except (ValueError, TypeError):
            return GateOutcome(self.name, GateResult.SKIP, f"Invalid exit_by: {exit_by}")


class EODExitGate(ExitGate):
    """Exit 5: End-of-day cutoff (15:45 ET) for 0DTE positions.

    Skipped for multi-day contracts (DTE > 0) since they don't expire today.
    """

    name = "eod_cutoff"

    async def evaluate(self, ctx: dict[str, Any]) -> GateOutcome:
        trade = ctx["trade"]
        now = ctx["now_et"]

        dte = _get_dte(trade, now)
        if dte > 0:
            return GateOutcome(self.name, GateResult.SKIP,
                               f"Multi-day contract (DTE={dte}), no EOD cutoff")

        cutoff = now.replace(hour=15, minute=45, second=0, microsecond=0)
        if now >= cutoff:
            return GateOutcome(self.name, GateResult.FAIL,
                               "EOD cutoff 15:45 ET reached")
        return GateOutcome(self.name, GateResult.PASS, "Before EOD cutoff")


class ThetaDecayExitGate(ExitGate):
    """Exit 6: Theta decay exit for near-expiry positions."""

    name = "theta_decay"

    async def evaluate(self, ctx: dict[str, Any]) -> GateOutcome:
        settings = ctx["settings"]
        if not getattr(settings, "ENABLE_THETA_DECAY_EXIT", False):
            return GateOutcome(self.name, GateResult.SKIP, "Theta decay exit disabled")

        from options_owl.risk.theta_manager import should_theta_exit
        trade = ctx["trade"]
        exit_premium = ctx["exit_premium"]

        should_exit, reason = should_theta_exit(trade, exit_premium, settings)
        if should_exit:
            return GateOutcome(self.name, GateResult.FAIL, reason)
        return GateOutcome(self.name, GateResult.PASS, "Theta decay within limits")


class NoMomentumExitGate(ExitGate):
    """Exit 7: No-momentum exit.

    If the trade hasn't moved favorably after N minutes, cut it.
    Avoids holding dead positions that bleed theta while waiting for a move
    that isn't coming.

    For multi-day contracts (DTE > 0), doubles the patience threshold since
    theta decay is negligible and the underlying may recover next day.
    """

    name = "no_momentum"

    async def evaluate(self, ctx: dict[str, Any]) -> GateOutcome:
        settings = ctx.get("settings")
        if not settings or not getattr(settings, "ENABLE_NO_MOMENTUM_EXIT", False):
            return GateOutcome(self.name, GateResult.SKIP, "No-momentum exit disabled")

        trade = ctx["trade"]
        exit_premium = ctx.get("exit_premium")
        entry_premium = trade["premium_per_contract"]

        if exit_premium is None or entry_premium <= 0:
            return GateOutcome(self.name, GateResult.SKIP, "Missing premium data")

        # Check elapsed time — multi-day gets 2x patience
        dte = _get_dte(trade, ctx.get("now_et"))
        min_minutes = settings.NO_MOMENTUM_MINUTES
        if dte > 0:
            min_minutes *= 2  # multi-day: double patience (theta negligible)
        opened_at = trade.get("opened_at")
        if not opened_at:
            return GateOutcome(self.name, GateResult.SKIP, "No opened_at timestamp")

        try:
            from datetime import datetime
            opened_dt = datetime.fromisoformat(opened_at)
            now = ctx.get("now_et") or _now_et()
            if now.tzinfo and opened_dt.tzinfo is None:
                opened_dt = opened_dt.replace(tzinfo=now.tzinfo)
            elapsed = (now - opened_dt).total_seconds() / 60
        except (ValueError, TypeError):
            return GateOutcome(self.name, GateResult.SKIP, "Cannot parse opened_at")

        if elapsed < min_minutes:
            return GateOutcome(
                self.name, GateResult.PASS,
                f"Only {elapsed:.0f}m elapsed (check at {min_minutes}m)"
            )

        # After N minutes, check if premium has gained enough
        gain_pct = (exit_premium - entry_premium) / entry_premium * 100
        min_gain = settings.NO_MOMENTUM_MIN_GAIN_PCT

        if gain_pct < min_gain:
            return GateOutcome(
                self.name, GateResult.FAIL,
                f"No momentum: {elapsed:.0f}m elapsed, premium {gain_pct:+.1f}% "
                f"(need +{min_gain:.0f}%)"
            )

        return GateOutcome(
            self.name, GateResult.PASS,
            f"Momentum OK: {gain_pct:+.1f}% after {elapsed:.0f}m"
        )


# ---------------------------------------------------------------------------
# Vinny strategy exit gates
# ---------------------------------------------------------------------------


class ProfitRetraceExitGate(ExitGate):
    """Exit: Profit-based retracement — protect gains in the adaptive trail dormant zone.

    When peak gain is between MIN_GAIN and adaptive trail activation (10-40%),
    exit if the trade gives back RETRACE_PCT of its profit from peak.

    Example: entry $1.00, peak $1.50 (+50% = $0.50 profit), 35% retrace
    → exit at $1.325 (keeps 65% of the move = $0.325 profit).

    This fills the gap where adaptive trail is dormant and no trail protects
    gains. Tested on 28 live signals: +$670 improvement, 0 trades worsened.
    """

    name = "profit_retrace"

    async def evaluate(self, ctx: dict[str, Any]) -> GateOutcome:
        settings = ctx.get("settings")
        if not settings or not getattr(settings, "ENABLE_PROFIT_RETRACE", False):
            return GateOutcome(self.name, GateResult.SKIP, "Profit retrace disabled")

        trade = ctx["trade"]
        exit_premium = ctx.get("exit_premium")
        entry_premium = trade["premium_per_contract"]
        mfe_premium = trade.get("mfe_premium")

        if exit_premium is None or entry_premium <= 0 or mfe_premium is None:
            return GateOutcome(self.name, GateResult.SKIP, "Missing premium data")

        if mfe_premium <= entry_premium:
            return GateOutcome(self.name, GateResult.SKIP, "No profit to protect")

        peak_gain_pct = (mfe_premium - entry_premium) / entry_premium * 100
        min_gain = settings.PROFIT_RETRACE_MIN_GAIN_PCT

        if peak_gain_pct < min_gain:
            return GateOutcome(
                self.name, GateResult.PASS,
                f"Peak +{peak_gain_pct:.1f}% < min +{min_gain:.0f}%"
            )

        # Calculate retrace
        profit_at_peak = mfe_premium - entry_premium
        profit_now = exit_premium - entry_premium
        profit_given_back = profit_at_peak - profit_now
        retrace_pct = (profit_given_back / profit_at_peak) * 100

        threshold = settings.PROFIT_RETRACE_PCT
        if retrace_pct >= threshold:
            retrace_exit = mfe_premium - profit_at_peak * (threshold / 100)
            return GateOutcome(
                self.name, GateResult.FAIL,
                f"Profit retrace: gave back {retrace_pct:.1f}% of profit "
                f"(${profit_given_back:.3f}/${profit_at_peak:.3f}), "
                f"prem ${exit_premium:.3f} < retrace line ${retrace_exit:.3f} "
                f"(peak +{peak_gain_pct:.0f}%)"
            )

        return GateOutcome(
            self.name, GateResult.PASS,
            f"Retrace {retrace_pct:.1f}% < {threshold:.0f}% "
            f"(peak +{peak_gain_pct:.0f}%, prem ${exit_premium:.3f})"
        )


class ProfitFloorExitGate(ExitGate):
    """Exit: Ratcheting profit floor (v3).

    Activates at +15% gain. Floor = entry + (peak_gain * 60%).
    Floor only ratchets UP, never down. Tightens with time urgency near expiry.

    Backtested: +$120 P&L improvement, +4% win rate over v2.1.
    """

    name = "profit_floor"

    async def evaluate(self, ctx: dict[str, Any]) -> GateOutcome:
        settings = ctx.get("settings")
        if not settings or not getattr(settings, "ENABLE_PROFIT_FLOOR", False):
            return GateOutcome(self.name, GateResult.SKIP, "Profit floor disabled")

        trade = ctx["trade"]
        exit_premium = ctx.get("exit_premium")
        entry_premium = trade["premium_per_contract"]
        mfe_premium = trade.get("mfe_premium")

        if exit_premium is None or entry_premium <= 0 or mfe_premium is None:
            return GateOutcome(self.name, GateResult.SKIP, "Missing premium data")

        if mfe_premium <= entry_premium:
            return GateOutcome(self.name, GateResult.SKIP, "No profit to protect")

        peak_gain_pct = (mfe_premium - entry_premium) / entry_premium * 100
        activation = getattr(settings, "PROFIT_FLOOR_ACTIVATION_PCT", 15.0)

        if peak_gain_pct < activation:
            return GateOutcome(
                self.name, GateResult.PASS,
                f"Peak +{peak_gain_pct:.1f}% < activation +{activation:.0f}%"
            )

        # Determine ratchet % — tighten with time urgency near expiry
        base_ratchet = getattr(settings, "PROFIT_FLOOR_RATCHET_PCT", 60.0)
        ratchet_pct = base_ratchet

        # Time urgency: tighten floor as option approaches expiry
        expiry_date = trade.get("expiry_date")
        now_et = ctx.get("now_et")
        if expiry_date and now_et:
            try:
                from datetime import datetime
                expiry_dt = datetime.strptime(expiry_date, "%Y-%m-%d").replace(
                    hour=16, minute=0, second=0, microsecond=0,
                    tzinfo=now_et.tzinfo,
                )
                time_remaining_min = max(0, (expiry_dt - now_et).total_seconds() / 60)

                if time_remaining_min < 15:
                    ratchet_pct = 95.0
                elif time_remaining_min < 30:
                    ratchet_pct = 90.0
                elif time_remaining_min < 60:
                    ratchet_pct = 80.0
                elif time_remaining_min < 120:
                    ratchet_pct = 70.0
            except (ValueError, TypeError):
                pass

        # Calculate floor: entry + (peak_gain * ratchet%)
        peak_gain = mfe_premium - entry_premium
        floor = entry_premium + peak_gain * (ratchet_pct / 100)

        if exit_premium <= floor:
            return GateOutcome(
                self.name, GateResult.FAIL,
                f"Profit floor hit: prem ${exit_premium:.3f} <= floor ${floor:.3f} "
                f"(peak +{peak_gain_pct:.0f}%, ratchet {ratchet_pct:.0f}%)"
            )

        return GateOutcome(
            self.name, GateResult.PASS,
            f"Above floor: prem ${exit_premium:.3f} > ${floor:.3f} "
            f"(peak +{peak_gain_pct:.0f}%, ratchet {ratchet_pct:.0f}%)"
        )


class BounceFadeExitGate(ExitGate):
    """Exit: Bounce-and-fade detection (v3).

    On deep dips (>50% from entry), instead of a hard catastrophic stop,
    wait for a bounce (10%+ recovery from the low) then sell when it fades
    15% from the bounce high. Better than selling at the absolute bottom.

    Tightens thresholds near option expiry (smaller bounce needed, tighter fade).

    Backtested: saved $903 across 16 trades vs hard catastrophic stop.
    """

    name = "bounce_fade"

    async def evaluate(self, ctx: dict[str, Any]) -> GateOutcome:
        settings = ctx.get("settings")
        if not settings or not getattr(settings, "ENABLE_BOUNCE_FADE", False):
            return GateOutcome(self.name, GateResult.SKIP, "Bounce-fade disabled")

        trade = ctx["trade"]
        exit_premium = ctx.get("exit_premium")
        entry_premium = trade["premium_per_contract"]

        if exit_premium is None or entry_premium <= 0:
            return GateOutcome(self.name, GateResult.SKIP, "Missing premium data")

        drop_pct = (entry_premium - exit_premium) / entry_premium * 100 if exit_premium < entry_premium else 0
        watch_threshold = getattr(settings, "BOUNCE_FADE_WATCH_PCT", 50.0)

        if drop_pct < watch_threshold:
            return GateOutcome(
                self.name, GateResult.PASS,
                f"Drop {drop_pct:.1f}% < watch threshold {watch_threshold:.0f}%"
            )

        # We're in deep dip territory — check bounce state
        # Bounce state is tracked in ctx across poll cycles by position_monitor
        bounce = ctx.get("bounce_state", {})
        bounce_low = bounce.get("low", exit_premium)
        bounce_detected = bounce.get("detected", False)
        bounce_high = bounce.get("high", exit_premium)

        # Update bounce low
        if exit_premium < bounce_low:
            bounce_low = exit_premium

        # Determine thresholds based on time to expiry
        min_recovery = getattr(settings, "BOUNCE_FADE_MIN_RECOVERY_PCT", 10.0)
        fade_pct = getattr(settings, "BOUNCE_FADE_PCT", 15.0)

        expiry_date = trade.get("expiry_date")
        now_et = ctx.get("now_et")
        if expiry_date and now_et:
            try:
                from datetime import datetime
                expiry_dt = datetime.strptime(expiry_date, "%Y-%m-%d").replace(
                    hour=16, minute=0, second=0, microsecond=0,
                    tzinfo=now_et.tzinfo,
                )
                time_remaining_min = max(0, (expiry_dt - now_et).total_seconds() / 60)

                if time_remaining_min < 15:
                    min_recovery = 3.0
                    fade_pct = 5.0
                elif time_remaining_min < 30:
                    min_recovery = 5.0
                    fade_pct = 8.0
                elif time_remaining_min < 60:
                    min_recovery = 7.0
                    fade_pct = 10.0
            except (ValueError, TypeError):
                pass

        # Check for bounce
        if not bounce_detected and bounce_low > 0:
            recovery_pct = (exit_premium - bounce_low) / bounce_low * 100
            if recovery_pct >= min_recovery:
                bounce_detected = True
                bounce_high = exit_premium

        if bounce_detected:
            bounce_high = max(bounce_high, exit_premium)

            # Check for fade from bounce high
            if bounce_high > 0:
                fade_from_bounce = (bounce_high - exit_premium) / bounce_high * 100
                if fade_from_bounce >= fade_pct:
                    # Update bounce state before exiting
                    ctx["bounce_state"] = {
                        "low": bounce_low,
                        "detected": True,
                        "high": bounce_high,
                    }
                    return GateOutcome(
                        self.name, GateResult.FAIL,
                        f"Bounce-fade: drop {drop_pct:.0f}% from entry, "
                        f"bounced to ${bounce_high:.3f}, faded {fade_from_bounce:.1f}% "
                        f"(threshold {fade_pct:.0f}%) — exit at ${exit_premium:.3f}"
                    )

        # Update bounce state in ctx for next poll cycle
        ctx["bounce_state"] = {
            "low": bounce_low,
            "detected": bounce_detected,
            "high": bounce_high,
        }

        status = "watching" if not bounce_detected else f"bounce high ${bounce_high:.3f}"
        return GateOutcome(
            self.name, GateResult.PASS,
            f"Bounce-fade {status}: drop {drop_pct:.0f}%, low ${bounce_low:.3f}"
        )


class ThesisCutExitGate(ExitGate):
    """Exit: Continuous thesis cut — trend-confirmed loss cutting (v3).

    Replaces hard stop + grace period + catastrophic stop. When a trade drops
    below -40% from entry, evaluates trend health every poll cycle:
      - Counting new lows in a rolling window (making lower lows = dead trend)
      - Checking for bounces from the low (buying pressure = support)
      - Deceleration detection (decline slowing = finding support)

    If trend is dead (3+ new lows in 8 ticks, no bounce) → cut losses.
    If showing support (deceleration, bounce > 5%) → hold for recovery.
    With < 30 min to expiry and still down 40%+ → cut regardless (theta death).

    Backtested: +$324 P&L improvement over hard stop, 70.1% WR vs 63.6%.
    Correctly held V-shaped recovery trades (NVDA -44% → +2%) while cutting
    truly dead trades (IWM -61%, AVGO -64%).
    """

    name = "thesis_cut"

    async def evaluate(self, ctx: dict[str, Any]) -> GateOutcome:
        settings = ctx.get("settings")
        if not settings or not getattr(settings, "ENABLE_THESIS_CUT", False):
            return GateOutcome(self.name, GateResult.SKIP, "Thesis cut disabled")

        trade = ctx["trade"]
        exit_premium = ctx.get("exit_premium")
        entry_premium = trade["premium_per_contract"]

        if exit_premium is None or entry_premium <= 0:
            return GateOutcome(self.name, GateResult.SKIP, "Missing premium data")

        drop_pct = (entry_premium - exit_premium) / entry_premium * 100 if exit_premium < entry_premium else 0
        threshold = getattr(settings, "THESIS_CUT_THRESHOLD_PCT", 40.0)

        if drop_pct < threshold:
            return GateOutcome(
                self.name, GateResult.PASS,
                f"Drop {drop_pct:.1f}% < thesis check threshold {threshold:.0f}%"
            )

        # We're in the danger zone — analyze trend from premium history
        history = ctx.get("premium_history", [])
        lookback = getattr(settings, "THESIS_CUT_LOOKBACK_TICKS", 8)
        min_ticks = getattr(settings, "THESIS_CUT_MIN_TICKS", 4)

        # Track ticks in danger zone using thesis_cut state
        thesis_state = ctx.get("thesis_cut_state", {"ticks_in_zone": 0})
        thesis_state["ticks_in_zone"] = thesis_state.get("ticks_in_zone", 0) + 1
        ctx["thesis_cut_state"] = thesis_state

        if thesis_state["ticks_in_zone"] < min_ticks:
            return GateOutcome(
                self.name, GateResult.PASS,
                f"Down {drop_pct:.0f}%, waiting ({thesis_state['ticks_in_zone']}/{min_ticks} ticks)"
            )

        if len(history) < lookback:
            return GateOutcome(
                self.name, GateResult.PASS,
                f"Down {drop_pct:.0f}%, not enough history ({len(history)}/{lookback})"
            )

        # Get time remaining for urgency check
        expiry_date = trade.get("expiry_date")
        now_et = ctx.get("now_et")
        time_remaining_min = 300  # default: 5 hours
        if expiry_date and now_et:
            try:
                from datetime import datetime
                expiry_dt = datetime.strptime(expiry_date, "%Y-%m-%d").replace(
                    hour=16, minute=0, second=0, microsecond=0,
                    tzinfo=now_et.tzinfo,
                )
                time_remaining_min = max(0, (expiry_dt - now_et).total_seconds() / 60)
            except (ValueError, TypeError):
                pass

        # Time urgency: with < 30 min left, cut if still deeply negative
        time_urgency_min = getattr(settings, "THESIS_CUT_TIME_URGENCY_MIN", 30.0)
        time_cut_drop = getattr(settings, "THESIS_CUT_TIME_CUT_DROP_PCT", 40.0)
        if time_remaining_min < time_urgency_min and drop_pct >= time_cut_drop:
            return GateOutcome(
                self.name, GateResult.FAIL,
                f"Thesis time cut: down {drop_pct:.0f}% with {time_remaining_min:.0f}min left "
                f"(threshold -{time_cut_drop:.0f}% @ <{time_urgency_min:.0f}min)"
            )

        # Analyze recent premium ticks for trend health
        window = [p for _, p in history[-lookback:]]

        # 1. Count new lows in the window
        new_low_count = 0
        running_low = window[0]
        for wp in window[1:]:
            if wp < running_low:
                new_low_count += 1
                running_low = wp

        # 2. Check for bounce from recent low
        recent_low = min(window)
        bounce_pct = (exit_premium - recent_low) / recent_low * 100 if recent_low > 0 else 0
        bounce_hold = getattr(settings, "THESIS_CUT_BOUNCE_HOLD_PCT", 5.0)

        if bounce_pct >= bounce_hold:
            return GateOutcome(
                self.name, GateResult.PASS,
                f"Down {drop_pct:.0f}% but bouncing {bounce_pct:.1f}% from low "
                f"${recent_low:.3f} (threshold {bounce_hold:.0f}%)"
            )

        # 3. Check deceleration (second half declining slower than first half)
        half = len(window) // 2
        if half >= 2:
            first_change = (window[half - 1] - window[0])
            second_change = (window[-1] - window[half])
            decelerating = second_change > first_change  # less negative
            if decelerating and bounce_pct > 2:
                return GateOutcome(
                    self.name, GateResult.PASS,
                    f"Down {drop_pct:.0f}%, decline decelerating (finding support)"
                )

        # 4. New low count threshold — adjust for time urgency
        new_low_threshold = getattr(settings, "THESIS_CUT_NEW_LOW_EXIT", 3)
        if time_remaining_min < 60:
            new_low_threshold = max(2, new_low_threshold - 1)

        if new_low_count >= new_low_threshold:
            return GateOutcome(
                self.name, GateResult.FAIL,
                f"Thesis dead: down {drop_pct:.0f}%, {new_low_count} new lows "
                f"in last {lookback} ticks, no bounce "
                f"({time_remaining_min:.0f}min left)"
            )

        return GateOutcome(
            self.name, GateResult.PASS,
            f"Down {drop_pct:.0f}%, trend inconclusive ({new_low_count} new lows, "
            f"bounce {bounce_pct:.1f}%)"
        )


class DecelExitGate(ExitGate):
    """Exit: Premium deceleration — exit when short-term momentum collapses.

    Compares short-term premium velocity (last N readings) against long-term
    velocity (last M readings). When short_vel - long_vel drops below a
    threshold, momentum has collapsed and the trade is likely bleeding theta.

    Catches stalling trades 15-20 min before no_momentum gate.
    Backtested: +$92 SPY, +$263 QQQ vs baseline.
    """

    name = "decel_exit"

    async def evaluate(self, ctx: dict[str, Any]) -> GateOutcome:
        settings = ctx.get("settings")
        if not settings or not getattr(settings, "ENABLE_DECEL_EXIT", False):
            return GateOutcome(self.name, GateResult.SKIP, "Decel exit disabled")

        trade = ctx["trade"]
        exit_premium = ctx.get("exit_premium")
        entry_premium = trade["premium_per_contract"]

        if exit_premium is None or entry_premium <= 0:
            return GateOutcome(self.name, GateResult.SKIP, "Missing premium data")

        # Check minimum hold time
        opened_at = trade.get("opened_at")
        if opened_at:
            try:
                from datetime import datetime
                if isinstance(opened_at, str):
                    opened_dt = datetime.fromisoformat(opened_at)
                else:
                    opened_dt = opened_at
                now = ctx.get("now_et") or datetime.now()
                if now.tzinfo and opened_dt.tzinfo is None:
                    opened_dt = opened_dt.replace(tzinfo=now.tzinfo)
                held_seconds = (now - opened_dt).total_seconds()
                min_hold = getattr(settings, "DECEL_MIN_HOLD_SECONDS", 480)
                if held_seconds < min_hold:
                    return GateOutcome(
                        self.name, GateResult.PASS,
                        f"Held {held_seconds:.0f}s < min {min_hold}s"
                    )
            except (ValueError, TypeError):
                pass

        # Check minimum gain was reached at some point
        gain_pct = (exit_premium - entry_premium) / entry_premium * 100
        min_gain = getattr(settings, "DECEL_MIN_GAIN_PCT", 5.0)
        mfe_premium = trade.get("mfe_premium")
        peak_gain_pct = 0.0
        if mfe_premium and mfe_premium > entry_premium:
            peak_gain_pct = (mfe_premium - entry_premium) / entry_premium * 100
        if peak_gain_pct < min_gain:
            return GateOutcome(
                self.name, GateResult.PASS,
                f"Peak gain +{peak_gain_pct:.1f}% < min +{min_gain:.0f}%"
            )

        # Get premium history — list of (timestamp, premium) tuples
        history = ctx.get("premium_history", [])
        short_window = getattr(settings, "DECEL_SHORT_WINDOW", 5)
        long_window = getattr(settings, "DECEL_LONG_WINDOW", 15)

        if len(history) < long_window:
            return GateOutcome(
                self.name, GateResult.PASS,
                f"Not enough history ({len(history)}/{long_window})"
            )

        # Compute short-term and long-term velocity (% change over window)
        def _velocity(hist, window):
            if len(hist) < window or window < 2:
                return 0.0
            start_prem = hist[-window][1]
            end_prem = hist[-1][1]
            if start_prem <= 0:
                return 0.0
            return (end_prem - start_prem) / start_prem * 100

        v_short = _velocity(history, short_window)
        v_long = _velocity(history, long_window)
        accel = v_short - v_long

        threshold = getattr(settings, "DECEL_THRESHOLD", -3.0)
        if accel < threshold:
            return GateOutcome(
                self.name, GateResult.FAIL,
                f"Decel exit: short_vel={v_short:+.1f}% long_vel={v_long:+.1f}% "
                f"accel={accel:+.1f} < threshold {threshold} "
                f"(prem ${exit_premium:.3f}, peak +{peak_gain_pct:.0f}%)"
            )

        return GateOutcome(
            self.name, GateResult.PASS,
            f"Momentum OK: accel={accel:+.1f} >= {threshold} "
            f"(short={v_short:+.1f}% long={v_long:+.1f}%)"
        )


class AdaptiveTrailingStopExitGate(ExitGate):
    """Exit: 3-stage adaptive trailing stop (v2.1).

    Replaces phase trail when enabled. Three stages based on peak gain:
      DORMANT  (below +40%): no trail — let the trade develop
      ACTIVE   (+40% to +150%): 35% trail width — standard protection
      RUNNER   (+150% to +400%): 45% trail width — wider, let winners run
      MOONSHOT (+400%+): 30% trail width — tighter, lock in huge gains

    Backtested: 2x P&L improvement ($17K vs $8.3K) at same 80% win rate.
    """

    name = "adaptive_trailing_stop"

    async def evaluate(self, ctx: dict[str, Any]) -> GateOutcome:
        settings = ctx.get("settings")
        if not settings or not getattr(settings, "ENABLE_ADAPTIVE_TRAIL", False):
            return GateOutcome(self.name, GateResult.SKIP, "Adaptive trail disabled")

        trade = ctx["trade"]
        exit_premium = ctx.get("exit_premium")
        entry_premium = trade["premium_per_contract"]
        mfe_premium = trade.get("mfe_premium")

        if exit_premium is None or entry_premium <= 0:
            return GateOutcome(self.name, GateResult.SKIP, "Missing premium data")

        peak = mfe_premium if mfe_premium is not None else entry_premium

        from options_owl.risk.vinny_strategy import evaluate_adaptive_trail

        # Apply volume-peak tighten modifier (v2.1 §6)
        vol_tighten = ctx.get("volume_peak_tighten", False)
        tighten_factor = 1.0
        if vol_tighten:
            tighten_factor = getattr(settings, "VOLUME_PEAK_TIGHTEN_FACTOR", 0.7)

        result = evaluate_adaptive_trail(
            entry_premium=entry_premium,
            current_premium=exit_premium,
            peak_premium=peak,
            activation_pct=settings.ADAPTIVE_TRAIL_ACTIVATION_PCT,
            active_width=settings.ADAPTIVE_TRAIL_ACTIVE_WIDTH * tighten_factor,
            runner_threshold=settings.ADAPTIVE_TRAIL_RUNNER_THRESHOLD,
            runner_width=settings.ADAPTIVE_TRAIL_RUNNER_WIDTH * tighten_factor,
            moonshot_threshold=settings.ADAPTIVE_TRAIL_MOONSHOT_THRESHOLD,
            moonshot_width=settings.ADAPTIVE_TRAIL_MOONSHOT_WIDTH * tighten_factor,
        )

        if result.should_exit:
            tighten_note = " [vol-peak tightened]" if vol_tighten else ""
            return GateOutcome(self.name, GateResult.FAIL, result.reason + tighten_note)
        return GateOutcome(self.name, GateResult.PASS, result.reason)


class UnderlyingTrailExitGate(ExitGate):
    """Exit: Underlying-anchored trail (v2.1 §5).

    Trails on the underlying stock price instead of option premium.
    Tighter than premium trail since underlying moves less; catches reversals faster.
    Only active after adaptive trail activation threshold is reached.
    """

    name = "underlying_trail"

    async def evaluate(self, ctx: dict[str, Any]) -> GateOutcome:
        settings = ctx.get("settings")
        if not settings or not getattr(settings, "ENABLE_UNDERLYING_TRAIL", False):
            return GateOutcome(self.name, GateResult.SKIP, "Underlying trail disabled")

        trade = ctx["trade"]
        exit_premium = ctx.get("exit_premium")
        current_price = ctx.get("current_price")
        entry_premium = trade["premium_per_contract"]
        mfe_premium = trade.get("mfe_premium") or entry_premium
        peak_underlying = trade.get("peak_underlying_price")
        direction = trade.get("option_type", "call")

        if exit_premium is None or entry_premium <= 0:
            return GateOutcome(self.name, GateResult.SKIP, "Missing premium data")
        if not current_price or current_price <= 0:
            return GateOutcome(self.name, GateResult.SKIP, "No underlying price")
        if not peak_underlying or peak_underlying <= 0:
            return GateOutcome(self.name, GateResult.SKIP, "No peak underlying")

        from options_owl.risk.vinny_strategy import (
            evaluate_underlying_trail,
            parse_underlying_trail_tiers,
        )

        tiers_str = getattr(settings, "UNDERLYING_TRAIL_TIERS", "100:0.50,50:0.40,15:0.30,0:0.20")
        tiers = parse_underlying_trail_tiers(tiers_str)
        activation = getattr(settings, "ADAPTIVE_TRAIL_ACTIVATION_PCT", 35.0)

        should_exit, reason = evaluate_underlying_trail(
            entry_premium=entry_premium,
            current_premium=exit_premium,
            peak_premium=mfe_premium,
            current_underlying=current_price,
            peak_underlying=peak_underlying,
            direction=direction,
            tiers=tiers,
            activation_pct=activation,
        )

        if should_exit:
            return GateOutcome(self.name, GateResult.FAIL, reason)
        return GateOutcome(self.name, GateResult.PASS, reason)


class VolumePeakExitGate(ExitGate):
    """Exit: Volume-peak modifier (v2.1 §6).

    Detects exhaustion via multi-timeframe candle data (RSI, OBV, candle
    patterns, volume trend). When real candle data is available, uses the
    full exhaustion check from candle_cache. Falls back to underlying price
    momentum divergence when candle data is unavailable.

    When triggered, tightens the adaptive trail width by VOLUME_PEAK_TIGHTEN_FACTOR.
    Does not directly trigger exit — sets a flag in ctx for adaptive trail to use.
    Always PASSes; modifies ctx['volume_peak_tighten'] for the adaptive trail gate.
    """

    name = "volume_peak"

    async def evaluate(self, ctx: dict[str, Any]) -> GateOutcome:
        settings = ctx.get("settings")
        if not settings or not getattr(settings, "ENABLE_VOLUME_PEAK", False):
            return GateOutcome(self.name, GateResult.SKIP, "Volume peak disabled")

        trade = ctx["trade"]
        exit_premium = ctx.get("exit_premium")
        entry_premium = trade["premium_per_contract"]
        mfe_premium = trade.get("mfe_premium") or entry_premium
        direction = trade.get("option_type", "call")

        if exit_premium is None or entry_premium <= 0:
            return GateOutcome(self.name, GateResult.SKIP, "Missing data")

        peak_gain_pct = (mfe_premium - entry_premium) / entry_premium * 100
        min_gain = getattr(settings, "VOLUME_PEAK_MIN_GAIN_PCT", 35.0)

        if peak_gain_pct < min_gain:
            return GateOutcome(self.name, GateResult.PASS,
                               f"Peak gain +{peak_gain_pct:.1f}% < min {min_gain:.0f}%")

        # Try real candle data first (multi-TF with RSI, OBV, patterns)
        candle_data = ctx.get("candle_data", {})
        if candle_data and candle_data.get("indicators"):
            from options_owl.collectors.candle_cache import check_exhaustion
            is_exhausted, reason = check_exhaustion(
                candle_data, direction, peak_gain_pct, min_gain,
            )
            if is_exhausted:
                ctx["volume_peak_tighten"] = True
                return GateOutcome(self.name, GateResult.PASS,
                                   f"Trail tightened — {reason}")
            return GateOutcome(self.name, GateResult.PASS, reason)

        # Fallback: underlying price momentum divergence (no candle data)
        underlying_prices = ctx.get("underlying_price_history", [])

        from options_owl.risk.vinny_strategy import check_volume_peak
        result = check_volume_peak(underlying_prices, direction)

        if result == "tighten":
            ctx["volume_peak_tighten"] = True
            return GateOutcome(self.name, GateResult.PASS,
                               "Volume peak detected (price momentum) — trail tightened")

        return GateOutcome(self.name, GateResult.PASS, "No divergence detected")


class TrancheScaleOutExitGate(ExitGate):
    """Exit: Tranche scale-out (v2.1 §4).

    When premium gain reaches TRANCHE_LOCK_GAIN_PCT (+25%) and trade has
    enough contracts, close 1/3 to lock profit. Uses description prefix
    '[TRANCHE_SCALEOUT]' so position_monitor knows to do a partial close.
    """

    name = "tranche_scaleout"

    async def evaluate(self, ctx: dict[str, Any]) -> GateOutcome:
        settings = ctx.get("settings")
        if not settings or not getattr(settings, "ENABLE_TRANCHE_SCALEOUT", False):
            return GateOutcome(self.name, GateResult.SKIP, "Tranche scale-out disabled")

        trade = ctx["trade"]
        exit_premium = ctx.get("exit_premium")
        entry_premium = trade["premium_per_contract"]
        contracts = trade.get("contracts", 1)
        min_contracts = getattr(settings, "TRANCHE_MIN_CONTRACTS", 3)
        lock_pct = getattr(settings, "TRANCHE_LOCK_GAIN_PCT", 25.0)

        if exit_premium is None or entry_premium <= 0:
            return GateOutcome(self.name, GateResult.SKIP, "Missing data")

        if contracts < min_contracts:
            return GateOutcome(self.name, GateResult.PASS,
                               f"Only {contracts} contracts (need {min_contracts})")

        # Check if already did a tranche scale-out (scale_out_count tracks partials)
        scale_outs = trade.get("scale_out_count", 0) or 0
        if scale_outs > 0:
            return GateOutcome(self.name, GateResult.PASS,
                               "Already did tranche scale-out")

        gain_pct = (exit_premium - entry_premium) / entry_premium * 100
        if gain_pct >= lock_pct:
            close_qty = max(1, contracts // 3)
            return GateOutcome(
                self.name, GateResult.FAIL,
                f"[TRANCHE_SCALEOUT:{close_qty}] Lock {close_qty}/{contracts} "
                f"at +{gain_pct:.1f}% (threshold +{lock_pct:.0f}%)",
            )

        return GateOutcome(self.name, GateResult.PASS,
                           f"Gain +{gain_pct:.1f}% < lock +{lock_pct:.0f}%")


class ThetaBleedExitGate(ExitGate):
    """Exit: Theta bleed — held too long and losing too much.

    Skipped for multi-day contracts (DTE > 0) — theta is 43-69x slower,
    so time-based loss exits don't apply intraday.
    """

    name = "theta_bleed"

    async def evaluate(self, ctx: dict[str, Any]) -> GateOutcome:
        settings = ctx.get("settings")
        if not settings or not getattr(settings, "ENABLE_VINNY_STRATEGY", False):
            return GateOutcome(self.name, GateResult.SKIP, "Vinny strategy disabled")

        trade = ctx["trade"]
        dte = _get_dte(trade, ctx.get("now_et"))
        if dte > 0:
            return GateOutcome(self.name, GateResult.SKIP,
                               f"Multi-day (DTE={dte}), theta bleed negligible")

        trade = ctx["trade"]
        exit_premium = ctx.get("exit_premium")
        entry_premium = trade["premium_per_contract"]
        opened_at = trade.get("opened_at")

        if exit_premium is None or entry_premium <= 0 or not opened_at:
            return GateOutcome(self.name, GateResult.SKIP, "Missing data")

        from options_owl.risk.vinny_strategy import check_theta_bleed
        should_exit, reason = check_theta_bleed(
            entry_premium=entry_premium,
            current_premium=exit_premium,
            opened_at=opened_at,
            now=ctx.get("now_et"),
            max_hold_minutes=settings.THETA_BLEED_HOLD_MINUTES,
            max_loss_pct=settings.THETA_BLEED_MAX_LOSS_PCT,
        )

        if should_exit:
            return GateOutcome(self.name, GateResult.FAIL, reason)
        return GateOutcome(self.name, GateResult.PASS, reason)


class MLSellExitGate(ExitGate):
    """Exit: ML-powered sell timing.

    Uses pre-trained LightGBM models (per-ticker for SPY/QQQ/IWM,
    generic fallback for others) to predict optimal exit timing.

    Runs a classifier (should I sell?) + regressor (what's the expected
    future PnL?) and exits when both agree there's no upside left.

    Trained on 2 years of 1-minute options data across 13 tickers.
    """

    name = "ml_sell"

    async def evaluate(self, ctx: dict[str, Any]) -> GateOutcome:
        settings = ctx.get("settings")
        if not settings or not getattr(settings, "ENABLE_ML_EXIT", False):
            return GateOutcome(self.name, GateResult.SKIP, "ML exit disabled")

        trade = ctx["trade"]
        exit_premium = ctx.get("exit_premium")
        current_price = ctx.get("current_price")
        now_et = ctx.get("now_et")

        if exit_premium is None or now_et is None:
            return GateOutcome(self.name, GateResult.SKIP, "Missing premium/time data")

        entry_premium = trade["premium_per_contract"]
        if entry_premium <= 0:
            return GateOutcome(self.name, GateResult.SKIP, "No entry premium")

        mfe_premium = trade.get("mfe_premium") or entry_premium
        opened_at = trade.get("opened_at")
        if not opened_at:
            return GateOutcome(self.name, GateResult.SKIP, "No opened_at")

        # Calculate minutes since entry
        try:
            if isinstance(opened_at, str):
                from datetime import datetime as dt
                opened_dt = dt.fromisoformat(opened_at)
            else:
                opened_dt = opened_at
            minutes_held = (now_et - opened_dt).total_seconds() / 60
        except (ValueError, TypeError):
            return GateOutcome(self.name, GateResult.SKIP, "Cannot parse opened_at")

        ticker = trade["ticker"]
        is_call = trade.get("option_type", "").upper() in ("CALL", "C")

        # Get premium history from context (if position_monitor provides it)
        premium_history = ctx.get("premium_history")

        from options_owl.risk.ml_exit import predict_sell

        signal = predict_sell(
            ticker=ticker,
            entry_premium=entry_premium,
            current_premium=exit_premium,
            peak_premium=mfe_premium,
            minutes_since_entry=minutes_held,
            now_hour=now_et.hour,
            now_minute=now_et.minute,
            is_call=is_call,
            premium_history=premium_history,
            underlying_entry=trade.get("entry_price"),
            underlying_current=current_price,
        )

        if signal.should_sell:
            pnl_pct = (exit_premium - entry_premium) / entry_premium * 100
            return GateOutcome(
                self.name, GateResult.FAIL,
                f"{signal.reason} | pnl={pnl_pct:+.1f}% held={minutes_held:.0f}m "
                f"model={signal.model_used}",
            )

        return GateOutcome(
            self.name, GateResult.PASS,
            f"ML hold: P(sell)={signal.sell_probability:.2f} "
            f"E[future]={signal.expected_future_pnl:+.1f}% "
            f"model={signal.model_used}",
        )


class TimeDecayZoneExitGate(ExitGate):
    """Exit: Time decay zone — no new premium high in N minutes.

    Activates after 45 min hold OR after 3 PM ET.
    If no new premium high in 5 minutes, exit to avoid theta decay.
    """

    name = "time_decay_zone"

    async def evaluate(self, ctx: dict[str, Any]) -> GateOutcome:
        settings = ctx.get("settings")
        if not settings or not getattr(settings, "ENABLE_VINNY_STRATEGY", False):
            return GateOutcome(self.name, GateResult.SKIP, "Vinny strategy disabled")
        if not getattr(settings, "ENABLE_TIME_DECAY_ZONE", True):
            return GateOutcome(self.name, GateResult.SKIP, "Time decay zone disabled")

        trade = ctx["trade"]
        exit_premium = ctx.get("exit_premium")
        now_et = ctx.get("now_et")
        opened_at = trade.get("opened_at")

        if exit_premium is None or not opened_at or not now_et:
            return GateOutcome(self.name, GateResult.SKIP, "Missing data")

        from options_owl.risk.vinny_strategy import is_time_decay_zone
        in_decay = is_time_decay_zone(
            opened_at, now_et,
            max_hold_minutes=settings.TIME_DECAY_HOLD_MINUTES,
            afternoon_hour=settings.TIME_DECAY_AFTERNOON_HOUR,
            afternoon_minute=settings.TIME_DECAY_AFTERNOON_MINUTE,
        )

        if not in_decay:
            return GateOutcome(self.name, GateResult.PASS,
                               "Not in time decay zone")

        # In decay zone — check if premium is making new highs
        mfe_premium = trade.get("mfe_premium")
        last_new_high_at = trade.get("last_new_high_at")

        if mfe_premium is None:
            return GateOutcome(self.name, GateResult.SKIP, "No MFE data")

        # If current premium IS the peak (or very close), it's still making highs
        if exit_premium >= mfe_premium * 0.99:
            return GateOutcome(self.name, GateResult.PASS,
                               "Time decay zone but still making highs")

        from options_owl.risk.vinny_strategy import check_time_decay_no_new_high
        should_exit, reason = check_time_decay_no_new_high(
            current_premium=exit_premium,
            peak_premium=mfe_premium,
            last_new_high_at=last_new_high_at,
            now=now_et,
            stale_minutes=settings.TIME_DECAY_STALE_MINUTES,
        )

        if should_exit:
            return GateOutcome(self.name, GateResult.FAIL, reason)
        return GateOutcome(self.name, GateResult.PASS, reason)


class DollarTrailExitGate(ExitGate):
    """Exit: Dollar-based stair-step trailing stop.

    Activates at 10% profit, then ratchets the stop up in dollar increments
    per contract.  Below $50 profit: $20 steps.  Above $50: $10 steps.
    Replaces the old velocity exit.
    """

    name = "dollar_trail"

    async def evaluate(self, ctx: dict[str, Any]) -> GateOutcome:
        settings = ctx.get("settings")
        if not settings or not getattr(settings, "ENABLE_DOLLAR_TRAIL", False):
            return GateOutcome(self.name, GateResult.SKIP, "Dollar trail disabled")

        trade = ctx["trade"]
        exit_premium = ctx.get("exit_premium")
        if exit_premium is None:
            return GateOutcome(self.name, GateResult.SKIP, "No exit premium")

        entry_premium = trade.get("premium_per_contract", 0)
        mfe_premium = trade.get("mfe_premium")
        if not entry_premium or entry_premium <= 0:
            return GateOutcome(self.name, GateResult.SKIP, "No entry premium")
        if mfe_premium is None or mfe_premium <= 0:
            return GateOutcome(self.name, GateResult.SKIP, "No MFE data")

        from options_owl.risk.vinny_strategy import evaluate_dollar_trail

        result = evaluate_dollar_trail(
            entry_premium=entry_premium,
            current_premium=exit_premium,
            peak_premium=mfe_premium,
            activation_pct=getattr(settings, "DOLLAR_TRAIL_ACTIVATION_PCT", 10.0),
            small_step_pct=getattr(settings, "DOLLAR_TRAIL_SMALL_STEP_PCT", 10.0),
            step_threshold_pct=getattr(settings, "DOLLAR_TRAIL_STEP_THRESHOLD_PCT", 25.0),
            large_step_pct=getattr(settings, "DOLLAR_TRAIL_LARGE_STEP_PCT", 5.0),
        )

        if result.should_exit:
            return GateOutcome(self.name, GateResult.FAIL, result.reason)

        return GateOutcome(self.name, GateResult.PASS, result.reason)


class ProfitLockExitGate(ExitGate):
    """Exit: Profit lock ratchet — lock in minimum profit after reaching gain thresholds.

    Tiers: e.g. after +80% gain, lock in +30% minimum; after +150%, lock in +70%.
    Prevents the AMZN scenario: MFE +154% → exit at -34%.
    """

    name = "profit_lock"

    async def evaluate(self, ctx: dict[str, Any]) -> GateOutcome:
        settings = ctx.get("settings")
        if not settings or not getattr(settings, "ENABLE_PROFIT_LOCK", False):
            return GateOutcome(self.name, GateResult.SKIP, "Profit lock disabled")

        trade = ctx["trade"]
        exit_premium = ctx.get("exit_premium")

        if exit_premium is None:
            return GateOutcome(self.name, GateResult.SKIP, "No exit premium")

        entry_premium = trade.get("premium_per_contract", 0)
        mfe_premium = trade.get("mfe_premium")

        if not entry_premium or entry_premium <= 0 or not mfe_premium:
            return GateOutcome(self.name, GateResult.SKIP, "Missing entry/MFE data")

        # Parse tier config: "80:30,150:70,250:150"
        tiers_str = getattr(settings, "PROFIT_LOCK_TIERS", "80:30,150:70,250:150")
        tiers = []
        for pair in tiers_str.split(","):
            pair = pair.strip()
            if ":" in pair:
                threshold, lock = pair.split(":", 1)
                try:
                    tiers.append((float(threshold), float(lock)))
                except ValueError:
                    continue

        if not tiers:
            return GateOutcome(self.name, GateResult.SKIP, "No valid tiers")

        # Check peak gain (MFE) — did we ever reach a tier threshold?
        peak_gain_pct = (mfe_premium - entry_premium) / entry_premium * 100
        current_gain_pct = (exit_premium - entry_premium) / entry_premium * 100

        # Find the highest tier the peak gain reached
        applicable_lock = None
        applicable_threshold = None
        for threshold, lock in sorted(tiers, reverse=True):
            if peak_gain_pct >= threshold:
                applicable_lock = lock
                applicable_threshold = threshold
                break

        if applicable_lock is None:
            return GateOutcome(
                self.name, GateResult.PASS,
                f"Peak gain {peak_gain_pct:.0f}% hasn't reached any tier"
            )

        # Check if current gain has dropped below the lock floor
        if current_gain_pct <= applicable_lock:
            return GateOutcome(
                self.name, GateResult.FAIL,
                f"Profit lock: peaked at +{peak_gain_pct:.0f}% "
                f"(tier {applicable_threshold:.0f}%), now at +{current_gain_pct:.0f}% "
                f"≤ lock floor +{applicable_lock:.0f}%"
            )

        return GateOutcome(
            self.name, GateResult.PASS,
            f"Current +{current_gain_pct:.0f}% above lock floor +{applicable_lock:.0f}% "
            f"(peak +{peak_gain_pct:.0f}%)"
        )


class AdaptiveTimeTightenExitGate(ExitGate):
    """Exit: Adaptive trail tightening based on hold time.

    After TIME_TIGHTEN_AFTER_MINUTES, tightens the effective trail by
    TIME_TIGHTEN_FACTOR (e.g., 0.7 means trail becomes 70% of normal width).
    Implements as an override on the phase trail's drop threshold.
    """

    name = "adaptive_time_tighten"

    async def evaluate(self, ctx: dict[str, Any]) -> GateOutcome:
        settings = ctx.get("settings")
        if not settings or not getattr(settings, "ENABLE_TIME_TIGHTEN", False):
            return GateOutcome(self.name, GateResult.SKIP, "Time tighten disabled")

        if not getattr(settings, "ENABLE_VINNY_STRATEGY", False):
            return GateOutcome(self.name, GateResult.SKIP, "Vinny strategy disabled")

        trade = ctx["trade"]
        exit_premium = ctx.get("exit_premium")
        now_et = ctx.get("now_et")
        opened_at = trade.get("opened_at")

        if exit_premium is None or not opened_at or not now_et:
            return GateOutcome(self.name, GateResult.SKIP, "Missing data")

        # Check if we've passed the tightening threshold
        try:
            from datetime import datetime
            opened_dt = datetime.fromisoformat(opened_at)
            elapsed_min = (now_et - opened_dt).total_seconds() / 60
        except (ValueError, TypeError):
            return GateOutcome(self.name, GateResult.SKIP, "Cannot parse opened_at")

        tighten_after = settings.TIME_TIGHTEN_AFTER_MINUTES
        if elapsed_min < tighten_after:
            return GateOutcome(
                self.name, GateResult.PASS,
                f"{elapsed_min:.0f}m < {tighten_after:.0f}m threshold"
            )

        # Apply tightened trail
        mfe_premium = trade.get("mfe_premium")
        trade.get("premium_per_contract", 0)
        if not mfe_premium or mfe_premium <= 0:
            return GateOutcome(self.name, GateResult.SKIP, "No MFE data")

        from options_owl.risk.vinny_strategy import PHASE_TRAILS, get_current_phase

        last_target = trade.get("last_target_hit")
        phase = get_current_phase(last_target)
        base_trail = PHASE_TRAILS.get(phase, 25.0)

        # Apply tightening factor
        factor = settings.TIME_TIGHTEN_FACTOR
        tightened_trail = base_trail * factor

        drop_from_peak = (mfe_premium - exit_premium) / mfe_premium * 100 if mfe_premium > 0 else 0

        if drop_from_peak >= tightened_trail:
            return GateOutcome(
                self.name, GateResult.FAIL,
                f"Adaptive tighten: {elapsed_min:.0f}m held, trail {base_trail:.0f}%→"
                f"{tightened_trail:.0f}% (×{factor}), drop {drop_from_peak:.1f}% "
                f"from peak ${mfe_premium:.2f}"
            )

        return GateOutcome(
            self.name, GateResult.PASS,
            f"Time-tightened trail: drop {drop_from_peak:.1f}% < "
            f"{tightened_trail:.0f}% (×{factor} after {elapsed_min:.0f}m)"
        )


# ---------------------------------------------------------------------------
# v5 dynamic exit gates (signal-driven, no fixed time windows)
# ---------------------------------------------------------------------------


class ScalpTrailExitGate(ExitGate):
    """Exit: Dynamic scalp — take profit when premium peaked, faded, AND underlying
    is NOT confirming the trade direction.

    No time window. Fires at 5min or 50min — whenever premium spikes and fades
    without underlying support. If underlying IS confirming (moved >0.2% in
    trade direction), holds even if premium dips (it's real movement, not IV noise).

    Decision logic:
      premium peaked >20% AND faded to <60% of peak AND gain > 0:
        - underlying confirms (>0.2% in direction) → HOLD (real move, let it run)
        - underlying NOT confirming → SCALP (IV-driven spike, take profit)
    """

    name = "scalp_trail"

    async def evaluate(self, ctx: dict[str, Any]) -> GateOutcome:
        settings = ctx.get("settings")
        if not settings or not getattr(settings, "ENABLE_SCALP_TRAIL", False):
            return GateOutcome(self.name, GateResult.SKIP, "Scalp trail disabled")

        trade = ctx["trade"]
        exit_premium = ctx.get("exit_premium")
        entry_premium = trade["premium_per_contract"]
        mfe_premium = trade.get("mfe_premium")

        if exit_premium is None or entry_premium <= 0 or mfe_premium is None:
            return GateOutcome(self.name, GateResult.SKIP, "Missing premium data")

        peak_pct = getattr(settings, "SCALP_TRAIL_PEAK_PCT", 20.0)
        fade_pct = getattr(settings, "SCALP_TRAIL_FADE_PCT", 60.0)
        confirm_pct = getattr(settings, "SCALP_TRAIL_UNDERLYING_CONFIRM_PCT", 0.2)

        peak_gain = (mfe_premium - entry_premium) / entry_premium * 100
        current_gain = (exit_premium - entry_premium) / entry_premium * 100

        if peak_gain < peak_pct:
            return GateOutcome(self.name, GateResult.PASS,
                               f"Peak +{peak_gain:.1f}% < scalp threshold +{peak_pct:.0f}%")

        # Peak was high enough — check if it's faded
        if current_gain > 0 and current_gain < peak_gain * (fade_pct / 100):
            # Premium has faded — check underlying confirmation
            current_price = ctx.get("current_price")
            entry_price = trade.get("entry_price")
            option_type = trade.get("option_type", "call").lower()

            if current_price and entry_price and entry_price > 0:
                u_move = (current_price - entry_price) / entry_price * 100
                # Check if underlying confirms trade direction
                if option_type in ("call", "bullish", "long"):
                    confirms = u_move > confirm_pct
                    against = u_move < -0.5  # meaningful move against
                else:
                    confirms = u_move < -confirm_pct
                    against = u_move > 0.5

                if confirms:
                    return GateOutcome(
                        self.name, GateResult.PASS,
                        f"Scalp held: peaked +{peak_gain:.0f}%, faded to +{current_gain:.1f}% "
                        f"but underlying {u_move:+.2f}% confirms — HOLD"
                    )

                # Multi-day: only scalp if underlying actively AGAINST (more patient)
                dte = _get_dte(trade, ctx.get("now_et"))
                if dte > 0 and not against:
                    return GateOutcome(
                        self.name, GateResult.PASS,
                        f"Scalp held (multi-day DTE={dte}): peaked +{peak_gain:.0f}%, "
                        f"faded to +{current_gain:.1f}% but underlying {u_move:+.2f}% "
                        f"not against — holding"
                    )

            # Underlying NOT confirming (or no data) → scalp
            return GateOutcome(
                self.name, GateResult.FAIL,
                f"Scalp: peaked +{peak_gain:.0f}%, faded to +{current_gain:.1f}% "
                f"(< {fade_pct:.0f}% of peak), underlying not confirming — taking profit"
            )

        return GateOutcome(
            self.name, GateResult.PASS,
            f"Scalp: +{current_gain:.1f}% of +{peak_gain:.0f}% peak "
            f"(threshold {fade_pct:.0f}%)"
        )


class CheckpointExitGate(ExitGate):
    """Exit: Dynamic checkpoint — cut when BOTH premium AND underlying agree
    the trade is dead. No fixed time window.

    Fires when ALL conditions met:
      1. Premium down >15% from entry
      2. Underlying moved >0.3% against trade direction
      3. At least 5min elapsed (avoid open noise)

    This replaces the fixed 30-min checkpoint. Cuts losers at 5-10min when
    BOTH signals agree, but holds when only premium is down (IV crush) or
    only underlying is against (noise).
    """

    name = "checkpoint"

    async def evaluate(self, ctx: dict[str, Any]) -> GateOutcome:
        settings = ctx.get("settings")
        if not settings or not getattr(settings, "ENABLE_CHECKPOINT", False):
            return GateOutcome(self.name, GateResult.SKIP, "Checkpoint disabled")

        trade = ctx["trade"]

        # Skip checkpoint for multi-day contracts — temporary dips recover
        dte = _get_dte(trade, ctx.get("now_et"))
        if dte > 0:
            return GateOutcome(self.name, GateResult.SKIP,
                               f"Checkpoint skipped for multi-day (DTE={dte})")

        exit_premium = ctx.get("exit_premium")
        entry_premium = trade["premium_per_contract"]

        if exit_premium is None or entry_premium <= 0:
            return GateOutcome(self.name, GateResult.SKIP, "Missing premium data")

        # Minimum elapsed time to avoid noise
        opened_at = trade.get("opened_at")
        if not opened_at:
            return GateOutcome(self.name, GateResult.SKIP, "No opened_at")

        try:
            from datetime import datetime
            opened_dt = datetime.fromisoformat(opened_at)
            now = ctx.get("now_et") or datetime.now()
            if now.tzinfo and opened_dt.tzinfo is None:
                opened_dt = opened_dt.replace(tzinfo=now.tzinfo)
            elapsed = (now - opened_dt).total_seconds() / 60
        except (ValueError, TypeError):
            return GateOutcome(self.name, GateResult.SKIP, "Cannot parse opened_at")

        min_elapsed = getattr(settings, "CHECKPOINT_MIN_ELAPSED_MINUTES", 5.0)
        if elapsed < min_elapsed:
            return GateOutcome(self.name, GateResult.PASS,
                               f"Too early for checkpoint ({elapsed:.0f}m < {min_elapsed:.0f}m)")

        # Check premium drop
        drop_pct = (entry_premium - exit_premium) / entry_premium * 100 if exit_premium < entry_premium else 0
        premium_threshold = getattr(settings, "CHECKPOINT_PREMIUM_DROP_PCT", 15.0)

        if drop_pct < premium_threshold:
            return GateOutcome(
                self.name, GateResult.PASS,
                f"Premium drop {drop_pct:.1f}% < threshold {premium_threshold:.0f}%"
            )

        # Premium is down enough — check underlying confirmation
        current_price = ctx.get("current_price")
        entry_price = trade.get("entry_price")
        option_type = trade.get("option_type", "call").lower()
        u_threshold = getattr(settings, "CHECKPOINT_UNDERLYING_AGAINST_PCT", 0.3)

        if current_price and entry_price and entry_price > 0:
            u_move = (current_price - entry_price) / entry_price * 100
            if option_type in ("call", "bullish", "long"):
                against = u_move < -u_threshold
            else:
                against = u_move > u_threshold

            if not against:
                return GateOutcome(
                    self.name, GateResult.PASS,
                    f"Premium down {drop_pct:.1f}% but underlying {u_move:+.2f}% "
                    f"not against (need {u_threshold:.1f}%) — holding"
                )

            return GateOutcome(
                self.name, GateResult.FAIL,
                f"Checkpoint cut: premium -{drop_pct:.1f}% AND underlying {u_move:+.2f}% "
                f"both against at {elapsed:.0f}m — trade is dead"
            )

        # No underlying data — fall back to premium-only with higher threshold
        if drop_pct >= premium_threshold + 10:
            return GateOutcome(
                self.name, GateResult.FAIL,
                f"Checkpoint cut: premium -{drop_pct:.1f}% (no underlying data, "
                f"using higher threshold -{premium_threshold + 10:.0f}%) at {elapsed:.0f}m"
            )

        return GateOutcome(
            self.name, GateResult.PASS,
            f"Premium down {drop_pct:.1f}% but no underlying data — holding"
        )


class GraduatedStopExitGate(ExitGate):
    """Exit: Dynamic stop loss driven by underlying confirmation.

    Two stop levels based on underlying movement (not time):
      - WIDE (40%): underlying hasn't confirmed against trade → premium drop is
        likely IV crush / noise, not a real reversal. Be patient.
      - TIGHT (25%): underlying moved 0.4%+ against trade → real reversal,
        cut losses quickly.

    Absolute backstop at wide + 15% (55%) regardless — safety net.

    Minimum 5min grace to avoid open noise.
    """

    name = "graduated_stop"

    async def evaluate(self, ctx: dict[str, Any]) -> GateOutcome:
        settings = ctx.get("settings")
        if not settings or not getattr(settings, "ENABLE_GRADUATED_STOP", False):
            return GateOutcome(self.name, GateResult.SKIP, "Graduated stop disabled")

        trade = ctx["trade"]
        exit_premium = ctx.get("exit_premium")
        entry_premium = trade["premium_per_contract"]

        if exit_premium is None or entry_premium <= 0:
            return GateOutcome(self.name, GateResult.SKIP, "Missing premium data")

        # Minimum grace to avoid open noise
        opened_at = trade.get("opened_at")
        if not opened_at:
            return GateOutcome(self.name, GateResult.SKIP, "No opened_at")

        try:
            from datetime import datetime
            opened_dt = datetime.fromisoformat(opened_at)
            now = ctx.get("now_et") or datetime.now()
            if now.tzinfo and opened_dt.tzinfo is None:
                opened_dt = opened_dt.replace(tzinfo=now.tzinfo)
            elapsed = (now - opened_dt).total_seconds() / 60
        except (ValueError, TypeError):
            return GateOutcome(self.name, GateResult.SKIP, "Cannot parse opened_at")

        grace = getattr(settings, "GRADUATED_STOP_GRACE_MINUTES", 5.0)
        if elapsed < grace:
            return GateOutcome(self.name, GateResult.PASS,
                               f"Grace ({elapsed:.0f}m < {grace:.0f}m)")

        drop_pct = (entry_premium - exit_premium) / entry_premium * 100 if exit_premium < entry_premium else 0
        wide_stop = getattr(settings, "GRADUATED_STOP_WIDE_PCT", 50.0)
        tight_stop = getattr(settings, "GRADUATED_STOP_TIGHT_PCT", 35.0)

        # Multi-day: widen stops by 50% (theta negligible, underlying may recover)
        dte = _get_dte(trade, ctx.get("now_et"))
        if dte > 0:
            wide_stop = min(wide_stop * 1.5, 75.0)
            tight_stop = min(tight_stop * 1.5, 55.0)
        confirm_pct = getattr(settings, "STOP_UNDERLYING_CONFIRM_PCT", 0.4)
        backstop_extra = getattr(settings, "STOP_BACKSTOP_EXTRA_PCT", 15.0)

        # Determine underlying state
        current_price = ctx.get("current_price")
        entry_price = trade.get("entry_price")
        option_type = trade.get("option_type", "call").lower()
        u_move = 0.0
        has_underlying = False
        underlying_against = False

        if current_price and entry_price and entry_price > 0:
            has_underlying = True
            u_move = (current_price - entry_price) / entry_price * 100
            if option_type in ("call", "bullish", "long"):
                underlying_against = u_move < -confirm_pct
            else:
                underlying_against = u_move > confirm_pct

        if underlying_against:
            # Underlying confirms against — use TIGHT stop
            if drop_pct >= tight_stop:
                return GateOutcome(
                    self.name, GateResult.FAIL,
                    f"Confirmed stop: drop {drop_pct:.1f}% >= {tight_stop:.0f}% (tight), "
                    f"underlying {u_move:+.2f}% against ({elapsed:.0f}m)"
                )
            return GateOutcome(
                self.name, GateResult.PASS,
                f"Tight mode: drop {drop_pct:.1f}% < {tight_stop:.0f}%, "
                f"underlying {u_move:+.2f}% against ({elapsed:.0f}m)"
            )

        # Underlying NOT against — use WIDE stop (patient)
        if drop_pct >= wide_stop:
            if has_underlying:
                # Absolute backstop regardless of underlying
                if drop_pct >= wide_stop + backstop_extra:
                    return GateOutcome(
                        self.name, GateResult.FAIL,
                        f"Hard backstop: drop {drop_pct:.1f}% >= {wide_stop + backstop_extra:.0f}% "
                        f"(underlying {u_move:+.2f}% not confirmed but backstop hit) ({elapsed:.0f}m)"
                    )
                return GateOutcome(
                    self.name, GateResult.PASS,
                    f"Wide stop: drop {drop_pct:.1f}% >= {wide_stop:.0f}% but underlying "
                    f"{u_move:+.2f}% not against — holding ({elapsed:.0f}m)"
                )
            else:
                # No underlying data — use wide stop as-is
                return GateOutcome(
                    self.name, GateResult.FAIL,
                    f"Stop: drop {drop_pct:.1f}% >= {wide_stop:.0f}% "
                    f"(no underlying data) ({elapsed:.0f}m)"
                )

        return GateOutcome(
            self.name, GateResult.PASS,
            f"Stop OK: drop {drop_pct:.1f}% < {wide_stop:.0f}% ({elapsed:.0f}m)"
        )


# ---------------------------------------------------------------------------
# v5 exit gate list
# ---------------------------------------------------------------------------

V5_EXIT_GATES: list[type[ExitGate]] = [
    # --- Tier 1: Dynamic scalp (underlying-confirmed profit taking) ---
    ScalpTrailExitGate,               # 1. Scalp: peaked+faded AND underlying not confirming

    # --- Tier 2: Dynamic loss cutting (underlying-confirmed stops) ---
    GraduatedStopExitGate,            # 2. Stop: tight when underlying against, wide when not
    CheckpointExitGate,               # 3. Checkpoint: cut when BOTH premium AND underlying against

    # --- Tier 3: Scale-out ---
    TrancheScaleOutExitGate,          # 4. Tranche lock: close 1/3 at +25%

    # --- Tier 4: Trail modifiers (set flags, don't exit) ---
    VolumePeakExitGate,               # 5. Volume peak: tighten trail on exhaustion

    # --- Tier 5: Trail gates (PRIMARY exit mechanism for winners) ---
    SoftTrailExitGate,                # 6. Soft trail 15-50% band: floor = 50% of peak gain
    AdaptiveTrailingStopExitGate,     # 7. Adaptive 3-stage trail (dormant/active/runner/moonshot)

    # --- Tier 6: Time-based loss cutting (last resort) ---
    ThetaBleedExitGate,               # 8. Theta bleed: 120min+ and down 30%+

    # --- Tier 7: Final exits ---
    EODExitGate,                      # 9. End-of-day cutoff (3:45 PM ET)
]


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


# Default gate ordering — defines the standard entry pipeline.
# Gates run in this order; first failure short-circuits logging but all
# gates still run to collect full diagnostics.
DEFAULT_ENTRY_GATES: list[type[EntryGate]] = [
    BlockedTickerGate,   # 0. Blocked tickers (historically unprofitable)
    PutTickerExclusionGate,  # 0a. Block PUTs on losing tickers (PLTR, AMD, MSTR, AVGO)
    PutMarketDirectionGate,  # 0c. PUTs only when SPY is green (or bear mode)
    DirectionalRegimeGate,  # 0b. Confirm direction matches market regime (candle-based)
    ScoreGate,           # 1. Minimum score
    PremiumGate,         # 2. Valid premium
    PremiumCapGate,      # 2b. V6: reject non-index > $5 premium (skip when disabled)
    SpreadCostGate,      # 2c. V6: reject wide bid-ask spreads (skip when disabled)
    StopPriceGate,       # 3. Stop price exists
    AntiChaseGate,       # 4. Anti-chase (Vinny: reject if price moved >0.3%)
    MomentumConfirmGate, # 4b. Candle momentum confirmation (reject if underlying fading)
    TimeOfDayGate,       # 5. Time-of-day score thresholds (Vinny)
    ConsecutiveLoserGate,  # 6. Consecutive loser pause (Vinny)
    DailyLossGate,       # 7. Daily loss limit
    ConcurrentPositionsGate,  # 8. Max concurrent
    DuplicateTickerGate,  # 9. No duplicate ticker
    CorrelationCapGate,  # 10. Correlation cap (max 3 same-direction per group)
    CircuitBreakerGate,  # 11. Circuit breakers (time buffers, streaks, drawdown)
    PortfolioRiskGate,   # 11. Portfolio-level risk
    PerTradeRiskGate,    # 12. Per-trade risk
    LiquidityGate,       # 13. Liquidity (OI / volume / spread)
    WeeklyLossGate,      # 14. Weekly loss limit
    IVFilterGate,        # 15. IV rank/percentile
    VIXRegimeGate,       # 16. VIX regime
    AnalystFilterGate,   # 17. Bot performance
    BalanceGate,         # 18. Sufficient balance
]

# Exit gates run in priority order — first FAIL determines the exit reason.
#
# v2.2 Phase 1 reorder rationale (per Vince's feedback):
#   - Defensive gates (hard stop, BE clamp, soft trail) fire first — zero downside
#   - Trail gates (adaptive, underlying) are the PRIMARY exit mechanism for winners
#   - Time-based exits (theta bleed, no momentum, decel) are SUBORDINATED to trails
#     so they don't preempt the trail on winning trades
#   - Dollar trail DISABLED (replaced by soft trail + adaptive trail to extend hold times)
DEFAULT_EXIT_GATES: list[type[ExitGate]] = [
    # --- Tier 1: Emergency / defensive (always fire) ---
    ENRGExitGate,                 # 0. ENRG: early negative thesis revalidation (grace period)
    StopLossExitGate,             # 1. Hard stop -30% from entry (v2.2 §3)
    BEClampExitGate,              # 2. BE clamp: never go red after +15% green (v2.2 §4)

    # --- Tier 2: ML and scale-out ---
    MLSellExitGate,               # 3. ML-powered sell timing (disabled)
    TrancheScaleOutExitGate,      # 4. Tranche lock: close 1/3 at +25%

    # --- Tier 3: Trail modifiers (set flags, don't exit directly) ---
    VolumePeakExitGate,           # 5. Volume peak: tighten trail on exhaustion (sets flag only)

    # --- Tier 4: Trail gates — PRIMARY exit mechanism for winners ---
    SoftTrailExitGate,            # 6. Soft trail 15-35% band (v2.2 §11): floor = 50% of peak gain
    AdaptiveTrailingStopExitGate, # 7. Adaptive 3-stage trail (v2.1 — 2x P&L improvement)
    UnderlyingTrailExitGate,      # 8. Underlying-anchored trail (v2.1 §5)
    AdaptiveTimeTightenExitGate,  # 9. Adaptive tightening: narrow trail after 60min
    TrailingStopExitGate,         # 10. Simple trailing stop (fallback, skipped when adaptive enabled)

    # --- Tier 5: Profit protection (only fire after trails have had their say) ---
    ProfitFloorExitGate,          # 11. Ratcheting profit floor (v3)
    ProfitLockExitGate,           # 12. Profit lock ratchet (disabled)
    ProfitRetraceExitGate,        # 13. Profit retrace (disabled, superseded by soft trail)

    # --- Tier 6: Loss-cutting (thesis dead, momentum gone) ---
    BounceFadeExitGate,           # 14. Bounce-fade (disabled)
    ThesisCutExitGate,            # 15. Thesis cut: trend-confirmed loss cutting
    DecelExitGate,                # 16. Decel exit: momentum collapse

    # --- Tier 7: Target-based exits ---
    Target5ExitGate,              # 17. T5
    Target4ExitGate,              # 18. T4
    Target3ExitGate,              # 19. T3
    Target2ExitGate,              # 20. T2
    Target1ExitGate,              # 21. T1

    # --- Tier 8: Time-based exits (SUBORDINATED to trails) ---
    ThetaBleedExitGate,           # 22. Theta bleed: held too long + losing
    TimeDecayZoneExitGate,        # 23. Time decay zone: no new high
    NoMomentumExitGate,           # 24. No momentum after 45min
    DollarTrailExitGate,          # 25. Dollar trail: MOVED LAST (v2.2: don't preempt adaptive trail)

    # --- Tier 9: Final exits ---
    TimeExpiryExitGate,           # 26. Signal exit-by time
    EODExitGate,                  # 27. End-of-day cutoff (3:45 PM)
    ThetaDecayExitGate,           # 28. Theta decay (legacy)
]

# Map exit gate names to position_monitor exit reason codes
EXIT_GATE_TO_REASON: dict[str, str] = {
    # v5 gates
    "scalp_trail": "scalp_trail",
    "graduated_stop": "graduated_stop",
    "checkpoint": "checkpoint_cut",
    # v3 gates
    "enrg": "enrg_exit",
    "stop_loss": "stop_hit",
    "be_clamp": "be_clamp",
    "ml_sell": "ml_sell",
    "tranche_scaleout": "tranche_lock",
    "volume_peak": "volume_peak",
    "soft_trail": "soft_trail",
    "dollar_trail": "dollar_trail",
    "profit_floor": "profit_floor",
    "bounce_fade": "bounce_fade",
    "thesis_cut": "thesis_cut",
    "profit_retrace": "profit_retrace",
    "decel_exit": "decel_exit",
    "profit_lock": "profit_lock",
    "underlying_trail": "underlying_trail",
    "adaptive_trailing_stop": "adaptive_trail",
    "trailing_stop": "trailing_stop",
    "adaptive_time_tighten": "time_tighten",
    "target_5": "t5_hit",
    "target_4": "t4_hit",
    "target_3": "t3_hit",
    "target_2": "t2_hit",
    "target_1": "t1_hit",
    "theta_bleed": "theta_bleed",
    "time_decay_zone": "time_decay_zone",
    "no_momentum": "no_momentum",
    "time_expiry": "time_expiry",
    "eod_cutoff": "eod_expiry",
    "theta_decay": "theta_decay",
}


async def run_entry_pipeline(
    ctx: dict[str, Any],
    gates: list[type[EntryGate]] | None = None,
) -> PipelineResult:
    """Run the entry pipeline: evaluate all gates and return the aggregate result.

    All gates run (no short-circuit) so we get complete diagnostics.
    """
    gate_classes = gates or DEFAULT_ENTRY_GATES
    outcomes: list[GateOutcome] = []

    signal = ctx.get("signal", None)
    ticker_str = getattr(signal, "ticker", "?") if signal else "?"

    for gate_cls in gate_classes:
        gate = gate_cls()
        try:
            outcome = await gate.evaluate(ctx)
        except Exception as exc:
            outcome = GateOutcome(gate.name, GateResult.SKIP, f"Error: {exc}")
            logger.warning(f"  [{ticker_str}] gate '{gate.name}' raised: {exc}")
        outcomes.append(outcome)

        # Log every gate verdict at INFO for full traceability
        icon = {"pass": "✓", "fail": "✗", "skip": "−"}[outcome.result.value]
        logger.info(f"  [{ticker_str}] {icon} {outcome.gate_name}: {outcome.reason}")

    approved = all(o.result != GateResult.FAIL for o in outcomes)
    result = PipelineResult(approved=approved, outcomes=outcomes)

    # Log the pipeline verdict
    if approved:
        passed = sum(1 for o in outcomes if o.result == GateResult.PASS)
        skipped = sum(1 for o in outcomes if o.result == GateResult.SKIP)
        logger.info(f"Pipeline APPROVED: {ticker_str} ({passed} passed, {skipped} skipped)")
    else:
        failures = ", ".join(f"{o.gate_name}({o.reason})" for o in result.failures)
        logger.info(f"Pipeline REJECTED: {ticker_str} — {failures}")

    return result


_TARGET_GATE_NAMES = {"target_1", "target_2", "target_3", "target_4", "target_5"}
# Trail gates that ML hold can override (dollar_trail exits that ML thinks are premature)
_TRAIL_GATE_NAMES = {"dollar_trail", "profit_retrace", "adaptive_trailing_stop", "trailing_stop", "adaptive_time_tighten"}


async def run_exit_pipeline(
    ctx: dict[str, Any],
    gates: list[type[ExitGate]] | None = None,
) -> tuple[str | None, str]:
    """Run the exit pipeline: evaluate gates in priority order.

    Returns (exit_reason, description) on the FIRST gate that triggers,
    or (None, "") if all gates pass (hold the position).

    Unlike entry pipeline, this short-circuits on first FAIL because
    exit priority matters (stop > trailing > targets > time > EOD).

    **ML override**: When ML says HOLD with high confidence (expected future
    PnL above threshold), target gates (T1-T5) are converted to scale-out
    only — they still trigger partial closes but don't override ML's hold
    decision for the remaining position. This lets ML manage the bulk of
    the position while targets lock in incremental profits.
    """
    # Select gate list based on EXIT_ENGINE setting (v3 or v5)
    if gates:
        gate_classes = gates
    else:
        settings = ctx.get("settings")
        engine = getattr(settings, "EXIT_ENGINE", "v3") if settings else "v3"
        if engine == "v5":
            gate_classes = V5_EXIT_GATES
        else:
            gate_classes = DEFAULT_EXIT_GATES

    trade = ctx.get("trade", {})
    ticker = trade.get("ticker", "?")
    trade_id = trade.get("id", "?")

    settings = ctx.get("settings")
    ml_override_targets = (
        settings
        and getattr(settings, "ML_OVERRIDE_TARGETS", False)
        and getattr(settings, "ENABLE_ML_EXIT", False)
    )
    ml_override_trails = (
        settings
        and getattr(settings, "ML_OVERRIDE_TRAILS", False)
        and getattr(settings, "ENABLE_ML_EXIT", False)
    )
    ml_holding = False  # set True when ML says hold with confidence

    for gate_cls in gate_classes:
        gate = gate_cls()
        try:
            outcome = await gate.evaluate(ctx)
        except Exception as exc:
            logger.warning(f"  [#{trade_id} {ticker}] exit gate '{gate.name}' raised: {exc}")
            continue

        # Track ML's hold decision for downstream gates
        if gate.name == "ml_sell" and outcome.result == GateResult.PASS and (ml_override_targets or ml_override_trails):
            # Parse expected future PnL from ML outcome reason
            # Format: "ML hold: P(sell)=0.XX E[future]=+YY.Y% model=..."
            min_future = getattr(settings, "ML_OVERRIDE_MIN_FUTURE_PNL", 5.0)
            try:
                future_str = outcome.reason.split("E[future]=")[1].split("%")[0]
                expected_future = float(future_str)
                if expected_future >= min_future:
                    ml_holding = True
                    logger.debug(
                        f"  [#{trade_id} {ticker}] ML override active: "
                        f"E[future]={expected_future:+.1f}% >= {min_future}%"
                    )
            except (IndexError, ValueError):
                pass  # can't parse — don't override

        if outcome.result == GateResult.FAIL:
            # ML override: when ML says HOLD with high confidence, suppress
            # trail exits (dollar_trail, adaptive_trail) and convert target
            # gates to scale-out-only. ML trained on 2yr of data sees patterns
            # that simple trails miss — trust it over mechanical trailing stops.
            if ml_holding and ml_override_targets and gate.name in _TARGET_GATE_NAMES:
                reason_code = EXIT_GATE_TO_REASON.get(gate.name, gate.name)
                logger.info(
                    f"  [#{trade_id} {ticker}] {gate.name} hit but ML holding — "
                    f"scale-out only: {outcome.reason}"
                )
                # Return with ml_scale_out prefix so position_monitor can
                # do a partial close but keep monitoring
                return reason_code, f"[ML_HOLD] {outcome.reason}"

            if ml_holding and ml_override_trails and gate.name in _TRAIL_GATE_NAMES:
                logger.info(
                    f"  [#{trade_id} {ticker}] {gate.name} would exit but ML "
                    f"overrides — holding: {outcome.reason}"
                )
                continue  # skip this trail exit, ML says hold

            reason_code = EXIT_GATE_TO_REASON.get(gate.name, gate.name)
            logger.info(
                f"  [#{trade_id} {ticker}] EXIT triggered by {gate.name}: {outcome.reason}"
            )
            return reason_code, outcome.reason

        # Log non-triggering gates at DEBUG for full audit trail
        if outcome.result == GateResult.PASS:
            logger.debug(f"  [#{trade_id} {ticker}] ✓ {gate.name}: {outcome.reason}")

    return None, ""
