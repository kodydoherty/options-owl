"""Unified risk manager — aggregates all risk checks before approving a trade."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from loguru import logger

from options_owl.journal.db import connect as _connect_db

try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")
except ImportError:
    _ET = timezone(timedelta(hours=-5))

from options_owl.models.signals import TradeSignal

# Optional feature-module imports — the risk manager degrades gracefully
# if any of these modules have not been created yet.
try:
    from options_owl.signals.iv_filter import check_iv_filter  # type: ignore[import-untyped]
except ImportError:
    check_iv_filter = None

try:
    from options_owl.risk.vix_regime import check_vix_regime  # type: ignore[import-untyped]
except ImportError:
    check_vix_regime = None

try:
    from options_owl.signals.analyst_tracker import check_analyst_filter
except ImportError:
    check_analyst_filter = None  # type: ignore[assignment]

try:
    from options_owl.risk.circuit_breaker import CircuitBreaker
except ImportError:
    CircuitBreaker = None  # type: ignore[assignment,misc]


class RiskManager:
    """Central risk gate that runs all enabled checks on an incoming signal."""

    def __init__(self, settings: object) -> None:
        self.settings = settings

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def check_trade(
        self,
        signal: TradeSignal,
        db_path: str,
    ) -> tuple[bool, list[str]]:
        """Run every enabled risk check and return (approved, reasons).

        *reasons* collects human-readable strings for each check that
        blocked the trade.  If *approved* is True the list is empty.
        """
        if not getattr(self.settings, "ENABLE_RISK_MANAGER", False):
            return True, []

        reasons: list[str] = []

        # Check 1 — portfolio risk limit
        reason = await self._check_portfolio_risk(signal, db_path)
        if reason:
            reasons.append(reason)

        # Check 2 — per-trade risk limit
        reason = self._check_per_trade_risk(signal)
        if reason:
            reasons.append(reason)

        # Check 3 — weekly loss limit
        reason = await self._check_weekly_loss(db_path)
        if reason:
            reasons.append(reason)

        # Check 4 — IV filter (optional module)
        if getattr(self.settings, "ENABLE_IV_FILTER", False) and check_iv_filter is not None:
            try:
                passes, iv_reason = check_iv_filter(signal.ticker, self.settings)
                if not passes:
                    reasons.append(f"IV filter: {iv_reason}")
            except Exception as exc:
                logger.warning(f"IV filter check failed: {exc}")

        # Check 5 — VIX regime (optional module)
        if getattr(self.settings, "ENABLE_VIX_FILTER", False) and check_vix_regime is not None:
            try:
                regime = check_vix_regime(self.settings)
                if not regime.can_trade:
                    reasons.append(f"VIX regime: {regime.reason}")
            except Exception as exc:
                logger.warning(f"VIX regime check failed: {exc}")

        # Check 6 — analyst / bot filter (optional module)
        if getattr(self.settings, "ENABLE_ANALYST_FILTER", False) and check_analyst_filter is not None:
            try:
                passes, analyst_reason, _stats = await check_analyst_filter(
                    db_path, signal.bot_source.value, self.settings,
                )
                if not passes:
                    reasons.append(f"Analyst filter: {analyst_reason}")
            except Exception as exc:
                logger.warning(f"Analyst filter check failed: {exc}")

        # Check 7 — circuit breakers (optional module)
        if getattr(self.settings, "ENABLE_CIRCUIT_BREAKERS", False) and CircuitBreaker is not None:
            try:
                cb_approved, cb_reasons = await CircuitBreaker.check_all(
                    db_path, self.settings,
                )
                if not cb_approved:
                    reasons.extend(f"Circuit breaker: {r}" for r in cb_reasons)
            except Exception as exc:
                logger.warning(f"Circuit breaker check failed: {exc}")

        approved = len(reasons) == 0

        if approved:
            logger.debug(
                f"RiskManager APPROVED: {signal.ticker} {signal.direction.value} "
                f"score={signal.score} (all 7 checks passed)"
            )
        else:
            logger.info(
                f"RiskManager REJECTED: {signal.ticker} — {'; '.join(reasons)}"
            )

        return approved, reasons

    def get_position_size_multiplier(self) -> float:
        """Return a sizing multiplier based on the current VIX regime.

        Falls back to 1.0 if the VIX module is unavailable or disabled.
        """
        if not getattr(self.settings, "ENABLE_VIX_FILTER", False):
            return 1.0

        if check_vix_regime is None:
            return 1.0

        try:
            regime = check_vix_regime(self.settings)
            return regime.position_size_multiplier
        except Exception as exc:
            logger.warning(f"VIX regime check failed, using 1.0x multiplier: {exc}")
            return 1.0

    async def get_risk_summary(self, db_path: str) -> str:
        """Return a formatted string describing current risk exposure."""
        lines: list[str] = ["=== Risk Summary ==="]

        portfolio_size: float = getattr(self.settings, "PORTFOLIO_SIZE", 0.0)
        max_portfolio_pct: float = getattr(self.settings, "MAX_PORTFOLIO_RISK_PCT", 20.0)
        max_trade_pct: float = getattr(self.settings, "MAX_LOSS_PER_TRADE_PCT", 2.0)
        weekly_limit_pct: float = getattr(self.settings, "WEEKLY_LOSS_LIMIT_PCT", 20.0)

        # Open-position exposure
        open_cost = await self._sum_open_trade_costs(db_path)
        exposure_pct = (open_cost / portfolio_size * 100.0) if portfolio_size > 0 else 0.0
        lines.append(
            f"  Open exposure: ${open_cost:.2f} "
            f"({exposure_pct:.1f}% / {max_portfolio_pct:.0f}% limit)"
        )

        # Weekly losses
        weekly_loss = await self._sum_weekly_losses(db_path)
        weekly_pct = (abs(weekly_loss) / portfolio_size * 100.0) if portfolio_size > 0 else 0.0
        lines.append(
            f"  Weekly losses:  ${weekly_loss:.2f} "
            f"({weekly_pct:.1f}% / {weekly_limit_pct:.0f}% limit)"
        )

        lines.append(f"  Max per trade: {max_trade_pct:.1f}% of portfolio")
        lines.append(f"  Size multiplier: {self.get_position_size_multiplier():.2f}x")
        lines.append("=" * 20)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal checks
    # ------------------------------------------------------------------

    async def _check_portfolio_risk(
        self, signal: TradeSignal, db_path: str,
    ) -> str | None:
        """Check 1: total open-position cost vs portfolio limit."""
        portfolio_size: float = getattr(self.settings, "PORTFOLIO_SIZE", 0.0)
        max_pct: float = getattr(self.settings, "MAX_PORTFOLIO_RISK_PCT", 20.0)
        if portfolio_size <= 0:
            return None

        open_cost = await self._sum_open_trade_costs(db_path)
        # Estimate this new trade's cost (1 contract minimum)
        premium = signal.atm_premium or 0.0
        est_cost = premium * 100.0
        total_risk = open_cost + est_cost
        risk_pct = total_risk / portfolio_size * 100.0

        if risk_pct > max_pct:
            return (
                f"Portfolio risk {risk_pct:.1f}% exceeds "
                f"{max_pct:.0f}% limit (open=${open_cost:.0f} + new=${est_cost:.0f})"
            )
        return None

    def _check_per_trade_risk(self, signal: TradeSignal) -> str | None:
        """Check 2: single trade cost vs portfolio limit."""
        portfolio_size: float = getattr(self.settings, "PORTFOLIO_SIZE", 0.0)
        max_pct: float = getattr(self.settings, "MAX_LOSS_PER_TRADE_PCT", 2.0)
        if portfolio_size <= 0:
            return None

        premium = signal.atm_premium or 0.0
        est_cost = premium * 100.0
        trade_pct = est_cost / portfolio_size * 100.0

        if trade_pct > max_pct:
            return (
                f"Trade risk {trade_pct:.1f}% exceeds "
                f"{max_pct:.1f}% per-trade limit (cost=${est_cost:.0f})"
            )
        return None

    async def _check_weekly_loss(self, db_path: str) -> str | None:
        """Check 3: cumulative closed-trade losses this week."""
        portfolio_size: float = getattr(self.settings, "PORTFOLIO_SIZE", 0.0)
        max_pct: float = getattr(self.settings, "WEEKLY_LOSS_LIMIT_PCT", 20.0)
        if portfolio_size <= 0:
            return None

        weekly_loss = await self._sum_weekly_losses(db_path)
        loss_pct = abs(weekly_loss) / portfolio_size * 100.0

        if loss_pct >= max_pct:
            return (
                f"Weekly loss {loss_pct:.1f}% hit "
                f"{max_pct:.0f}% limit (${weekly_loss:.2f})"
            )
        return None

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    @staticmethod
    async def _sum_open_trade_costs(db_path: str) -> float:
        """Sum total_cost of all currently open paper trades."""
        try:
            async with _connect_db(db_path) as conn:
                cursor = await conn.execute(
                    "SELECT COALESCE(SUM(total_cost), 0) FROM paper_trades WHERE status = 'open'"
                )
                row = await cursor.fetchone()
                return float(row[0]) if row else 0.0  # type: ignore[index]
        except Exception as exc:
            logger.warning(f"Failed to sum open trade costs: {exc}")
            return 0.0

    @staticmethod
    async def _sum_weekly_losses(db_path: str) -> float:
        """Sum negative pnl_dollars for trades closed in the current ISO week."""
        try:
            now = datetime.now(tz=_ET)
            week_start = (now - timedelta(days=now.weekday())).strftime("%Y-%m-%d")
            async with _connect_db(db_path) as conn:
                cursor = await conn.execute(
                    "SELECT COALESCE(SUM(pnl_dollars), 0) FROM paper_trades "
                    "WHERE status = 'closed' AND pnl_dollars < 0 "
                    "AND closed_at >= ?",
                    (week_start,),
                )
                row = await cursor.fetchone()
                return float(row[0]) if row else 0.0  # type: ignore[index]
        except Exception as exc:
            logger.warning(f"Failed to sum weekly losses: {exc}")
            return 0.0
