"""Signal scoring engine — evaluates signals through the risk manager pipeline."""

from __future__ import annotations

from loguru import logger

from options_owl.models.signals import TradeSignal
from options_owl.risk.manager import RiskManager


class SignalEngine:
    """Wraps the unified :class:`RiskManager` and provides a single
    entry-point for signal evaluation.
    """

    def __init__(self, settings: object) -> None:
        self.settings = settings
        self.risk_manager = RiskManager(settings)

    async def evaluate_signal(
        self,
        signal: TradeSignal,
        db_path: str,
        settings: object | None = None,
    ) -> tuple[bool, list[str]]:
        """Run *signal* through all enabled risk checks.

        Args:
            signal: The parsed trade signal to evaluate.
            db_path: Path to the sqlite database.
            settings: Optional override; falls back to ``self.settings``.

        Returns:
            (approved, reasons) — *approved* is True when the signal passes
            every check.  *reasons* lists human-readable rejection messages.
        """
        _settings = settings or self.settings

        # Basic score gate (non-risk-manager check)
        min_score: int = getattr(_settings, "MIN_SCORE", 0)
        if signal.score < min_score:
            reason = f"Score {signal.score} < minimum {min_score}"
            logger.info(f"Signal rejected: {reason}")
            return False, [reason]

        # Delegate to unified risk manager
        approved, reasons = await self.risk_manager.check_trade(signal, db_path)

        if approved:
            logger.info(
                f"Signal APPROVED: {signal.ticker} "
                f"{signal.direction.value.upper()} score={signal.score}"
            )
        else:
            logger.info(
                f"Signal REJECTED: {signal.ticker} "
                f"{signal.direction.value.upper()} — {'; '.join(reasons)}"
            )

        return approved, reasons
