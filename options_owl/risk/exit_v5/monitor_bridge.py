"""Bridge between position_monitor and v5 FSM exit engine.

Manages per-trade TradeState instances and translates FSM ExitAction
results into the (reason, description) tuple format that position_monitor
expects.

The bridge is stateful (holds TradeState per trade_id) but the FSM
itself is stateless — all logic lives in ExitFSM.evaluate().
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from loguru import logger

from options_owl.risk.exit_v5.config import V5Config, get_ticker_config
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState
from options_owl.risk.exit_v5.types import ExitReason

if TYPE_CHECKING:
    from options_owl.config.settings import Settings


# Map ExitReason → position_monitor reason strings (for DB compatibility)
_REASON_MAP = {
    ExitReason.HOLD: None,
    ExitReason.BID_DISAPPEARANCE: "bid_disappearance",
    ExitReason.HARD_STOP: "stop_loss",
    ExitReason.SOFT_TRAIL: "soft_trail",
    ExitReason.EOD_CUTOFF: "eod_cutoff",
    ExitReason.SCALP_TRAIL: "scalp_trail",
    ExitReason.CHECKPOINT_CUT: "checkpoint_cut",
    ExitReason.CONFIRMED_STOP: "confirmed_stop",
    ExitReason.ADAPTIVE_TRAIL: "adaptive_trail",
    ExitReason.THETA_BLEED: "theta_bleed",
    ExitReason.THETA_TIMER: "theta_timer",
    ExitReason.PROFIT_TARGET: "profit_target",
    ExitReason.BREAKEVEN_RATCHET: "breakeven_ratchet",
    ExitReason.SCALEOUT: "scaleout_20",
    ExitReason.SIDEWAYS_SCALP: "sideways_scalp",
}


class V5MonitorBridge:
    """Manages v5 FSM state for all open trades.

    One instance per bot, created once at startup when EXIT_ENGINE=v5.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.cfg = V5Config.from_settings(settings)
        self._use_per_ticker = getattr(settings, "ENABLE_V6_PER_TICKER_CONFIG", False)
        # Default FSM for tickers without per-ticker config
        self.fsm = ExitFSM(self.cfg, settings=settings)
        self._states: dict[int, TradeState] = {}
        # Cache per-ticker FSMs to avoid re-creating each cycle
        self._ticker_fsms: dict[str, ExitFSM] = {}

    def get_or_create_state(self, trade: dict, now_et: datetime) -> TradeState:
        """Get existing TradeState or create one from a DB trade dict."""
        trade_id = trade["id"]
        if trade_id in self._states:
            return self._states[trade_id]

        # Parse entry time — DB stores UTC, monitor passes ET
        entry_time = now_et  # fallback
        opened_at = trade.get("opened_at", "")
        if opened_at:
            try:
                from zoneinfo import ZoneInfo
                entry_time = datetime.fromisoformat(opened_at)
                # DB timestamps are UTC (naive). Convert to ET so elapsed
                # time calculations work correctly against now_et (which is ET).
                if entry_time.tzinfo is None:
                    entry_time = entry_time.replace(tzinfo=ZoneInfo("UTC"))
                entry_time = entry_time.astimezone(ZoneInfo("America/New_York"))
                # Strip tzinfo so subtraction in FSM works (both naive ET)
                entry_time = entry_time.replace(tzinfo=None)
            except (ValueError, TypeError):
                pass

        # Compute DTE from expiry_date
        expiry_date = trade.get("expiry_date", "")
        dte = 0
        if expiry_date:
            try:
                from datetime import date as _date
                exp = datetime.strptime(expiry_date, "%Y-%m-%d").date()
                today = now_et.date() if hasattr(now_et, "date") else _date.today()
                dte = max(0, (exp - today).days)
            except (ValueError, TypeError):
                pass

        # Use actual Webull fill price when available — signal premium can be
        # stale/inflated, causing the FSM to undercount gain% and miss exits.
        # e.g. AVGO #135: signal $2.11, fill $1.83, peak $2.26 → +7% vs +23%.
        entry_prem = (
            trade.get("webull_entry_fill_price")
            or trade.get("premium_per_contract", 0.0)
        )

        state = TradeState(
            trade_id=trade_id,
            ticker=trade["ticker"],
            option_type=trade.get("option_type", "call"),
            entry_premium=entry_prem,
            entry_time=entry_time,
            contracts=trade.get("contracts", 1),
            peak_premium=trade.get("mfe_premium") or entry_prem,
            entry_underlying_price=trade.get("entry_price", 0.0),
            dte=dte,
            expiry_date=expiry_date,
        )

        self._states[trade_id] = state
        logger.info(
            f"EXIT_FSM: Created TradeState #{trade_id} {trade['ticker']} "
            f"entry=${state.entry_premium:.2f} contracts={state.contracts} "
            f"dte={dte} expiry={expiry_date}"
        )
        return state

    def _get_fsm(self, ticker: str) -> ExitFSM:
        """Get the FSM for a ticker, using per-ticker config if V6 enabled."""
        if not self._use_per_ticker:
            return self.fsm
        if ticker not in self._ticker_fsms:
            cfg = get_ticker_config(ticker, use_per_ticker=True)
            self._ticker_fsms[ticker] = ExitFSM(cfg, settings=self.settings)
        return self._ticker_fsms[ticker]

    def evaluate(
        self,
        trade: dict,
        exit_premium: float,
        current_price: float,
        now_et: datetime,
        candle_data: dict | None = None,
    ) -> tuple[str | None, str]:
        """Evaluate v5/v6 exit conditions for one trade.

        Returns (reason, description) matching position_monitor's expected format.
        reason=None means HOLD.
        """
        state = self.get_or_create_state(trade, now_et)

        # Sync contracts from DB (may have changed via partial close)
        db_contracts = trade.get("contracts", 1)
        if db_contracts != state.contracts:
            state.contracts = db_contracts

        # Use real bid/ask from trade dict if available.
        # Fall back to spread estimate based on premium level.
        bid = trade.get("bid", 0.0) or 0.0
        ask = trade.get("ask", 0.0) or 0.0
        if bid <= 0 or ask <= 0:
            spread_pct = 0.05 if exit_premium < 1.0 else 0.02
            bid = exit_premium * (1 - spread_pct)
            ask = exit_premium * (1 + spread_pct)

        # Compute minutes to close (market closes at 4:00 PM ET)
        minutes_to_close = max(0, (16 * 60) - (now_et.hour * 60 + now_et.minute))

        # V6: per-ticker FSM config
        fsm = self._get_fsm(state.ticker)

        action = fsm.evaluate(
            state=state,
            current_premium=exit_premium,
            bid=bid,
            ask=ask,
            now_et=now_et,
            current_underlying=current_price,
            minutes_to_close=float(minutes_to_close),
            candle_data=candle_data or {},
        )

        # Log every evaluation — exits at INFO, holds at DEBUG
        gain_pct = (
            (exit_premium - state.entry_premium) / state.entry_premium * 100
            if state.entry_premium > 0 else 0
        )
        if action.should_exit:
            logger.info(
                f"EXIT_FSM: #{state.trade_id} {state.ticker} "
                f"state={state.state.value} reason={action.reason.value} "
                f"prem=${exit_premium:.2f} ({gain_pct:+.1f}%) "
                f"peak=${state.peak_premium:.2f} | {action.detail}"
            )
        else:
            logger.debug(
                f"EXIT_FSM: #{state.trade_id} {state.ticker} "
                f"state={state.state.value} HOLD "
                f"prem=${exit_premium:.2f} ({gain_pct:+.1f}%) "
                f"peak=${state.peak_premium:.2f}"
            )

        if not action.should_exit:
            return None, ""

        reason = _REASON_MAP.get(action.reason, action.reason.value)

        # V6: encode scale-out quantity in description so position_monitor
        # can dispatch a partial close (matches tranche pattern)
        if action.contracts_to_close > 0:
            description = f"[V6_SCALEOUT:{action.contracts_to_close}] {action.detail}"
        else:
            description = f"[V5] {action.detail}"

        return reason, description

    def cleanup_trade(self, trade_id: int) -> None:
        """Remove state for a closed trade."""
        removed = self._states.pop(trade_id, None)
        if removed:
            logger.debug(f"EXIT_FSM: Cleaned up state for trade #{trade_id}")


# Backward compat alias
V4MonitorBridge = V5MonitorBridge
