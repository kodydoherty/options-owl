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

from options_owl.risk.exit_v5.config import (
    V5Config,
    apply_v7_wide_trail_exits,
    get_ticker_config,
)
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
    ExitReason.SCALP_TARGET: "scalp_target",
}


class V5MonitorBridge:
    """Manages v5 FSM state for all open trades.

    One instance per bot, created once at startup when EXIT_ENGINE=v5.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        # V7 wide-trail exits. CALL + PUT FSMs use a settings copy with scaleout +
        # 2PM-tighten OFF (breakeven ratchet stays ON). The per-side cfg transform
        # differs: CALLs get the faster stall-cut (theta 60); PUTs keep no-hold-limit
        # so they can ride slow-building crashes (see apply_v7_wide_trail_exits).
        self._v7_wide_trail = getattr(settings, "ENABLE_V7_WIDE_TRAIL", False)
        self._v7_settings = settings
        if self._v7_wide_trail:
            self._v7_settings = settings.model_copy(
                update={
                    "ENABLE_V6_SCALEOUT": False,
                    "ENABLE_V6_2PM_TIGHTEN": False,
                    "ENABLE_V6_BREAKEVEN_RATCHET": True,
                }
            )
        self.cfg = V5Config.from_settings(settings)
        self._use_per_ticker = getattr(settings, "ENABLE_V6_PER_TICKER_CONFIG", False)
        # Default FSM for CALL tickers without per-ticker config
        _default_cfg = (
            apply_v7_wide_trail_exits(self.cfg, is_put=False)
            if self._v7_wide_trail
            else self.cfg
        )
        self.fsm = ExitFSM(_default_cfg, settings=self._v7_settings)
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
        webull_fill = trade.get("webull_entry_fill_price") or 0.0
        blended = trade.get("premium_per_contract", 0.0) or 0.0

        # RESTART DURABILITY (FIX 3a): if this trade was DCA'd, webull_entry_fill_price
        # holds only the ORIGINAL first fill — using it would overstate gain% and
        # break the FSM after a restart. premium_per_contract is the blended average
        # (updated by dca_add_contracts). Detect a DCA via dca_last_add_at (set on
        # every add) or the blended price diverging from the first fill beyond a
        # penny of rounding. When detected, prefer the blended average.
        dca_occurred = bool(trade.get("dca_last_add_at")) or (
            webull_fill > 0 and blended > 0 and abs(blended - webull_fill) > 0.01
        )
        if dca_occurred and blended > 0:
            entry_prem = blended
            entry_source = "blended(DCA)"
        elif webull_fill > 0:
            entry_prem = webull_fill
            entry_source = "webull_fill"
        else:
            entry_prem = blended
            entry_source = "premium_per_contract"

        peak_premium = trade.get("mfe_premium") or entry_prem
        # Guard: peak can never be below entry (mfe may be stale/null pre-DCA).
        if peak_premium < entry_prem:
            peak_premium = entry_prem

        state = TradeState(
            trade_id=trade_id,
            ticker=trade["ticker"],
            option_type=trade.get("option_type", "call"),
            entry_premium=entry_prem,
            entry_time=entry_time,
            contracts=trade.get("contracts", 1),
            peak_premium=peak_premium,
            entry_underlying_price=trade.get("entry_price", 0.0),
            dte=dte,
            expiry_date=expiry_date,
        )

        # RESTART DURABILITY (FIX 3c): restore scaled_out so a trade that already
        # scaled out before a restart does NOT scale out again. The monitor injects
        # `_scaled_out_restore=True` when a scaleout child row exists for this parent.
        if trade.get("_scaled_out_restore"):
            state.scaled_out = True

        # RESTART DURABILITY (FIX 3b): arm the breakeven ratchet from PEAK gain, not
        # current gain. A trade that peaked >= the trigger but now sits at a loss
        # must keep its break-even protection across a restart. check_breakeven_ratchet
        # will (re-)arm on current gain too, so the normal in-process path is unchanged.
        if self.settings is not None and getattr(
            self.settings, "ENABLE_V6_BREAKEVEN_RATCHET", False
        ):
            trigger_pct = getattr(self.settings, "V6_BREAKEVEN_TRIGGER_PCT", 20.0)
            peak_gain = (
                (peak_premium - entry_prem) / entry_prem * 100
                if entry_prem > 0 else 0.0
            )
            if peak_gain >= trigger_pct:
                state.breakeven_ratchet_armed = True

        self._states[trade_id] = state
        logger.info(
            f"EXIT_FSM: Created TradeState #{trade_id} {trade['ticker']} "
            f"entry=${state.entry_premium:.2f} ({entry_source}) "
            f"peak=${state.peak_premium:.2f} contracts={state.contracts} "
            f"scaled_out={state.scaled_out} "
            f"ratchet_armed={state.breakeven_ratchet_armed} "
            f"dte={dte} expiry={expiry_date}"
        )
        return state

    def _get_fsm(self, ticker: str, option_type: str = "call") -> ExitFSM:
        """Get the FSM for a ticker+direction, using per-ticker config if V6 enabled.

        PUT trades always get PUT_SCALP_CONFIG regardless of per-ticker setting.
        CALL trades without per-ticker config return the default FSM.
        """
        is_put = option_type.lower() == "put"

        # CALL trades: use default FSM if per-ticker is disabled
        if not is_put and not self._use_per_ticker:
            return self.fsm

        cache_key = f"{ticker}:{option_type.lower()}"
        if cache_key not in self._ticker_fsms:
            cfg = get_ticker_config(
                ticker,
                use_per_ticker=self._use_per_ticker,
                option_type=option_type,
            )
            # V7 wide-trail applies to both CALL and PUT; the per-side cfg transform
            # differs (CALLs get theta 60 stall-cut; PUTs keep no-hold-limit).
            if self._v7_wide_trail:
                cfg = apply_v7_wide_trail_exits(cfg, is_put=is_put)
                fsm_settings = self._v7_settings
            else:
                fsm_settings = self.settings
            self._ticker_fsms[cache_key] = ExitFSM(cfg, settings=fsm_settings)
        return self._ticker_fsms[cache_key]

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

        # Use real bid/ask from the trade dict when the monitor supplied them
        # (position_monitor injects only when a quote source actually returned
        # NBBO). A real bid of 0.0 is meaningful — it's exactly the "no buyers"
        # signal the bid-disappearance gate exists to catch — so it must pass
        # through, NOT be replaced by a synthetic positive bid. Only estimate a
        # spread when no real quote was provided (bid/ask absent or None).
        raw_bid = trade.get("bid")
        raw_ask = trade.get("ask")
        if raw_bid is None or raw_ask is None:
            spread_pct = 0.05 if exit_premium < 1.0 else 0.02
            bid = exit_premium * (1 - spread_pct)
            ask = exit_premium * (1 + spread_pct)
        else:
            bid = raw_bid
            ask = raw_ask

        # Compute minutes to close (market closes at 4:00 PM ET)
        seconds_left = max(0, (16 * 60 * 60) - (now_et.hour * 3600 + now_et.minute * 60 + now_et.second))
        minutes_to_close = seconds_left / 60.0

        # V6: per-ticker FSM config (PUT trades get PUT_SCALP_CONFIG)
        fsm = self._get_fsm(state.ticker, state.option_type)

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
