from datetime import datetime, timezone

from options_owl.models.signals import PriceBar, TradeOutcome
from options_owl.signals.outcome_resolver import resolve_signal


def _bar(ts_minute: int, o: float, h: float, lo: float, c: float) -> PriceBar:
    """Helper to create a PriceBar at a given minute offset."""
    return PriceBar(
        timestamp=datetime(2026, 3, 27, 9, 30 + ts_minute, tzinfo=timezone.utc),
        open=o,
        high=h,
        low=lo,
        close=c,
        volume=1000,
    )


# Base signal dicts
PUT_SIGNAL = {
    "direction": "put",
    "entry_price": 170.0,
    "target_1": 168.0,
    "target_2": 167.0,
    "stop_price": 171.0,
    "atm_premium": 1.70,
    "otm_premium": 0.46,
    "ticker": "NVDA",
    "created_at": "2026-03-27T09:30:00",
}

CALL_SIGNAL = {
    "direction": "call",
    "entry_price": 368.0,
    "target_1": 369.5,
    "target_2": 370.5,
    "stop_price": 367.0,
    "atm_premium": 0.93,
    "otm_premium": 0.39,
    "ticker": "TSLA",
    "created_at": "2026-03-27T09:30:00",
}


class TestPutSignalResolution:
    def test_put_hits_t1_then_t2(self):
        bars = [
            _bar(0, 170.0, 170.5, 169.0, 169.5),  # hovering
            _bar(1, 169.5, 169.5, 167.8, 168.0),   # hits T1 (168.0)
            _bar(2, 168.0, 168.2, 166.5, 166.8),   # hits T2 (167.0)
        ]
        result = resolve_signal(PUT_SIGNAL, bars, signal_id=1)
        assert result.outcome == TradeOutcome.T2_HIT
        assert result.hit_price == 167.0
        assert result.pnl_underlying_pct > 0

    def test_put_hits_t1_only(self):
        bars = [
            _bar(0, 170.0, 170.2, 167.8, 168.5),   # hits T1
            _bar(1, 168.5, 169.0, 168.0, 168.5),    # stays above T2
        ]
        result = resolve_signal(PUT_SIGNAL, bars, signal_id=2)
        assert result.outcome == TradeOutcome.T1_HIT
        assert result.hit_price == 168.0

    def test_put_hits_stop(self):
        bars = [
            _bar(0, 170.0, 171.5, 169.5, 171.0),   # hits stop (171.0)
        ]
        result = resolve_signal(PUT_SIGNAL, bars, signal_id=3)
        assert result.outcome == TradeOutcome.STOP_HIT
        assert result.hit_price == 171.0
        assert result.pnl_underlying_pct < 0

    def test_put_expires_flat(self):
        bars = [
            _bar(0, 170.0, 170.5, 169.0, 169.5),
            _bar(1, 169.5, 170.5, 169.0, 170.0),
            _bar(2, 170.0, 170.5, 169.0, 169.8),
        ]
        result = resolve_signal(PUT_SIGNAL, bars, signal_id=4)
        assert result.outcome == TradeOutcome.EXPIRED
        assert result.max_favorable_pct > 0


class TestCallSignalResolution:
    def test_call_hits_t2(self):
        bars = [
            _bar(0, 368.0, 369.8, 367.5, 369.5),   # hits T1
            _bar(1, 369.5, 371.0, 369.0, 370.5),    # hits T2
        ]
        result = resolve_signal(CALL_SIGNAL, bars, signal_id=5)
        assert result.outcome == TradeOutcome.T2_HIT
        assert result.pnl_underlying_pct > 0

    def test_call_hits_stop(self):
        bars = [
            _bar(0, 368.0, 368.2, 366.5, 366.8),   # hits stop
        ]
        result = resolve_signal(CALL_SIGNAL, bars, signal_id=6)
        assert result.outcome == TradeOutcome.STOP_HIT
        assert result.pnl_underlying_pct < 0


class TestEdgeCases:
    def test_no_bars_returns_unknown(self):
        result = resolve_signal(PUT_SIGNAL, [], signal_id=7)
        assert result.outcome == TradeOutcome.UNKNOWN

    def test_atm_pnl_estimated(self):
        bars = [
            _bar(0, 170.0, 170.0, 167.8, 168.0),   # hits T1
        ]
        result = resolve_signal(PUT_SIGNAL, bars, signal_id=8)
        assert result.pnl_atm_est is not None
        assert result.pnl_atm_est > 0

    def test_otm_pnl_estimated(self):
        bars = [
            _bar(0, 170.0, 170.0, 167.8, 168.0),
        ]
        result = resolve_signal(PUT_SIGNAL, bars, signal_id=9)
        assert result.pnl_otm_est is not None
        assert result.pnl_otm_est > 0

    def test_max_excursion_tracked(self):
        bars = [
            _bar(0, 170.0, 170.3, 168.5, 169.0),
            _bar(1, 169.0, 170.8, 169.0, 170.5),   # adverse
            _bar(2, 170.5, 170.5, 168.0, 168.5),    # favorable, hits T1
        ]
        result = resolve_signal(PUT_SIGNAL, bars, signal_id=10)
        assert result.max_favorable_pct > 0
        assert result.max_adverse_pct > 0


# ---------------------------------------------------------------------------
# Expanded tests: same-bar stop+T1, signal timestamp filtering, PnL accuracy,
# max favorable/adverse excursion values
# ---------------------------------------------------------------------------


class TestSameBarStopAndT1:
    """When stop and T1 hit in the same bar, conservative logic should call it a stop."""

    def test_put_stop_and_t1_same_bar_conservative_is_stop(self):
        # For a put: T1=168.0, stop=171.0
        # Bar where low <= T1 AND high >= stop simultaneously
        bars = [
            _bar(0, 170.0, 171.5, 167.5, 169.0),  # low hits T1 (167.5<=168), high hits stop (171.5>=171)
        ]
        result = resolve_signal(PUT_SIGNAL, bars, signal_id=100)
        # Same timestamp for both => conservative = STOP_HIT
        assert result.outcome == TradeOutcome.STOP_HIT
        assert result.hit_price == 171.0

    def test_call_stop_and_t1_same_bar_conservative_is_stop(self):
        # For a call: T1=369.5, stop=367.0
        # Bar where high >= T1 AND low <= stop simultaneously
        bars = [
            _bar(0, 368.0, 370.0, 366.5, 368.5),  # high hits T1 (370>=369.5), low hits stop (366.5<=367)
        ]
        result = resolve_signal(CALL_SIGNAL, bars, signal_id=101)
        assert result.outcome == TradeOutcome.STOP_HIT
        assert result.hit_price == 367.0

    def test_t1_before_stop_different_bars(self):
        """T1 hit on earlier bar, stop hit later => T1 wins."""
        bars = [
            _bar(0, 170.0, 170.2, 167.8, 168.0),  # T1 hit (low<=168)
            _bar(1, 168.0, 171.5, 168.0, 171.0),   # stop hit later
        ]
        result = resolve_signal(PUT_SIGNAL, bars, signal_id=102)
        # T1 came first (minute 0 < minute 1)
        assert result.outcome == TradeOutcome.T1_HIT
        assert result.hit_price == 168.0

    def test_stop_before_t1_different_bars(self):
        """Stop hit on earlier bar, T1 hit later => stop wins."""
        bars = [
            _bar(0, 170.0, 171.5, 170.0, 171.2),  # stop hit (high>=171)
            _bar(1, 171.2, 171.2, 167.5, 168.0),    # T1 hit later
        ]
        result = resolve_signal(PUT_SIGNAL, bars, signal_id=103)
        assert result.outcome == TradeOutcome.STOP_HIT
        assert result.hit_price == 171.0


class TestSignalTimestampFiltering:
    """Bars before the signal creation time should be ignored."""

    def test_bars_before_signal_are_ignored(self):
        # Signal created at 09:32, bars at 09:30 and 09:31 should be ignored
        signal = {
            **PUT_SIGNAL,
            "created_at": "2026-03-27T09:32:00",
        }
        bars = [
            _bar(0, 170.0, 170.0, 166.0, 167.0),  # 09:30 - would hit T1/T2 but before signal
            _bar(1, 167.0, 167.5, 166.5, 167.0),   # 09:31 - still before signal
            _bar(2, 170.0, 170.5, 169.0, 169.5),   # 09:32 - after signal, no targets hit
            _bar(3, 169.5, 170.5, 169.0, 170.0),   # 09:33 - no targets hit
        ]
        result = resolve_signal(signal, bars, signal_id=110)
        # Only bars at 09:32 and 09:33 should be considered — neither hits T1/T2/stop
        assert result.outcome == TradeOutcome.EXPIRED

    def test_all_bars_before_signal_returns_unknown(self):
        signal = {
            **PUT_SIGNAL,
            "created_at": "2026-03-27T10:00:00",
        }
        bars = [
            _bar(0, 170.0, 170.0, 166.0, 167.0),  # 09:30
            _bar(1, 167.0, 167.5, 166.5, 167.0),   # 09:31
        ]
        result = resolve_signal(signal, bars, signal_id=111)
        assert result.outcome == TradeOutcome.UNKNOWN

    def test_no_created_at_uses_all_bars(self):
        signal = {
            "direction": "put",
            "entry_price": 170.0,
            "target_1": 168.0,
            "target_2": 167.0,
            "stop_price": 171.0,
            "atm_premium": 1.70,
            "otm_premium": 0.46,
            "ticker": "NVDA",
            # no created_at
        }
        bars = [
            _bar(0, 170.0, 170.2, 167.8, 168.0),  # hits T1
        ]
        result = resolve_signal(signal, bars, signal_id=112)
        assert result.outcome == TradeOutcome.T1_HIT


class TestPnLAccuracy:
    """Test PnL calculation accuracy for both calls and puts."""

    def test_put_t1_pnl_accuracy(self):
        # Entry=170, T1=168 -> pnl = (170-168)/170 * 100 = 1.1765%
        bars = [
            _bar(0, 170.0, 170.2, 167.8, 168.0),
        ]
        result = resolve_signal(PUT_SIGNAL, bars, signal_id=120)
        assert result.outcome == TradeOutcome.T1_HIT
        expected_pnl = (170.0 - 168.0) / 170.0 * 100
        assert abs(result.pnl_underlying_pct - round(expected_pnl, 4)) < 0.01

    def test_put_stop_pnl_accuracy(self):
        # Entry=170, stop=171 -> pnl = (170-171)/170 * 100 = -0.5882%
        bars = [
            _bar(0, 170.0, 171.5, 169.5, 171.0),
        ]
        result = resolve_signal(PUT_SIGNAL, bars, signal_id=121)
        assert result.outcome == TradeOutcome.STOP_HIT
        expected_pnl = (170.0 - 171.0) / 170.0 * 100
        assert abs(result.pnl_underlying_pct - round(expected_pnl, 4)) < 0.01

    def test_call_t2_pnl_accuracy(self):
        # Entry=368, T2=370.5 -> pnl = (370.5-368)/368 * 100 = 0.6793%
        bars = [
            _bar(0, 368.0, 370.8, 367.5, 370.0),  # hits T1 and T2
        ]
        result = resolve_signal(CALL_SIGNAL, bars, signal_id=122)
        assert result.outcome == TradeOutcome.T2_HIT
        expected_pnl = (370.5 - 368.0) / 368.0 * 100
        assert abs(result.pnl_underlying_pct - round(expected_pnl, 4)) < 0.01

    def test_call_stop_pnl_accuracy(self):
        # Entry=368, stop=367 -> pnl = (367-368)/368 * 100 = -0.2717%
        bars = [
            _bar(0, 368.0, 368.2, 366.5, 367.0),
        ]
        result = resolve_signal(CALL_SIGNAL, bars, signal_id=123)
        assert result.outcome == TradeOutcome.STOP_HIT
        expected_pnl = (367.0 - 368.0) / 368.0 * 100
        assert abs(result.pnl_underlying_pct - round(expected_pnl, 4)) < 0.01

    def test_expired_put_uses_last_close(self):
        # Entry=170, last close=169.5 -> pnl = (170-169.5)/170 * 100 = 0.2941%
        bars = [
            _bar(0, 170.0, 170.5, 169.0, 169.5),
            _bar(1, 169.5, 170.5, 169.0, 169.5),
        ]
        result = resolve_signal(PUT_SIGNAL, bars, signal_id=124)
        assert result.outcome == TradeOutcome.EXPIRED
        expected_pnl = (170.0 - 169.5) / 170.0 * 100
        assert abs(result.pnl_underlying_pct - round(expected_pnl, 4)) < 0.01

    def test_expired_call_uses_last_close(self):
        # Entry=368, last close=368.5 -> pnl = (368.5-368)/368 * 100 = 0.1359%
        bars = [
            _bar(0, 368.0, 369.0, 367.5, 368.5),
        ]
        result = resolve_signal(CALL_SIGNAL, bars, signal_id=125)
        assert result.outcome == TradeOutcome.EXPIRED
        expected_pnl = (368.5 - 368.0) / 368.0 * 100
        assert abs(result.pnl_underlying_pct - round(expected_pnl, 4)) < 0.01


class TestExcursionValues:
    """Test that max favorable and adverse excursion values are computed correctly."""

    def test_put_favorable_excursion(self):
        # For put: favorable = (entry - low) / entry * 100
        # entry=170, lowest low=166.0 -> favorable = (170-166)/170*100 = 2.3529%
        bars = [
            _bar(0, 170.0, 170.5, 168.0, 169.0),
            _bar(1, 169.0, 169.5, 166.0, 167.0),  # lowest low
            _bar(2, 167.0, 168.0, 167.0, 167.5),
        ]
        result = resolve_signal(PUT_SIGNAL, bars, signal_id=130)
        expected_mfe = (170.0 - 166.0) / 170.0 * 100
        assert abs(result.max_favorable_pct - round(expected_mfe, 4)) < 0.01

    def test_put_adverse_excursion(self):
        # For put: adverse = (high - entry) / entry * 100
        # entry=170, highest high=172.0 -> adverse = (172-170)/170*100 = 1.1765%
        bars = [
            _bar(0, 170.0, 172.0, 169.5, 170.5),  # highest high
            _bar(1, 170.5, 171.0, 167.5, 168.0),
        ]
        result = resolve_signal(PUT_SIGNAL, bars, signal_id=131)
        expected_mae = (172.0 - 170.0) / 170.0 * 100
        assert abs(result.max_adverse_pct - round(expected_mae, 4)) < 0.01

    def test_call_favorable_excursion(self):
        # For call: favorable = (high - entry) / entry * 100
        # entry=368, highest high=372.0 -> favorable = (372-368)/368*100 = 1.0870%
        bars = [
            _bar(0, 368.0, 372.0, 367.5, 371.0),  # highest high
            _bar(1, 371.0, 371.5, 369.0, 370.0),
        ]
        result = resolve_signal(CALL_SIGNAL, bars, signal_id=132)
        expected_mfe = (372.0 - 368.0) / 368.0 * 100
        assert abs(result.max_favorable_pct - round(expected_mfe, 4)) < 0.01

    def test_call_adverse_excursion(self):
        # For call: adverse = (entry - low) / entry * 100
        # entry=368, lowest low=365.0 -> adverse = (368-365)/368*100 = 0.8152%
        bars = [
            _bar(0, 368.0, 369.0, 365.0, 367.5),  # lowest low
            _bar(1, 367.5, 369.5, 367.0, 369.0),
        ]
        result = resolve_signal(CALL_SIGNAL, bars, signal_id=133)
        expected_mae = (368.0 - 365.0) / 368.0 * 100
        assert abs(result.max_adverse_pct - round(expected_mae, 4)) < 0.01

    def test_atm_option_pnl_for_stop(self):
        """ATM option PnL should be negative for stop hits."""
        bars = [
            _bar(0, 170.0, 171.5, 169.5, 171.0),
        ]
        result = resolve_signal(PUT_SIGNAL, bars, signal_id=134)
        assert result.outcome == TradeOutcome.STOP_HIT
        assert result.pnl_atm_est is not None
        assert result.pnl_atm_est < 0

    def test_otm_option_pnl_for_winner(self):
        """OTM option PnL should be positive for T1/T2 hits."""
        bars = [
            _bar(0, 170.0, 170.2, 167.8, 168.0),
        ]
        result = resolve_signal(PUT_SIGNAL, bars, signal_id=135)
        assert result.outcome == TradeOutcome.T1_HIT
        assert result.pnl_otm_est is not None
        assert result.pnl_otm_est > 0
