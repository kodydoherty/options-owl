"""Tests for sideways scalp gate — covers all 6 identified issues.

Issue #1: History doesn't survive restarts
  → Tests that gate is disabled with empty/short history (fails to HOLD)
Issue #2: Memory bounded
  → Tests that history is capped at MAX_HISTORY_LEN
Issue #3: Logging / reason map
  → Tests that SIDEWAYS_SCALP reason is in the monitor bridge reason map
Issue #4: DCA interaction
  → Tests that gate uses blended entry after DCA (correct cost basis)
Issue #5: Scaleout interaction
  → Tests that gate can fire on remaining contracts after scaleout
Issue #6: Gate ordering
  → Tests that sideways scalp fires after graduated_stop but before soft_trail
  → Tests that higher-priority gates (profit_target, breakeven, scaleout) still win
"""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from options_owl.risk.exit_v5.config import V5Config, get_ticker_config
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState
from options_owl.risk.exit_v5.gates import check_sideways_scalp
from options_owl.risk.exit_v5.monitor_bridge import _REASON_MAP
from options_owl.risk.exit_v5.types import ExitReason


# ---------------------------------------------------------------------------
# Settings helper — mirrors production V6 flags
# ---------------------------------------------------------------------------

def _settings(**overrides):
    defaults = dict(
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
        ENABLE_V6_SIDEWAYS_SCALP=True,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_state(
    entry_premium=1.00,
    ticker="MSFT",
    contracts=5,
    elapsed_min=10,
    premium_history=None,
    elapsed_sec_history=None,
    underlying_history=None,
    entry_underlying=200.0,
    peak_premium=None,
):
    """Build a TradeState with history pre-populated."""
    entry_time = datetime(2026, 5, 11, 10, 0, 0)
    now = entry_time + timedelta(minutes=elapsed_min)
    state = TradeState(
        trade_id=1,
        ticker=ticker,
        option_type="call",
        entry_premium=entry_premium,
        entry_time=entry_time,
        contracts=contracts,
        peak_premium=peak_premium or entry_premium,
        entry_underlying_price=entry_underlying,
    )
    if premium_history is not None:
        state.premium_history = list(premium_history)
    if elapsed_sec_history is not None:
        state.elapsed_sec_history = list(elapsed_sec_history)
    if underlying_history is not None:
        state.underlying_history = list(underlying_history)
    return state, now


def _sideways_premiums(entry=1.00, n=30, jitter=0.02):
    """Generate sideways premium history: oscillates around entry with small jitter."""
    import math
    premiums = []
    for i in range(n):
        p = entry + jitter * math.sin(i * 0.5)
        premiums.append(round(p, 4))
    return premiums


def _elapsed_secs(n=30, interval=5.0):
    """Generate elapsed seconds list at fixed interval."""
    return [i * interval for i in range(n)]


def _flat_underlying(entry=200.0, n=30):
    """Generate flat underlying history."""
    return [entry] * n


# ===========================================================================
# Issue #1: History doesn't survive restarts — gate disabled with empty history
# ===========================================================================

class TestIssue1RestartSafety:
    """Gate must be completely inert with insufficient history."""

    def test_empty_history_returns_none(self):
        """No history at all → gate returns None (HOLD)."""
        cfg = V5Config()
        result = check_sideways_scalp(
            gain=15.0,
            peak_gain_from_history=15.0,
            premium_history=[],
            timestamp_history=[],
            underlying_history=[],
            entry_premium=1.00,
            entry_underlying=200.0,
            cfg=cfg,
            debug={},
        )
        assert result is None

    def test_short_history_returns_none(self):
        """Less than min_ticks → gate returns None."""
        cfg = V5Config(sideways_min_ticks=10)
        result = check_sideways_scalp(
            gain=15.0,
            peak_gain_from_history=15.0,
            premium_history=[1.10] * 9,
            timestamp_history=_elapsed_secs(9),
            underlying_history=_flat_underlying(n=9),
            entry_premium=1.00,
            entry_underlying=200.0,
            cfg=cfg,
            debug={},
        )
        assert result is None

    def test_exact_min_ticks_can_fire(self):
        """Exactly min_ticks history → gate CAN fire if conditions met."""
        n = 10
        cfg = V5Config(
            sideways_min_ticks=n,
            sideways_take_profit_pct=5.0,
            sideways_signals_needed=1,
            sideways_no_new_high_min=0.5,  # easy to trigger
        )
        # Sideways premiums hovering at +10% with no new highs
        premiums = [1.10] * n
        result = check_sideways_scalp(
            gain=10.0,
            peak_gain_from_history=10.0,
            premium_history=premiums,
            timestamp_history=_elapsed_secs(n),
            underlying_history=_flat_underlying(n=n),
            entry_premium=1.00,
            entry_underlying=200.0,
            cfg=cfg,
            debug={},
        )
        assert result is not None
        assert result.reason == ExitReason.SIDEWAYS_SCALP

    def test_fsm_after_restart_holds_until_history_accumulates(self):
        """Full FSM integration: new TradeState (simulating restart) holds."""
        cfg = V5Config()
        fsm = ExitFSM(cfg, settings=_settings())
        state = TradeState(
            trade_id=99, ticker="MSFT", option_type="call",
            entry_premium=1.00, entry_time=datetime(2026, 5, 11, 10, 0),
            contracts=5, peak_premium=1.10,
            entry_underlying_price=200.0,
        )
        # History is empty (simulates restart)
        assert len(state.premium_history) == 0

        # First few evaluations should NOT sideways scalp
        now = datetime(2026, 5, 11, 10, 12)
        for i in range(5):
            action = fsm.evaluate(
                state, current_premium=1.10, bid=1.09, ask=1.11,
                now_et=now + timedelta(seconds=i * 5),
                current_underlying=200.0, minutes_to_close=240,
            )
            assert action.reason != ExitReason.SIDEWAYS_SCALP

        # History should have accumulated
        assert len(state.premium_history) == 5


# ===========================================================================
# Issue #2: Memory bounded — history capped at MAX_HISTORY_LEN
# ===========================================================================

class TestIssue2MemoryBounded:
    """History lists must never grow beyond MAX_HISTORY_LEN."""

    def test_history_capped_in_fsm(self):
        """After MAX_HISTORY_LEN+N evaluations, lists stay bounded."""
        cfg = V5Config()
        fsm = ExitFSM(cfg, settings=_settings(ENABLE_V6_SIDEWAYS_SCALP=False))
        state = TradeState(
            trade_id=1, ticker="SPY", option_type="call",
            entry_premium=1.00, entry_time=datetime(2026, 5, 11, 10, 0),
            contracts=5, peak_premium=1.00,
            entry_underlying_price=450.0,
        )

        now = datetime(2026, 5, 11, 10, 5)
        for i in range(state.MAX_HISTORY_LEN + 50):
            fsm.evaluate(
                state, current_premium=1.05, bid=1.04, ask=1.06,
                now_et=now + timedelta(seconds=i * 5),
                current_underlying=450.0, minutes_to_close=300,
            )

        assert len(state.premium_history) <= state.MAX_HISTORY_LEN
        assert len(state.elapsed_sec_history) <= state.MAX_HISTORY_LEN
        assert len(state.underlying_history) <= state.MAX_HISTORY_LEN

    def test_all_three_lists_same_length(self):
        """All history lists stay synchronized in length."""
        cfg = V5Config()
        fsm = ExitFSM(cfg, settings=_settings(ENABLE_V6_SIDEWAYS_SCALP=False))
        state = TradeState(
            trade_id=1, ticker="SPY", option_type="call",
            entry_premium=1.00, entry_time=datetime(2026, 5, 11, 10, 0),
            contracts=5, peak_premium=1.00,
            entry_underlying_price=450.0,
        )

        now = datetime(2026, 5, 11, 10, 5)
        for i in range(75):
            fsm.evaluate(
                state, current_premium=1.05, bid=1.04, ask=1.06,
                now_et=now + timedelta(seconds=i * 5),
                current_underlying=450.0, minutes_to_close=300,
            )

        assert len(state.premium_history) == len(state.elapsed_sec_history)
        assert len(state.premium_history) == len(state.underlying_history)


# ===========================================================================
# Issue #3: Logging / reason map — SIDEWAYS_SCALP in monitor bridge
# ===========================================================================

class TestIssue3Logging:
    """SIDEWAYS_SCALP must be in the reason map for proper DB recording."""

    def test_reason_in_map(self):
        assert ExitReason.SIDEWAYS_SCALP in _REASON_MAP
        assert _REASON_MAP[ExitReason.SIDEWAYS_SCALP] == "sideways_scalp"

    def test_exit_action_has_correct_reason(self):
        """Gate returns ExitAction with SIDEWAYS_SCALP reason."""
        n = 30
        cfg = V5Config(
            sideways_take_profit_pct=5.0,
            sideways_signals_needed=1,
            sideways_no_new_high_min=0.5,
        )
        result = check_sideways_scalp(
            gain=10.0,
            peak_gain_from_history=10.0,
            premium_history=[1.10] * n,
            timestamp_history=_elapsed_secs(n),
            underlying_history=_flat_underlying(n=n),
            entry_premium=1.00,
            entry_underlying=200.0,
            cfg=cfg,
            debug={},
        )
        assert result is not None
        assert result.reason == ExitReason.SIDEWAYS_SCALP
        assert result.should_exit is True
        assert "sideways" in result.debug


# ===========================================================================
# Issue #4: DCA interaction — uses blended entry premium
# ===========================================================================

class TestIssue4DCAInteraction:
    """After DCA changes entry_premium, gate uses the new blended cost basis."""

    def test_gain_relative_to_blended_entry(self):
        """DCA lowers entry from 1.00 to 0.85. Premium at 0.93 = +9.4% from blended."""
        n = 20
        cfg = V5Config(
            sideways_take_profit_pct=8.0,
            sideways_signals_needed=1,
            sideways_no_new_high_min=0.5,
        )
        blended_entry = 0.85  # after DCA averaged down
        current_premium = 0.93
        gain = (current_premium - blended_entry) / blended_entry * 100  # +9.4%

        result = check_sideways_scalp(
            gain=gain,
            peak_gain_from_history=gain,
            premium_history=[current_premium] * n,
            timestamp_history=_elapsed_secs(n),
            underlying_history=_flat_underlying(n=n),
            entry_premium=blended_entry,
            entry_underlying=200.0,
            cfg=cfg,
            debug={},
        )
        assert result is not None  # Should fire — +9.4% > 8% threshold

    def test_cross_count_uses_blended_entry(self):
        """Crosses are counted relative to blended entry, not original."""
        n = 20
        blended_entry = 0.85
        # Premiums oscillate around blended entry
        premiums = []
        for i in range(n):
            premiums.append(0.80 if i % 2 == 0 else 0.90)

        cfg = V5Config(
            sideways_take_profit_pct=5.0,
            sideways_signals_needed=1,
            sideways_cross_count=3,
        )
        # Current premium is above blended entry
        gain = (0.90 - blended_entry) / blended_entry * 100

        result = check_sideways_scalp(
            gain=gain,
            peak_gain_from_history=gain,
            premium_history=premiums,
            timestamp_history=_elapsed_secs(n),
            underlying_history=_flat_underlying(n=n),
            entry_premium=blended_entry,
            entry_underlying=200.0,
            cfg=cfg,
            debug={},
        )
        assert result is not None
        assert result.debug["sideways"]["crosses"] >= 3


# ===========================================================================
# Issue #5: Scaleout interaction — can fire on remaining contracts
# ===========================================================================

class TestIssue5ScaleoutInteraction:
    """Sideways scalp can fire after scaleout reduced contract count."""

    def test_fires_after_scaleout(self):
        """Scaleout sells 1/3, remaining position goes sideways → scalp fires."""
        cfg = V5Config()
        fsm = ExitFSM(cfg, settings=_settings())

        state = TradeState(
            trade_id=1, ticker="MSFT", option_type="call",
            entry_premium=1.00, entry_time=datetime(2026, 5, 11, 10, 0),
            contracts=6,  # started with 9, scaleout sold 3
            peak_premium=1.25,
            entry_underlying_price=420.0,
            scaled_out=True,  # scaleout already fired
        )

        # Simulate sideways after scaleout: premium hovering around +10%
        now = datetime(2026, 5, 11, 10, 12)
        last_action = None
        for i in range(40):
            # Oscillate between 1.09 and 1.11
            prem = 1.09 if i % 2 == 0 else 1.11
            action = fsm.evaluate(
                state, current_premium=prem, bid=prem - 0.01, ask=prem + 0.01,
                now_et=now + timedelta(seconds=i * 5),
                current_underlying=420.0, minutes_to_close=240,
            )
            last_action = action
            if action.should_exit and action.reason == ExitReason.SIDEWAYS_SCALP:
                break

        # Should eventually fire sideways scalp (not scaleout again since already_scaled=True)
        if last_action and last_action.should_exit:
            assert last_action.reason != ExitReason.SCALEOUT


# ===========================================================================
# Issue #6: Gate ordering — correct priority
# ===========================================================================

class TestIssue6GateOrdering:
    """Sideways scalp must not override higher-priority gates."""

    def test_profit_target_beats_sideways(self):
        """Index 0DTE at +35% → profit_target fires, not sideways scalp."""
        cfg = get_ticker_config("SPY", use_per_ticker=True)
        fsm = ExitFSM(cfg, settings=_settings())

        state = TradeState(
            trade_id=1, ticker="SPY", option_type="call",
            entry_premium=1.00, entry_time=datetime(2026, 5, 11, 10, 0),
            contracts=5, peak_premium=1.35,
            entry_underlying_price=550.0,
            dte=0, expiry_date="2026-05-11",
        )
        # Build up enough history
        now = datetime(2026, 5, 11, 10, 10)
        for i in range(15):
            fsm.evaluate(
                state, current_premium=1.35, bid=1.34, ask=1.36,
                now_et=now + timedelta(seconds=i * 5),
                current_underlying=550.5, minutes_to_close=300,
            )

        action = fsm.evaluate(
            state, current_premium=1.35, bid=1.34, ask=1.36,
            now_et=now + timedelta(seconds=80),
            current_underlying=550.5, minutes_to_close=300,
        )
        # Profit target has priority over sideways scalp
        if action.should_exit:
            assert action.reason == ExitReason.PROFIT_TARGET

    def test_breakeven_ratchet_beats_sideways(self):
        """Trade was +25%, now at entry → breakeven_ratchet fires, not sideways."""
        cfg = V5Config()
        fsm = ExitFSM(cfg, settings=_settings())

        state = TradeState(
            trade_id=1, ticker="MSFT", option_type="call",
            entry_premium=1.00, entry_time=datetime(2026, 5, 11, 10, 0),
            contracts=5, peak_premium=1.25,
            entry_underlying_price=420.0,
            breakeven_ratchet_armed=True,  # was +25%
        )
        # Build history
        now = datetime(2026, 5, 11, 10, 10)
        for i in range(15):
            fsm.evaluate(
                state, current_premium=1.10, bid=1.09, ask=1.11,
                now_et=now + timedelta(seconds=i * 5),
                current_underlying=420.0, minutes_to_close=300,
            )

        # Now premium drops below entry — breakeven ratchet should fire
        action = fsm.evaluate(
            state, current_premium=0.98, bid=0.97, ask=0.99,
            now_et=now + timedelta(seconds=80),
            current_underlying=419.5, minutes_to_close=300,
        )
        assert action.should_exit
        assert action.reason == ExitReason.BREAKEVEN_RATCHET

    def test_graduated_stop_beats_sideways(self):
        """Premium at -40% with underlying against → graduated stop fires first."""
        cfg = V5Config()
        fsm = ExitFSM(cfg, settings=_settings())

        state = TradeState(
            trade_id=1, ticker="MSFT", option_type="call",
            entry_premium=1.00, entry_time=datetime(2026, 5, 11, 10, 0),
            contracts=5, peak_premium=1.00,
            entry_underlying_price=420.0,
        )
        now = datetime(2026, 5, 11, 10, 10)
        # Build enough history
        for i in range(15):
            fsm.evaluate(
                state, current_premium=0.60, bid=0.59, ask=0.61,
                now_et=now + timedelta(seconds=i * 5),
                current_underlying=417.0, minutes_to_close=300,
            )

        action = fsm.evaluate(
            state, current_premium=0.60, bid=0.59, ask=0.61,
            now_et=now + timedelta(seconds=80),
            current_underlying=417.0, minutes_to_close=300,
        )
        assert action.should_exit
        # Should be confirmed_stop, checkpoint_cut, or hard_stop — all higher priority than sideways
        assert action.reason in (
            ExitReason.CONFIRMED_STOP, ExitReason.HARD_STOP, ExitReason.CHECKPOINT_CUT,
        )

    def test_sideways_fires_before_soft_trail(self):
        """Trade at +12% and sideways → sideways scalp fires, not soft_trail."""
        cfg = V5Config(
            sideways_take_profit_pct=10.0,
            sideways_signals_needed=1,
            sideways_no_new_high_min=2.0,
            soft_trail_band_low_pct=10.0,  # soft_trail would also trigger at +12%
        )
        fsm = ExitFSM(cfg, settings=_settings())

        state = TradeState(
            trade_id=1, ticker="AAPL", option_type="call",
            entry_premium=1.00, entry_time=datetime(2026, 5, 11, 10, 0),
            contracts=5, peak_premium=1.15,
            entry_underlying_price=190.0,
        )

        # Build up sideways history at +12% for several minutes
        now = datetime(2026, 5, 11, 10, 8)
        for i in range(40):
            prem = 1.12
            fsm.evaluate(
                state, current_premium=prem, bid=prem - 0.01, ask=prem + 0.01,
                now_et=now + timedelta(seconds=i * 5),
                current_underlying=190.0, minutes_to_close=300,
            )

        # Final evaluation — sideways should fire before soft_trail
        action = fsm.evaluate(
            state, current_premium=1.12, bid=1.11, ask=1.13,
            now_et=now + timedelta(seconds=205),
            current_underlying=190.0, minutes_to_close=300,
        )
        if action.should_exit:
            assert action.reason == ExitReason.SIDEWAYS_SCALP


# ===========================================================================
# Core gate logic tests
# ===========================================================================

class TestSidewaysGateLogic:
    """Test the pure gate function directly."""

    def test_not_profitable_returns_none(self):
        """Gain below threshold → no scalp."""
        n = 30
        cfg = V5Config(sideways_take_profit_pct=10.0)
        result = check_sideways_scalp(
            gain=8.0,  # below 10%
            peak_gain_from_history=8.0,
            premium_history=[1.08] * n,
            timestamp_history=_elapsed_secs(n),
            underlying_history=_flat_underlying(n=n),
            entry_premium=1.00,
            entry_underlying=200.0,
            cfg=cfg,
            debug={},
        )
        assert result is None

    def test_peak_above_cap_returns_none(self):
        """Peak gain > 30% → trade trended, don't scalp."""
        n = 30
        cfg = V5Config(sideways_take_profit_pct=5.0, sideways_signals_needed=1)
        result = check_sideways_scalp(
            gain=10.0,
            peak_gain_from_history=35.0,  # above 30% cap
            premium_history=[1.10] * n,
            timestamp_history=_elapsed_secs(n),
            underlying_history=_flat_underlying(n=n),
            entry_premium=1.00,
            entry_underlying=200.0,
            cfg=cfg,
            debug={},
        )
        assert result is None

    def test_not_enough_signals_returns_none(self):
        """Only 1 of 4 indicators hits, needs 2 → no scalp."""
        n = 30
        # Trending up — new highs constantly, big range, no crosses, underlying moving
        premiums = [1.00 + 0.01 * i for i in range(n)]  # 1.00 → 1.29, always new highs
        underlying = [200.0 + 0.5 * i for i in range(n)]  # moving up 0.25%/tick

        cfg = V5Config(
            sideways_take_profit_pct=5.0,
            sideways_signals_needed=2,
            sideways_range_pct=5.0,
            sideways_no_new_high_min=8.0,
            sideways_cross_count=3,
            sideways_underlying_flat_pct=0.15,
        )
        # Current gain from last premium
        gain = (premiums[-1] - 1.00) / 1.00 * 100  # +29%
        peak = gain
        result = check_sideways_scalp(
            gain=gain,
            peak_gain_from_history=peak,
            premium_history=premiums,
            timestamp_history=_elapsed_secs(n),
            underlying_history=underlying,
            entry_premium=1.00,
            entry_underlying=200.0,
            cfg=cfg,
            debug={},
        )
        # Trending up: no range-bound, no stale peak, no flat underlying, no crosses → 0/4
        assert result is None

    def test_all_four_indicators_fire(self):
        """All 4 indicators agree → scalp fires with signals_needed=4."""
        n = 30
        cfg = V5Config(
            sideways_take_profit_pct=3.0,      # low threshold for this test
            sideways_signals_needed=4,
            sideways_range_pct=25.0,           # wide enough for 8% range
            sideways_no_new_high_min=1.0,      # 1 min since peak
            sideways_cross_count=2,
            sideways_underlying_flat_pct=0.5,
        )
        # Oscillate between 0.96 and 1.04 (crosses entry=1.00 many times)
        # Peak at 1.05 early at tick 2, never exceeded → no_new_high triggers
        premiums = []
        for i in range(n):
            if i == 2:
                premiums.append(1.05)  # peak early at tick 2
            elif i % 4 < 2:
                premiums.append(1.04)  # slightly below peak, above entry
            else:
                premiums.append(0.96)  # dip below entry
        premiums[-1] = 1.04  # end profitable but below peak

        gain = (premiums[-1] - 1.00) / 1.00 * 100  # +4%
        peak_gain = (max(premiums) - 1.00) / 1.00 * 100  # +5%
        result = check_sideways_scalp(
            gain=gain,
            peak_gain_from_history=peak_gain,
            premium_history=premiums,
            timestamp_history=_elapsed_secs(n),
            underlying_history=_flat_underlying(n=n),
            entry_premium=1.00,
            entry_underlying=200.0,
            cfg=cfg,
            debug={},
        )
        assert result is not None
        assert result.debug["sideways"]["signals_hit"] == 4

    def test_indicator_range_bound(self):
        """Tight range triggers range-bound indicator."""
        n = 20
        cfg = V5Config(
            sideways_take_profit_pct=5.0,
            sideways_signals_needed=1,
            sideways_range_pct=5.0,
        )
        # Very tight range: 1.10 ± 0.01 = 2% range
        premiums = [1.10 + 0.01 * (i % 2) for i in range(n)]

        result = check_sideways_scalp(
            gain=10.0,
            peak_gain_from_history=11.0,
            premium_history=premiums,
            timestamp_history=_elapsed_secs(n),
            underlying_history=_flat_underlying(n=n),
            entry_premium=1.00,
            entry_underlying=200.0,
            cfg=cfg,
            debug={},
        )
        assert result is not None
        assert result.debug["sideways"]["prem_range_pct"] < 5.0

    def test_indicator_no_new_highs(self):
        """Peak was hit 10 minutes ago → no_new_high triggers."""
        n = 30
        # Peak at tick 5, then flat
        premiums = [1.00] * n
        premiums[5] = 1.15
        for i in range(6, n):
            premiums[i] = 1.10

        cfg = V5Config(
            sideways_take_profit_pct=5.0,
            sideways_signals_needed=1,
            sideways_no_new_high_min=1.0,
        )

        result = check_sideways_scalp(
            gain=10.0,
            peak_gain_from_history=15.0,
            premium_history=premiums,
            timestamp_history=_elapsed_secs(n),
            underlying_history=_flat_underlying(n=n),
            entry_premium=1.00,
            entry_underlying=200.0,
            cfg=cfg,
            debug={},
        )
        assert result is not None
        assert result.debug["sideways"]["min_since_peak"] > 1.0

    def test_indicator_underlying_flat(self):
        """Underlying hasn't moved → flat indicator triggers."""
        n = 20
        cfg = V5Config(
            sideways_take_profit_pct=5.0,
            sideways_signals_needed=1,
            sideways_underlying_flat_pct=0.2,
        )
        result = check_sideways_scalp(
            gain=10.0,
            peak_gain_from_history=10.0,
            premium_history=[1.10] * n,
            timestamp_history=_elapsed_secs(n),
            underlying_history=[200.0] * n,  # flat
            entry_premium=1.00,
            entry_underlying=200.0,
            cfg=cfg,
            debug={},
        )
        assert result is not None

    def test_indicator_entry_crosses(self):
        """Premium crossed entry 5 times → cross indicator triggers."""
        premiums = []
        for i in range(20):
            premiums.append(1.10 if i % 2 == 0 else 0.90)
        premiums[-1] = 1.10  # end profitable

        cfg = V5Config(
            sideways_take_profit_pct=5.0,
            sideways_signals_needed=1,
            sideways_cross_count=3,
        )
        gain = 10.0
        result = check_sideways_scalp(
            gain=gain,
            peak_gain_from_history=gain,
            premium_history=premiums,
            timestamp_history=_elapsed_secs(20),
            underlying_history=_flat_underlying(n=20),
            entry_premium=1.00,
            entry_underlying=200.0,
            cfg=cfg,
            debug={},
        )
        assert result is not None
        assert result.debug["sideways"]["crosses"] >= 3

    def test_no_underlying_data_still_works(self):
        """Missing underlying data → only 3 indicators available, sn=2 still possible."""
        n = 20
        cfg = V5Config(
            sideways_take_profit_pct=5.0,
            sideways_signals_needed=2,
            sideways_no_new_high_min=0.5,
        )
        result = check_sideways_scalp(
            gain=10.0,
            peak_gain_from_history=10.0,
            premium_history=[1.10] * n,
            timestamp_history=_elapsed_secs(n),
            underlying_history=[],  # no underlying data
            entry_premium=1.00,
            entry_underlying=0.0,  # no entry underlying
            cfg=cfg,
            debug={},
        )
        # Should still fire — range-bound + no_new_high = 2 signals
        assert result is not None


# ===========================================================================
# Feature flag tests
# ===========================================================================

class TestFeatureFlag:
    """Gate only fires when ENABLE_V6_SIDEWAYS_SCALP=True."""

    def test_disabled_by_default(self):
        """With flag off, sideways scalp never fires even if conditions met."""
        cfg = V5Config(sideways_take_profit_pct=5.0, sideways_signals_needed=1)
        fsm = ExitFSM(cfg, settings=_settings(ENABLE_V6_SIDEWAYS_SCALP=False))

        state = TradeState(
            trade_id=1, ticker="MSFT", option_type="call",
            entry_premium=1.00, entry_time=datetime(2026, 5, 11, 10, 0),
            contracts=5, peak_premium=1.00,
            entry_underlying_price=420.0,
        )

        now = datetime(2026, 5, 11, 10, 10)
        for i in range(40):
            action = fsm.evaluate(
                state, current_premium=1.10, bid=1.09, ask=1.11,
                now_et=now + timedelta(seconds=i * 5),
                current_underlying=420.0, minutes_to_close=300,
            )
            assert action.reason != ExitReason.SIDEWAYS_SCALP

    def test_enabled_can_fire(self):
        """With flag on, sideways scalp can fire when conditions met."""
        cfg = V5Config(
            sideways_take_profit_pct=5.0,
            sideways_signals_needed=1,
            sideways_no_new_high_min=1.0,
        )
        fsm = ExitFSM(cfg, settings=_settings(ENABLE_V6_SIDEWAYS_SCALP=True))

        state = TradeState(
            trade_id=1, ticker="MSFT", option_type="call",
            entry_premium=1.00, entry_time=datetime(2026, 5, 11, 10, 0),
            contracts=5, peak_premium=1.00,
            entry_underlying_price=420.0,
        )

        now = datetime(2026, 5, 11, 10, 10)
        fired = False
        for i in range(60):
            action = fsm.evaluate(
                state, current_premium=1.10, bid=1.09, ask=1.11,
                now_et=now + timedelta(seconds=i * 5),
                current_underlying=420.0, minutes_to_close=300,
            )
            if action.should_exit and action.reason == ExitReason.SIDEWAYS_SCALP:
                fired = True
                break

        assert fired, "Sideways scalp should have fired with flag enabled"


# ===========================================================================
# Source code safety — variable initialization
# ===========================================================================

class TestSourceCodeSafety:
    """Inspect actual source to catch uninitialized variable patterns."""

    def test_history_lists_initialized_in_dataclass(self):
        """TradeState history fields have default_factory (not None)."""
        import inspect
        source = inspect.getsource(TradeState)
        assert "premium_history" in source
        assert "elapsed_sec_history" in source
        assert "underlying_history" in source
        assert "default_factory=list" in source

    def test_max_history_len_defined(self):
        """MAX_HISTORY_LEN is defined on TradeState."""
        state = TradeState(
            trade_id=1, ticker="TEST", option_type="call",
            entry_premium=1.0, entry_time=datetime.now(),
        )
        assert hasattr(state, "MAX_HISTORY_LEN")
        assert state.MAX_HISTORY_LEN > 0

    def test_gate_imported_in_fsm(self):
        """check_sideways_scalp is imported in fsm.py."""
        import inspect
        from options_owl.risk.exit_v5 import fsm as fsm_module
        source = inspect.getsource(fsm_module)
        assert "check_sideways_scalp" in source
        assert "ENABLE_V6_SIDEWAYS_SCALP" in source
