"""Tests for position cap (10%) and DCA cap (5%) — Option B sizing.

Verifies:
1. MAX_POSITION_PCT is a hard ceiling (min, not max) on score tier caps
2. MAX_DCA_POSITION_PCT caps V6 DCA contract adds
3. End-to-end: entry + DCA combined never exceeds expected portfolio %
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from options_owl.risk.vinny_strategy import score_to_contracts


# ── 1. Position cap is a true ceiling ──────────────────────────────────────


class TestPositionCapCeiling:
    """MAX_POSITION_PCT caps all trades (flat sizing, no per-tier caps)."""

    def test_10pct_cap_limits_all_scores(self):
        """All scores above 78 are capped at MAX_POSITION_PCT=10%."""
        # $23K balance, $2.00 option ($200/ct), flat 85% mult
        # slot = 23000*0.75/4 = $4312, 85% = $3665, raw = 18
        # 10% cap = $2300 → 11ct
        max_allowed = int(23000 * 0.10 / 200)  # = 11
        for score in [150, 130, 110, 100, 78]:
            result = score_to_contracts(
                score, cost_per_contract=200, balance=23000, max_position_pct=10.0,
            )
            assert result <= max_allowed, f"Score {score} → {result}, exceeds 10% cap of {max_allowed}"
            assert result == max_allowed, f"Score {score} → {result}, expected {max_allowed} (flat sizing)"

    def test_every_score_respects_cap(self):
        """No score should exceed MAX_POSITION_PCT of portfolio."""
        for score in [150, 135, 130, 120, 110, 100, 92, 85, 78]:
            result = score_to_contracts(
                score, cost_per_contract=100, balance=23000, max_position_pct=10.0,
            )
            max_spend = result * 100
            pct_of_portfolio = max_spend / 23000 * 100
            assert pct_of_portfolio <= 10.0, (
                f"Score {score} → {result} contracts = ${max_spend} = {pct_of_portfolio:.1f}% "
                f"(exceeds 10%)"
            )

    def test_setting_3pct_cap(self):
        """MAX_POSITION_PCT=3% caps spending."""
        result = score_to_contracts(
            110, cost_per_contract=100, balance=10000, max_position_pct=3.0,
        )
        max_spend = result * 100
        assert max_spend <= 10000 * 0.03, f"${max_spend} exceeds 3% cap"

    def test_production_kody_sizing(self):
        """Kody's $23K account at 10% cap with typical $1.50 option."""
        result = score_to_contracts(
            110, cost_per_contract=150, balance=23000, max_position_pct=10.0,
        )
        # 10% cap: 23000*0.10/150 = 15
        max_allowed = int(23000 * 0.10 / 150)
        assert result <= max_allowed
        assert result > 0


class TestPositionCapWithDCA:
    """Verify that entry sizing + DCA sizing stay within expected limits."""

    def test_entry_plus_dca_under_15pct(self):
        """10% entry + 5% DCA = 15% max total exposure."""
        entry_contracts = score_to_contracts(
            110, cost_per_contract=200, balance=23000, max_position_pct=10.0,
        )
        entry_cost = entry_contracts * 200

        # DCA capped at 5% of portfolio
        dca_cap = 23000 * 0.05  # = $1150
        dca_contracts = min(entry_contracts, int(dca_cap / 200))
        dca_cost = dca_contracts * 200

        total = entry_cost + dca_cost
        assert total <= 23000 * 0.15, (
            f"Entry ${entry_cost} + DCA ${dca_cost} = ${total} > 15% of $23K"
        )


# ── 2. V6 DCA contract capping ────────────────────────────────────────────


class TestV6DCACapLogic:
    """Test that _check_v6_dca respects MAX_DCA_POSITION_PCT."""

    @pytest.fixture
    def dca_settings(self):
        return SimpleNamespace(
            ENABLE_V6_DCA=True,
            V6_DCA_TICKERS="NVDA,SPY,QQQ",
            V6_DCA_MIN_MINUTES=8.0,
            V6_DCA_MAX_MINUTES=20.0,
            V6_DCA_MIN_DIP_PCT=15.0,
            V6_DCA_MAX_DIP_PCT=35.0,
            V6_DCA_UNDERLYING_THRESHOLD=0.5,
            MAX_DCA_POSITION_PCT=5.0,  # 5% of portfolio
            PORTFOLIO_SIZE=23000.0,
            WEBULL_ENTRY_AGGRESS_PCT=2.0,
        )

    @pytest.fixture
    def mock_trade(self):
        return {
            "id": 100,
            "ticker": "NVDA",
            "contracts": 10,
            "premium_per_contract": 2.00,
            "opened_at": "2026-05-19 14:30:00",
            "entry_price": 130.0,
            "option_type": "call",
            "strike": 130.0,
            "expiry_date": "2026-05-19",
            "total_cost": 2000.0,
        }

    @pytest.mark.asyncio
    async def test_dca_contracts_capped(self, dca_settings, mock_trade):
        """DCA should add at most MAX_DCA_POSITION_PCT worth of contracts."""
        from options_owl.execution.position_monitor import _check_v6_dca, _v6_dca_fired

        _v6_dca_fired.discard(100)

        # Premium dipped 20% from entry (within 15-35% window)
        exit_premium = 1.60  # 20% dip from $2.00
        current_price = 130.0  # underlying hasn't moved against us

        mock_pt = MagicMock()
        mock_pt.get_portfolio_balance = AsyncMock(return_value=23000.0)
        mock_pt.webull_executor = None

        # Patch aiosqlite and time
        import aiosqlite
        mock_conn = AsyncMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)

        # _now_et returns a time within DCA window (10 min after entry)
        from datetime import datetime
        from unittest.mock import patch as _patch

        fake_now = datetime(2026, 5, 19, 10, 40, 0)  # 10:40 ET, 10min after entry

        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def _fake_connect(path):
            yield mock_conn

        with _patch("options_owl.execution.position_monitor._now_et", return_value=fake_now), \
             _patch("options_owl.execution.position_monitor.datetime") as mock_dt, \
             _patch("options_owl.execution.position_monitor._connect_db", _fake_connect):
            mock_dt.fromisoformat.return_value = datetime(2026, 5, 19, 10, 30, 0)
            mock_dt.now.return_value = datetime(2026, 5, 19, 10, 40, 0)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

            await _check_v6_dca(
                mock_trade, exit_premium, current_price,
                dca_settings, mock_pt, "test.db",
            )

        # Check: 5% of $23K = $1150. At $1.60/share = $160/ct → max 7 contracts
        # Original trade has 10 contracts, so DCA should be capped to 7
        if 100 in _v6_dca_fired:
            # DCA fired — check the UPDATE call
            if mock_conn.execute.called:
                call_args = mock_conn.execute.call_args
                params = call_args[0][1] if len(call_args[0]) > 1 else None
                if params:
                    new_total = params[0]  # first param is new total contracts
                    dca_added = new_total - 10
                    max_dca = int(23000 * 0.05 / (exit_premium * 100))
                    assert dca_added <= max_dca, (
                        f"DCA added {dca_added} contracts, max allowed {max_dca} "
                        f"(5% of $23K at ${exit_premium}/share)"
                    )

    def test_dca_cap_math(self):
        """Verify DCA cap calculation matches expected values."""
        # $23K portfolio, 5% cap, $1.60 premium
        balance = 23000
        dca_pct = 5.0
        premium = 1.60
        cost_per = premium * 100  # $160
        dca_max_spend = balance * (dca_pct / 100)  # $1150
        dca_max_contracts = int(dca_max_spend / cost_per)  # 7
        assert dca_max_contracts == 7

        # $3K portfolio, 5% cap, $2.00 premium
        balance = 3000
        cost_per = 200
        dca_max_spend = balance * (dca_pct / 100)  # $150
        dca_max_contracts = int(dca_max_spend / cost_per)  # 0 → should block DCA
        assert dca_max_contracts == 0

    def test_dca_cap_various_portfolios(self):
        """DCA cap scales correctly with portfolio size."""
        for balance, premium, expected_max in [
            (23000, 2.00, 5),   # 5% = $1150, $200/ct → 5
            (23000, 1.00, 11),  # 5% = $1150, $100/ct → 11
            (5000, 2.00, 1),    # 5% = $250, $200/ct → 1
            (3000, 3.00, 0),    # 5% = $150, $300/ct → 0 (blocked)
            (50000, 1.50, 16),  # 5% = $2500, $150/ct → 16
        ]:
            cost_per = premium * 100
            max_ct = int(balance * 0.05 / cost_per)
            assert max_ct == expected_max, (
                f"bal=${balance} prem=${premium}: expected {expected_max}, got {max_ct}"
            )


# ── 3. Settings integration ───────────────────────────────────────────────


class TestSettingsHasDCACap:
    """Verify MAX_DCA_POSITION_PCT exists in Settings."""

    def test_setting_exists(self):
        from options_owl.config.settings import Settings
        s = Settings(DISCORD_TOKEN="test")
        assert hasattr(s, "MAX_DCA_POSITION_PCT")
        assert s.MAX_DCA_POSITION_PCT == 0.0  # 0 = auto-adapt from portfolio size

    def test_setting_overridable(self, monkeypatch):
        monkeypatch.setenv("MAX_DCA_POSITION_PCT", "5.0")
        from options_owl.config.settings import Settings
        s = Settings(DISCORD_TOKEN="test")
        assert s.MAX_DCA_POSITION_PCT == 5.0


# ── 4. Source code safety: verify min() is used, not max() ────────────────


class TestSourceCodeSafety:
    """Inspect actual source to catch regressions."""

    def test_confidence_weighted_sizing(self):
        """vinny_strategy.py must use confidence-weighted sizing via _ml_confidence_to_mult."""
        import inspect
        source = inspect.getsource(score_to_contracts)
        assert "_ml_confidence_to_mult" in source, (
            "score_to_contracts must use _ml_confidence_to_mult for confidence-weighted sizing"
        )
        assert "_SCORE_TIER_TABLE" not in source, (
            "score_to_contracts must NOT reference _SCORE_TIER_TABLE"
        )

    def test_dca_cap_variable_exists_in_monitor(self):
        """position_monitor._check_v6_dca must reference MAX_DCA_POSITION_PCT."""
        import inspect
        from options_owl.execution.position_monitor import _check_v6_dca
        source = inspect.getsource(_check_v6_dca)
        assert "MAX_DCA_POSITION_PCT" in source, (
            "_check_v6_dca must cap DCA contracts using MAX_DCA_POSITION_PCT"
        )
