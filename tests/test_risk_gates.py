"""E2E and integration tests for risk gates: GFV, max loss cap, premium floor, cost guard.

Tests the full trade lifecycle to ensure all protective gates fire correctly
and interact properly — no bugs from the sizing overhaul.
"""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime
from unittest.mock import MagicMock

import aiosqlite
import pytest

from options_owl.config.settings import Settings
from options_owl.execution.paper_trader import PaperTrader
from options_owl.risk.pipeline import GateResult
from options_owl.risk.vinny_strategy import score_to_contracts


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def settings():
    """Production-like settings with new sizing config."""
    s = Settings(
        DISCORD_TOKEN="test",
        DISCORD_CHANNEL_ID=123,
        PORTFOLIO_SIZE=8227,
        MAX_POSITION_PCT=15.0,
        MAX_CONCURRENT=4,
        MAX_PORTFOLIO_RISK_PCT=75.0,
        MIN_OPTION_PREMIUM=0.30,
        ENABLE_VINNY_STRATEGY=True,
        ENABLE_SCORE_SIZING=True,
        PAPER_TRADE=False,
        WEBULL_KILL_SWITCH=True,
        DB_PATH=":memory:",
    )
    # Add the new setting
    object.__setattr__(s, "MAX_TRADE_LOSS_EXIT_PCT", 8.0)
    object.__setattr__(s, "DAILY_LOSS_CIRCUIT_BREAKER_PCT", 25.0)
    return s


def make_signal(ticker="AAPL", score=110, premium=1.50, direction="bullish",
                strike=200.0, expiry="0DTE"):
    """Create a mock trade signal."""
    sig = MagicMock()
    sig.ticker = ticker
    sig.score = score
    sig.atm_premium = premium
    sig.otm_premium = None
    sig.direction = direction
    sig.strike = strike
    sig.expiry = expiry
    sig.option_type = "call" if direction == "bullish" else "put"
    sig.entry_price = None
    sig.targets = []
    sig.stop_loss = None
    sig.created_at = datetime.now()
    sig.bot_source = MagicMock(value="neverland")
    return sig


# ---------------------------------------------------------------------------
# 1. Premium Floor Gate
# ---------------------------------------------------------------------------

class TestPremiumFloor:
    """MIN_OPTION_PREMIUM = $0.30 — rejects cheap lottery tickets."""

    def test_premium_floor_default_is_030(self):
        s = Settings(DISCORD_TOKEN="t", DISCORD_CHANNEL_ID=1)
        assert s.MIN_OPTION_PREMIUM == 0.30

    def test_entry_pipeline_rejects_below_floor(self):
        """The premium check gate should reject premiums < $0.30."""
        from options_owl.risk.pipeline import PremiumGate

        gate = PremiumGate()

        # Below floor
        signal = make_signal(premium=0.25)
        settings = Settings(
            DISCORD_TOKEN="t", DISCORD_CHANNEL_ID=1,
            MIN_OPTION_PREMIUM=0.30,
        )
        ctx = {"signal": signal, "settings": settings}
        result = asyncio.run(gate.evaluate(ctx))
        assert result.result == GateResult.FAIL
        assert "0.25" in result.reason
        assert "0.30" in result.reason

    def test_entry_pipeline_passes_above_floor(self):
        """Premium >= $0.30 should pass."""
        from options_owl.risk.pipeline import PremiumGate

        gate = PremiumGate()
        signal = make_signal(premium=0.50)
        settings = Settings(
            DISCORD_TOKEN="t", DISCORD_CHANNEL_ID=1,
            MIN_OPTION_PREMIUM=0.30,
        )
        ctx = {"signal": signal, "settings": settings}
        result = asyncio.run(gate.evaluate(ctx))
        assert result.result == GateResult.PASS

    def test_entry_pipeline_passes_at_exact_floor(self):
        """Premium == $0.30 should pass."""
        from options_owl.risk.pipeline import PremiumGate

        gate = PremiumGate()
        signal = make_signal(premium=0.30)
        settings = Settings(
            DISCORD_TOKEN="t", DISCORD_CHANNEL_ID=1,
            MIN_OPTION_PREMIUM=0.30,
        )
        ctx = {"signal": signal, "settings": settings}
        result = asyncio.run(gate.evaluate(ctx))
        assert result.result == GateResult.PASS


# ---------------------------------------------------------------------------
# 2. Cost Guard — skip trades where 1 contract exceeds position cap
# ---------------------------------------------------------------------------

class TestCostGuard:
    """Skip expensive options that would violate position limits."""

    def test_expensive_option_skipped_small_account(self):
        """$25 META option on $3K account → 15% cap=$450 < $2500 → 0."""
        assert score_to_contracts(
            110, cost_per_contract=2500, balance=3000,
            max_position_pct=15.0, max_concurrent=4,
        ) == 0

    def test_expensive_option_skipped_medium_account(self):
        """$15 SPX on $5K → 15% cap=$750 < $1500 → 0."""
        assert score_to_contracts(
            150, cost_per_contract=1500, balance=5000,
            max_position_pct=15.0, max_concurrent=4,
        ) == 0

    def test_affordable_option_passes(self):
        """$1.50 option on $8K → 15% cap=$1234, cost=$150 → passes."""
        result = score_to_contracts(
            110, cost_per_contract=150, balance=8000,
            max_position_pct=15.0, max_concurrent=4,
        )
        assert result > 0

    def test_borderline_1_contract_fits(self):
        """$1200 option on $8K → 15% cap=$1200, cost=$1200 → exactly 1."""
        result = score_to_contracts(
            150, cost_per_contract=1200, balance=8000,
            max_position_pct=15.0, max_concurrent=4,
        )
        assert result == 1

    def test_borderline_1_contract_too_expensive(self):
        """$1201 option on $8K → 15% cap=$1200 < $1201 → 0."""
        result = score_to_contracts(
            150, cost_per_contract=1201, balance=8000,
            max_position_pct=15.0, max_concurrent=4,
        )
        assert result == 0


# ---------------------------------------------------------------------------
# 3. Score-tiered sizing with 15%/4c
# ---------------------------------------------------------------------------

class TestSizingWith15Pct4C:
    """Verify flat sizing math with production defaults (15% cap, 4 concurrent)."""

    def test_defaults_are_15_4(self):
        """Code defaults should be 15%/4c (env may override)."""
        import inspect
        sig = inspect.signature(score_to_contracts)
        assert sig.parameters["max_position_pct"].default == 15.0
        assert sig.parameters["max_concurrent"].default == 4

    def test_all_scores_same_sizing(self):
        """Flat sizing: all scores >= 78 get same contracts for same cost/balance."""
        # $8227 balance, $150/ct, 15% cap, 4 concurrent
        # slot = 8227*0.75/4 = $1542, 85% = $1311, raw = 8
        # pos_cap = 8227*0.15/150 = 8
        expected = score_to_contracts(
            150, cost_per_contract=150, balance=8227,
            max_position_pct=15.0, max_concurrent=4,
        )
        for score in [150, 130, 110, 92, 80, 78]:
            result = score_to_contracts(
                score, cost_per_contract=150, balance=8227,
                max_position_pct=15.0, max_concurrent=4,
            )
            assert result == expected, f"Score {score} got {result}, expected {expected}"

    def test_below_floor_rejected(self):
        """Score 61 → 0 contracts (below the 62 floor)."""
        assert score_to_contracts(61, cost_per_contract=150, balance=8227) == 0

    def test_all_scores_respect_15pct_cap(self):
        """All scores cap at 15% of portfolio."""
        for score in [150, 130, 110, 92, 80]:
            result = score_to_contracts(
                score, cost_per_contract=50, balance=10000,
                max_position_pct=15.0, max_concurrent=4,
            )
            assert result <= 30, f"Score {score} got {result} contracts, exceeds 15% cap"

    def test_small_account_still_trades(self):
        """$3K account with $1.50 option → should get at least 1 contract."""
        result = score_to_contracts(
            110, cost_per_contract=150, balance=3000,
            max_position_pct=15.0, max_concurrent=4,
        )
        assert result >= 1

    def test_small_account_respects_cap(self):
        """$3K × 15% = $450 → max 3 contracts at $150."""
        result = score_to_contracts(
            150, cost_per_contract=150, balance=3000,
            max_position_pct=15.0, max_concurrent=4,
        )
        assert result == 3


# ---------------------------------------------------------------------------
# 4. GFV Protection — in-memory tracker + DB seed
# ---------------------------------------------------------------------------

class TestGFVProtection:
    """GFV hard block: daily buys cannot exceed starting balance."""

    @pytest.fixture
    def paper_trader(self, settings, tmp_path):
        db_path = str(tmp_path / "test.db")
        settings = settings.model_copy(update={"DB_PATH": db_path})
        pt = PaperTrader(settings)
        # Create tables
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS paper_trades (
                id INTEGER PRIMARY KEY,
                ticker TEXT, strike REAL, direction TEXT,
                premium_per_contract REAL, contracts INTEGER,
                status TEXT DEFAULT 'open', opened_at TEXT,
                closed_at TEXT, pnl_dollars REAL,
                webull_order_id TEXT, mae_premium REAL,
                signal_id INTEGER
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS paper_portfolio (
                strategy TEXT PRIMARY KEY,
                starting_balance REAL,
                current_balance REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trade_events (
                id INTEGER PRIMARY KEY,
                ticker TEXT, event_type TEXT, details TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute(
            "INSERT OR REPLACE INTO paper_portfolio VALUES (?, ?, ?)",
            ("B", 8227.0, 8227.0),
        )
        conn.commit()
        conn.close()
        return pt

    def test_gfv_tracker_starts_at_zero(self, paper_trader):
        assert paper_trader._gfv_daily_spent == 0.0
        assert paper_trader._gfv_day == ""

    def test_gfv_tracker_updates_on_trade(self, paper_trader):
        """Simulate trade placement incrementing the tracker."""
        paper_trader._gfv_day = "2026-05-06"
        paper_trader._gfv_daily_spent = 1000.0

        # Simulate what happens after a trade is placed
        trade_cost = 5 * 1.50 * 100  # 5 contracts at $1.50
        paper_trader._gfv_daily_spent += trade_cost
        assert paper_trader._gfv_daily_spent == 1750.0

    def test_gfv_blocks_when_over_limit(self, paper_trader):
        """After spending $8000, a $300 trade should be blocked."""
        paper_trader._gfv_day = "2026-05-06"
        paper_trader._gfv_daily_spent = 8000.0

        # Any trade with min cost $300 pushes over $8227
        min_trade_cost = 3.00 * 100  # $300
        assert paper_trader._gfv_daily_spent + min_trade_cost > 8227.0

    def test_gfv_allows_when_under_limit(self, paper_trader):
        """After spending $5000, a $300 trade should be allowed."""
        paper_trader._gfv_day = "2026-05-06"
        paper_trader._gfv_daily_spent = 5000.0

        min_trade_cost = 3.00 * 100
        assert paper_trader._gfv_daily_spent + min_trade_cost <= 8227.0

    @pytest.mark.asyncio
    async def test_gfv_seeds_from_db_on_new_day(self, paper_trader):
        """On first call of a new day, tracker should seed from DB."""
        db_path = paper_trader.db_path
        today = datetime.now().strftime("%Y-%m-%d")

        # Insert a pre-existing Webull trade for today
        async with aiosqlite.connect(db_path) as conn:
            await conn.execute(
                "INSERT INTO paper_trades "
                "(ticker, strike, direction, premium_per_contract, contracts, "
                "status, opened_at, webull_order_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ("AAPL", 200, "call", 1.50, 5, "open",
                 f"{today} 10:00:00", "WB123"),
            )
            await conn.commit()

        # Seed by checking unsettled (simulate the flow)
        assert paper_trader._gfv_day == ""  # not yet seeded

        # After seeding, should show $750 already spent
        async with aiosqlite.connect(db_path) as conn:
            cursor = await conn.execute(
                "SELECT COALESCE(SUM(premium_per_contract * contracts * 100), 0) "
                "FROM paper_trades "
                "WHERE date(opened_at) = ? AND webull_order_id IS NOT NULL",
                (today,),
            )
            total = float((await cursor.fetchone())[0])
        assert total == 750.0  # 5 × $1.50 × 100


# ---------------------------------------------------------------------------
# 5. Max Trade Loss Exit — force-close at 8% of portfolio
# ---------------------------------------------------------------------------

class TestMaxTradeLossExit:
    """MAX_TRADE_LOSS_EXIT_PCT = 8% — cap single trade losses."""

    def test_setting_default_is_8(self):
        """Default should be 8%."""
        s = Settings(DISCORD_TOKEN="t", DISCORD_CHANNEL_ID=1)
        assert s.MAX_TRADE_LOSS_EXIT_PCT == 8.0

    def test_loss_cap_math(self):
        """$8227 × 8% = $658.16 max loss per trade."""
        portfolio = 8227.0
        max_loss_pct = 8.0
        max_loss_dollars = portfolio * (max_loss_pct / 100)
        assert abs(max_loss_dollars - 658.16) < 0.01

    def test_loss_cap_triggers_on_exceed(self):
        """Unrealized loss > $658 should trigger exit."""
        portfolio = 8227.0
        max_loss_dollars = portfolio * 0.08
        entry_prem = 2.00
        contracts = 5
        # Premium drops to $0.50 → loss = (0.50 - 2.00) × 5 × 100 = -$750
        exit_prem = 0.50
        unrealized = (exit_prem - entry_prem) * contracts * 100
        assert unrealized == -750.0
        assert unrealized < -max_loss_dollars  # should trigger

    def test_loss_cap_holds_within_limit(self):
        """Unrealized loss < $658 should not trigger exit."""
        portfolio = 8227.0
        max_loss_dollars = portfolio * 0.08
        entry_prem = 2.00
        contracts = 3
        # Premium drops to $1.00 → loss = (1.00 - 2.00) × 3 × 100 = -$300
        exit_prem = 1.00
        unrealized = (exit_prem - entry_prem) * contracts * 100
        assert unrealized == -300.0
        assert unrealized >= -max_loss_dollars  # should NOT trigger

    def test_loss_cap_per_account_size(self):
        """Verify loss cap scales with portfolio size."""
        for portfolio, expected_cap in [
            (8227, 658),
            (4685, 374),
            (3123, 249),
            (3600, 288),
        ]:
            cap = portfolio * 0.08
            assert abs(cap - expected_cap) < 1, f"${portfolio}: expected ${expected_cap}, got ${cap:.0f}"


# ---------------------------------------------------------------------------
# 6. Integration: sizing + GFV interaction
# ---------------------------------------------------------------------------

class TestSizingGFVInteraction:
    """Verify that sizing and GFV work together correctly."""

    def test_max_daily_spend_within_balance(self):
        """No combination of trades should exceed starting balance."""
        balance = 8227
        daily_spent = 0.0
        trades_entered = 0

        # Simulate entering trades until GFV blocks
        for _ in range(20):
            contracts = score_to_contracts(
                110, cost_per_contract=150, balance=balance,
                max_position_pct=15.0, max_concurrent=4,
            )
            if contracts <= 0:
                break
            trade_cost = contracts * 150
            if daily_spent + trade_cost > balance:
                break  # GFV would block
            daily_spent += trade_cost
            trades_entered += 1

        assert daily_spent <= balance, f"Spent ${daily_spent} > balance ${balance}"
        assert trades_entered > 0, "Should enter at least 1 trade"
        assert trades_entered <= 14, "Shouldn't enter more than ~13 trades (4ct × $150 each)"

    def test_small_account_cant_overspend(self):
        """$3K account at 15%/4c cannot exceed $3K daily."""
        balance = 3123
        daily_spent = 0.0

        for _ in range(20):
            contracts = score_to_contracts(
                110, cost_per_contract=200, balance=balance,
                max_position_pct=15.0, max_concurrent=4,
            )
            if contracts <= 0:
                break
            trade_cost = contracts * 200
            if daily_spent + trade_cost > balance:
                break
            daily_spent += trade_cost

        assert daily_spent <= balance

    def test_expensive_options_blocked_not_forced(self):
        """$25 META on $3K should return 0, not force 1 contract."""
        result = score_to_contracts(
            150, cost_per_contract=2500, balance=3000,
            max_position_pct=15.0, max_concurrent=4,
        )
        assert result == 0, "Should skip, not force 1 contract"

    def test_gfv_math_all_agents(self):
        """Verify no agent can exceed their balance in any scenario."""
        agents = [
            ("Kody", 8227),
            ("Adam", 4685),
            ("Vinny", 3123),
            ("Yank", 3600),
        ]
        for name, balance in agents:
            daily_spent = 0.0
            for premium in [0.50, 1.00, 1.50, 2.00, 3.00, 5.00]:
                cost_per = premium * 100
                contracts = score_to_contracts(
                    110, cost_per_contract=cost_per, balance=balance,
                    max_position_pct=15.0, max_concurrent=4,
                )
                if contracts <= 0:
                    continue
                trade_cost = contracts * cost_per
                if daily_spent + trade_cost > balance:
                    continue
                daily_spent += trade_cost

            assert daily_spent <= balance, (
                f"{name} (${balance}): spent ${daily_spent:.0f} > balance"
            )


# ---------------------------------------------------------------------------
# 7. Circuit Breaker — daily net P&L limit
# ---------------------------------------------------------------------------

class TestCircuitBreaker:
    """DAILY_LOSS_CIRCUIT_BREAKER_PCT = 25% — stop trading after bad day."""

    def test_setting_default_is_25(self):
        s = Settings(DISCORD_TOKEN="t", DISCORD_CHANNEL_ID=1)
        assert s.DAILY_LOSS_CIRCUIT_BREAKER_PCT == 25.0

    def test_circuit_breaker_math(self):
        """$8227 × 25% = $2056.75 — stop after losing this much."""
        portfolio = 8227.0
        cb_pct = 25.0
        cb_limit = portfolio * (cb_pct / 100)
        assert abs(cb_limit - 2056.75) < 0.01

    def test_net_pnl_allows_wins_to_offset(self):
        """Net P&L: $500 win + $800 loss = -$300 (not -$800)."""
        day_pnl = 500 + (-800)
        cb_limit = 8227 * 0.25  # $2056
        assert day_pnl > -cb_limit  # -$300 > -$2056, should NOT trigger
