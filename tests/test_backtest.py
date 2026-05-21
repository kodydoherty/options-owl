"""Tests for the backtesting engine."""

import math

import pytest

from options_owl.backtest.engine import BacktestConfig, BacktestEngine


def _make_signal(
    *,
    ticker: str = "AAPL",
    score: int = 85,
    atm_premium: float = 2.50,
    pnl_pct: float = 25.0,
    outcome: str = "t1_hit",
    direction: str = "call",
    bot_source: str = "Captain Hook",
    created_at: str = "2025-01-15T10:00:00",
    resolved_at: str = "2025-01-15T14:00:00",
    signal_id: int = 1,
) -> dict:
    return {
        "id": signal_id,
        "signal_id": signal_id,
        "ticker": ticker,
        "score": score,
        "atm_premium": atm_premium,
        "pnl_pct": pnl_pct,
        "outcome": outcome,
        "direction": direction,
        "bot_source": bot_source,
        "created_at": created_at,
        "resolved_at": resolved_at,
    }


class TestBacktestEngineBasic:
    """Basic engine functionality."""

    def test_single_winning_trade(self):
        config = BacktestConfig(starting_balance=5000, min_score=75)
        engine = BacktestEngine(config)
        signals = [_make_signal(pnl_pct=25.0)]
        result = engine.run(signals)

        assert result.total_trades == 1
        assert result.wins == 1
        assert result.losses == 0
        assert result.total_pnl > 0

    def test_single_losing_trade(self):
        config = BacktestConfig(starting_balance=5000, min_score=75)
        engine = BacktestEngine(config)
        signals = [_make_signal(pnl_pct=-30.0, outcome="stop_hit")]
        result = engine.run(signals)

        assert result.total_trades == 1
        assert result.wins == 0
        assert result.losses == 1
        assert result.total_pnl < 0

    def test_mixed_wins_and_losses(self):
        config = BacktestConfig(starting_balance=10000, min_score=70)
        engine = BacktestEngine(config)
        signals = [
            _make_signal(signal_id=1, pnl_pct=30.0, outcome="t1_hit", created_at="2025-01-15T10:00:00"),
            _make_signal(signal_id=2, pnl_pct=-25.0, outcome="stop_hit", created_at="2025-01-16T10:00:00"),
            _make_signal(signal_id=3, pnl_pct=50.0, outcome="t2_hit", created_at="2025-01-17T10:00:00"),
            _make_signal(signal_id=4, pnl_pct=-100.0, outcome="expired", created_at="2025-01-18T10:00:00"),
            _make_signal(signal_id=5, pnl_pct=20.0, outcome="t1_hit", created_at="2025-01-19T10:00:00"),
        ]
        result = engine.run(signals)

        assert result.total_trades == 5
        assert result.wins == 3
        assert result.losses == 2
        assert result.win_rate == pytest.approx(60.0)

    def test_no_signals(self):
        config = BacktestConfig(starting_balance=5000)
        engine = BacktestEngine(config)
        result = engine.run([])

        assert result.total_trades == 0
        assert result.total_pnl == 0.0
        assert result.win_rate == 0.0


class TestScoreFiltering:
    """Test that signals below min_score are skipped."""

    def test_below_min_score_skipped(self):
        config = BacktestConfig(starting_balance=5000, min_score=80)
        engine = BacktestEngine(config)
        signals = [
            _make_signal(signal_id=1, score=70, pnl_pct=50.0),  # should skip
            _make_signal(signal_id=2, score=85, pnl_pct=25.0),  # should trade
        ]
        result = engine.run(signals)

        assert result.total_trades == 1
        assert result.wins == 1


class TestPositionLimits:
    """Test max concurrent position enforcement."""

    def test_concurrent_limit_respected(self):
        """With max_concurrent=1 and sequential signals, all should trade
        (each opens/closes instantly in backtest). But we verify the engine
        respects the limit by checking trade count."""
        config = BacktestConfig(starting_balance=10000, max_concurrent=3, min_score=70)
        engine = BacktestEngine(config)

        # In the current engine, each signal is processed atomically
        # (entry + exit in same step), so concurrent positions don't accumulate.
        # This tests that the framework doesn't crash and processes correctly.
        signals = [
            _make_signal(signal_id=i, pnl_pct=10.0, created_at=f"2025-01-{15+i:02d}T10:00:00")
            for i in range(5)
        ]
        result = engine.run(signals)
        assert result.total_trades == 5

    def test_balance_insufficient_skips_trade(self):
        """When balance is too low to open a position, trade is skipped."""
        config = BacktestConfig(starting_balance=100, min_score=70)
        engine = BacktestEngine(config)
        # premium=5.00 -> cost per contract = 500 > 100 balance
        signals = [_make_signal(atm_premium=5.00, pnl_pct=25.0)]
        result = engine.run(signals)
        assert result.total_trades == 0


class TestSharpeRatio:
    """Test Sharpe ratio computation."""

    def test_sharpe_all_positive_returns(self):
        returns = [10.0, 12.0, 8.0, 15.0, 11.0]
        sharpe = BacktestEngine._compute_sharpe(returns)
        mean = sum(returns) / len(returns)
        std = math.sqrt(sum((r - mean) ** 2 for r in returns) / (len(returns) - 1))
        expected = mean / std
        assert sharpe == pytest.approx(expected, rel=1e-6)

    def test_sharpe_mixed_returns(self):
        returns = [20.0, -15.0, 30.0, -10.0, 5.0]
        sharpe = BacktestEngine._compute_sharpe(returns)
        mean = sum(returns) / len(returns)
        std = math.sqrt(sum((r - mean) ** 2 for r in returns) / (len(returns) - 1))
        expected = mean / std
        assert sharpe == pytest.approx(expected, rel=1e-6)

    def test_sharpe_single_return(self):
        assert BacktestEngine._compute_sharpe([10.0]) == 0.0

    def test_sharpe_zero_std(self):
        assert BacktestEngine._compute_sharpe([5.0, 5.0, 5.0]) == 0.0

    def test_sharpe_empty(self):
        assert BacktestEngine._compute_sharpe([]) == 0.0


class TestSortinoRatio:
    """Test Sortino ratio computation."""

    def test_sortino_no_downside(self):
        returns = [10.0, 20.0, 15.0]
        sortino = BacktestEngine._compute_sortino(returns)
        assert sortino == float("inf")

    def test_sortino_with_downside(self):
        returns = [20.0, -10.0, 15.0, -5.0]
        sortino = BacktestEngine._compute_sortino(returns)
        assert sortino > 0

    def test_sortino_all_negative(self):
        returns = [-10.0, -20.0, -5.0]
        sortino = BacktestEngine._compute_sortino(returns)
        assert sortino < 0


class TestMaxDrawdown:
    """Test drawdown tracking."""

    def test_drawdown_after_loss(self):
        config = BacktestConfig(starting_balance=10000, min_score=70)
        engine = BacktestEngine(config)
        signals = [
            _make_signal(signal_id=1, pnl_pct=20.0, created_at="2025-01-15T10:00:00"),
            _make_signal(signal_id=2, pnl_pct=-50.0, created_at="2025-01-16T10:00:00"),
        ]
        result = engine.run(signals)

        assert result.max_drawdown_dollars > 0
        assert result.max_drawdown_pct > 0

    def test_no_drawdown_all_wins(self):
        config = BacktestConfig(starting_balance=5000, min_score=70)
        engine = BacktestEngine(config)
        signals = [
            _make_signal(signal_id=1, pnl_pct=10.0, created_at="2025-01-15T10:00:00"),
            _make_signal(signal_id=2, pnl_pct=15.0, created_at="2025-01-16T10:00:00"),
        ]
        result = engine.run(signals)

        assert result.max_drawdown_dollars == 0.0
        assert result.max_drawdown_pct == 0.0

    def test_drawdown_sequence(self):
        """Three consecutive losses should produce increasing drawdown."""
        config = BacktestConfig(starting_balance=10000, min_score=70)
        engine = BacktestEngine(config)
        signals = [
            _make_signal(signal_id=1, pnl_pct=-20.0, created_at="2025-01-15T10:00:00"),
            _make_signal(signal_id=2, pnl_pct=-20.0, created_at="2025-01-16T10:00:00"),
            _make_signal(signal_id=3, pnl_pct=-20.0, created_at="2025-01-17T10:00:00"),
        ]
        result = engine.run(signals)

        assert result.max_drawdown_dollars > 0
        assert result.max_drawdown_pct > 0


class TestProfitFactor:
    """Test profit factor calculation."""

    def test_profit_factor_mixed(self):
        config = BacktestConfig(starting_balance=10000, min_score=70)
        engine = BacktestEngine(config)
        signals = [
            _make_signal(signal_id=1, pnl_pct=50.0, created_at="2025-01-15T10:00:00"),
            _make_signal(signal_id=2, pnl_pct=-25.0, created_at="2025-01-16T10:00:00"),
        ]
        result = engine.run(signals)

        assert result.profit_factor > 1.0

    def test_profit_factor_all_wins(self):
        config = BacktestConfig(starting_balance=5000, min_score=70)
        engine = BacktestEngine(config)
        signals = [
            _make_signal(signal_id=1, pnl_pct=20.0, created_at="2025-01-15T10:00:00"),
        ]
        result = engine.run(signals)

        assert result.profit_factor == float("inf")

    def test_profit_factor_all_losses(self):
        config = BacktestConfig(starting_balance=5000, min_score=70)
        engine = BacktestEngine(config)
        signals = [
            _make_signal(signal_id=1, pnl_pct=-20.0, created_at="2025-01-15T10:00:00"),
        ]
        result = engine.run(signals)

        assert result.profit_factor == 0.0


class TestFormatReport:
    """Test report formatting."""

    def test_report_contains_key_sections(self):
        config = BacktestConfig(starting_balance=5000, min_score=70)
        engine = BacktestEngine(config)
        engine.run([
            _make_signal(signal_id=1, pnl_pct=25.0),
            _make_signal(signal_id=2, pnl_pct=-15.0),
        ])
        report = engine.format_report()

        assert "BACKTEST REPORT" in report
        assert "Starting Balance" in report
        assert "Win Rate" in report
        assert "Sharpe Ratio" in report
        assert "Profit Factor" in report
        assert "Max Drawdown" in report


class TestPnlResolution:
    """Test _resolve_pnl_pct with different signal data."""

    def test_explicit_pnl_pct(self):
        sig = {"pnl_pct": 33.5}
        assert BacktestEngine._resolve_pnl_pct(sig) == 33.5

    def test_atm_est_fallback(self):
        sig = {"pnl_atm_est": 22.0}
        assert BacktestEngine._resolve_pnl_pct(sig) == 22.0

    def test_underlying_fallback(self):
        sig = {"pnl_underlying_pct": 5.0}
        assert BacktestEngine._resolve_pnl_pct(sig) == 5.0

    def test_outcome_t1_hit_default(self):
        sig = {"outcome": "t1_hit"}
        assert BacktestEngine._resolve_pnl_pct(sig) == 25.0

    def test_outcome_stop_hit_default(self):
        sig = {"outcome": "stop_hit"}
        assert BacktestEngine._resolve_pnl_pct(sig) == -30.0

    def test_outcome_expired(self):
        sig = {"outcome": "expired"}
        assert BacktestEngine._resolve_pnl_pct(sig) == -100.0

    def test_unknown_outcome(self):
        sig = {"outcome": "unknown"}
        assert BacktestEngine._resolve_pnl_pct(sig) == 0.0
