"""Tests for scripts/optimize_filters.py — combinatorial filter optimization."""

import sys
from pathlib import Path

import pytest

# Add scripts/ to path so we can import the module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from optimize_filters import (
    CANDIDATE_RULES,
    apply_rules,
    apply_rules_ids,
    compute_stats,
    evaluate_combination_fast,
    minutes_since_open,
    rank_results,
    run_optimization,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_trade(
    id: int = 1,
    ticker: str = "NVDA",
    direction: str = "call",
    score: int = 95,
    entry_premium: float = 2.0,
    pnl_dollars: float = 100.0,
    hour: int = 10,
    minute: int = 0,
    day_of_week: int = 2,  # Wednesday
    contracts: int = 5,
    strike: float = 200.0,
    duration_minutes: float = 30.0,
    **overrides,
) -> dict:
    mso = minutes_since_open(hour, minute)
    trade = {
        "id": id,
        "ticker": ticker,
        "direction": direction,
        "score": score,
        "entry_premium": entry_premium,
        "strike": strike,
        "contracts": contracts,
        "total_cost": entry_premium * contracts * 100,
        "pnl_dollars": pnl_dollars,
        "pnl_pct": pnl_dollars / (entry_premium * contracts * 100) * 100 if entry_premium > 0 else 0,
        "exit_reason": "adaptive_trail",
        "exit_source": "ai",
        "webull_order_id": "W123",
        "is_winner": pnl_dollars > 0,
        "hour": hour,
        "minute": minute,
        "minutes_since_open": mso,
        "day_of_week": day_of_week,
        "date": "2026-05-20",
        "duration_minutes": duration_minutes,
        "parent_trade_id": None,
    }
    trade.update(overrides)
    return trade


def make_synthetic_dataset():
    """5 winners and 5 losers with distinct features for testing."""
    winners = [
        make_trade(id=1, ticker="NVDA", score=100, entry_premium=2.0, pnl_dollars=200, hour=10, minute=0, day_of_week=1),
        make_trade(id=2, ticker="TSLA", score=95, entry_premium=1.5, pnl_dollars=150, hour=10, minute=30, day_of_week=2),
        make_trade(id=3, ticker="AAPL", score=90, entry_premium=3.0, pnl_dollars=300, hour=11, minute=0, day_of_week=3),
        make_trade(id=4, ticker="SPY", score=85, entry_premium=0.8, pnl_dollars=80, hour=13, minute=0, day_of_week=4, direction="call"),
        make_trade(id=5, ticker="META", score=110, entry_premium=4.0, pnl_dollars=400, hour=14, minute=0, day_of_week=0),
    ]
    losers = [
        make_trade(id=6, ticker="AMD", score=80, entry_premium=1.0, pnl_dollars=-100, hour=9, minute=31, day_of_week=0),
        make_trade(id=7, ticker="MSTR", score=78, entry_premium=6.0, pnl_dollars=-600, hour=15, minute=30, day_of_week=1),
        make_trade(id=8, ticker="QQQ", score=82, entry_premium=0.2, pnl_dollars=-20, hour=12, minute=30, day_of_week=2, direction="put"),
        make_trade(id=9, ticker="PLTR", score=88, entry_premium=0.4, pnl_dollars=-40, hour=9, minute=35, day_of_week=3),
        make_trade(id=10, ticker="SPY", score=92, entry_premium=2.5, pnl_dollars=-250, hour=15, minute=45, day_of_week=4, direction="put"),
    ]
    return winners + losers


# ── Tests ────────────────────────────────────────────────────────────────────

class TestMinutesSinceOpen:
    def test_market_open(self):
        assert minutes_since_open(9, 30) == 0

    def test_one_hour_after_open(self):
        assert minutes_since_open(10, 30) == 60

    def test_935(self):
        assert minutes_since_open(9, 35) == 5

    def test_eod(self):
        assert minutes_since_open(16, 0) == 390


class TestIndividualFilterRules:
    """Test that individual candidate rules work correctly on mock trades."""

    def test_skip_before_935(self):
        rule_fn = dict(CANDIDATE_RULES)["skip_before_935"]
        early_trade = make_trade(hour=9, minute=31)  # 1 min after open
        assert early_trade["minutes_since_open"] == 1
        assert rule_fn(early_trade) is False

        ok_trade = make_trade(hour=9, minute=36)  # 6 min after open
        assert rule_fn(ok_trade) is True

    def test_min_score_90(self):
        rule_fn = dict(CANDIDATE_RULES)["min_score_90"]
        assert rule_fn(make_trade(score=89)) is False
        assert rule_fn(make_trade(score=90)) is True
        assert rule_fn(make_trade(score=100)) is True

    def test_max_premium_3(self):
        rule_fn = dict(CANDIDATE_RULES)["max_premium_3"]
        assert rule_fn(make_trade(entry_premium=2.5)) is True
        assert rule_fn(make_trade(entry_premium=3.0)) is True
        assert rule_fn(make_trade(entry_premium=3.01)) is False

    def test_skip_index_puts(self):
        rule_fn = dict(CANDIDATE_RULES)["skip_index_puts"]
        # SPY put should be filtered
        assert rule_fn(make_trade(ticker="SPY", direction="put")) is False
        # SPY call should pass
        assert rule_fn(make_trade(ticker="SPY", direction="call")) is True
        # NVDA put should pass (not an index)
        assert rule_fn(make_trade(ticker="NVDA", direction="put")) is True

    def test_skip_monday(self):
        rule_fn = dict(CANDIDATE_RULES)["skip_monday"]
        assert rule_fn(make_trade(day_of_week=0)) is False  # Monday
        assert rule_fn(make_trade(day_of_week=2)) is True   # Wednesday

    def test_only_calls(self):
        rule_fn = dict(CANDIDATE_RULES)["only_calls"]
        assert rule_fn(make_trade(direction="call")) is True
        assert rule_fn(make_trade(direction="put")) is False

    def test_skip_mstr(self):
        rule_fn = dict(CANDIDATE_RULES)["skip_mstr"]
        assert rule_fn(make_trade(ticker="MSTR")) is False
        assert rule_fn(make_trade(ticker="NVDA")) is True

    def test_max_premium_5(self):
        rule_fn = dict(CANDIDATE_RULES)["max_premium_5"]
        assert rule_fn(make_trade(entry_premium=4.99)) is True
        assert rule_fn(make_trade(entry_premium=5.0)) is True
        assert rule_fn(make_trade(entry_premium=5.01)) is False

    def test_min_premium_050(self):
        rule_fn = dict(CANDIDATE_RULES)["min_premium_0.50"]
        assert rule_fn(make_trade(entry_premium=0.49)) is False
        assert rule_fn(make_trade(entry_premium=0.50)) is True


class TestComputeStats:
    def test_empty_trades(self):
        stats = compute_stats([])
        assert stats["count"] == 0
        assert stats["win_rate"] == 0.0

    def test_all_winners(self):
        trades = [make_trade(id=i, pnl_dollars=100) for i in range(5)]
        stats = compute_stats(trades)
        assert stats["count"] == 5
        assert stats["winners"] == 5
        assert stats["losers"] == 0
        assert stats["win_rate"] == 100.0
        assert stats["total_pnl"] == 500.0

    def test_mixed(self):
        trades = [
            make_trade(id=1, pnl_dollars=200),
            make_trade(id=2, pnl_dollars=-50),
            make_trade(id=3, pnl_dollars=100),
        ]
        stats = compute_stats(trades)
        assert stats["count"] == 3
        assert stats["winners"] == 2
        assert stats["losers"] == 1
        assert stats["win_rate"] == pytest.approx(66.67, abs=0.1)
        assert stats["total_pnl"] == 250.0
        assert stats["avg_winner"] == 150.0
        assert stats["avg_loser"] == -50.0


class TestApplyRules:
    def test_single_rule_filters_correctly(self):
        trades = make_synthetic_dataset()
        rules = [("min_score_90", dict(CANDIDATE_RULES)["min_score_90"])]
        filtered = apply_rules(trades, rules)
        assert all(t["score"] >= 90 for t in filtered)

    def test_multiple_rules_conjunction(self):
        """Multiple rules act as AND — all must pass."""
        trades = make_synthetic_dataset()
        rules = [
            ("min_score_85", dict(CANDIDATE_RULES)["min_score_85"]),
            ("max_premium_3", dict(CANDIDATE_RULES)["max_premium_3"]),
        ]
        filtered = apply_rules(trades, rules)
        assert all(t["score"] >= 85 and t["entry_premium"] <= 3.0 for t in filtered)

    def test_no_rules_returns_all(self):
        trades = make_synthetic_dataset()
        filtered = apply_rules(trades, [])
        assert len(filtered) == len(trades)


class TestApplyRulesIds:
    def test_returns_correct_ids(self):
        trades = make_synthetic_dataset()
        rules = [("skip_mstr", dict(CANDIDATE_RULES)["skip_mstr"])]
        ids = apply_rules_ids(trades, rules)
        # Trade 7 is MSTR, should be excluded
        assert 7 not in ids
        assert len(ids) == 9


class TestEvaluateCombinationFast:
    def test_removes_losers_keeps_winners(self):
        trades = make_synthetic_dataset()
        winner_ids = {t["id"] for t in trades if t["is_winner"]}
        loser_ids = {t["id"] for t in trades if not t["is_winner"]}

        # skip_mstr removes MSTR (id=7, a loser with -$600)
        rules = [("skip_mstr", dict(CANDIDATE_RULES)["skip_mstr"])]
        result = evaluate_combination_fast(
            trades, 5, 5, rules, min_trades=1,
            winner_ids=winner_ids, loser_ids=loser_ids,
        )
        assert result is not None
        assert result["losers_removed"] == 1
        assert result["winners_removed"] == 0
        assert result["trades_remaining"] == 9

    def test_min_trades_filters(self):
        trades = make_synthetic_dataset()
        winner_ids = {t["id"] for t in trades if t["is_winner"]}
        loser_ids = {t["id"] for t in trades if not t["is_winner"]}

        # This combo will leave some trades, but require 100 min
        rules = [("min_score_90", dict(CANDIDATE_RULES)["min_score_90"])]
        result = evaluate_combination_fast(
            trades, 5, 5, rules, min_trades=100,
            winner_ids=winner_ids, loser_ids=loser_ids,
        )
        assert result is None


class TestRankResults:
    def test_zero_casualties_ranked_first(self):
        """Zero-winner-casualty combos should always rank above non-zero."""
        results = [
            {
                "rules": ["rule_a"], "num_rules": 1, "trades_remaining": 8,
                "winners": 5, "losers": 3, "win_rate": 62.5, "avg_pnl": 50,
                "total_pnl": 400, "losers_removed": 2, "loser_removal_rate": 40,
                "winners_removed": 0, "winner_casualty_rate": 0,
                "avg_winner": 100, "avg_loser": -50,
            },
            {
                "rules": ["rule_b"], "num_rules": 1, "trades_remaining": 6,
                "winners": 4, "losers": 2, "win_rate": 66.7, "avg_pnl": 75,
                "total_pnl": 450, "losers_removed": 3, "loser_removal_rate": 60,
                "winners_removed": 1, "winner_casualty_rate": 20,
                "avg_winner": 120, "avg_loser": -30,
            },
        ]
        ranked = rank_results(results)
        # rule_a has 0 casualties, should be first even though rule_b has higher loser removal
        assert ranked[0]["rules"] == ["rule_a"]
        assert ranked[1]["rules"] == ["rule_b"]

    def test_among_zero_casualties_higher_loser_removal_wins(self):
        results = [
            {
                "rules": ["low_removal"], "num_rules": 1, "trades_remaining": 9,
                "winners": 5, "losers": 4, "win_rate": 55.6, "avg_pnl": 30,
                "total_pnl": 270, "losers_removed": 1, "loser_removal_rate": 20,
                "winners_removed": 0, "winner_casualty_rate": 0,
                "avg_winner": 100, "avg_loser": -50,
            },
            {
                "rules": ["high_removal"], "num_rules": 1, "trades_remaining": 7,
                "winners": 5, "losers": 2, "win_rate": 71.4, "avg_pnl": 60,
                "total_pnl": 420, "losers_removed": 3, "loser_removal_rate": 60,
                "winners_removed": 0, "winner_casualty_rate": 0,
                "avg_winner": 100, "avg_loser": -40,
            },
        ]
        ranked = rank_results(results)
        assert ranked[0]["rules"] == ["high_removal"]

    def test_tiebreak_by_win_rate(self):
        """Same loser removal rate, zero casualties — higher WR wins."""
        base = {
            "num_rules": 1, "trades_remaining": 8, "losers_removed": 2,
            "loser_removal_rate": 40, "winners_removed": 0,
            "winner_casualty_rate": 0, "avg_winner": 100, "avg_loser": -50,
        }
        r1 = {**base, "rules": ["wr_low"], "winners": 5, "losers": 3,
               "win_rate": 62.5, "avg_pnl": 40, "total_pnl": 320}
        r2 = {**base, "rules": ["wr_high"], "winners": 6, "losers": 2,
               "win_rate": 75.0, "avg_pnl": 50, "total_pnl": 400}
        ranked = rank_results([r1, r2])
        assert ranked[0]["rules"] == ["wr_high"]


class TestRunOptimization:
    def test_finds_good_filters_on_synthetic_data(self):
        """Run on synthetic dataset — should find combos that remove losers."""
        trades = make_synthetic_dataset()
        # Use a small subset of rules to keep test fast
        test_rules = [
            ("skip_mstr", dict(CANDIDATE_RULES)["skip_mstr"]),
            ("max_premium_5", dict(CANDIDATE_RULES)["max_premium_5"]),
            ("skip_index_puts", dict(CANDIDATE_RULES)["skip_index_puts"]),
            ("min_score_85", dict(CANDIDATE_RULES)["min_score_85"]),
        ]
        ranked = run_optimization(
            trades, max_combo=2, min_trades=3, rules=test_rules,
        )
        assert len(ranked) > 0
        # Best result should have 0 or few winner casualties
        best = ranked[0]
        assert best["winners_removed"] <= 1

    def test_respects_min_trades(self):
        """Combos that leave too few trades are excluded."""
        trades = make_synthetic_dataset()  # 10 trades
        test_rules = [
            ("min_score_100", dict(CANDIDATE_RULES)["min_score_100"]),
        ]
        ranked = run_optimization(
            trades, max_combo=1, min_trades=8, rules=test_rules,
        )
        # min_score_100 leaves only 2 trades (score 100 and 110), which < 8
        for r in ranked:
            assert r["trades_remaining"] >= 8

    def test_empty_trades_returns_empty(self):
        ranked = run_optimization([], max_combo=1, min_trades=1)
        assert ranked == []


class TestSyntheticDatasetIntegrity:
    def test_has_5_winners_5_losers(self):
        trades = make_synthetic_dataset()
        assert len(trades) == 10
        winners = [t for t in trades if t["is_winner"]]
        losers = [t for t in trades if not t["is_winner"]]
        assert len(winners) == 5
        assert len(losers) == 5

    def test_unique_ids(self):
        trades = make_synthetic_dataset()
        ids = [t["id"] for t in trades]
        assert len(set(ids)) == 10

    def test_feature_diversity(self):
        """Trades cover different tickers, days, times."""
        trades = make_synthetic_dataset()
        tickers = {t["ticker"] for t in trades}
        days = {t["day_of_week"] for t in trades}
        assert len(tickers) >= 5
        assert len(days) >= 3
