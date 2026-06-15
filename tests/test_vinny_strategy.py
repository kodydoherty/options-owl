"""Tests for Vinny's strategy — phase-based trailing stops, anti-chase, scoring."""

from datetime import datetime

from options_owl.risk.vinny_strategy import (
    check_anti_chase,
    check_consecutive_loser_pause,
    check_theta_bleed,
    check_time_decay_no_new_high,
    compute_vix_adjusted_trail,
    evaluate_dollar_trail,
    get_current_phase,
    is_time_decay_zone,
    score_to_contracts,
)


class TestVIXAdjustment:
    def test_vix_at_20_no_change(self):
        assert compute_vix_adjusted_trail(25.0, 20.0) == 25.0

    def test_high_vix_widens_trail(self):
        # VIX 30: adjustment = (30-20)*0.5 = +5%
        assert compute_vix_adjusted_trail(25.0, 30.0) == 30.0

    def test_low_vix_tightens_trail(self):
        # VIX 14: adjustment = (14-20)*0.5 = -3%
        assert compute_vix_adjusted_trail(25.0, 14.0) == 22.0

    def test_floor_at_5_pct(self):
        # VIX 5: adjustment = (5-20)*0.5 = -7.5, base 8 → 0.5, floored to 5
        assert compute_vix_adjusted_trail(8.0, 5.0) == 5.0


# ---------------------------------------------------------------------------
# Anti-chase
# ---------------------------------------------------------------------------


class TestAntiChase:
    def test_no_chase(self):
        passed, _ = check_anti_chase(550.0, 550.50)
        assert passed is True

    def test_chase_detected(self):
        # 0.5% move > 0.3% threshold
        passed, reason = check_anti_chase(550.0, 552.80)
        assert passed is False
        assert "Anti-chase" in reason

    def test_custom_threshold(self):
        passed, _ = check_anti_chase(550.0, 553.0, max_move_pct=1.0)
        # 0.55% < 1.0% → ok
        assert passed is True

    def test_no_alert_price(self):
        passed, _ = check_anti_chase(0, 550.0)
        assert passed is True


# ---------------------------------------------------------------------------
# Score-based sizing
# ---------------------------------------------------------------------------


class TestScoreSizing:
    """Flat sizing: all scores >= 78 get 85% budget multiplier."""

    def test_all_qualifying_scores_equal(self):
        """Every score above 78 gets the same fallback: int(5*0.85) = 4."""
        assert score_to_contracts(150) == 4
        assert score_to_contracts(135) == 4
        assert score_to_contracts(120) == 4
        assert score_to_contracts(100) == 4
        assert score_to_contracts(90) == 4
        assert score_to_contracts(78) == 4
        assert score_to_contracts(75) == 4

    def test_reject(self):
        # Score floor aligned to the 0.62 pattern gate (2026-06-15): 62+ trades, below rejected.
        assert score_to_contracts(74) == 4   # was rejected under old 75 floor; now trades
        assert score_to_contracts(62) == 4   # at the floor
        assert score_to_contracts(61) == 0   # just below the 62 floor
        assert score_to_contracts(50) == 0


class TestScoreSizingAffordability:
    """Verify contracts scale with portfolio size (flat 85% mult, 15% pos cap)."""

    def test_expensive_option_skipped_small_account(self):
        """SPX at $15/contract on a $1K account → 15% cap=$150 < $1500 → SKIP."""
        assert score_to_contracts(150, cost_per_contract=1500, balance=1000) == 0

    def test_expensive_option_skipped_medium_account(self):
        """SPX at $15/contract on $5K account → 15% cap=$750 < $1500 → SKIP."""
        assert score_to_contracts(150, cost_per_contract=1500, balance=5000) == 0

    def test_expensive_option_large_account(self):
        """SPX at $15/contract on $50K — slot=$9375, 85%=$7968, 15% cap=$7500 → 5."""
        assert score_to_contracts(150, cost_per_contract=1500, balance=50000) == 5
        assert score_to_contracts(150, cost_per_contract=1500, balance=50000, max_concurrent=4, max_position_pct=20.0) == 5

    def test_cheap_option_small_account(self):
        """$0.50 on $1K → slot=$187, 85%=$159, 15% cap=$150 → 3."""
        assert score_to_contracts(150, cost_per_contract=50, balance=1000) == 3

    def test_cheap_option_scales_with_portfolio(self):
        """$0.50 on $5K → slot=$937, 85%=$796, 15% cap=$750 → 15."""
        assert score_to_contracts(150, cost_per_contract=50, balance=5000) == 15

    def test_score_doesnt_affect_sizing(self):
        """All scores above 78 get same result with same cost/balance."""
        # $1 option on $5K → slot=$937, 85%=$796, 15% cap=$750 → 7
        assert score_to_contracts(150, cost_per_contract=100, balance=5000) == 7
        assert score_to_contracts(100, cost_per_contract=100, balance=5000) == 7
        assert score_to_contracts(78, cost_per_contract=100, balance=5000) == 7

    def test_expensive_option_skip(self):
        """$20 option on $5K → 15% cap=$750 < $2000 → SKIP."""
        assert score_to_contracts(110, cost_per_contract=2000, balance=5000) == 0

    def test_reject_score_ignores_affordability(self):
        """Score <78 returns 0 even if options are cheap."""
        assert score_to_contracts(60, cost_per_contract=10, balance=100000) == 0

    def test_custom_max_position_pct(self):
        """Position cap is binding constraint."""
        # $500/contract, $5K balance, 50% position, 4 concurrent
        # slot: 5K*75%/4=$937, 85%=$796, raw=1, pos_cap: 5K*50%/500=5 → 1
        assert score_to_contracts(
            150, cost_per_contract=500, balance=5000, max_position_pct=50.0, max_concurrent=4,
        ) == 1

    def test_tsla_on_1k_account(self):
        """TSLA at $4.50/contract on $1K → 15% cap=$150 < $450 → SKIP."""
        assert score_to_contracts(164, cost_per_contract=450, balance=1000) == 0

    def test_tsla_on_10k_account(self):
        """TSLA at $4.50/contract on $10K → slot=$1875, 85%=$1593, 15% cap=$1500 → 3."""
        assert score_to_contracts(164, cost_per_contract=450, balance=10000) == 3

    def test_no_balance_info_returns_base(self):
        """Without balance/cost info, returns flat fallback: int(5*0.85) = 4."""
        assert score_to_contracts(150) == 4
        assert score_to_contracts(110) == 4
        assert score_to_contracts(78) == 4

    def test_zero_cost_returns_base(self):
        """Zero cost per contract skips affordability check."""
        assert score_to_contracts(150, cost_per_contract=0, balance=1000) == 4

    def test_scales_with_large_portfolio(self):
        """$80K portfolio should get proportionally more contracts."""
        # $1.00 premium ($100/contract), $80K, 4 slots, slot=$15000, 85%=$12750, 15% cap=$12000 → 120
        assert score_to_contracts(150, cost_per_contract=100, balance=80000) == 120
        # Same for score 130 — flat sizing, same result
        assert score_to_contracts(130, cost_per_contract=100, balance=80000) == 120


# ---------------------------------------------------------------------------
# Time decay zone
# ---------------------------------------------------------------------------


class TestTimeDecayZone:
    def test_not_in_zone_fresh_trade(self):
        now = datetime(2024, 6, 3, 10, 30)
        opened = datetime(2024, 6, 3, 10, 10)  # 20 min ago
        assert is_time_decay_zone(opened, now) is False

    def test_in_zone_by_hold_time(self):
        now = datetime(2024, 6, 3, 10, 50)
        opened = datetime(2024, 6, 3, 10, 0)  # 50 min ago
        assert is_time_decay_zone(opened, now) is True

    def test_in_zone_by_afternoon(self):
        now = datetime(2024, 6, 3, 15, 5)
        opened = datetime(2024, 6, 3, 14, 55)  # 10 min ago
        assert is_time_decay_zone(opened, now) is True

    def test_no_new_high_exit(self):
        now = datetime(2024, 6, 3, 15, 10)
        last_high = datetime(2024, 6, 3, 15, 3)  # 7 min ago
        should_exit, _ = check_time_decay_no_new_high(
            current_premium=2.50, peak_premium=3.00,
            last_new_high_at=last_high, now=now,
        )
        assert should_exit is True

    def test_recent_high_hold(self):
        now = datetime(2024, 6, 3, 15, 10)
        last_high = datetime(2024, 6, 3, 15, 7)  # 3 min ago
        should_exit, _ = check_time_decay_no_new_high(
            current_premium=2.50, peak_premium=3.00,
            last_new_high_at=last_high, now=now,
        )
        assert should_exit is False


# ---------------------------------------------------------------------------
# Theta bleed
# ---------------------------------------------------------------------------


class TestThetaBleed:
    def test_early_trade_no_exit(self):
        now = datetime(2024, 6, 3, 10, 20)
        opened = datetime(2024, 6, 3, 10, 0)
        should_exit, _ = check_theta_bleed(2.00, 1.20, opened, now)
        assert should_exit is False

    def test_long_hold_and_losing(self):
        now = datetime(2024, 6, 3, 11, 0)
        opened = datetime(2024, 6, 3, 10, 0)  # 60 min
        should_exit, _ = check_theta_bleed(2.00, 1.30, opened, now)
        # Down 35% > 30% after 60 min > 45 min
        assert should_exit is True

    def test_long_hold_but_profitable(self):
        now = datetime(2024, 6, 3, 11, 0)
        opened = datetime(2024, 6, 3, 10, 0)
        should_exit, _ = check_theta_bleed(2.00, 2.50, opened, now)
        # Up 25% — not a theta bleed
        assert should_exit is False


# ---------------------------------------------------------------------------
# Consecutive loser pause
# ---------------------------------------------------------------------------


class TestConsecutiveLoserPause:
    def test_no_losses(self):
        can_trade, _ = check_consecutive_loser_pause(0, None)
        assert can_trade is True

    def test_one_loss(self):
        can_trade, _ = check_consecutive_loser_pause(1, None)
        assert can_trade is True

    def test_two_losses_cooling_down(self):
        now = datetime(2024, 6, 3, 10, 10)
        last_loss = datetime(2024, 6, 3, 10, 5)  # 5 min ago
        can_trade, _ = check_consecutive_loser_pause(2, last_loss, now)
        assert can_trade is False

    def test_two_losses_cooldown_expired(self):
        now = datetime(2024, 6, 3, 10, 25)
        last_loss = datetime(2024, 6, 3, 10, 5)  # 20 min ago
        can_trade, _ = check_consecutive_loser_pause(2, last_loss, now)
        assert can_trade is True


# ---------------------------------------------------------------------------
# Phase helper
# ---------------------------------------------------------------------------


class TestGetCurrentPhase:
    def test_no_targets(self):
        assert get_current_phase(None) == 0
        assert get_current_phase(0) == 0

    def test_targets_hit(self):
        assert get_current_phase(1) == 1
        assert get_current_phase(3) == 3
        assert get_current_phase(5) == 5

    def test_capped_at_6(self):
        assert get_current_phase(7) == 6
        assert get_current_phase(10) == 6


# ---------------------------------------------------------------------------
# Dollar-based stair-step trailing stop
# ---------------------------------------------------------------------------


class TestDollarTrail:
    """Vinny's dollar trail: activate at 10%, $20 steps up to $50, $10 steps above."""

    def test_dormant_below_activation(self):
        """Peak profit below 10% of entry → dormant, no exit."""
        # Entry $2.00 ($200/contract), activation = $20
        # Peak $2.15 ($15 profit) < $20 activation
        r = evaluate_dollar_trail(entry_premium=2.00, current_premium=2.10, peak_premium=2.15)
        assert r.should_exit is False
        assert r.active is False

    def test_activates_at_10_pct(self):
        """Peak above 10% → trail activates, stop at activation level."""
        # Entry $2.00, peak $2.25 ($25 profit), current $2.22 ($22 profit)
        # Activation = $20, stop = $20. $22 > $20 → hold
        r = evaluate_dollar_trail(entry_premium=2.00, current_premium=2.22, peak_premium=2.25)
        assert r.active is True
        assert r.should_exit is False
        assert r.stop_level == 20.0  # activation = $20

    def test_exit_at_activation_level(self):
        """Profit drops below activation level → exit."""
        # Entry $2.00, peak $2.30 ($30), current $2.19 ($19)
        # Steps: (30-20)/20 = 0 → stop at $20
        # Profit $19 < stop $20 → exit
        r = evaluate_dollar_trail(entry_premium=2.00, current_premium=2.19, peak_premium=2.30)
        assert r.should_exit is True
        assert r.stop_level == 20.0

    def test_20_dollar_step_holds(self):
        """Peak $45, stop at $40 (one $20 step). Current $42 → hold."""
        # Entry $2.00, peak $2.45 ($45), current $2.42 ($42)
        # Steps: (45-20)/20 = 1 → stop = 20 + 1*20 = $40
        r = evaluate_dollar_trail(entry_premium=2.00, current_premium=2.42, peak_premium=2.45)
        assert r.should_exit is False
        assert r.stop_level == 40.0

    def test_20_dollar_step_exits(self):
        """Peak $45, stop at $40. Current $38 → exit."""
        r = evaluate_dollar_trail(entry_premium=2.00, current_premium=2.38, peak_premium=2.45)
        assert r.should_exit is True
        assert r.stop_level == 40.0

    def test_switches_to_10_steps_above_50(self):
        """Peak $65, stop at $60 ($10 steps above threshold)."""
        # Entry $2.00, peak $2.65 ($65)
        # Phase 1: (50-20)/20 = 1 → phase1_top = $40
        # Phase 2: (65-40)/10 = 2 → stop = 40 + 2*10 = $60
        r = evaluate_dollar_trail(entry_premium=2.00, current_premium=2.62, peak_premium=2.65)
        assert r.should_exit is False
        assert r.stop_level == 60.0

    def test_10_step_exit(self):
        """Peak $65, stop $60. Profit drops to $55 → exit."""
        r = evaluate_dollar_trail(entry_premium=2.00, current_premium=2.55, peak_premium=2.65)
        assert r.should_exit is True
        assert r.stop_level == 60.0

    def test_high_profit_70_stop_at_60(self):
        """Peak $75. Phase1_top=$40, phase2 steps=(75-40)/10=3 → stop=$70."""
        r = evaluate_dollar_trail(entry_premium=2.00, current_premium=2.72, peak_premium=2.75)
        assert r.should_exit is False
        assert r.stop_level == 70.0

    def test_high_profit_exit_at_70(self):
        """Peak $75, stop $70. Profit drops to $65 → exit."""
        r = evaluate_dollar_trail(entry_premium=2.00, current_premium=2.65, peak_premium=2.75)
        assert r.should_exit is True
        assert r.stop_level == 70.0

    def test_expensive_option(self):
        """$5.00 option ($500/contract). 10% activation = $50, 10% step = $50, 25% threshold = $125."""
        # Peak $5.85 ($85 profit), activation $50
        # small_step = $50, threshold = $125
        # Phase 1: (85-50)/50 = 0 → stop = $50
        # Current $5.82 ($82 profit) > $50 → hold
        r = evaluate_dollar_trail(entry_premium=5.00, current_premium=5.82, peak_premium=5.85)
        assert r.active is True
        assert r.stop_level == 50.0

    def test_cheap_option(self):
        """$0.50 option ($50/contract). 10% activation = $5, 10% step = $5, 25% threshold = $12.50."""
        # Peak $0.80 ($30 profit), activation $5
        # small_step = $5, threshold = $12.50
        # Phase 1: (12.50-5)/5 = 1 → phase1_top = $10
        # Phase 2: (30-10)/2.50 = 8 → stop = 10 + 8*2.50 = $30
        # Wait, peak is $30 and stop is $30 → exit. Need higher current.
        # Actually peak $30, phase1_top = $10, above = 20, large_step = $2.50
        # steps = int(20/2.50) = 8 → stop = 10 + 20 = $30. Hmm stop == peak.
        # Let me use a peak that gives a stop below it.
        # Peak $0.77 ($27), phase2 above $10 = $17, steps = int(17/2.50) = 6 → stop = 10+15 = $25
        r = evaluate_dollar_trail(entry_premium=0.50, current_premium=0.76, peak_premium=0.77)
        assert r.should_exit is False
        assert r.stop_level == 25.0

    def test_cheap_option_exit(self):
        """$0.50 option, peak $27, stop $25. Profit drops to $20 → exit."""
        r = evaluate_dollar_trail(entry_premium=0.50, current_premium=0.70, peak_premium=0.77)
        # Profit = $20, stop = $25 → exit
        assert r.should_exit is True
        assert r.stop_level == 25.0

    def test_zero_entry_no_crash(self):
        """Zero entry premium → safe no-op."""
        r = evaluate_dollar_trail(entry_premium=0.0, current_premium=1.00, peak_premium=1.50)
        assert r.should_exit is False
        assert r.active is False

    def test_custom_step_sizes(self):
        """Custom 30% small step, 15% large step, 80% threshold."""
        # Entry $1.00 ($100/contract), activation = 10% = $10
        # small_step = 30% of $100 = $30, threshold = 80% of $100 = $80
        # Peak $1.50 ($50 profit)
        # Phase 1: (50-10)/30 = 1 → stop = 10+30 = $40
        r = evaluate_dollar_trail(
            entry_premium=1.00, current_premium=1.45, peak_premium=1.50,
            activation_pct=10.0, small_step_pct=30.0, step_threshold_pct=80.0, large_step_pct=15.0,
        )
        assert r.stop_level == 40.0
        assert r.should_exit is False
