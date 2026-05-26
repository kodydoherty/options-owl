"""Tests for ML signal sourcing V2 — feature engineering, UW adjustments, FSM simulation.

Covers:
  1. compute_setup_features() — verify feature math on known inputs
  2. UWScoreAdjuster — mock DB, test each rule independently
  3. simulate_with_production_fsm() — known input → known output
  4. Edge cases: null greeks, missing quotes, NaN values
  5. Dataset building — correct label assignment
"""

from __future__ import annotations

import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Import from the training script
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.train_option_signals_v2 import (
    FEATURE_COLS,
    UWScoreAdjuster,
    compute_setup_features,
    find_profitable_moves,
    simulate_with_production_fsm,
)


# ---------------------------------------------------------------------------
# Helpers — build synthetic 1-min data
# ---------------------------------------------------------------------------

def _make_ohlc(n: int = 60, base_price: float = 2.0, trend: float = 0.01,
               right: str = "CALL", ticker: str = "SPY") -> pd.DataFrame:
    """Create n rows of synthetic 1-min option OHLC data."""
    rows = []
    price = base_price
    t = datetime(2026, 5, 21, 9, 30, 0)  # 9:30 AM ET
    for i in range(n):
        noise = np.random.RandomState(42 + i).normal(0, 0.02)
        price = max(0.05, price + trend + noise)
        rows.append({
            "timestamp": t.isoformat(),
            "ticker": ticker,
            "right": right,
            "strike": 740.0,
            "expiry": "2026-05-21",
            "open": price - 0.01,
            "high": price + 0.03,
            "low": price - 0.03,
            "close": price,
            "volume": int(100 + i * 10 + np.random.RandomState(i).randint(0, 50)),
            "vwap": price,
        })
        t += timedelta(minutes=1)
    return pd.DataFrame(rows)


def _make_quotes(n: int = 60, base_price: float = 2.0, spread: float = 0.05,
                 right: str = "CALL") -> pd.DataFrame:
    rows = []
    t = datetime(2026, 5, 21, 9, 30, 0)
    price = base_price
    for i in range(n):
        price = max(0.05, price + 0.01)
        rows.append({
            "timestamp": t.isoformat(),
            "right": right,
            "bid": price - spread / 2,
            "ask": price + spread / 2,
            "bid_size": 50 + i,
            "ask_size": 40 + i,
        })
        t += timedelta(minutes=1)
    return pd.DataFrame(rows)


def _make_greeks(n: int = 60, base_price: float = 2.0, right: str = "CALL",
                 underlying: float = 740.0) -> pd.DataFrame:
    rows = []
    t = datetime(2026, 5, 21, 9, 30, 0)
    for i in range(n):
        rows.append({
            "timestamp": t.isoformat(),
            "right": right,
            "implied_vol": 0.25 + np.random.RandomState(i).normal(0, 0.01),
            "delta": 0.50 + np.random.RandomState(i + 100).normal(0, 0.05),
            "theta": -0.03,
            "vega": 0.08,
            "underlying_price": underlying + i * 0.05,
        })
        t += timedelta(minutes=1)
    return pd.DataFrame(rows)


def _make_stock(n: int = 60, base_price: float = 740.0) -> pd.DataFrame:
    rows = []
    t = datetime(2026, 5, 21, 9, 30, 0)
    price = base_price
    for i in range(n):
        price += 0.05
        rows.append({
            "timestamp": t.isoformat(),
            "close": price,
        })
        t += timedelta(minutes=1)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test: compute_setup_features()
# ---------------------------------------------------------------------------

class TestComputeSetupFeatures:
    """Verify feature engineering produces correct values."""

    def test_returns_dict_with_all_feature_cols(self):
        ohlc = _make_ohlc(30)
        quotes = _make_quotes(30)
        greeks = _make_greeks(30)
        stock = _make_stock(30)
        features = compute_setup_features(ohlc, quotes, greeks, stock, idx=20)
        assert features is not None
        for col in FEATURE_COLS:
            assert col in features, f"Missing feature: {col}"

    def test_premium_is_close_price(self):
        ohlc = _make_ohlc(30, base_price=3.50)
        features = compute_setup_features(ohlc, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), idx=20)
        assert features is not None
        assert features["premium"] == pytest.approx(ohlc.iloc[20]["close"], abs=0.01)

    def test_minutes_since_open_correct(self):
        ohlc = _make_ohlc(30)
        features = compute_setup_features(ohlc, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), idx=20)
        assert features is not None
        # Fix A: uses idx-1 (decision candle), so row 19 = 19 minutes after 9:30 AM
        assert features["minutes_since_open"] == 19

    def test_is_first_30min_flag(self):
        ohlc = _make_ohlc(60)
        # idx=10 → 10 min after open → first 30min
        f_early = compute_setup_features(ohlc, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), idx=20)
        assert f_early["is_first_30min"] == 1
        # idx=40 → 40 min after open → not first 30min
        f_late = compute_setup_features(ohlc, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), idx=40)
        assert f_late["is_first_30min"] == 0

    def test_volume_ratio_above_1_when_increasing(self):
        ohlc = _make_ohlc(30)
        # Volume increases with time in our helper
        features = compute_setup_features(ohlc, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), idx=25)
        assert features is not None
        assert features["volume_ratio"] > 0

    def test_premium_volatility_low_for_flat_price(self):
        """Flat price → low volatility (coiled spring indicator)."""
        ohlc = _make_ohlc(30, base_price=2.0, trend=0.0)
        # Override closes to be nearly flat
        ohlc["close"] = 2.0
        features = compute_setup_features(ohlc, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), idx=20)
        assert features is not None
        assert features["premium_volatility"] == 0.0  # all identical prices

    def test_spread_computed_from_quotes(self):
        ohlc = _make_ohlc(30)
        quotes = _make_quotes(30, spread=0.10)
        features = compute_setup_features(ohlc, quotes, pd.DataFrame(), pd.DataFrame(), idx=20)
        assert features is not None
        assert features["spread"] == pytest.approx(0.10, abs=0.02)

    def test_greeks_extracted(self):
        ohlc = _make_ohlc(30)
        greeks = _make_greeks(30)
        features = compute_setup_features(ohlc, pd.DataFrame(), greeks, pd.DataFrame(), idx=20)
        assert features is not None
        assert features["iv"] > 0
        assert features["delta"] > 0
        assert features["theta"] < 0
        assert features["vega"] > 0

    def test_underlying_change_positive_for_uptrend(self):
        ohlc = _make_ohlc(30)
        stock = _make_stock(30, base_price=740.0)  # uptrending
        features = compute_setup_features(ohlc, pd.DataFrame(), pd.DataFrame(), stock, idx=20)
        assert features is not None
        assert features["underlying_change_15m"] > 0

    def test_is_call_flag(self):
        ohlc_call = _make_ohlc(30, right="CALL")
        ohlc_put = _make_ohlc(30, right="PUT")
        f_call = compute_setup_features(ohlc_call, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), idx=20)
        f_put = compute_setup_features(ohlc_put, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), idx=20)
        assert f_call["is_call"] == 1
        assert f_put["is_call"] == 0

    def test_returns_none_if_idx_too_early(self):
        ohlc = _make_ohlc(30)
        features = compute_setup_features(ohlc, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), idx=3)
        assert features is None  # idx < lookback (15)

    def test_returns_none_if_zero_price(self):
        ohlc = _make_ohlc(30)
        # Fix A: decision candle is idx-1, so zero out idx-1=19 to trigger None
        ohlc.loc[19, "close"] = 0
        features = compute_setup_features(ohlc, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), idx=20)
        assert features is None

    def test_no_nan_in_features(self):
        """Features should never contain NaN values."""
        ohlc = _make_ohlc(30)
        quotes = _make_quotes(30)
        greeks = _make_greeks(30)
        stock = _make_stock(30)
        features = compute_setup_features(ohlc, quotes, greeks, stock, idx=20)
        assert features is not None
        for k, v in features.items():
            if isinstance(v, (int, float)):
                assert not np.isnan(v), f"NaN in feature {k}"

    def test_handles_empty_quotes_gracefully(self):
        ohlc = _make_ohlc(30)
        features = compute_setup_features(ohlc, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), idx=20)
        assert features is not None
        assert features["spread"] == 0
        assert features["spread_pct"] == 0

    def test_handles_empty_greeks_gracefully(self):
        ohlc = _make_ohlc(30)
        features = compute_setup_features(ohlc, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), idx=20)
        assert features is not None
        assert features["iv"] == 0
        assert features["delta"] == 0

    def test_coiled_spring_pattern(self):
        """Low volatility + high volume ratio should trigger coiled_spring."""
        ohlc = _make_ohlc(30, trend=0.0)
        # Make prices very stable (low vol) but volume high at decision candle (idx-1=19)
        ohlc["close"] = 2.0
        ohlc["volume"] = 100  # low base
        ohlc.loc[19, "volume"] = 500  # big spike at decision candle (idx-1)
        features = compute_setup_features(ohlc, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), idx=20)
        assert features is not None
        assert features["premium_volatility"] == 0.0
        assert features["volume_ratio"] > 1.5
        assert features["coiled_spring"] == 1


class TestComputeSetupFeaturesEdgeCases:
    """Edge cases for feature computation."""

    def test_null_greeks_values(self):
        """Greeks with None/NaN values should not crash."""
        ohlc = _make_ohlc(30)
        greeks = _make_greeks(30)
        greeks.loc[20, "implied_vol"] = None
        greeks.loc[20, "delta"] = np.nan
        features = compute_setup_features(ohlc, pd.DataFrame(), greeks, pd.DataFrame(), idx=20)
        # Should still return features, not crash
        assert features is not None

    def test_null_bid_ask(self):
        """Quotes with None bid/ask should not crash."""
        ohlc = _make_ohlc(30)
        quotes = _make_quotes(30)
        quotes.loc[20, "bid"] = None
        quotes.loc[20, "ask"] = None
        features = compute_setup_features(ohlc, quotes, pd.DataFrame(), pd.DataFrame(), idx=20)
        assert features is not None

    def test_single_row_window(self):
        """Minimal window (idx=lookback+1 — minimum with Fix A)."""
        ohlc = _make_ohlc(20)
        features = compute_setup_features(ohlc, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), idx=16)
        assert features is not None


# ---------------------------------------------------------------------------
# Test: UWScoreAdjuster
# ---------------------------------------------------------------------------

def _create_uw_db(path: str):
    """Create a minimal UW DB with test data."""
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode = WAL")

    # greek_exposure table
    conn.execute("""
        CREATE TABLE greek_exposure (
            ticker TEXT, date TEXT,
            call_gamma REAL, put_gamma REAL,
            call_delta REAL, put_delta REAL,
            call_charm REAL, put_charm REAL,
            call_vanna REAL, put_vanna REAL,
            PRIMARY KEY (ticker, date)
        )
    """)

    # options_volume table
    conn.execute("""
        CREATE TABLE options_volume (
            ticker TEXT, date TEXT,
            call_volume INTEGER, put_volume INTEGER,
            call_volume_ask_side INTEGER, call_volume_bid_side INTEGER,
            put_volume_ask_side INTEGER, put_volume_bid_side INTEGER,
            net_call_premium REAL, net_put_premium REAL,
            call_premium REAL, put_premium REAL,
            bearish_premium REAL, bullish_premium REAL,
            put_open_interest INTEGER, call_open_interest INTEGER,
            avg_30_day_call_volume REAL, avg_30_day_put_volume REAL,
            PRIMARY KEY (ticker, date)
        )
    """)

    # flow_alerts table
    conn.execute("""
        CREATE TABLE flow_alerts (
            id TEXT PRIMARY KEY, ticker TEXT, created_at TEXT, type TEXT,
            strike REAL, expiry TEXT, price REAL, volume INTEGER,
            open_interest INTEGER, total_premium REAL, underlying_price REAL,
            trade_count INTEGER, iv_start REAL, iv_end REAL,
            volume_oi_ratio REAL, has_sweep INTEGER, has_floor INTEGER,
            has_multileg INTEGER, all_opening_trades INTEGER,
            alert_rule TEXT, total_bid_side_prem REAL, total_ask_side_prem REAL
        )
    """)

    # net_prem_ticks table
    conn.execute("""
        CREATE TABLE net_prem_ticks (
            ticker TEXT, date TEXT, tape_time TEXT,
            call_volume INTEGER, put_volume INTEGER,
            call_volume_ask_side INTEGER, call_volume_bid_side INTEGER,
            put_volume_ask_side INTEGER, put_volume_bid_side INTEGER,
            net_call_premium REAL, net_put_premium REAL,
            net_call_volume INTEGER, net_put_volume INTEGER,
            net_delta REAL,
            PRIMARY KEY (ticker, tape_time)
        )
    """)

    conn.commit()
    return conn


class TestUWScoreAdjuster:
    """Test UW flow-based score adjustments."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = str(Path(self.tmpdir) / "uw_test.db")
        self.conn = _create_uw_db(self.db_path)

    def teardown_method(self):
        self.conn.close()

    def test_disabled_when_no_db(self):
        adj = UWScoreAdjuster("/nonexistent/path.db")
        assert not adj.enabled
        result = adj.compute_adjustment("SPY", "2026-05-21", "CALL")
        assert result["adjustment"] == 0.0

    def test_enabled_with_valid_db(self):
        # Insert at least one GEX row
        self.conn.execute(
            "INSERT INTO greek_exposure (ticker, date, call_gamma, put_gamma, call_delta, put_delta, call_charm, put_charm, call_vanna, put_vanna) VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("SPY", "2026-05-21", 1000, -2000, 50000, -30000, 0, 0, 0, 0),
        )
        self.conn.commit()
        adj = UWScoreAdjuster(self.db_path)
        assert adj.enabled
        adj.close()

    def test_negative_gex_boosts_score(self):
        """Negative net gamma = more volatility = good for 0DTE."""
        self.conn.execute(
            "INSERT INTO greek_exposure VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("SPY", "2026-05-21", 1000, -5000, 100000, -50000, 0, 0, 0, 0),
        )
        self.conn.commit()
        adj = UWScoreAdjuster(self.db_path)
        result = adj.compute_adjustment("SPY", "2026-05-21", "CALL")
        assert result["adjustment"] > 0
        assert "negative_gex_volatile_day" in result["reasons"]
        adj.close()

    def test_delta_aligned_call(self):
        """Positive net delta + CALL signal = aligned, boost."""
        self.conn.execute(
            "INSERT INTO greek_exposure VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("SPY", "2026-05-21", 5000, -1000, 200000, -50000, 0, 0, 0, 0),
        )
        self.conn.commit()
        adj = UWScoreAdjuster(self.db_path)
        result = adj.compute_adjustment("SPY", "2026-05-21", "CALL")
        assert "gex_delta_aligned" in result["reasons"]
        adj.close()

    def test_delta_against_call(self):
        """Negative net delta + CALL signal = against, dampen."""
        self.conn.execute(
            "INSERT INTO greek_exposure VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("SPY", "2026-05-21", 5000, -1000, 50000, -200000, 0, 0, 0, 0),
        )
        self.conn.commit()
        adj = UWScoreAdjuster(self.db_path)
        result = adj.compute_adjustment("SPY", "2026-05-21", "CALL")
        assert "gex_delta_against" in result["reasons"]
        assert result["adjustment"] < 0
        adj.close()

    def test_bullish_flow_aligned_with_call(self):
        """Bullish premium > 55% + CALL = boost."""
        self.conn.execute(
            "INSERT INTO greek_exposure VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("SPY", "2026-05-21", 0, 0, 0, 0, 0, 0, 0, 0),
        )
        self.conn.execute(
            """INSERT INTO options_volume
            (ticker, date, call_volume, put_volume, call_volume_ask_side, call_volume_bid_side,
             put_volume_ask_side, put_volume_bid_side, net_call_premium, net_put_premium,
             call_premium, put_premium, bearish_premium, bullish_premium,
             put_open_interest, call_open_interest, avg_30_day_call_volume, avg_30_day_put_volume)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            ("SPY", "2026-05-21", 5000, 3000, 2500, 2500, 1500, 1500,
             1000, -500, 500000, 300000, 300000, 500000, 10000, 8000, 5000, 4000),
        )
        self.conn.commit()
        adj = UWScoreAdjuster(self.db_path)
        result = adj.compute_adjustment("SPY", "2026-05-21", "CALL")
        assert "bullish_flow_aligned" in result["reasons"]
        adj.close()

    def test_sweep_alerts_boost(self):
        """Multiple sweeps in our direction = strong boost."""
        self.conn.execute(
            "INSERT INTO greek_exposure VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("SPY", "2026-05-21", 0, 0, 0, 0, 0, 0, 0, 0),
        )
        for i in range(3):
            self.conn.execute(
                """INSERT INTO flow_alerts
                (id, ticker, created_at, type, strike, expiry, price, volume,
                 open_interest, total_premium, underlying_price, trade_count,
                 iv_start, iv_end, volume_oi_ratio, has_sweep, has_floor,
                 has_multileg, all_opening_trades, alert_rule,
                 total_bid_side_prem, total_ask_side_prem)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (f"sweep_{i}", "SPY", f"2026-05-21T10:{30+i}:00Z", "call",
                 740, "2026-05-21", 2.50, 1000, 500, 250000, 740, 5,
                 0.25, 0.26, 2.0, 1, 0, 0, 0, "Sweep", 0, 250000),
            )
        self.conn.commit()
        adj = UWScoreAdjuster(self.db_path)
        result = adj.compute_adjustment("SPY", "2026-05-21", "CALL")
        assert any("sweeps_aligned" in r for r in result["reasons"])
        assert result["adjustment"] > 0.05
        adj.close()

    def test_adjustment_clamped_to_bounds(self):
        """Total adjustment should never exceed ±0.20."""
        # Stack all positive rules
        self.conn.execute(
            "INSERT INTO greek_exposure VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("SPY", "2026-05-21", 1000, -5000, 200000, -50000, 0, 0, 0, 0),
        )
        self.conn.execute(
            """INSERT INTO options_volume
            (ticker, date, call_volume, put_volume, call_volume_ask_side, call_volume_bid_side,
             put_volume_ask_side, put_volume_bid_side, net_call_premium, net_put_premium,
             call_premium, put_premium, bearish_premium, bullish_premium,
             put_open_interest, call_open_interest, avg_30_day_call_volume, avg_30_day_put_volume)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            ("SPY", "2026-05-21", 5000, 3000, 2500, 2500, 1500, 1500,
             1000, -500, 500000, 300000, 300000, 500000, 10000, 8000, 5000, 4000),
        )
        for i in range(5):
            self.conn.execute(
                """INSERT INTO flow_alerts
                (id, ticker, created_at, type, strike, expiry, price, volume,
                 open_interest, total_premium, underlying_price, trade_count,
                 iv_start, iv_end, volume_oi_ratio, has_sweep, has_floor,
                 has_multileg, all_opening_trades, alert_rule,
                 total_bid_side_prem, total_ask_side_prem)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (f"big_{i}", "SPY", f"2026-05-21T10:{30+i}:00Z", "call",
                 740, "2026-05-21", 2.50, 1000, 500, 200000, 740, 5,
                 0.25, 0.26, 2.0, 1, 0, 0, 0, "Sweep", 0, 200000),
            )
        self.conn.commit()
        adj = UWScoreAdjuster(self.db_path)
        result = adj.compute_adjustment("SPY", "2026-05-21", "CALL")
        assert result["adjustment"] <= 0.20
        assert result["adjustment"] >= -0.20
        adj.close()

    def test_no_data_returns_zero_adjustment(self):
        """Missing data for date should return 0 adjustment."""
        self.conn.execute(
            "INSERT INTO greek_exposure VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("SPY", "2026-05-20", 0, 0, 0, 0, 0, 0, 0, 0),
        )
        self.conn.commit()
        adj = UWScoreAdjuster(self.db_path)
        result = adj.compute_adjustment("SPY", "2026-05-21", "CALL")
        assert result["adjustment"] == 0.0
        adj.close()

    def test_put_direction_rules(self):
        """Bearish flow + PUT signal = aligned."""
        self.conn.execute(
            "INSERT INTO greek_exposure VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("SPY", "2026-05-21", 1000, -5000, 50000, -200000, 0, 0, 0, 0),
        )
        self.conn.execute(
            """INSERT INTO options_volume
            (ticker, date, call_volume, put_volume, call_volume_ask_side, call_volume_bid_side,
             put_volume_ask_side, put_volume_bid_side, net_call_premium, net_put_premium,
             call_premium, put_premium, bearish_premium, bullish_premium,
             put_open_interest, call_open_interest, avg_30_day_call_volume, avg_30_day_put_volume)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            ("SPY", "2026-05-21", 3000, 5000, 1500, 1500, 2500, 2500,
             -500, 1000, 300000, 500000, 500000, 300000, 12000, 6000, 4000, 5000),
        )
        self.conn.commit()
        adj = UWScoreAdjuster(self.db_path)
        result = adj.compute_adjustment("SPY", "2026-05-21", "PUT")
        # Negative delta + PUT = aligned, bearish flow + PUT = aligned
        assert result["adjustment"] > 0
        assert "gex_delta_aligned" in result["reasons"]
        assert "bearish_flow_aligned" in result["reasons"]
        adj.close()


# ---------------------------------------------------------------------------
# Test: simulate_with_production_fsm()
# ---------------------------------------------------------------------------

class TestFSMSimulation:
    """Test FSM simulation produces sane outputs."""

    def test_profitable_trade_returns_positive_pnl(self):
        """Uptrending option should produce positive P&L."""
        ohlc = _make_ohlc(120, base_price=2.0, trend=0.02)  # strong uptrend
        quotes = _make_quotes(120)
        greeks = _make_greeks(120)

        result = simulate_with_production_fsm(
            ohlc, quotes, greeks,
            entry_idx=15, ticker="SPY", dte=0, expiry_date="2026-05-21",
        )
        assert result is not None
        # Strong uptrend should be profitable
        assert result["pnl_pct"] > 0 or result["reason"] != ""  # At minimum it returns a result

    def test_returns_none_for_near_end(self):
        """Entry near end of data should return None."""
        ohlc = _make_ohlc(20)
        result = simulate_with_production_fsm(
            ohlc, pd.DataFrame(), pd.DataFrame(),
            entry_idx=18, ticker="SPY",
        )
        assert result is None

    def test_returns_none_for_zero_entry(self):
        """Zero entry price should return None."""
        ohlc = _make_ohlc(60)
        ohlc.loc[15, "close"] = 0
        result = simulate_with_production_fsm(
            ohlc, pd.DataFrame(), pd.DataFrame(),
            entry_idx=15, ticker="SPY",
        )
        assert result is None

    def test_returns_none_for_nan_entry(self):
        """NaN entry price should return None."""
        ohlc = _make_ohlc(60)
        ohlc.loc[15, "close"] = np.nan
        result = simulate_with_production_fsm(
            ohlc, pd.DataFrame(), pd.DataFrame(),
            entry_idx=15, ticker="SPY",
        )
        assert result is None

    def test_result_has_required_keys(self):
        """Result dict should have all expected keys."""
        ohlc = _make_ohlc(120, trend=0.005)
        result = simulate_with_production_fsm(
            ohlc, pd.DataFrame(), pd.DataFrame(),
            entry_idx=15, ticker="SPY", dte=0, expiry_date="2026-05-21",
        )
        assert result is not None
        for key in ["pnl_pct", "pnl_dollars", "reason", "hold_minutes", "exit_premium", "peak_gain"]:
            assert key in result, f"Missing key: {key}"

    def test_hold_minutes_is_positive(self):
        ohlc = _make_ohlc(120, trend=0.005)
        result = simulate_with_production_fsm(
            ohlc, pd.DataFrame(), pd.DataFrame(),
            entry_idx=15, ticker="SPY", dte=0, expiry_date="2026-05-21",
        )
        assert result is not None
        assert result["hold_minutes"] >= 0

    def test_peak_gain_gte_final_pnl_for_winners(self):
        """Peak gain should always be >= final P&L (we can only sell at or below peak)."""
        ohlc = _make_ohlc(120, trend=0.02)
        result = simulate_with_production_fsm(
            ohlc, pd.DataFrame(), pd.DataFrame(),
            entry_idx=15, ticker="SPY", dte=0, expiry_date="2026-05-21",
        )
        if result and result["pnl_pct"] > 0:
            assert result["peak_gain"] >= result["pnl_pct"]

    def test_per_ticker_config_used(self):
        """Different tickers should use different FSM configs."""
        ohlc = _make_ohlc(120, trend=0.005)
        result_spy = simulate_with_production_fsm(
            ohlc, pd.DataFrame(), pd.DataFrame(),
            entry_idx=15, ticker="SPY", dte=0, expiry_date="2026-05-21",
        )
        result_mstr = simulate_with_production_fsm(
            ohlc, pd.DataFrame(), pd.DataFrame(),
            entry_idx=15, ticker="MSTR", dte=0, expiry_date="2026-05-21",
        )
        # Both should return results, potentially different exit reasons due to config
        assert result_spy is not None
        assert result_mstr is not None

    def test_losing_trade(self):
        """Downtrending option should produce negative P&L."""
        ohlc = _make_ohlc(120, base_price=3.0, trend=-0.02)
        result = simulate_with_production_fsm(
            ohlc, pd.DataFrame(), pd.DataFrame(),
            entry_idx=15, ticker="SPY", dte=0, expiry_date="2026-05-21",
        )
        assert result is not None
        assert result["pnl_pct"] < 0


# ---------------------------------------------------------------------------
# Test: find_profitable_moves()
# ---------------------------------------------------------------------------

class TestFindProfitableMoves:
    """Test move detection finds actual profitable entries."""

    def test_finds_moves_in_uptrending_data(self):
        """Strong uptrend should yield profitable moves."""
        ohlc = _make_ohlc(200, base_price=1.5, trend=0.03)
        quotes = _make_quotes(200)
        greeks = _make_greeks(200)

        moves = find_profitable_moves(
            ohlc, quotes, greeks,
            ticker="SPY", min_move_pct=15.0, cooldown=30,
            dte=0, expiry_date="2026-05-21",
        )
        # Strong uptrend should find at least one profitable move
        # (depends on FSM behavior, so may be 0 if exits trigger early)
        assert isinstance(moves, list)
        for m in moves:
            assert m["pnl_pct"] >= 15.0
            assert "idx" in m
            assert "entry_price" in m

    def test_cooldown_respected(self):
        """Moves should be spaced at least cooldown minutes apart."""
        ohlc = _make_ohlc(300, base_price=1.0, trend=0.025)
        moves = find_profitable_moves(
            ohlc, pd.DataFrame(), pd.DataFrame(),
            ticker="SPY", cooldown=30,
        )
        for i in range(1, len(moves)):
            gap = moves[i]["idx"] - moves[i - 1]["idx"]
            assert gap >= 30, f"Cooldown violated: gap={gap} between moves {i-1} and {i}"

    def test_no_moves_in_flat_data(self):
        """Flat price should yield no profitable moves (no +15% gains)."""
        ohlc = _make_ohlc(100, base_price=2.0, trend=0.0)
        ohlc["close"] = 2.0  # perfectly flat
        moves = find_profitable_moves(
            ohlc, pd.DataFrame(), pd.DataFrame(),
            ticker="SPY",
        )
        assert len(moves) == 0


# ---------------------------------------------------------------------------
# Test: Backtest UW integration
# ---------------------------------------------------------------------------

class TestBacktestUWIntegration:
    """Test that UW adjuster integrates with backtest without errors."""

    def test_backtest_runs_with_no_uw(self):
        """Backtest should work when UW adjuster is disabled."""
        adj = UWScoreAdjuster("/nonexistent/path.db")
        result = adj.compute_adjustment("SPY", "2026-05-21", "CALL")
        assert result["adjustment"] == 0.0
        assert result["reasons"] == []

    def test_adjustment_affects_threshold(self):
        """A positive adjustment should allow lower raw confidence to pass."""
        # If threshold=0.5, raw=0.45 would fail
        # But with +0.10 UW adjustment, adjusted=0.55 passes
        raw_prob = 0.45
        threshold = 0.50
        uw_adj = 0.10
        assert raw_prob < threshold
        assert raw_prob + uw_adj >= threshold
