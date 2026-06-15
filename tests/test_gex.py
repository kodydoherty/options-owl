"""Tests for the harvester's per-ticker dealer-positioning (GEX) aggregation.

Spec: specs/active/2026-06-10_feature-pipeline-expansion_v1.html section 4.1:
    gamma_exposure = gamma * open_interest * 100 * spot
    net_gamma = SUM(call gamma_exposure) - SUM(put gamma_exposure)
Charm/vanna aggregated analogously. Contracts with NULL open_interest are
EXCLUDED from the sums (missing data != zero positioning). Empty/sparse
chains return 0, never NaN.
"""

from __future__ import annotations

import asyncio
import math
from unittest.mock import AsyncMock

from options_owl.harvester import compute_gex_aggregate


def _contract(
    option_type: str = "call",
    gamma: float | None = 0.02,
    charm: float | None = -0.001,
    vanna: float | None = 0.003,
    open_interest: int | None = 1000,
) -> dict:
    return {
        "option_type": option_type,
        "gamma": gamma,
        "charm": charm,
        "vanna": vanna,
        "open_interest": open_interest,
    }


class TestGexMath:
    SPOT = 100.0

    def test_single_call_gamma_exposure(self):
        # gamma * OI * 100 * spot = 0.02 * 1000 * 100 * 100 = 200,000
        agg = compute_gex_aggregate("SPY", [_contract()], self.SPOT)
        assert agg["call_gamma"] == 200_000.0
        assert agg["put_gamma"] == 0.0
        assert agg["net_gamma"] == 200_000.0
        assert agg["total_oi"] == 1000
        assert agg["n_contracts"] == 1
        assert agg["spot"] == self.SPOT
        assert agg["ticker"] == "SPY"

    def test_net_gamma_is_calls_minus_puts(self):
        rows = [
            _contract("call", gamma=0.02, open_interest=1000),  # 200,000
            _contract("put", gamma=0.01, open_interest=500),    # 50,000
        ]
        agg = compute_gex_aggregate("SPY", rows, self.SPOT)
        assert agg["call_gamma"] == 200_000.0
        assert agg["put_gamma"] == 50_000.0
        assert agg["net_gamma"] == 150_000.0
        assert agg["total_oi"] == 1500
        assert agg["n_contracts"] == 2

    def test_put_heavy_chain_yields_negative_net_gamma(self):
        rows = [
            _contract("call", gamma=0.01, open_interest=100),   # 10,000
            _contract("put", gamma=0.03, open_interest=2000),   # 600,000
        ]
        agg = compute_gex_aggregate("QQQ", rows, self.SPOT)
        assert agg["net_gamma"] == 10_000.0 - 600_000.0
        assert agg["net_gamma"] < 0

    def test_charm_and_vanna_aggregated_call_minus_put(self):
        rows = [
            _contract("call", charm=-0.001, vanna=0.003, open_interest=1000),
            _contract("put", charm=-0.002, vanna=0.001, open_interest=1000),
        ]
        agg = compute_gex_aggregate("SPY", rows, self.SPOT)
        scale = 1000 * 100 * self.SPOT
        assert math.isclose(agg["net_charm"], (-0.001 * scale) - (-0.002 * scale))
        assert math.isclose(agg["net_vanna"], (0.003 * scale) - (0.001 * scale))

    def test_values_are_finite(self):
        rows = [_contract("call"), _contract("put")]
        agg = compute_gex_aggregate("SPY", rows, self.SPOT)
        for key in ("net_gamma", "call_gamma", "put_gamma", "net_charm", "net_vanna"):
            assert math.isfinite(agg[key]), f"{key} not finite"


class TestGexEdgeCases:
    def test_empty_chain_returns_zero_not_nan(self):
        agg = compute_gex_aggregate("SPY", [], 100.0)
        assert agg["net_gamma"] == 0.0
        assert agg["call_gamma"] == 0.0
        assert agg["put_gamma"] == 0.0
        assert agg["net_charm"] == 0.0
        assert agg["net_vanna"] == 0.0
        assert agg["total_oi"] == 0
        assert agg["n_contracts"] == 0
        for v in agg.values():
            if isinstance(v, float):
                assert not math.isnan(v)

    def test_null_oi_contracts_excluded(self):
        """NULL open_interest = missing data — excluded entirely from sums."""
        rows = [
            _contract("call", gamma=0.02, open_interest=1000),
            _contract("call", gamma=99.0, open_interest=None),  # must NOT contribute
        ]
        agg = compute_gex_aggregate("SPY", rows, 100.0)
        assert agg["call_gamma"] == 0.02 * 1000 * 100 * 100.0
        assert agg["n_contracts"] == 1
        assert agg["total_oi"] == 1000

    def test_null_gamma_contracts_excluded(self):
        rows = [
            _contract("call", gamma=None, open_interest=1000),
            _contract("put", gamma=0.01, open_interest=500),
        ]
        agg = compute_gex_aggregate("SPY", rows, 100.0)
        assert agg["call_gamma"] == 0.0
        assert agg["put_gamma"] == 0.01 * 500 * 100 * 100.0
        assert agg["n_contracts"] == 1

    def test_null_charm_vanna_treated_as_zero_but_contract_counted(self):
        """Gamma+OI present but charm/vanna missing → contract still in GEX sum."""
        rows = [_contract("call", charm=None, vanna=None, open_interest=1000)]
        agg = compute_gex_aggregate("SPY", rows, 100.0)
        assert agg["call_gamma"] == 0.02 * 1000 * 100 * 100.0
        assert agg["net_charm"] == 0.0
        assert agg["net_vanna"] == 0.0
        assert agg["n_contracts"] == 1

    def test_missing_spot_returns_zero_aggregate(self):
        agg = compute_gex_aggregate("SPY", [_contract()], None)
        assert agg["net_gamma"] == 0.0
        assert agg["n_contracts"] == 0
        assert agg["spot"] is None

    def test_zero_spot_returns_zero_aggregate(self):
        agg = compute_gex_aggregate("SPY", [_contract()], 0.0)
        assert agg["net_gamma"] == 0.0
        assert agg["n_contracts"] == 0

    def test_unknown_option_type_excluded(self):
        rows = [_contract("weird"), _contract("call")]
        agg = compute_gex_aggregate("SPY", rows, 100.0)
        assert agg["n_contracts"] == 1


class TestHarvesterGexWiring:
    """_persist_rows computes the aggregate and writes it via write_gex_ticks_batch."""

    def test_persist_rows_writes_gex_aggregate(self, monkeypatch):
        from options_owl import harvester
        from options_owl.db import postgres as pg
        from options_owl.db import redis_client

        monkeypatch.setattr(pg, "is_connected", lambda: True)
        monkeypatch.setattr(pg, "write_option_ticks_batch", AsyncMock())
        monkeypatch.setattr(pg, "write_stock_tick", AsyncMock())
        gex_mock = AsyncMock()
        monkeypatch.setattr(pg, "write_gex_ticks_batch", gex_mock)
        monkeypatch.setattr(redis_client, "is_connected", lambda: False)

        rows = [
            {
                "contract_ticker": "O:SPY260610C00550000",
                "underlying": "SPY",
                "strike": 550.0,
                "expiry_date": "2026-06-10",
                "option_type": "call",
                "underlying_price": 550.0,
                "bid": 2.45, "ask": 2.55, "bid_size": 10, "ask_size": 12,
                "midpoint": 2.50, "last_trade_price": 2.50,
                "last_trade_ts_ns": None,
                "day_open": None, "day_high": None, "day_low": None,
                "day_close": None, "day_volume": 1000, "day_vwap": None,
                "open_interest": 2000,
                "implied_volatility": 0.30,
                "delta": 0.5, "gamma": 0.04, "theta": -0.05, "vega": 0.1,
                "charm": -0.001, "vanna": 0.002,
            }
        ]
        written = asyncio.run(harvester._persist_rows("SPY", rows))
        assert written == 1

        gex_mock.assert_awaited_once()
        (gex_rows,) = gex_mock.await_args.args
        assert len(gex_rows) == 1
        agg = gex_rows[0]
        assert agg["ticker"] == "SPY"
        # 0.04 * 2000 * 100 * 550 = 4,400,000
        assert agg["net_gamma"] == 0.04 * 2000 * 100 * 550.0
        assert agg["n_contracts"] == 1

        # charm/vanna also flow into the option_ticks write
        ot_call = pg.write_option_ticks_batch.await_args.args[0]
        assert ot_call[0]["charm"] == -0.001
        assert ot_call[0]["vanna"] == 0.002
