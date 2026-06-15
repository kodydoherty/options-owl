"""Unit tests for the UW flow collector filter + signal builder + gate bypass (Track 4)."""

import pytest

from options_owl.collectors.uw_flow_collector import (
    FlowSignal,
    evaluate_flow_alert,
    flow_signal_to_trade_signal,
)
from options_owl.config.settings import Settings
from options_owl.models.signals import BotSource, Direction, Sentiment


def _settings(**ov):
    base = dict(
        UW_FLOW_PUT_TICKERS="META,AMZN,AAPL,TSLA",
        UW_FLOW_CALL_TICKERS="TSLA,AAPL,AMD,AVGO,PLTR",
        UW_FLOW_MIN_PREMIUM=250_000.0,
        UW_FLOW_ASK_FRAC=0.60,
        UW_FLOW_REQUIRE_SWEEP=True,
    )
    base.update(ov)
    return Settings(**base)


def _alert(**ov):
    a = dict(ticker="META", type="put", total_premium="500000",
             total_ask_side_prem="400000", has_sweep=True, volume_oi_ratio="2.5",
             strike="567.5", expiry="2026-06-26", option_chain="META260626P00567500")
    a.update(ov)
    return a


class TestFlowFilter:
    def test_qualifying_put_sweep(self):
        fs = evaluate_flow_alert(_alert(), _settings())
        assert isinstance(fs, FlowSignal)
        assert fs.ticker == "META" and fs.direction == Direction.PUT
        assert fs.ask_frac == 0.8 and fs.strike == 567.5

    def test_qualifying_call_sweep(self):
        fs = evaluate_flow_alert(_alert(ticker="AMD", type="call"), _settings())
        assert fs is not None and fs.direction == Direction.CALL

    def test_put_on_non_put_whitelist_rejected(self):
        # NVDA is excluded from the PUT whitelist (phase-5 loser)
        assert evaluate_flow_alert(_alert(ticker="NVDA"), _settings()) is None

    def test_call_on_non_call_whitelist_rejected(self):
        # META is a PUT name, not a CALL name
        assert evaluate_flow_alert(_alert(ticker="META", type="call"), _settings()) is None

    def test_below_premium_floor_rejected(self):
        assert evaluate_flow_alert(_alert(total_premium="100000", total_ask_side_prem="90000"), _settings()) is None

    def test_bid_side_dominant_rejected(self):
        # puts SOLD (bid-side) = not bearish conviction
        assert evaluate_flow_alert(_alert(total_ask_side_prem="100000"), _settings()) is None

    def test_no_sweep_rejected_when_required(self):
        assert evaluate_flow_alert(_alert(has_sweep=False), _settings()) is None

    def test_no_sweep_allowed_when_not_required(self):
        assert evaluate_flow_alert(_alert(has_sweep=False), _settings(UW_FLOW_REQUIRE_SWEEP=False)) is not None

    def test_malformed_alert_no_crash(self):
        assert evaluate_flow_alert({}, _settings()) is None
        assert evaluate_flow_alert({"type": "put", "ticker": "META", "total_premium": None}, _settings()) is None

    def test_unknown_option_type_rejected(self):
        # no valid type AND no parseable OCC chain → rejected
        assert evaluate_flow_alert(_alert(type="spread", option_chain=""), _settings()) is None
        assert evaluate_flow_alert(_alert(type="spread", option_chain="GARBAGE"), _settings()) is None

    def test_ws_format_type_from_occ_chain(self):
        # live WS has no 'type' field — derive put/call from the option_chain OCC symbol
        ws = _alert(ticker="TSLA", option_chain="TSLA260626C00415000")  # TSLA in default call WL
        for k in ("type", "strike", "expiry"):  # real WS has none of these — only the OCC chain
            ws.pop(k, None)
        fs = evaluate_flow_alert(ws, _settings())
        assert fs is not None and fs.direction == Direction.CALL and fs.strike == 415.0
        assert fs.expiry == "2026-06-26"


class TestSignalBuilder:
    def test_put_signal_shape(self):
        fs = evaluate_flow_alert(_alert(), _settings())
        sig = flow_signal_to_trade_signal(fs, underlying=560.0)
        assert sig.bot_source == BotSource.UW_FLOW
        assert sig.direction == Direction.PUT and sig.sentiment == Sentiment.BEARISH
        assert sig.ticker == "META" and sig.entry_price == 560.0

    def test_call_signal_shape(self):
        fs = evaluate_flow_alert(_alert(ticker="TSLA", type="call"), _settings())
        sig = flow_signal_to_trade_signal(fs)
        assert sig.direction == Direction.CALL and sig.sentiment == Sentiment.BULLISH


class TestGateBypass:
    """Flow signals must SKIP the directional/signal-quality gates; others must not."""

    def _flow_put(self):
        return flow_signal_to_trade_signal(evaluate_flow_alert(_alert(), _settings()))

    def _ml_put(self):
        s = self._flow_put()
        return s.model_copy(update={"bot_source": BotSource.ML_SOURCING})

    @pytest.mark.asyncio
    async def test_flow_bypasses_all_four_gates(self):
        from options_owl.risk.pipeline import (
            DirectionalRegimeGate, GateResult, PutBearishConfirmGate,
            PutMarketDirectionGate, PutTickerExclusionGate,
        )
        st = Settings(ENABLE_DIRECTIONAL_REGIME=True, PUT_EXCLUDED_TICKERS="META,AMZN")
        ctx = {"signal": self._flow_put(), "settings": st}
        for G in (PutTickerExclusionGate, PutMarketDirectionGate,
                  PutBearishConfirmGate, DirectionalRegimeGate):
            out = await G().evaluate(ctx)
            assert out.result == GateResult.SKIP, f"{G.__name__} did not bypass flow"
            assert "flow" in out.reason.lower()

    @pytest.mark.asyncio
    async def test_non_flow_put_is_NOT_bypassed_by_exclusion(self):
        # an ML-sourced PUT on an excluded ticker must still be blocked (not bypassed)
        from options_owl.risk.pipeline import GateResult, PutTickerExclusionGate
        st = Settings(PUT_EXCLUDED_TICKERS="META,AMZN")
        out = await PutTickerExclusionGate().evaluate({"signal": self._ml_put(), "settings": st})
        assert out.result != GateResult.SKIP or "flow" not in out.reason.lower()

    @pytest.mark.asyncio
    async def test_flow_bypasses_blocklist_but_ml_does_not(self):
        # MU is blocklisted for general-ML, but flow has its own validated whitelist → bypass.
        from options_owl.risk.pipeline import BlockedTickerGate, GateResult
        st = Settings(BLOCKED_TICKERS="MU,MSFT")
        flow = self._flow_put().model_copy(update={"ticker": "MU"})
        ml = self._ml_put().model_copy(update={"ticker": "MU"})
        out_flow = await BlockedTickerGate().evaluate({"signal": flow, "settings": st})
        out_ml = await BlockedTickerGate().evaluate({"signal": ml, "settings": st})
        assert out_flow.result == GateResult.SKIP and "flow" in out_flow.reason.lower()
        assert out_ml.result == GateResult.FAIL  # general-ML MU still blocked


class TestCollectorWSIntegration:
    """WS message → parse → filter → on_signal dispatch (mocked socket)."""

    @pytest.mark.asyncio
    async def test_ws_message_dispatches_qualifying_signal(self, monkeypatch):
        import json as _json

        from options_owl.collectors import uw_flow_collector as mod

        qualifying = _json.dumps(_alert())                      # META put sweep → should fire
        nonqualifying = _json.dumps(_alert(ticker="NVDA"))      # not on PUT whitelist → ignored

        class _FakeWS:
            def __init__(self):
                self.sent = []
            async def send(self, m):
                self.sent.append(m)
            def __aiter__(self):
                async def gen():
                    yield qualifying
                    yield nonqualifying
                    raise StopExitLoop  # break the infinite reconnect after the feed drains
                return gen()

        class _FakeConn:
            async def __aenter__(self):
                return _FakeWS()
            async def __aexit__(self, *a):
                return False

        class StopExitLoop(Exception):
            pass

        monkeypatch.setattr(mod, "websockets", type("W", (), {"connect": staticmethod(lambda *a, **k: _FakeConn())}))

        captured = []
        async def on_signal(fs):
            captured.append(fs)

        st = _settings(UNUSUAL_WHALES_API_KEY="test")
        # run_uw_flow_collector loops forever on reconnect; the StopExitLoop from the
        # async-iter ends up in its except → sleeps → we cancel via timeout.
        import asyncio
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(mod.run_uw_flow_collector(st, on_signal), timeout=0.5)

        # only the qualifying META put should have dispatched
        assert len(captured) == 1
        assert captured[0].ticker == "META" and captured[0].direction == Direction.PUT


class TestConvictionSizing:
    """Stage D: conviction multiplier + propagation + sizing integration."""

    def test_propagates_conviction_fields(self):
        fs = evaluate_flow_alert(_alert(), _settings())
        fs.cluster_count = 3
        ts = flow_signal_to_trade_signal(fs, underlying=100.0)
        assert ts.flow_cluster_count == 3
        assert ts.flow_total_premium == fs.total_premium
        assert ts.flow_ask_frac == fs.ask_frac

    def test_mult_single_stock_vs_index_inversion(self):
        from options_owl.risk.vinny_strategy import flow_conviction_mult
        # single-stock $1M+ clustered high-ask = big bet
        big = flow_conviction_mult(5, 1_200_000, 0.9, False)[0]
        # index $1M+ = hedge = small bet (inverted)
        hedge = flow_conviction_mult(2, 1_200_000, 0.9, True)[0]
        assert big > 2.0 and hedge < 0.6 and big > hedge

    def test_mult_single_sweep_is_trimmed(self):
        from options_owl.risk.vinny_strategy import flow_conviction_mult
        single = flow_conviction_mult(1, 300_000, 0.7, False)[0]
        clustered = flow_conviction_mult(4, 300_000, 0.7, False)[0]
        assert single < 0.5 and clustered > single

    def test_mult_clamped(self):
        from options_owl.risk.vinny_strategy import flow_conviction_mult
        m = flow_conviction_mult(9, 5_000_000, 0.99, False)[0]
        assert 0.25 <= m <= 2.5  # base clamp (no p_runner)
        mr = flow_conviction_mult(9, 5_000_000, 0.99, False, 0.99)[0]
        assert 0.2 <= mr <= 3.0  # with p_runner

    def test_score_to_contracts_honors_conviction_mult(self):
        from options_owl.risk.vinny_strategy import score_to_contracts
        kw = dict(cost_per_contract=200.0, balance=20000.0, max_position_pct=50.0,
                  max_concurrent=4, max_portfolio_risk_pct=75.0, ml_confidence=0.75)
        base = score_to_contracts(90, conviction_mult=1.0, **kw)
        big = score_to_contracts(90, conviction_mult=2.0, **kw)
        assert big > base  # higher conviction → more contracts

    def test_conviction_mult_capped_by_position_pct(self):
        from options_owl.risk.vinny_strategy import score_to_contracts
        # tiny position cap: even huge conviction can't exceed MAX_POSITION_PCT
        kw = dict(cost_per_contract=200.0, balance=20000.0, max_position_pct=2.0,
                  max_concurrent=4, max_portfolio_risk_pct=75.0, ml_confidence=0.75)
        capped = score_to_contracts(90, conviction_mult=2.5, **kw)
        assert capped <= int(20000.0 * 0.02 / 200.0)  # <= 2 contracts


class TestLiquidityCap:
    """MAX_POSITION_DOLLARS absolute per-trade $ ceiling (0DTE fill realism)."""

    def test_dollar_cap_binds_at_scale(self):
        from options_owl.risk.vinny_strategy import score_to_contracts
        # large balance: 15% pos cap = $150k, but $50k dollar cap should bind
        kw = dict(cost_per_contract=200.0, balance=1_000_000.0, max_position_pct=15.0,
                  max_concurrent=8, max_portfolio_risk_pct=75.0, ml_confidence=0.75, conviction_mult=2.5)
        uncapped = score_to_contracts(90, max_position_dollars=0.0, **kw)
        capped = score_to_contracts(90, max_position_dollars=50_000.0, **kw)
        assert capped < uncapped
        assert capped <= int(50_000.0 / 200.0)  # <= 250 contracts ($50k cap)

    def test_dollar_cap_noop_for_small_account(self):
        from options_owl.risk.vinny_strategy import score_to_contracts
        # small account: 15% of $23k = $3.45k < $50k cap → cap never binds
        kw = dict(cost_per_contract=200.0, balance=23_000.0, max_position_pct=15.0,
                  max_concurrent=8, max_portfolio_risk_pct=75.0, ml_confidence=0.75)
        assert score_to_contracts(90, max_position_dollars=50_000.0, **kw) == score_to_contracts(90, max_position_dollars=0.0, **kw)


class TestFlowPRunner:
    """Serve-time P(runner) helper: real features path, safe None fallbacks."""

    def _sig(self):
        s = flow_signal_to_trade_signal(evaluate_flow_alert(_alert(), _settings()))
        s.strike = 567.5
        s.entry_price = 565.0
        return s

    @pytest.mark.asyncio
    async def test_returns_score_with_good_data(self, monkeypatch):
        import options_owl.collectors.polygon_options as po
        import options_owl.sourcing.scoring.ml_gates.signal_model as sm
        from options_owl.risk.flow_runner import compute_flow_p_runner

        async def _snap(*a, **k):
            return {"bid": 4.9, "ask": 5.1, "mid": 5.0, "iv": 0.45, "delta": 0.5,
                    "theta": -0.3, "vega": 0.2, "volume": 1200, "bid_size": 10, "ask_size": 12}
        async def _bars(*a, **k):
            return [{"close": 4.5 + i * 0.05, "volume": 100 + i} for i in range(20)]
        monkeypatch.setattr(po, "polygon_option_snapshot_greeks", _snap)
        monkeypatch.setattr(po, "polygon_intraday_1m", _bars)
        monkeypatch.setattr(sm, "predict_entry_confidence",
                            lambda t, f, d="": {"runner_score": 0.73, "model_source": "per_ticker"})
        s = self._sig()
        score = await compute_flow_p_runner(s, _settings(POLYGON_API_KEY="k"))
        assert score == pytest.approx(0.73)

    @pytest.mark.asyncio
    async def test_none_when_greeks_missing(self, monkeypatch):
        import options_owl.collectors.polygon_options as po
        from options_owl.risk.flow_runner import compute_flow_p_runner

        async def _snap(*a, **k):
            return None  # no greeks
        async def _bars(*a, **k):
            return []
        monkeypatch.setattr(po, "polygon_option_snapshot_greeks", _snap)
        monkeypatch.setattr(po, "polygon_intraday_1m", _bars)
        assert await compute_flow_p_runner(self._sig(), _settings(POLYGON_API_KEY="k")) is None

    @pytest.mark.asyncio
    async def test_none_when_no_api_key(self):
        from options_owl.risk.flow_runner import compute_flow_p_runner
        assert await compute_flow_p_runner(self._sig(), _settings(POLYGON_API_KEY="")) is None


class TestTideGate:
    """B2 market-tide gate: PUT against a bullish tide is sized down; calls unaffected."""

    def test_put_misaligned_bullish_tide_sized_down(self):
        from options_owl.risk.vinny_strategy import flow_conviction_mult
        # same flow params, put: bearish tide (aligned) vs bullish tide (misaligned)
        aligned = flow_conviction_mult(2, 600_000, 0.9, False, is_put=True, tide_bias=-1e6)[0]
        misaligned = flow_conviction_mult(2, 600_000, 0.9, False, is_put=True, tide_bias=+1e6)[0]
        assert misaligned < aligned
        assert misaligned == pytest.approx(aligned * 0.30, rel=0.01)

    def test_call_unaffected_by_tide(self):
        from options_owl.risk.vinny_strategy import flow_conviction_mult
        bull = flow_conviction_mult(2, 600_000, 0.9, False, is_put=False, tide_bias=+1e6)[0]
        bear = flow_conviction_mult(2, 600_000, 0.9, False, is_put=False, tide_bias=-1e6)[0]
        assert bull == bear  # tide doesn't touch calls

    def test_no_tide_no_change(self):
        from options_owl.risk.vinny_strategy import flow_conviction_mult
        with_none = flow_conviction_mult(2, 600_000, 0.9, False, is_put=True, tide_bias=None)[0]
        plain = flow_conviction_mult(2, 600_000, 0.9, False)[0]
        assert with_none == plain

    @pytest.mark.asyncio
    async def test_tide_bias_none_when_no_key(self):
        from options_owl.risk.flow_runner import get_market_tide_bias
        assert await get_market_tide_bias(_settings(UNUSUAL_WHALES_API_KEY="")) is None
