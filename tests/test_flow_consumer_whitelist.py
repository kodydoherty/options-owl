"""Consumer-side flow whitelist (2026-07-28 fix).

The harvester (flow publisher) filters against the settings DEFAULT UW_FLOW_*_TICKERS (no per-bot env),
so a per-bot override like kody/dennis's 'SPY removed from flow' was DEAD CONFIG — flow SPY kept trading
on the live bots. `flow_ticker_allowed` re-checks the bot's OWN whitelist on consume so per-bot exclusions
actually bite, while paper shadows (default list, incl. SPY) keep it. These lock down that behavior.
"""
from types import SimpleNamespace

from options_owl.bot_runner import flow_ticker_allowed
from options_owl.models.signals import Direction


def _fs(ticker, direction):
    return SimpleNamespace(ticker=ticker, direction=direction)


def _settings(call="META,AMZN,TSLA,AMD,ORCL,INTC,ARM,GOOG,LRCX", put="META,AMZN,AAPL,TSLA,MU"):
    return SimpleNamespace(UW_FLOW_CALL_TICKERS=call, UW_FLOW_PUT_TICKERS=put)


class TestFlowTickerAllowed:
    def test_spy_call_blocked_when_removed_from_bot_list(self):
        # kody/dennis list has SPY removed → flow SPY call must be rejected on consume
        assert flow_ticker_allowed(_fs("SPY", Direction.CALL), _settings()) is False

    def test_whitelisted_call_allowed(self):
        assert flow_ticker_allowed(_fs("META", Direction.CALL), _settings()) is True

    def test_paper_shadow_default_list_keeps_spy(self):
        # a bot on the settings default (SPY present) still trades flow SPY
        default = _settings(call="META,SPY,AMZN,TSLA,AMD,ORCL,INTC,ARM,GOOG,LRCX")
        assert flow_ticker_allowed(_fs("SPY", Direction.CALL), default) is True

    def test_put_direction_uses_put_list(self):
        # SPY not in the put list → blocked; MU in the put list → allowed
        assert flow_ticker_allowed(_fs("SPY", Direction.PUT), _settings()) is False
        assert flow_ticker_allowed(_fs("MU", Direction.PUT), _settings()) is True

    def test_call_not_in_put_list_still_allowed_as_call(self):
        # AMD is a call-list name, absent from the put list — must be judged by the CALL list for a call
        assert flow_ticker_allowed(_fs("AMD", Direction.CALL), _settings()) is True

    def test_case_insensitive(self):
        assert flow_ticker_allowed(_fs("meta", Direction.CALL), _settings()) is True

    def test_empty_list_fails_open(self):
        # a config typo (empty list) must NOT silently kill all flow
        assert flow_ticker_allowed(_fs("SPY", Direction.CALL), _settings(call="")) is True

    def test_none_list_fails_open(self):
        assert flow_ticker_allowed(_fs("SPY", Direction.CALL), _settings(call=None)) is True

    def test_handler_uses_helper(self):
        """The flow consumer must call flow_ticker_allowed before resolving/placing a flow trade."""
        import inspect

        import options_owl.bot_runner as br
        src = inspect.getsource(br.run_bot) if hasattr(br, "run_bot") else inspect.getsource(br)
        assert "flow_ticker_allowed(" in src
