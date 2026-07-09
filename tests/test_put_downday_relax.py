"""Tests for the conditional PUT down-day relaxation (2026-07-09).

Prod bans PLTR/AMD/MSTR/AVGO/AMZN/GOOGL from puts, but already lifts the ban in bear mode
(SPY <= PUT_BEAR_MODE_THRESHOLD). This narrowly extends the lift to the EARLY-selloff window
(SPY red but not yet bear mode) for the subset whose big dips SUSTAIN — PLTR/MSTR/GOOGL — when
BOTH the name is down >= PUT_DOWNDAY_NAME_DROP% from its open AND SPY <= 0 (broad-tape confirm).
AMD and AMZN stay banned (AMD is a serial dip-then-rip whipsaw; AMZN a net put loser). Down-day
review 2026-07-09: intraday dips on AMD closed GREEN ~6 of the last 7 down-ish days.
"""
import asyncio
from unittest.mock import MagicMock

from options_owl.config.settings import Settings
from options_owl.models.signals import Direction
from options_owl.risk.pipeline import GateResult, PutTickerExclusionGate


def _put(ticker):
    sig = MagicMock()
    sig.ticker = ticker
    sig.direction = Direction.PUT
    sig.bot_source = MagicMock(value="discord")  # not flow — flow bypasses this gate
    return sig


def _settings(**ov):
    d = dict(DISCORD_TOKEN="t", DISCORD_CHANNEL_ID=1,
             PUT_EXCLUDED_TICKERS="PLTR,AMD,MSTR,AVGO,AMZN,GOOGL",
             PUT_BEAR_MODE_THRESHOLD=-0.5,
             ENABLE_PUT_DOWNDAY_RELAX=True,
             PUT_DOWNDAY_RELAX_TICKERS="PLTR,MSTR,GOOGL",
             PUT_DOWNDAY_NAME_DROP=2.0)
    d.update(ov)
    return Settings(**d)


def _run(sig, spy_change, name_change, **s):
    ctx = {"signal": sig, "settings": _settings(**s),
           "spy_change_from_open": spy_change,
           "ticker_change_from_open": name_change}
    return asyncio.run(PutTickerExclusionGate().evaluate(ctx))


class TestPutDownDayRelax:
    def test_relaxed_name_down_hard_spy_red_allowed(self):
        """PLTR down -3% from open, SPY -0.3% (not yet bear) -> relaxed PASS."""
        r = _run(_put("PLTR"), spy_change=-0.3, name_change=-3.0)
        assert r.result == GateResult.PASS
        assert "down-day relax" in r.reason.lower()

    def test_relaxed_name_below_threshold_blocked(self):
        """PLTR only down -1% (< 2% drop) -> still excluded."""
        r = _run(_put("PLTR"), spy_change=-0.3, name_change=-1.0)
        assert r.result == GateResult.FAIL

    def test_relaxed_name_but_spy_green_blocked(self):
        """MSTR down -4% but SPY GREEN (+0.4%) -> blocked (green-tape single-name bounce trap)."""
        r = _run(_put("MSTR"), spy_change=0.4, name_change=-4.0)
        assert r.result == GateResult.FAIL

    def test_amd_stays_banned_even_on_hard_dive(self):
        """AMD down -5%, SPY red -0.3% -> STILL banned (serial whipsaw, not in relax list)."""
        r = _run(_put("AMD"), spy_change=-0.3, name_change=-5.0)
        assert r.result == GateResult.FAIL

    def test_amzn_stays_banned_even_on_hard_dive(self):
        """AMZN down -5%, SPY red -0.3% -> STILL banned (net put loser, not in relax list)."""
        r = _run(_put("AMZN"), spy_change=-0.3, name_change=-5.0)
        assert r.result == GateResult.FAIL

    def test_bear_mode_still_allows_all_names(self):
        """SPY -0.8% (bear mode) -> AMD allowed via the pre-existing bear-mode bypass, unchanged."""
        r = _run(_put("AMD"), spy_change=-0.8, name_change=0.0)
        assert r.result == GateResult.PASS
        assert "bear mode" in r.reason.lower()

    def test_flag_off_keeps_ban(self):
        """With ENABLE_PUT_DOWNDAY_RELAX off, a hard-diving PLTR on a red-but-not-bear day is blocked."""
        r = _run(_put("PLTR"), spy_change=-0.3, name_change=-3.0, ENABLE_PUT_DOWNDAY_RELAX=False)
        assert r.result == GateResult.FAIL

    def test_missing_name_change_fails_safe(self):
        """No name candle data -> relaxation can't confirm -> stays banned (fail-closed)."""
        r = _run(_put("PLTR"), spy_change=-0.3, name_change=None)
        assert r.result == GateResult.FAIL

    def test_non_excluded_name_unaffected(self):
        """SPY (never excluded) always passes regardless of the relaxation."""
        r = _run(_put("SPY"), spy_change=-0.3, name_change=-3.0)
        assert r.result == GateResult.PASS

    def test_boundary_name_exactly_at_threshold(self):
        """Name exactly -2.0% (== threshold) with SPY red -> allowed (<= -2.0)."""
        r = _run(_put("GOOGL"), spy_change=-0.1, name_change=-2.0)
        assert r.result == GateResult.PASS

    def test_boundary_spy_exactly_zero(self):
        """SPY exactly 0.0 (flat) with name down hard -> allowed (SPY <= 0)."""
        r = _run(_put("PLTR"), spy_change=0.0, name_change=-3.0)
        assert r.result == GateResult.PASS
