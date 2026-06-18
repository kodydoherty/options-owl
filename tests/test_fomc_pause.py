"""Tests for the FOMC-day entry pause (block all new entries on Fed announcement days)."""

from types import SimpleNamespace

from options_owl.risk.vinny_strategy import is_fomc_pause

DATES = "2026-01-28,2026-03-18,2026-04-29,2026-06-17,2026-07-29,2026-09-16,2026-10-28,2026-12-16"


def _s(enabled=True, dates=DATES):
    return SimpleNamespace(ENABLE_FOMC_PAUSE=enabled, FOMC_PAUSE_DATES=dates)


class TestFomcPause:
    def test_blocks_on_fomc_day(self):
        assert is_fomc_pause(_s(), "2026-06-17") is True
        assert is_fomc_pause(_s(), "2026-12-16") is True

    def test_allows_on_normal_day(self):
        assert is_fomc_pause(_s(), "2026-06-18") is False
        assert is_fomc_pause(_s(), "2026-06-16") is False

    def test_disabled_flag_never_blocks(self):
        # even on a real FOMC day, disabled → no block
        assert is_fomc_pause(_s(enabled=False), "2026-06-17") is False

    def test_empty_dates_never_blocks(self):
        assert is_fomc_pause(_s(dates=""), "2026-06-17") is False

    def test_handles_whitespace_in_csv(self):
        assert is_fomc_pause(_s(dates=" 2026-06-17 , 2026-07-29 "), "2026-07-29") is True

    def test_missing_settings_safe(self):
        assert is_fomc_pause(SimpleNamespace(), "2026-06-17") is False
