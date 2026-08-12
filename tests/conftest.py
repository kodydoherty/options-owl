import os
import tempfile

import pytest


@pytest.fixture
def tmp_db_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield os.path.join(tmpdir, "test.db")


@pytest.fixture(autouse=True)
def _pin_market_clock():
    """Pin the circuit-breaker clock so the suite does not depend on when it runs.

    Discovered 2026-08-12: 28 tests across test_partial_profits / test_slippage /
    test_risk_manager failed for ~15 minutes a day and passed the rest of the time.
    They open a trade through the real entry pipeline, and CircuitBreaker's opening
    (09:30-09:40 ET) and closing (15:45-16:00 ET) buffers read the WALL CLOCK — so
    running the suite inside either window rejected every signal and the trades came
    back None.

    That is worse than a flaky test: rebuild.sh gates deploys on `pytest -x`, so the
    deploy pipeline was self-blocking during the last 15 minutes of every trading day
    — precisely when an urgent fix is most likely to be needed.

    Pinned to a Monday mid-morning, clear of both buffers. Tests that exercise the
    buffers themselves patch `_now_et` inside a `with` block, which still takes
    precedence over this fixture.
    """
    from datetime import datetime
    from unittest.mock import patch

    from options_owl.risk import circuit_breaker as cb_mod

    pinned = datetime(2026, 3, 30, 10, 30, 0, tzinfo=cb_mod.ET)  # Monday 10:30 ET
    with patch.object(cb_mod, "_now_et", return_value=pinned):
        yield
