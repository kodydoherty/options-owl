"""Fleet staggering: each signal is deterministically assigned to K-of-N bots (priority round-robin)
so the bots hold different books and don't all win/lose together. No coordinator — every bot computes
the same window from the same market key and checks its own rank."""
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from options_owl.risk.vinny_strategy import fleet_takes_signal

ET = ZoneInfo("America/New_York")
NOW = datetime(2026, 6, 16, 10, tzinfo=ET)


def _s(**ov):
    d = dict(ENABLE_FLEET_STAGGER=True, FLEET_SIZE=5, FLEET_OVERLAP=2, FLEET_RANK=0)
    d.update(ov)
    return SimpleNamespace(**d)


def test_disabled_always_takes():
    assert fleet_takes_signal("SPY", "call", NOW, _s(ENABLE_FLEET_STAGGER=False)) is True


def test_exactly_k_of_n_take_each_signal():
    """For any signal, exactly FLEET_OVERLAP (K=2) of the 5 bots take it."""
    takers = [r for r in range(5) if fleet_takes_signal("SPY", "call", NOW, _s(FLEET_RANK=r))]
    assert len(takers) == 2


def test_deterministic_same_inputs():
    a = fleet_takes_signal("TSLA", "put", NOW, _s(FLEET_RANK=3))
    b = fleet_takes_signal("TSLA", "put", NOW, _s(FLEET_RANK=3))
    assert a == b


def test_union_of_all_ranks_covers_every_signal():
    """Collectively the fleet still takes every signal (no signal dropped by everyone)."""
    for tk in ("SPY", "TSLA", "META", "AMD", "MU", "ARM"):
        assert any(fleet_takes_signal(tk, "call", NOW, _s(FLEET_RANK=r)) for r in range(5))


def test_each_rank_gets_fair_share():
    counts = {r: 0 for r in range(5)}
    total = 0
    for tk in ("SPY", "TSLA", "META", "AMD", "MU", "NVDA", "AMZN", "GOOG", "ARM", "QQQ"):
        for h in range(9, 16):
            now = datetime(2026, 6, 16, h, tzinfo=ET)
            total += 1
            for r in range(5):
                if fleet_takes_signal(tk, "call", now, _s(FLEET_RANK=r)):
                    counts[r] += 1
    for r in range(5):
        assert 0.25 < counts[r] / total < 0.55, f"rank {r} share {counts[r]/total:.2f} (~0.40 expected)"


def test_fail_open_on_misconfig():
    assert fleet_takes_signal("SPY", "call", NOW, _s(FLEET_OVERLAP=5)) is True   # K>=N
    assert fleet_takes_signal("SPY", "call", NOW, _s(FLEET_SIZE=1)) is True
    assert fleet_takes_signal("SPY", "call", NOW, _s(FLEET_OVERLAP=0)) is True
