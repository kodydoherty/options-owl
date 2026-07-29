"""Call-side ticker expansion (2026-07-18): ORCL/INTC/TSM/ARM/SMH added to the ML pattern scan,
flag-gated by ENABLE_EXPANSION_TICKERS, and kept CALL-only (puts weren't validated)."""
import importlib
import os

EXPANSION = ["ORCL", "INTC", "TSM", "ARM", "SMH", "USO", "SLV", "GDX"]


def _reload_with(env_val):
    prev = os.environ.get("ENABLE_EXPANSION_TICKERS")
    if env_val is None:
        os.environ.pop("ENABLE_EXPANSION_TICKERS", None)
    else:
        os.environ["ENABLE_EXPANSION_TICKERS"] = env_val
    try:
        import options_owl.sourcing.ml_pipeline as m
        importlib.reload(m)
        return list(m.TICKERS)
    finally:
        if prev is None:
            os.environ.pop("ENABLE_EXPANSION_TICKERS", None)
        else:
            os.environ["ENABLE_EXPANSION_TICKERS"] = prev
        import options_owl.sourcing.ml_pipeline as m
        importlib.reload(m)  # restore module to default state for other tests


def test_flag_off_excludes_expansion():
    tickers = _reload_with("false")
    for t in EXPANSION:
        assert t not in tickers, f"{t} must NOT be scanned when expansion flag is off"


def test_flag_on_adds_expansion():
    tickers = _reload_with("true")
    for t in EXPANSION:
        assert t in tickers, f"{t} must be scanned when expansion flag is on"
    # no duplicates introduced
    assert len(tickers) == len(set(tickers))


def test_default_is_off():
    """Safety: default (unset) must NOT scan the expansion names — deploy is opt-in."""
    tickers = _reload_with(None)
    for t in EXPANSION:
        assert t not in tickers


def test_expansion_names_are_put_excluded():
    """CALLS validated, puts were NOT — the expansion names must be excluded from PUT trading."""
    from options_owl.config.settings import Settings
    excl = Settings().PUT_EXCLUDED_TICKERS
    for t in EXPANSION:
        assert t in excl, f"{t} must be PUT-excluded (calls-only expansion)"
