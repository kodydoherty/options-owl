"""Winner-concentration (conf_linear) sizing: replace the legacy ML-confidence buckets with a
monotonic conf→budget curve. ML pattern trades only (flow passes ml_confidence=None → unaffected).
Validated 2.5yr (gold-standard sweep 2026-06-16): 0.3→3.0 gave PF 1.95→3.78, P&L 4.6x, ~same DD."""
from options_owl.risk.vinny_strategy import _ml_confidence_to_mult, score_to_contracts

LIN = dict(conf_linear=True, cb_min=0.3, cb_max=3.0, cr_min=0.74, cr_max=0.95)


def test_endpoints_and_monotonic():
    lo, _ = _ml_confidence_to_mult(0.74, **LIN)
    mid, _ = _ml_confidence_to_mult(0.845, **LIN)   # halfway
    hi, _ = _ml_confidence_to_mult(0.95, **LIN)
    assert abs(lo - 0.3) < 0.01
    assert abs(hi - 3.0) < 0.01
    assert abs(mid - 1.65) < 0.05
    assert lo < mid < hi                            # monotonic: more confidence = more size


def test_clamps_outside_ref_band():
    assert abs(_ml_confidence_to_mult(0.99, **LIN)[0] - 3.0) < 0.01   # above ref_max → cb_max
    assert abs(_ml_confidence_to_mult(0.70, **LIN)[0] - 0.3) < 0.01   # 0.70>floor but <ref_min → cb_min


def test_still_rejects_below_floor():
    assert _ml_confidence_to_mult(0.55, **LIN)[0] == 0.0             # < 0.62 CALL floor


def test_flow_unaffected_none():
    mult, desc = _ml_confidence_to_mult(None, **LIN)
    assert desc == "no_ml"                                          # flow (None) bypasses the curve


def test_legacy_buckets_when_off():
    assert _ml_confidence_to_mult(0.85, conf_linear=False)[0] == 0.60   # the backwards 0.80 tier
    assert _ml_confidence_to_mult(0.95, conf_linear=False)[0] == 0.95


def test_high_conf_sizes_bigger_than_legacy():
    kw = dict(cost_per_contract=200.0, balance=23000.0, max_position_pct=100.0,
              max_concurrent=8, max_portfolio_risk_pct=75.0)
    big = score_to_contracts(95, ml_confidence=0.95, conf_linear=True, **kw)
    legacy = score_to_contracts(95, ml_confidence=0.95, conf_linear=False, **kw)
    marginal = score_to_contracts(95, ml_confidence=0.74, conf_linear=True, **kw)
    assert big > legacy                       # high conf: 3.0x >> legacy 0.95x
    assert marginal < legacy                  # marginal starved: 0.3x < legacy 0.95x
