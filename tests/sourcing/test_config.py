"""Tests for sourcing configuration."""

from options_owl.sourcing.config import SourcingSettings


def test_default_settings():
    settings = SourcingSettings()
    assert settings.SCAN_INTERVAL_SECONDS == 180
    assert settings.SCORE_THRESHOLD == 60
    assert "SPY" in settings.ticker_list
    assert "NVDA" in settings.ticker_list
    assert len(settings.ticker_list) == 13


def test_ticker_list_parsing():
    settings = SourcingSettings(SOURCING_TICKERS="SPY,QQQ,NVDA")
    assert settings.ticker_list == ["SPY", "QQQ", "NVDA"]


def test_alpha_sources_enabled_by_default():
    settings = SourcingSettings()
    assert settings.ENABLE_SEC_INSIDER is True
    assert settings.ENABLE_UW_CONGRESS is True
    assert settings.ENABLE_STOCKTWITS_SENTIMENT is True


def test_ml_gates_disabled_by_default():
    settings = SourcingSettings()
    assert settings.ENABLE_ML_FLOW_CLASSIFIER is False
    assert settings.ENABLE_ML_ENTRY_OPTIMIZER is False
    assert settings.ENABLE_ML_QUALITY_PREDICTOR is False
    assert settings.ENABLE_ML_REGIME_WEIGHTER is False
    assert settings.ENABLE_ML_EXIT_ADVISOR is False
