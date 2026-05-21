"""Sourcing-specific settings (pydantic-settings, env-driven)."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class SourcingSettings(BaseSettings):
    """All owlet-sourcing configuration, loaded from environment / .env."""

    # --- Scan loop ---
    SCAN_INTERVAL_SECONDS: int = 180
    SOURCING_TICKERS: str = "SPY,QQQ,NVDA,TSLA,META,AAPL,AMZN,GOOGL,MSFT,AMD,MSTR,PLTR,AVGO"
    SCORE_THRESHOLD: int = 60

    # --- Data sources (technical) ---
    ENABLE_SOURCE_HARVESTER_CANDLES: bool = True
    ENABLE_SOURCE_TWELVE_DATA: bool = False
    ENABLE_SOURCE_POLYGON_OPTIONS: bool = True
    ENABLE_SOURCE_POLYGON_NEWS: bool = False
    ENABLE_SOURCE_UNUSUAL_WHALES: bool = False
    ENABLE_SOURCE_GROK_AI: bool = False

    # --- Alpha sources (smart money / insider / sentiment) ---
    ENABLE_SEC_INSIDER: bool = True
    ENABLE_UW_CONGRESS: bool = True
    ENABLE_CAPITOL_TRADES: bool = False
    ENABLE_STOCKTWITS_SENTIMENT: bool = True

    # --- ML gates ---
    ENABLE_ML_FLOW_CLASSIFIER: bool = False
    ENABLE_ML_ENTRY_OPTIMIZER: bool = False
    ENABLE_ML_QUALITY_PREDICTOR: bool = False
    ENABLE_ML_REGIME_WEIGHTER: bool = False
    ENABLE_ML_EXIT_ADVISOR: bool = False

    # --- Scoring features ---
    ENABLE_FAST_ALERT: bool = True
    ENABLE_GAP_FADE: bool = True
    ENABLE_GAP_DOWN_REVERSAL: bool = True
    ENABLE_CATALYST_BOOST: bool = True
    ENABLE_REGIME_GATE: bool = True
    ENABLE_SECTOR_ROTATION: bool = True

    # --- News sentinel (real-time monitoring for open positions) ---
    ENABLE_NEWS_SENTINEL: bool = True
    NEWS_SENTINEL_POLL_SECONDS: int = 60

    # --- Output ---
    SOURCING_DISCORD_OUTPUT: bool = True
    SOURCING_DB_OUTPUT: bool = True

    # --- API keys (injected via env) ---
    TWELVE_DATA_KEY: str = ""
    UW_KEY: str = ""
    GROK_KEY: str = ""
    FINNHUB_KEY: str = ""

    # --- Paths ---
    SHARED_CANDLE_DB: str = "/app/shared_harvester/options_data.db"
    STATE_DB: str = "/app/journal/state.db"
    SIGNAL_DB: str = "/app/journal/signals.db"

    @property
    def ticker_list(self) -> list[str]:
        return [t.strip() for t in self.SOURCING_TICKERS.split(",") if t.strip()]

    model_config = {"env_prefix": "", "env_file": ".env", "extra": "ignore"}
