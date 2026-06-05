"""Configuration management for the Crypto Trading Bot.

Loads settings from environment variables / .env file via pydantic-settings.
Provides a singleton Config object used across all modules.

Environment variables:
    BINANCE_API_KEY         — Binance API key
    BINANCE_API_SECRET      — Binance API secret
    BINANCE_TESTNET         — Use testnet (true/false, default: true)
    SYMBOLS                 — Comma-separated trading pairs
    RISK_PER_TRADE          — Fraction of capital risked per trade (default: 0.02)
    MAX_DAILY_DRAWDOWN      — Max daily drawdown before stopping (default: 0.20)
    POSITION_SIZE_USDT      — Fixed position size in USDT
    STOP_LOSS_PCT           — Stop-loss percentage (default: 0.05)
    TAKE_PROFIT_PCT         — Take-profit percentage (default: 0.15)
    TRAILING_STOP_ACTIVATE_PCT — Profit % to activate trailing stop (default: 0.05)
    TRAILING_STOP_DISTANCE_PCT — Trailing stop distance (default: 0.02)
    DATABASE_URL            — SQLite database path
    DEFAULT_STRATEGY        — Strategy name
    STRATEGY_INTERVAL_MINUTES — Interval between strategy runs
    API_HOST / API_PORT     — REST API bind address
    LOG_LEVEL / LOG_FILE    — Logging configuration
    XGBOOST_MODEL_PATH      — Path to serialised XGBoost model
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Binance
    binance_api_key: str = ""
    binance_api_secret: str = ""
    binance_testnet: bool = True

    # Trading
    symbols: str = "BTC/USDT,ETH/USDT,BNB/USDT,SOL/USDT,ADA/USDT"
    risk_per_trade: float = 0.02
    max_daily_drawdown: float = 0.20
    position_size_usdt: float = 100.0
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.15
    trailing_stop_activate_pct: float = 0.05
    trailing_stop_distance_pct: float = 0.02

    # Database
    database_url: str = "sqlite:///data/trading.db"

    # Strategy
    default_strategy: str = "multi_signal_weighted"
    strategy_interval_minutes: int = 5

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Logging
    log_level: str = "INFO"
    log_file: str = "data/trading.log"

    # ML
    xgboost_model_path: str = "crypto_trading/models/xgboost_model.json"

    @property
    def symbol_list(self) -> List[str]:
        return [s.strip() for s in self.symbols.split(",") if s.strip()]


settings = Settings()
