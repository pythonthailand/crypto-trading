# Changelog

## [0.2.0] — 2026-06-05

### Added
- Unit tests for all core modules (44 tests):
  - `tests/test_indicators.py` — RSI, MACD, BB, EMA crossover, ATR, compute_all
  - `tests/test_signal_generator.py` — Signal generation, TA voting, ML fallback, multi-symbol
  - `tests/test_risk_manager.py` — Position sizing, SL/TP prices, trailing stop, drawdown, concurrency
  - `tests/test_order_executor.py` — DB recording, fill parsing, multiple records
  - `tests/test_backtest.py` — Run simulation, exit checks (SL/TP), summary metrics

## [0.1.0] — 2026-06-05

### Added
- Project skeleton with full directory structure
- `crypto_trading/` package with all modules:
  - `config.py` — pydantic-settings based configuration
  - `database.py` — SQLite schema (trades, signals, performance, config) with CRUD operations
  - `indicators.py` — Technical indicators (RSI, MACD, BB, EMA crossover, ATR)
  - `signal_generator.py` — Weighted voting fusion of TA + XGBoost signals
  - `order_executor.py` — Binance market/limit order execution with DB recording
  - `risk_manager.py` — Position sizing, SL/TP, trailing stop, drawdown control
  - `backtest.py` — Historical backtesting engine with PnL reporting
  - `portfolio.py` — Portfolio tracking and daily performance aggregation
  - `bot.py` — Main bot loop with WebSocket kline feed and strategy execution
  - `api.py` — FastAPI REST API (14 endpoints)
  - `utils.py` — Utility functions
- `main.py` — CLI entry point (live / backtest / api modes)
- `requirements.txt` — All Python dependencies
- `.env.example` — Config template
- `ARCHITECTURE.md` — Full architecture documentation
- `README.md` — Project overview and setup instructions
