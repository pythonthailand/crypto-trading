# Crypto Trading Bot

AI-powered cryptocurrency trading bot for Binance. Combines technical analysis indicators (RSI, MACD, Bollinger Bands, EMA) with XGBoost machine learning for trade signal generation, with configurable risk management.

## Features

- **Multi-strategy signal fusion** — Weighted voting of TA indicators + XGBoost classifier
- **Real-time WebSocket feed** — Binance kline streams for live price data
- **Risk management** — Position sizing (2% risk), stop-loss, take-profit, trailing stop, max drawdown
- **Backtesting engine** — 30-day historical simulation with performance metrics
- **REST API** — FastAPI with 14 endpoints for monitoring and control
- **Multi-pair support** — BTC/USDT, ETH/USDT, BNB/USDT, SOL/USDT, ADA/USDT
- **SQLite database** — Trades, signals, performance tracking

## Quick Start

### 1. Clone and install

```bash
git clone <repo-url> && cd crypto-trading
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your Binance API keys (testnet recommended)
```

### 3. Run

```bash
python main.py --mode api        # Start REST API (default)
python main.py --mode backtest   # Run backtest
python main.py --mode live       # Start live trading bot
```

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| GET | `/api/balance` | Wallet balance |
| GET | `/api/positions` | Open positions |
| POST | `/api/trade` | Place order |
| GET | `/api/signals` | Current signals |
| GET | `/api/signals/{symbol}` | Signal for pair |
| GET | `/api/performance` | Performance stats |
| GET | `/api/performance/{symbol}` | Per-pair perf |
| GET | `/api/strategies` | List strategies |
| POST | `/api/strategies` | Switch strategy |
| GET | `/api/trades` | Trade history |
| GET | `/api/trades/{id}` | Trade detail |
| GET | `/api/config` | Current config |
| POST | `/api/config` | Update config |

## Supported Strategies

- `multi_signal_weighted` — TA + ML weighted voting (default)
- `ta_only` — Technical indicators only
- `ml_only` — XGBoost predictions only
- `conservative` — Higher confirmation threshold
- `aggressive` — Lower confirmation threshold

## License

MIT
