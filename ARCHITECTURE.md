# Crypto Trading Bot — Architecture

## Overview

An AI-powered cryptocurrency trading bot for Binance that combines technical analysis indicators with machine learning (XGBoost) to generate trading signals, manages risk with configurable stop-loss/take-profit/trailing-stop rules, and executes orders via the Binance API.

---

## Components Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Crypto Trading Bot                           │
│                                                                      │
│  ┌──────────┐   ┌──────────────┐   ┌───────────────┐                │
│  │  Binance  │   │  WebSocket   │   │   Candle      │                │
│  │  Streams  │──▶│  Receiver    │──▶│   Buffer      │                │
│  └──────────┘   └──────────────┘   └───────┬───────┘                │
│                                            │                        │
│                                            ▼                        │
│                                     ┌──────────────┐                │
│                                     │  Indicator    │                │
│                                     │  Engine       │                │
│                                     │  (RSI, MACD,  │                │
│                                     │   BB, EMA)    │                │
│                                     └───────┬──────┘                │
│                                             │                        │
│                                             ▼                        │
│  ┌──────────┐                        ┌──────────────┐                │
│  │ XGBoost  │                        │  Signal       │                │
│  │ Model    │◀───────────────────────│  Generator    │                │
│  └──────────┘   Feature Vector       │  (TA + ML)    │                │
│                                       └───────┬──────┘                │
│                                               │                      │
│                                               ▼                      │
│  ┌──────────┐                        ┌──────────────┐                │
│  │ Binance  │◀───────────────────────│  Risk         │                │
│  │ Order    │   Order                │  Manager     │                │
│  │ Executor │───────────────────────▶│  (SL/TP/TS)  │                │
│  └──────────┘   Execution            └──────────────┘                │
│       │                                                              │
│       ▼                                                              │
│  ┌──────────┐                                                        │
│  │ Database │◀──── Trades / Signals / Performance                    │
│  │ (SQLite) │                                                        │
│  └──────────┘                                                        │
│       ▲                                                              │
│       │                                                              │
│  ┌──────────┐                                                        │
│  │ FastAPI  │──── REST API endpoints                                 │
│  │ Server   │                                                        │
│  └──────────┘                                                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

```
Binance WebSocket (kline_1m)
        │
        ▼
┌──────────────────┐
│ Candle Buffer     │  Per-symbol DataFrame (last 100 candles)
│ (per symbol)      │
└───────┬──────────┘
        │  Every STRATEGY_INTERVAL_MINUTES (default 5)
        ▼
┌──────────────────┐
│ IndicatorEngine   │  Computes RSI(14), MACD(12,26,9), BB(20,2),
│                   │  EMA crossover(9,21), ATR(14)
└───────┬──────────┘
        │  Feature vector (13 features)
        ▼
┌──────────────────┐
│ SignalGenerator   │  Weighted voting:
│                   │    RSI         0.20
│   TA Rules        │    MACD        0.15
│   +               │    Bollinger   0.15
│   XGBoost Model   │    EMA Cross   0.20
│                   │    XGBoost     0.30
└───────┬──────────┘
        │  Signal: BUY / SELL / HOLD + strength 0–1
        ▼
┌──────────────────┐
│ RiskManager       │  Checks:
│                   │    1. Max daily drawdown (20%)
│                   │    2. Max concurrent positions (3)
│                   │    3. Max 1 per symbol
│                   │  Computes:
│                   │    - Position size (2% risk per trade)
│                   │    - Stop-loss (ATR-based, clamped 2-5%)
│                   │    - Take-profit (3× SL, clamped 6-15%)
└───────┬──────────┘
        │  Go / No-Go + size + SL/TP prices
        ▼
┌──────────────────┐
│ OrderExecutor     │  MARKET or LIMIT order on Binance
│                   │  Records trade in Database
└───────┬──────────┘
        │
        ▼
┌──────────────────┐
│ Database (SQLite) │  trades, signals, performance, config tables
└──────────────────┘
```

---

## AI Pipeline

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Feature     │───▶│  XGBoost     │───▶│  Signal      │───▶│  Decision    │
│  Engineering │    │  Classifier  │    │  Weighting   │    │  (argmax)    │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
      │                    │                    │
      │  13 features:      │  Output: 3-class   │  Weight 0.30
      │  - RSI             │  probabilities      │  combined with
      │  - MACD line       │  [HOLD, BUY, SELL]  │  TA weights
      │  - MACD signal     │                     │
      │  - MACD diff       │                     │
      │  - BB upper        │                     │
      │  - BB middle       │                     │
      │  - BB lower        │                     │
      │  - EMA fast        │                     │
      │  - EMA slow        │                     │
      │  - EMA diff        │                     │
      │  - EMA crossover   │                     │
      │  - ATR             │                     │
      │  - Volume          │                     │
      └──────────────────┘                     │
                                                │
                                   Future: LSTM / Transformer for
                                   price prediction (regression),
                                   then convert predictions to signals
```

---

## Risk Management Flow

```
                    Incoming Trade Signal (BUY / SELL)
                              │
                              ▼
              ┌─────────────────────────────┐
              │  Max Daily Drawdown Check   │
              │  Daily PnL < 20% of capital │
              └──────────┬──────────────────┘
                         │ Pass
                         ▼
              ┌─────────────────────────────┐
              │  Concurrency Check           │
              │  ≤ 3 open positions total    │
              │  ≤ 1 per symbol              │
              └──────────┬──────────────────┘
                         │ Pass
                         ▼
              ┌─────────────────────────────┐
              │  Position Sizing             │
              │  risk_amt = capital × 0.02   │
              │  quantity = risk_amt / SL    │
              └──────────┬──────────────────┘
                         │
                         ▼
              ┌─────────────────────────────┐
              │  Stop-Loss / Take-Profit     │
              │  SL = entry × (1 ∓ sl_pct)   │
              │  TP = entry × (1 ± tp_pct)   │
              │  sl_pct = ATR/close (2-5%)   │
              │  tp_pct = sl_pct × 3 (6-15%)│
              └──────────┬──────────────────┘
                         │
                         ▼
                    ┌──────────┐
                    │  Execute  │
                    └──────────┘
                         │
                         ▼
              ┌─────────────────────────────┐
              │  Position Monitoring         │
              │  - Check SL/TP every tick    │
              │  - Activate trailing at 5%   │
              │  - Trail by 2%               │
              └─────────────────────────────┘
```

---

## Directory Structure

```
crypto-trading/
├── crypto_trading/          # Main package
│   ├── __init__.py
│   ├── config.py            # Configuration (pydantic-settings)
│   ├── database.py          # SQLite models & DB wrapper
│   ├── indicators.py        # Technical indicators (RSI, MACD, BB, EMA, ATR)
│   ├── signal_generator.py  # ML + rule-based signal fusion
│   ├── order_executor.py    # Binance order execution
│   ├── risk_manager.py      # Risk management (SL/TP/TS/sizing)
│   ├── backtest.py          # Backtesting engine
│   ├── portfolio.py         # Portfolio tracking & PnL
│   ├── bot.py               # Main bot loop (WebSocket + strategy)
│   ├── api.py               # FastAPI REST API
│   ├── utils.py             # Utility functions
│   └── models/              # ML models directory
├── tests/                   # Test directory
├── requirements.txt
├── .env.example
├── main.py                  # Entry point
├── ARCHITECTURE.md          # This file
├── CHANGELOG.md
└── README.md
```

---

## Deployment

### Docker

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "main.py", "--mode", "api"]
```

### docker-compose

```yaml
version: "3.9"
services:
  bot:
    build: .
    env_file: .env
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
    restart: unless-stopped
```

### systemd (optional)

```
[Unit]
Description=Crypto Trading Bot
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/crypto-trading
ExecStart=/usr/bin/python main.py --mode live
Restart=always
EnvironmentFile=/opt/crypto-trading/.env

[Install]
WantedBy=multi-user.target
```
