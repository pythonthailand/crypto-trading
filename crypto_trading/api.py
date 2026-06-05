"""FastAPI REST API for the Crypto Trading Bot.

Provides a REST interface to query balance, positions, signals, performance,
trade history, and configuration.

Endpoints
=========

Info
----
- **`GET /`** — API info (version, uptime, status)

Balance & Positions
-------------------
- **`GET /api/balance`** — Wallet balance (all non-zero assets)
- **`GET /api/positions`** — Open positions with unrealised PnL

Trading
-------
- **`POST /api/trade`** — Place an order
    Body: ``{ symbol, side, quantity, order_type, price? }``

Signals
-------
- **`GET /api/signals`** — Current signals for all configured pairs
- **`GET /api/signals/{symbol}`** — Signal for a specific pair

Performance
-----------
- **`GET /api/performance`** — Aggregated performance stats
- **`GET /api/performance/{symbol}`** — Per-pair performance

Strategies
----------
- **`GET /api/strategies`** — List available strategies
- **`POST /api/strategies`** — Switch active strategy
    Body: ``{ name }``

Trades
------
- **`GET /api/trades`** — Trade history (optional ``?symbol=``, ``?limit=``)
- **`GET /api/trades/{id}`** — Specific trade detail

Configuration
-------------
- **`GET /api/config`** — Current runtime configuration
- **`POST /api/config`** — Update configuration values
    Body: ``{ key: value, ... }``
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from loguru import logger
from pydantic import BaseModel

from crypto_trading.config import settings
from crypto_trading.database import Database
from crypto_trading.portfolio import Portfolio
from crypto_trading.signal_generator import SignalGenerator

try:
    import uvicorn
except ImportError:
    uvicorn = None  # type: ignore


# ------------------------------------------------------------------
# Pydantic schemas
# ------------------------------------------------------------------

class TradeRequest(BaseModel):
    symbol: str
    side: str  # BUY / SELL
    quantity: float
    order_type: str = "MARKET"  # MARKET / LIMIT
    price: Optional[float] = None


class StrategySwitch(BaseModel):
    name: str


class ConfigUpdate(BaseModel):
    key: str
    value: str


# ------------------------------------------------------------------
# App factory
# ------------------------------------------------------------------

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Crypto Trading Bot API",
        version="0.1.0",
        description="REST API for AI-powered Crypto Trading Bot",
    )

    db = Database()
    portfolio = Portfolio(db=db)
    signal_gen = SignalGenerator()

    # --- State ---
    app.state.db = db
    app.state.portfolio = portfolio
    app.state.signal_gen = signal_gen
    app.state.start_time = __import__("time").time()
    app.state.strategies = [
        "multi_signal_weighted",
        "ta_only",
        "ml_only",
        "conservative",
        "aggressive",
    ]

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.get("/")
    def root():
        return {
            "name": "Crypto Trading Bot",
            "version": "0.1.0",
            "status": "running",
            "uptime_sec": int(__import__("time").time() - app.state.start_time),
        }

    @app.get("/api/balance")
    def get_balance():
        from crypto_trading.order_executor import OrderExecutor
        try:
            ex = OrderExecutor(db=db)
            return {"balances": ex.get_all_balances()}
        except Exception as e:
            raise HTTPException(status_code=503, detail=str(e))

    @app.get("/api/positions")
    def get_positions():
        return {"positions": portfolio.get_open_positions()}

    @app.post("/api/trade")
    def place_trade(req: TradeRequest):
        from crypto_trading.order_executor import OrderExecutor
        ex = OrderExecutor(db=db)
        if req.order_type.upper() == "MARKET":
            order = ex.market_order(
                symbol=req.symbol,
                side=req.side.upper(),
                quantity=req.quantity,
            )
        elif req.order_type.upper() == "LIMIT":
            if req.price is None:
                raise HTTPException(status_code=400, detail="price required for LIMIT order")
            order = ex.limit_order(
                symbol=req.symbol,
                side=req.side.upper(),
                quantity=req.quantity,
                price=req.price,
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unknown order_type: {req.order_type}")
        if order is None:
            raise HTTPException(status_code=502, detail="Order execution failed")
        return {"order": order}

    @app.get("/api/signals")
    def get_signals():
        return {"signals": db.get_recent_signals(limit=50)}

    @app.get("/api/signals/{symbol}")
    def get_signal(symbol: str):
        signals = db.get_recent_signals(symbol=symbol, limit=10)
        if not signals:
            raise HTTPException(status_code=404, detail=f"No signals for {symbol}")
        return {"symbol": symbol, "signals": signals}

    @app.get("/api/performance")
    def get_performance():
        return {"performance": db.get_performance()}

    @app.get("/api/performance/{symbol}")
    def get_performance_symbol(symbol: str):
        perf = db.get_performance(symbol=symbol)
        if not perf:
            raise HTTPException(status_code=404, detail=f"No performance data for {symbol}")
        return {"symbol": symbol, "performance": perf}

    @app.get("/api/strategies")
    def list_strategies():
        return {"strategies": app.state.strategies, "active": settings.default_strategy}

    @app.post("/api/strategies")
    def switch_strategy(req: StrategySwitch):
        if req.name not in app.state.strategies:
            raise HTTPException(status_code=400, detail=f"Unknown strategy: {req.name}")
        db.set_config("default_strategy", req.name)
        settings.default_strategy = req.name
        return {"strategy": req.name, "status": "activated"}

    @app.get("/api/trades")
    def get_trades(
        symbol: Optional[str] = Query(None),
        limit: int = Query(100, le=1000),
    ):
        return {"trades": db.get_trade_history(symbol=symbol, limit=limit)}

    @app.get("/api/trades/{trade_id}")
    def get_trade(trade_id: int):
        trades = db.get_trade_history(limit=10000)
        for t in trades:
            if t["id"] == trade_id:
                return {"trade": t}
        raise HTTPException(status_code=404, detail=f"Trade {trade_id} not found")

    @app.get("/api/config")
    def get_config():
        return {"config": db.get_all_config()}

    @app.post("/api/config")
    def update_config(req: ConfigUpdate):
        db.set_config(req.key, req.value)
        return {"key": req.key, "value": req.value, "status": "updated"}

    return app


# ------------------------------------------------------------------
# Startup helper
# ------------------------------------------------------------------

def start_api(config_path: str = ".env") -> None:
    """Start the FastAPI server."""
    import dotenv
    dotenv.load_dotenv(config_path)

    if uvicorn is None:
        logger.error("uvicorn not installed — cannot start API")
        return

    app = create_app()
    logger.info(
        "Starting API server on {}:{}",
        settings.api_host, settings.api_port,
    )
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower(),
    )


# Direct run
app = create_app()
