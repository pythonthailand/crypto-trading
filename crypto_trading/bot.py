"""Main bot loop — WebSocket real-time feed + periodic strategy execution.

Architecture
============
1. Connect to Binance WebSocket for real-time kline/candlestick data on all pairs.
2. On each new candle:
   a. Update local OHLCV buffer.
   b. Run signal generation every ``STRATEGY_INTERVAL_MINUTES``.
   c. If a trade signal is generated, pass through risk checks.
   d. If risk checks pass, execute via ``OrderExecutor``.
   e. Monitor open positions for stop-loss/take-profit/trailing-stop hits.
3. Periodically persist performance snapshots to the database.

Data Flow
---------
WebSocket → Candle Buffer → IndicatorEngine → SignalGenerator → RiskManager → OrderExecutor → Binance
                                    ↑                                          ↓
                                Database ←──────────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from loguru import logger

from crypto_trading.config import settings
from crypto_trading.database import Database
from crypto_trading.indicators import IndicatorEngine
from crypto_trading.order_executor import OrderExecutor
from crypto_trading.portfolio import Portfolio
from crypto_trading.risk_manager import RiskManager
from crypto_trading.signal_generator import SignalGenerator

try:
    from binance import AsyncClient, BinanceSocketManager
except ImportError:
    AsyncClient = None  # type: ignore
    BinanceSocketManager = None  # type: ignore


class TradingBot:
    """Main bot orchestrating WebSocket feed, signal generation, and trade execution."""

    def __init__(self, config_path: str = ".env"):
        import dotenv
        dotenv.load_dotenv(config_path)

        self.db = Database()
        self.executor = OrderExecutor(db=self.db)
        self.risk = RiskManager(db=self.db)
        self.signal_gen = SignalGenerator()
        self.portfolio = Portfolio(db=self.db, executor=self.executor)

        # Per-symbol OHLCV buffers (up to 100 candles each)
        self.buffers: Dict[str, pd.DataFrame] = {}
        self.last_signal_time: Dict[str, float] = defaultdict(float)
        self._running = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the bot (blocking)."""
        asyncio.run(self._run_async())

    async def _run_async(self) -> None:
        self._running = True
        logger.info("Starting trading bot in live mode")

        if AsyncClient is None:
            logger.error("python-binance not installed — cannot run live")
            return

        client = await AsyncClient.create(
            api_key=settings.binance_api_key,
            api_secret=settings.binance_api_secret,
            testnet=settings.binance_testnet,
        )
        bm = BinanceSocketManager(client)
        symbols_binance = [s.replace("/", "").lower() for s in settings.symbol_list]

        # Start kline socket for each symbol
        streams = []
        for sym in symbols_binance:
            stream = bm.kline_socket(symbol=sym, interval=Client.KLINE_INTERVAL_1MINUTE if hasattr(Client, 'KLINE_INTERVAL_1MINUTE') else "1m")
            streams.append(stream)

        # Multiplex socket (more efficient)
        socket = bm.multiplex_socket([f"{sym}@kline_1m" for sym in symbols_binance])

        logger.info("WebSocket connected — listening for klines")

        async with socket:
            while self._running:
                try:
                    msg = await socket.recv()
                    if msg:
                        await self._on_message(msg)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error("WebSocket error: {}", e)
                    await asyncio.sleep(1)

        await client.close_connection()
        logger.info("Bot stopped")

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # Message handler
    # ------------------------------------------------------------------

    async def _on_message(self, msg: Dict[str, Any]) -> None:
        """Process an incoming WebSocket kline message."""
        data = msg.get("data", msg)
        kline = data.get("k", data)
        if not kline:
            return

        symbol = data.get("s", "")
        is_final = kline.get("x", False)
        if not is_final:
            return  # only act on closed candles

        # Normalise symbol
        symbol_fmt = f"{symbol[:-3]}/{symbol[-3:]}" if symbol.endswith("USDT") else symbol

        ohlcv = {
            "timestamp": pd.to_datetime(kline["t"], unit="ms"),
            "open": float(kline["o"]),
            "high": float(kline["h"]),
            "low": float(kline["l"]),
            "close": float(kline["c"]),
            "volume": float(kline["v"]),
        }

        # Update buffer
        if symbol_fmt not in self.buffers:
            self.buffers[symbol_fmt] = pd.DataFrame(columns=[
                "timestamp", "open", "high", "low", "close", "volume"
            ])
        buf = self.buffers[symbol_fmt]
        buf = pd.concat([buf, pd.DataFrame([ohlcv])], ignore_index=True)
        if len(buf) > 100:
            buf = buf.iloc[-100:]
        self.buffers[symbol_fmt] = buf

        # Check strategy interval
        now = time.time()
        interval_sec = settings.strategy_interval_minutes * 60
        if now - self.last_signal_time[symbol_fmt] < interval_sec:
            return
        self.last_signal_time[symbol_fmt] = now

        if len(buf) < 30:
            return  # need enough data for indicators

        await self._execute_strategy(symbol_fmt, buf)

    async def _execute_strategy(
        self, symbol: str, df: pd.DataFrame
    ) -> None:
        """Run strategy for a single symbol."""
        # 1. Generate signal
        signal_result = self.signal_gen.generate(df)
        signal = signal_result["signal"]
        strength = signal_result["strength"]

        # 2. Log signal to DB
        self.db.insert_signal(
            symbol=symbol,
            signal=signal,
            strength=strength,
            strategy=settings.default_strategy,
            indicators=signal_result.get("indicators"),
        )

        if signal == "HOLD":
            return

        # 3. Risk check
        latest = df.iloc[-1]
        atr = signal_result["indicators"].get("atr", latest["close"] * 0.02)
        balance = self.portfolio.get_capital()

        risk_check = self.risk.check_all(
            symbol=symbol,
            side=signal,
            balance=balance,
            atr=atr,
            close_price=latest["close"],
        )

        if not risk_check["can_trade"]:
            logger.info("Risk check blocked {} {}: {}", signal, symbol, risk_check.get("reason"))
            return

        # 4. Execute
        order = self.executor.market_order(
            symbol=symbol,
            side=signal,
            quantity=risk_check["quantity"],
            strategy=settings.default_strategy,
            signal_strength=strength,
        )

        if order:
            logger.info(
                "Executed {} {} qty={:.4f} — SL={:.2f} TP={:.2f}",
                signal, symbol, risk_check["quantity"],
                risk_check["stop_loss"], risk_check["take_profit"],
            )

    # ------------------------------------------------------------------
    # Monitoring loop (for stop-loss / trailing checks)
    # ------------------------------------------------------------------

    async def _monitor_positions(self) -> None:
        """Periodic check for open positions — stop-loss / take-profit / trailing."""
        while self._running:
            open_trades = self.db.get_open_trades()
            for trade in open_trades:
                symbol = trade["symbol"]
                buf = self.buffers.get(symbol)
                if buf is None or buf.empty:
                    continue
                current_price = buf.iloc[-1]["close"]
                trade_id = trade["id"]
                side = trade["side"]
                entry_price = trade["price"]

                # Trailing stop
                self.risk.should_activate_trailing(trade_id, side, entry_price, current_price)
                if self.risk.check_trailing_stop(trade_id, side, current_price):
                    logger.info("Trailing stop hit for trade {}", trade_id)
                    self.executor.market_order(
                        symbol=symbol,
                        side="SELL" if side == "BUY" else "BUY",
                        quantity=trade["quantity"],
                    )
            await asyncio.sleep(10)
