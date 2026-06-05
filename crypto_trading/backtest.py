"""Backtesting engine.

Simulates trading over historical OHLCV data (30-day lookback) using the same
signal generation and risk management logic as the live bot.

Workflow
========
1. Fetch historical data from Binance (or load from CSV).
2. Iterate over each candle:
   a. Compute technical indicators on a rolling window.
   b. Generate signal via ``SignalGenerator``.
   c. If signal is BUY/SELL and risk checks pass → execute paper trade.
   d. Monitor open positions for stop-loss / take-profit / trailing-stop hits.
   e. Record PnL upon exit.
3. Report aggregate metrics: total return, win rate, Sharpe ratio, max drawdown.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from crypto_trading.config import settings
from crypto_trading.database import Database
from crypto_trading.indicators import IndicatorEngine
from crypto_trading.risk_manager import RiskManager
from crypto_trading.signal_generator import SignalGenerator

try:
    from binance.client import Client as BinanceClient
except ImportError:
    BinanceClient = None  # type: ignore


class BacktestEngine:
    """Simulates trading over historical data."""

    def __init__(
        self,
        initial_capital: float = 10_000.0,
        db: Optional[Database] = None,
    ):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.db = db or Database()
        self.risk = RiskManager(db=self.db)
        self.signal_gen = SignalGenerator()
        self.positions: Dict[str, Dict[str, Any]] = {}  # symbol -> position info
        self.trade_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def fetch_historical(
        self,
        symbol: str,
        interval: str = "1h",
        days: int = 30,
    ) -> pd.DataFrame:
        """Fetch OHLCV candles from Binance."""
        if BinanceClient is None:
            raise ImportError("python-binance is required for backtesting")

        client = BinanceClient(
            api_key=settings.binance_api_key,
            api_secret=settings.binance_api_secret,
            testnet=True,
        )
        start_str = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        klines = client.get_historical_klines(
            symbol=symbol.replace("/", ""),
            interval=interval,
            start_str=start_str,
        )
        df = pd.DataFrame(
            klines,
            columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_vol", "trades", "taker_buy_vol",
                "taker_buy_quote", "ignore",
            ],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        return df[["timestamp", "open", "high", "low", "close", "volume"]]

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(
        self,
        symbols: Optional[List[str]] = None,
        interval: str = "1h",
        days: int = 30,
    ) -> Dict[str, Any]:
        """Run backtest across one or more symbols."""
        symbols = symbols or settings.symbol_list
        logger.info(
            "Starting backtest: {} symbols, {} days, {} interval",
            len(symbols), days, interval,
        )

        for symbol in symbols:
            df = self.fetch_historical(symbol, interval, days)
            self._run_symbol(symbol, df)

        return self._summary()

    def _run_symbol(self, symbol: str, df: pd.DataFrame) -> None:
        """Backtest a single symbol."""
        for i in range(30, len(df)):
            window = df.iloc[: i + 1]
            candle = df.iloc[i]
            close = candle["close"]

            # --- Check existing positions first ---
            if symbol in self.positions:
                pos = self.positions[symbol]
                self._check_exit(symbol, candle, pos)

            # --- Generate signal ---
            signal_result = self.signal_gen.generate(window)
            signal = signal_result["signal"]
            strength = signal_result["strength"]
            indicators = signal_result["indicators"]

            # --- Risk check ---
            if signal != "HOLD" and symbol not in self.positions:
                atr = indicators.get("atr", close * 0.02)
                risk_check = self.risk.check_all(
                    symbol=symbol,
                    side=signal,
                    balance=self.capital,
                    atr=atr,
                    close_price=close,
                )

                if risk_check["can_trade"]:
                    qty = risk_check["quantity"]
                    self.positions[symbol] = {
                        "side": signal,
                        "entry_price": close,
                        "quantity": qty,
                        "entry_time": candle["timestamp"],
                        "stop_loss": risk_check["stop_loss"],
                        "take_profit": risk_check["take_profit"],
                        "trailing_active": False,
                        "peak": close if signal == "BUY" else None,
                        "valley": close if signal == "SELL" else None,
                    }
                    self.trade_log.append({
                        "symbol": symbol,
                        "side": signal,
                        "entry_price": close,
                        "quantity": qty,
                        "entry_time": candle["timestamp"],
                        "exit_price": None,
                        "exit_time": None,
                        "pnl": None,
                        "signal_strength": strength,
                    })
                    cost = qty * close
                    self.capital -= cost
                    logger.debug(
                        "[BT] {} {} {:.4f} @ {:.2f} (cost {:.2f})",
                        signal, symbol, qty, close, cost,
                    )

    def _check_exit(
        self, symbol: str, candle: pd.Series, pos: Dict[str, Any]
    ) -> None:
        """Check stop-loss, take-profit, trailing stop."""
        close = candle["close"]
        high = candle["high"]
        low = candle["low"]
        side = pos["side"]
        entry = pos["entry_price"]

        exit_triggered = False
        exit_reason = ""

        if side == "BUY":
            # Stop-loss
            if low <= pos["stop_loss"]:
                exit_price = pos["stop_loss"]
                exit_reason = "stop_loss"
                exit_triggered = True
            # Take-profit
            elif high >= pos["take_profit"]:
                exit_price = pos["take_profit"]
                exit_reason = "take_profit"
                exit_triggered = True
            # Trailing stop
            else:
                peak = max(pos.get("peak", entry), high)
                pos["peak"] = peak
                if peak >= entry * (1 + settings.trailing_stop_activate_pct):
                    trail_level = peak * (1 - settings.trailing_stop_distance_pct)
                    if low <= trail_level:
                        exit_price = trail_level
                        exit_reason = "trailing_stop"
                        exit_triggered = True
        else:  # SELL
            if high >= pos["stop_loss"]:
                exit_price = pos["stop_loss"]
                exit_reason = "stop_loss"
                exit_triggered = True
            elif low <= pos["take_profit"]:
                exit_price = pos["take_profit"]
                exit_reason = "take_profit"
                exit_triggered = True
            else:
                valley = min(pos.get("valley", entry), low)
                pos["valley"] = valley
                if valley <= entry * (1 - settings.trailing_stop_activate_pct):
                    trail_level = valley * (1 + settings.trailing_stop_distance_pct)
                    if high >= trail_level:
                        exit_price = trail_level
                        exit_reason = "trailing_stop"
                        exit_triggered = True

        if exit_triggered:
            qty = pos["quantity"]
            if side == "BUY":
                pnl = (exit_price - entry) * qty
            else:
                pnl = (entry - exit_price) * qty
            self.capital += qty * exit_price

            # Update the most recent trade log entry
            for t in reversed(self.trade_log):
                if t["symbol"] == symbol and t["exit_price"] is None:
                    t["exit_price"] = exit_price
                    t["exit_time"] = candle["timestamp"]
                    t["pnl"] = pnl
                    t["exit_reason"] = exit_reason
                    break

            del self.positions[symbol]
            logger.debug(
                "[BT] Exit {} {} @ {:.2f} ({}) pnl={:.2f}",
                symbol, side, exit_price, exit_reason, pnl,
            )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _summary(self) -> Dict[str, Any]:
        """Compute backtest performance summary."""
        closed = [t for t in self.trade_log if t["pnl"] is not None]
        pnls = [t["pnl"] for t in closed]
        wins = [p for p in pnls if p > 0]

        total_return = self.capital - self.initial_capital
        total_return_pct = total_return / self.initial_capital
        win_rate = len(wins) / len(pnls) if pnls else 0.0
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean([p for p in pnls if p < 0]) if any(p < 0 for p in pnls) else 0.0
        profit_factor = abs(sum(wins) / sum(p for p in pnls if p < 0)) if any(p < 0 for p in pnls) else float("inf")

        # Sharpe ratio (assuming risk-free rate = 0)
        if len(pnls) > 1 and np.std(pnls) > 0:
            sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(365 * 24)  # hourly
        else:
            sharpe = 0.0

        # Max drawdown
        cumulative = self.initial_capital
        peak_capital = cumulative
        max_dd = 0.0
        for t in closed:
            cumulative += t["pnl"]
            peak_capital = max(peak_capital, cumulative)
            dd = (peak_capital - cumulative) / peak_capital
            max_dd = max(max_dd, dd)

        return {
            "initial_capital": self.initial_capital,
            "final_capital": round(self.capital, 2),
            "total_return": round(total_return, 2),
            "total_return_pct": round(total_return_pct, 4),
            "total_trades": len(closed),
            "win_rate": round(win_rate, 4),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown": round(max_dd, 4),
            "open_positions": len(self.positions),
        }


def run_backtest(config_path: str = ".env") -> None:
    """CLI entry point for backtesting."""
    import dotenv
    dotenv.load_dotenv(config_path)

    engine = BacktestEngine()
    result = engine.run()
    logger.info("Backtest complete\n{}", _fmt(result))


def _fmt(result: Dict[str, Any]) -> str:
    lines = [
        "═══════════════════════════════════════",
        "         Backtest Results",
        "═══════════════════════════════════════",
        f"  Initial capital:  ${result['initial_capital']:>10.2f}",
        f"  Final capital:    ${result['final_capital']:>10.2f}",
        f"  Total return:     ${result['total_return']:>+10.2f}  ({result['total_return_pct']:>+.2%})",
        f"  Total trades:     {result['total_trades']:>10d}",
        f"  Win rate:         {result['win_rate']:>10.2%}",
        f"  Avg win:          ${result['avg_win']:>+10.2f}",
        f"  Avg loss:         ${result['avg_loss']:>+10.2f}",
        f"  Profit factor:    {result['profit_factor']:>10.2f}",
        f"  Sharpe ratio:     {result['sharpe_ratio']:>10.4f}",
        f"  Max drawdown:     {result['max_drawdown']:>10.2%}",
        "═══════════════════════════════════════",
    ]
    return "\n".join(lines)
