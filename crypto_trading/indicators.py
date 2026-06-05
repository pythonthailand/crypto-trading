"""Technical indicators calculator.

Computes common technical-analysis indicators from OHLCV data using the `ta` library.

Indicators
==========
- **RSI (14)** — Relative Strength Index; oversold < 30, overbought > 70.
- **MACD (12, 26, 9)** — Moving Average Convergence Divergence with signal line.
- **Bollinger Bands (20, 2)** — Middle SMA(20) ± 2 * std; overbought/oversold levels.
- **EMA Crossover (9, 21)** — Two exponential moving averages; crossover = bullish,
  crossunder = bearish.
- **ATR (14)** — Average True Range; volatility measure used in risk management.

All methods expect a pandas DataFrame with at least columns:
    open, high, low, close, volume

Usage
-----
    import pandas as pd
    from crypto_trading.indicators import IndicatorEngine

    df = pd.DataFrame(...)  # OHLCV data
    engine = IndicatorEngine(df)
    features = engine.compute_all()
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd


class IndicatorEngine:
    """Computes a suite of technical indicators from OHLCV data."""

    def __init__(self, df: pd.DataFrame):
        df = df.copy()
        # Ensure required columns exist
        for col in ("open", "high", "low", "close", "volume"):
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        self.df = df

    # ------------------------------------------------------------------
    # Individual indicators
    # ------------------------------------------------------------------

    def rsi(self, period: int = 14) -> pd.Series:
        """Relative Strength Index."""
        import ta.momentum
        return ta.momentum.RSIIndicator(close=self.df["close"], window=period).rsi()

    def macd(
        self, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Dict[str, pd.Series]:
        """MACD line, signal line, and histogram."""
        import ta.trend
        macd = ta.trend.MACD(
            close=self.df["close"],
            window_slow=slow,
            window_fast=fast,
            window_sign=signal,
        )
        return {
            "macd": macd.macd(),
            "macd_signal": macd.macd_signal(),
            "macd_diff": macd.macd_diff(),
        }

    def bollinger_bands(self, period: int = 20, std: int = 2) -> Dict[str, pd.Series]:
        """Bollinger Bands (upper, middle, lower)."""
        import ta.volatility
        bb = ta.volatility.BollingerBands(
            close=self.df["close"], window=period, window_dev=std
        )
        return {
            "bb_upper": bb.bollinger_hband(),
            "bb_middle": bb.bollinger_mavg(),
            "bb_lower": bb.bollinger_lband(),
        }

    def ema_crossover(self, fast: int = 9, slow: int = 21) -> Dict[str, Any]:
        """EMA values and crossover signal.

        Returns a dict with:
            ema_fast, ema_slow, crossover (1 = bullish, -1 = bearish, 0 = flat)
        """
        ema_fast = self.df["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = self.df["close"].ewm(span=slow, adjust=False).mean()

        diff = ema_fast - ema_slow
        prev_diff = diff.shift(1)
        crossover = pd.Series(0, index=self.df.index)
        crossover[(diff > 0) & (prev_diff <= 0)] = 1  # golden cross
        crossover[(diff < 0) & (prev_diff >= 0)] = -1  # death cross

        return {
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "ema_diff": diff,
            "ema_crossover": crossover,
        }

    def atr(self, period: int = 14) -> pd.Series:
        """Average True Range."""
        import ta.volatility
        return ta.volatility.AverageTrueRange(
            high=self.df["high"],
            low=self.df["low"],
            close=self.df["close"],
            window=period,
        ).average_true_range()

    # ------------------------------------------------------------------
    # Aggregated
    # ------------------------------------------------------------------

    def compute_all(self) -> pd.DataFrame:
        """Compute all indicators and return a single DataFrame with their latest values.

        The returned DataFrame has one row per input row with named columns.
        """
        result = self.df[["open", "high", "low", "close", "volume"]].copy()

        result["rsi"] = self.rsi()

        macd_data = self.macd()
        result["macd"] = macd_data["macd"]
        result["macd_signal"] = macd_data["macd_signal"]
        result["macd_diff"] = macd_data["macd_diff"]

        bb_data = self.bollinger_bands()
        result["bb_upper"] = bb_data["bb_upper"]
        result["bb_middle"] = bb_data["bb_middle"]
        result["bb_lower"] = bb_data["bb_lower"]

        ema_data = self.ema_crossover()
        result["ema_fast"] = ema_data["ema_fast"]
        result["ema_slow"] = ema_data["ema_slow"]
        result["ema_diff"] = ema_data["ema_diff"]
        result["ema_crossover"] = ema_data["ema_crossover"]

        result["atr"] = self.atr()

        return result
