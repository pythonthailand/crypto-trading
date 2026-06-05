"""Tests for backtest module — focused on core logic.

Backtesting depends on Binance API for data; we test the
non-network parts (risk checks, signal generation, summary calc).
"""

import numpy as np
import pandas as pd
import pytest

from crypto_trading.backtest import BacktestEngine
from crypto_trading.database import Database


class TestBacktestEngine:
    @pytest.fixture
    def engine(self, tmp_path):
        db_path = str(tmp_path / "test_backtest.db")
        db = Database(db_path=db_path)
        return BacktestEngine(initial_capital=10000, db=db)

    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        n = 200
        close = np.cumsum(np.random.randn(n)) + 100
        return pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="h"),
            "open": close - np.random.rand(n),
            "high": close + np.abs(np.random.randn(n)),
            "low": close - np.abs(np.random.randn(n)),
            "close": close,
            "volume": np.random.rand(n) * 1000,
        })

    def test_run_symbol_no_crash(self, engine, sample_df):
        engine._run_symbol("BTC/USDT", sample_df)
        assert engine.trade_log is not None

    def test_run_symbol_creates_trades(self, engine, sample_df):
        engine._run_symbol("BTC/USDT", sample_df)
        assert len(engine.trade_log) > 0

    def test_summary_returns_expected_keys(self, engine, sample_df):
        engine._run_symbol("BTC/USDT", sample_df)
        summary = engine._summary()
        expected_keys = {
            "initial_capital", "final_capital", "total_return",
            "total_return_pct", "total_trades", "win_rate",
            "avg_win", "avg_loss", "profit_factor",
            "sharpe_ratio", "max_drawdown", "open_positions",
        }
        assert expected_keys.issubset(set(summary.keys()))

    def test_summary_metrics_reasonable(self, engine, sample_df):
        engine._run_symbol("BTC/USDT", sample_df)
        summary = engine._summary()
        assert summary["initial_capital"] == 10000
        assert 0 <= summary["win_rate"] <= 1
        assert summary["total_trades"] >= 0
        assert summary["max_drawdown"] >= 0

    def test_check_exit_stop_loss_buy(self, engine):
        pos = {
            "side": "BUY",
            "entry_price": 100,
            "quantity": 1,
            "stop_loss": 95,
            "take_profit": 110,
            "trailing_active": False,
            "peak": 100,
        }
        engine.trade_log.append({
            "symbol": "TEST", "side": "BUY", "entry_price": 100,
            "quantity": 1, "exit_price": None, "exit_time": None,
            "pnl": None,
        })
        candle = pd.Series({"timestamp": pd.Timestamp("2026-01-01"), "high": 98, "low": 94, "close": 94})
        engine.positions["TEST"] = pos
        engine._check_exit("TEST", candle, pos)
        assert "TEST" not in engine.positions
        assert engine.trade_log[-1]["exit_reason"] == "stop_loss"

    def test_check_exit_take_profit_buy(self, engine):
        pos = {
            "side": "BUY",
            "entry_price": 100,
            "quantity": 1,
            "stop_loss": 95,
            "take_profit": 110,
            "trailing_active": False,
            "peak": 100,
        }
        engine.trade_log.append({
            "symbol": "TEST", "side": "BUY", "entry_price": 100,
            "quantity": 1, "exit_price": None, "exit_time": None,
            "pnl": None,
        })
        candle = pd.Series({"timestamp": pd.Timestamp("2026-01-01"), "high": 112, "low": 108, "close": 112})
        engine.positions["TEST"] = pos
        engine._check_exit("TEST", candle, pos)
        assert "TEST" not in engine.positions
        assert engine.trade_log[-1]["exit_reason"] == "take_profit"

    def test_check_exit_stop_loss_sell(self, engine):
        pos = {
            "side": "SELL",
            "entry_price": 100,
            "quantity": 1,
            "stop_loss": 105,
            "take_profit": 90,
            "trailing_active": False,
            "valley": 100,
        }
        engine.trade_log.append({
            "symbol": "TEST", "side": "SELL", "entry_price": 100,
            "quantity": 1, "exit_price": None, "exit_time": None,
            "pnl": None,
        })
        candle = pd.Series({"timestamp": pd.Timestamp("2026-01-01"), "high": 106, "low": 99, "close": 106})
        engine.positions["TEST"] = pos
        engine._check_exit("TEST", candle, pos)
        assert "TEST" not in engine.positions
        assert engine.trade_log[-1]["exit_reason"] == "stop_loss"
