"""Tests for order executor module — focused on DB recording logic.

Binance API calls are not made in unit tests; we test the
non-network parts and verify the DB recording works correctly.
"""

import json
from datetime import datetime, timezone

import pytest

from crypto_trading.database import Database
from crypto_trading.order_executor import OrderExecutor


class TestOrderExecutorDB:
    @pytest.fixture
    def db(self, tmp_path):
        db_path = str(tmp_path / "test_trading.db")
        return Database(db_path=db_path)

    @pytest.fixture
    def executor(self, db):
        return OrderExecutor(db=db)

    def test_record_order_sets_fields(self, executor):
        order = {
            "executedQty": "0.5",
            "cummulativeQuoteQty": "25000",
            "price": "50000",
            "commission": "5.0",
            "fills": [
                {"price": "50000", "qty": "0.5", "commission": "5.0", "commissionAsset": "BNB"}
            ],
        }
        executor._record_order(order, "BTC/USDT", "BUY", "test_strategy", 0.8)
        trades = executor.db.get_trade_history()
        assert len(trades) == 1
        t = trades[0]
        assert t["symbol"] == "BTC/USDT"
        assert t["side"] == "BUY"
        assert t["quantity"] == 0.5
        assert t["strategy"] == "test_strategy"
        assert t["signal_strength"] == 0.8
        assert t["status"] == "open"

    def test_record_order_empty_fills(self, executor):
        order = {
            "executedQty": "1.0",
            "cummulativeQuoteQty": "50000",
            "price": "50000",
            "commission": "0",
        }
        executor._record_order(order, "ETH/USDT", "SELL", "", 0.0)
        trades = executor.db.get_trade_history()
        assert len(trades) == 1
        t = trades[0]
        assert t["symbol"] == "ETH/USDT"
        assert t["side"] == "SELL"
        assert t["quantity"] == 1.0

    def test_multiple_records(self, executor):
        for i in range(3):
            order = {
                "executedQty": str(0.1 * (i + 1)),
                "price": str(100 * (i + 1)),
                "fills": [{"price": str(100 * (i + 1)), "qty": str(0.1 * (i + 1)), "commission": "0"}],
            }
            executor._record_order(order, "BTC/USDT", "BUY", "strat", 0.5)
        trades = executor.db.get_trade_history()
        assert len(trades) == 3
