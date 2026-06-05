"""Tests for risk manager module."""

import pytest

from crypto_trading.risk_manager import RiskManager


class TestRiskManager:
    def setup_method(self):
        self.rm = RiskManager()

    def test_stop_loss_pct_clamps_min(self):
        pct = RiskManager.stop_loss_pct(atr=0.1, close_price=100)
        assert pct == 0.02

    def test_stop_loss_pct_clamps_max(self):
        pct = RiskManager.stop_loss_pct(atr=10, close_price=100)
        assert pct == 0.05

    def test_stop_loss_pct_normal(self):
        pct = RiskManager.stop_loss_pct(atr=3, close_price=100)
        assert 0.02 < pct < 0.05
        assert pct == pytest.approx(0.03)

    def test_take_profit_pct(self):
        tp = RiskManager.take_profit_pct(0.03)
        assert tp == 0.09

    def test_take_profit_pct_clamps_min(self):
        tp = RiskManager.take_profit_pct(0.01)
        assert tp == 0.06

    def test_take_profit_pct_clamps_max(self):
        tp = RiskManager.take_profit_pct(0.06)
        assert tp == 0.15

    def test_stop_loss_price_buy(self):
        sl = self.rm.stop_loss_price("BUY", 100, 0.05)
        assert sl == 95.0

    def test_stop_loss_price_sell(self):
        sl = self.rm.stop_loss_price("SELL", 100, 0.05)
        assert sl == 105.0

    def test_take_profit_price_buy(self):
        tp = self.rm.take_profit_price("BUY", 100, 0.10)
        assert tp == pytest.approx(110.0)

    def test_take_profit_price_sell(self):
        tp = self.rm.take_profit_price("SELL", 100, 0.10)
        assert tp == 90.0

    def test_position_size_zero_balance(self):
        size = self.rm.position_size(balance=0, atr=2, close_price=100)
        assert size == 0

    def test_position_size_positive(self):
        size = self.rm.position_size(balance=10000, atr=2, close_price=100)
        assert size > 0
        expected_risk = 10000 * 0.02
        expected_sl = 0.02
        expected_size = expected_risk / (100 * expected_sl)
        assert size == pytest.approx(expected_size)

    def test_trailing_stop_activation(self):
        result = self.rm.should_activate_trailing(
            trade_id=1, side="BUY", entry_price=100, current_price=106
        )
        assert result is True
        assert 1 in self.rm._active_trailing

    def test_trailing_stop_not_activated_below_threshold(self):
        result = self.rm.should_activate_trailing(
            trade_id=2, side="BUY", entry_price=100, current_price=103
        )
        assert result is False

    def test_check_trailing_stop_hit(self):
        self.rm._active_trailing[1] = {
            "activate_price": 105,
            "trail_distance": 0.02,
            "peak": 110,
        }
        hit = self.rm.check_trailing_stop(trade_id=1, side="BUY", current_price=100)
        assert hit is True

    def test_check_trailing_stop_not_hit(self):
        self.rm._active_trailing[1] = {
            "activate_price": 105,
            "trail_distance": 0.02,
        }
        hit = self.rm.check_trailing_stop(trade_id=1, side="BUY", current_price=110)
        assert hit is False

    def test_can_open_position_no_trades(self):
        assert self.rm.can_open_position("BTC/USDT") is True

    def test_update_daily_pnl(self):
        self.rm.update_daily_pnl(100)
        today = __import__("datetime").date.today().isoformat()
        assert self.rm._daily_pnl[today] == 100.0
        self.rm.update_daily_pnl(-50)
        assert self.rm._daily_pnl[today] == 50.0
