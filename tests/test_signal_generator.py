"""Tests for signal generator module."""

import numpy as np
import pandas as pd
import pytest

from crypto_trading.signal_generator import SignalGenerator


@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 100
    close = np.cumsum(np.random.randn(n)) + 100
    return pd.DataFrame({
        "open": close - np.random.rand(n),
        "high": close + np.random.rand(n),
        "low": close - np.random.rand(n),
        "close": close,
        "volume": np.random.rand(n) * 1000,
    })


class TestSignalGenerator:
    def test_generate_returns_expected_keys(self, sample_df):
        gen = SignalGenerator()
        result = gen.generate(sample_df)
        assert "signal" in result
        assert "strength" in result
        assert "details" in result
        assert "indicators" in result
        assert result["signal"] in ("BUY", "SELL", "HOLD")
        assert 0 <= result["strength"] <= 1

    def test_generate_details_contains_sources(self, sample_df):
        gen = SignalGenerator()
        result = gen.generate(sample_df)
        for src in ("rsi", "macd", "bollinger", "ema_crossover"):
            assert src in result["details"]
            assert result["details"][src]["signal"] in ("BUY", "SELL", "HOLD")

    def test_generate_indicators_contains_ta_values(self, sample_df):
        gen = SignalGenerator()
        result = gen.generate(sample_df)
        ind = result["indicators"]
        for key in ("rsi", "macd", "macd_signal", "macd_diff", "bb_upper", "bb_middle", "bb_lower"):
            assert key in ind

    def test_ml_fallback_when_no_model(self, sample_df):
        gen = SignalGenerator()
        result = gen.generate(sample_df)
        assert result["ml"]["signal"] == "HOLD"
        assert result["ml"]["strength"] == 0.0

    def test_generate_all(self, sample_df):
        gen = SignalGenerator()
        data = {"BTC/USDT": sample_df, "ETH/USDT": sample_df}
        results = gen.generate_all(data)
        assert set(results.keys()) == {"BTC/USDT", "ETH/USDT"}

    def test_rsi_vote(self):
        assert SignalGenerator._rsi_vote(25)[0] == "BUY"
        assert SignalGenerator._rsi_vote(75)[0] == "SELL"
        assert SignalGenerator._rsi_vote(50)[0] == "HOLD"

    def test_macd_vote(self):
        assert SignalGenerator._macd_vote(1.0)[0] == "BUY"
        assert SignalGenerator._macd_vote(-1.0)[0] == "SELL"
        assert SignalGenerator._macd_vote(0)[0] == "HOLD"

    def test_bb_vote(self):
        close, lower, upper = 100, 95, 105
        assert SignalGenerator._bb_vote(lower - 1, lower, upper)[0] == "BUY"
        assert SignalGenerator._bb_vote(upper + 1, lower, upper)[0] == "SELL"
        assert SignalGenerator._bb_vote(100, lower, upper)[0] == "HOLD"

    def test_ema_vote(self):
        assert SignalGenerator._ema_vote(1)[0] == "BUY"
        assert SignalGenerator._ema_vote(-1)[0] == "SELL"
        assert SignalGenerator._ema_vote(0)[0] == "HOLD"
