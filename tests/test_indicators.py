"""Tests for indicators module."""

import numpy as np
import pandas as pd
import pytest

from crypto_trading.indicators import IndicatorEngine


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


class TestIndicatorEngine:
    def test_missing_column_raises(self):
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required column"):
            IndicatorEngine(df)

    def test_rsi(self, sample_df):
        engine = IndicatorEngine(sample_df)
        rsi = engine.rsi()
        assert len(rsi) == len(sample_df)
        assert rsi.min() >= 0
        assert rsi.max() <= 100

    def test_macd(self, sample_df):
        engine = IndicatorEngine(sample_df)
        macd = engine.macd()
        for key in ("macd", "macd_signal", "macd_diff"):
            assert key in macd
            assert len(macd[key]) == len(sample_df)

    def test_bollinger_bands(self, sample_df):
        engine = IndicatorEngine(sample_df)
        bb = engine.bollinger_bands()
        for key in ("bb_upper", "bb_middle", "bb_lower"):
            assert key in bb
            assert len(bb[key]) == len(sample_df)
        valid = bb["bb_middle"].notna()
        assert (bb["bb_upper"][valid] >= bb["bb_middle"][valid]).all()
        assert (bb["bb_middle"][valid] >= bb["bb_lower"][valid]).all()

    def test_ema_crossover(self, sample_df):
        engine = IndicatorEngine(sample_df)
        ema = engine.ema_crossover()
        for key in ("ema_fast", "ema_slow", "ema_diff", "ema_crossover"):
            assert key in ema
            assert len(ema[key]) == len(sample_df)
        assert set(ema["ema_crossover"].unique()).issubset({-1, 0, 1})

    def test_atr(self, sample_df):
        engine = IndicatorEngine(sample_df)
        atr = engine.atr()
        assert len(atr) == len(sample_df)
        assert (atr >= 0).all()

    def test_compute_all(self, sample_df):
        engine = IndicatorEngine(sample_df)
        result = engine.compute_all()
        expected_cols = {
            "open", "high", "low", "close", "volume",
            "rsi", "macd", "macd_signal", "macd_diff",
            "bb_upper", "bb_middle", "bb_lower",
            "ema_fast", "ema_slow", "ema_diff", "ema_crossover",
            "atr",
        }
        assert expected_cols.issubset(set(result.columns))
        assert len(result) == len(sample_df)
