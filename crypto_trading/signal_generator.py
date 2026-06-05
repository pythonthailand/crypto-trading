"""Signal generator — combines technical analysis with ML-based classification.

Architecture
============

1. **Technical Analysis Layer**
   RSI(14), MACD(12,26,9), Bollinger Bands(20,2), EMA crossover(9,21) computed
   by `IndicatorEngine`. Each indicator votes BUY / SELL / HOLD with a weight.

2. **AI Model Layer (XGBoost)**
   A pre-trained XGBoost classifier reads the same feature vector and outputs
   class probabilities for BUY / SELL / HOLD. The model is loaded from disk
   (``xgboost_model.json``).  If the file is missing the generator falls back
   to TA-only.

3. **Weighted Voting**
   Final signal = argmax of weighted sum across TA + ML scores.

   | Source        | Weight |
   |---------------|--------|
   | RSI           | 0.20   |
   | MACD          | 0.15   |
   | Bollinger     | 0.15   |
   | EMA crossover | 0.20   |
   | XGBoost       | 0.30   |

4. **Output**
   - signal: BUY / SELL / HOLD
   - strength: float 0–1
   - details: per-indicator breakdown for debugging
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from crypto_trading.config import settings
from crypto_trading.indicators import IndicatorEngine

try:
    import xgboost as xgb
except ImportError:
    xgb = None  # type: ignore


class SignalGenerator:
    """Generates trading signals by fusing TA indicators and XGBoost predictions."""

    # Indicator weights (must sum to 1.0)
    WEIGHTS = {
        "rsi": 0.20,
        "macd": 0.15,
        "bollinger": 0.15,
        "ema_crossover": 0.20,
        "ml": 0.30,
    }

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = Path(model_path or settings.xgboost_model_path)
        self._model: Optional["xgb.Booster"] = None
        self._load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        if xgb is None:
            logger.warning("xgboost not installed — ML signals disabled")
            return
        if self.model_path.exists():
            self._model = xgb.Booster()
            self._model.load_model(str(self.model_path))
            logger.info("XGBoost model loaded from {}", self.model_path)
        else:
            logger.warning(
                "Model {} not found — ML signals disabled", self.model_path
            )

    # ------------------------------------------------------------------
    # TA voting
    # ------------------------------------------------------------------

    @staticmethod
    def _rsi_vote(rsi_value: float) -> Tuple[str, float]:
        if rsi_value < 30:
            return "BUY", 0.8
        elif rsi_value > 70:
            return "SELL", 0.8
        return "HOLD", 0.5

    @staticmethod
    def _macd_vote(macd_diff: float) -> Tuple[str, float]:
        if macd_diff > 0:
            return "BUY", 0.6
        elif macd_diff < 0:
            return "SELL", 0.6
        return "HOLD", 0.3

    @staticmethod
    def _bb_vote(close: float, bb_lower: float, bb_upper: float) -> Tuple[str, float]:
        if close <= bb_lower:
            return "BUY", 0.7
        elif close >= bb_upper:
            return "SELL", 0.7
        return "HOLD", 0.4

    @staticmethod
    def _ema_vote(crossover_value: float) -> Tuple[str, float]:
        if crossover_value == 1:
            return "BUY", 0.8
        elif crossover_value == -1:
            return "SELL", 0.8
        return "HOLD", 0.3

    # ------------------------------------------------------------------
    # ML prediction
    # ------------------------------------------------------------------

    def _ml_predict(self, features: np.ndarray) -> Tuple[str, float]:
        """Return ML-based signal and confidence."""
        if self._model is None:
            return "HOLD", 0.0

        dmat = xgb.DMatrix(features.reshape(1, -1))
        preds = self._model.predict(dmat)[0]  # shape (3,) — [HOLD, BUY, SELL]
        classes = ["HOLD", "BUY", "SELL"]
        idx = int(np.argmax(preds))
        return classes[idx], float(preds[idx])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute a single combined signal from the latest OHLCV row.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data with at least 26 rows (for MACD). The last row is used.

        Returns
        -------
        dict with keys:
            signal, strength, details, indicators
        """
        engine = IndicatorEngine(df)
        indicators = engine.compute_all()
        last = indicators.iloc[-1]

        # TA votes
        rsi_sig, rsi_str = self._rsi_vote(last.get("rsi", 50))
        macd_sig, macd_str = self._macd_vote(last.get("macd_diff", 0))
        bb_sig, bb_str = self._bb_vote(
            last["close"],
            last.get("bb_lower", last["close"]),
            last.get("bb_upper", last["close"]),
        )
        ema_sig, ema_str = self._ema_vote(last.get("ema_crossover", 0))

        votes: Dict[str, Tuple[str, float]] = {
            "rsi": (rsi_sig, rsi_str),
            "macd": (macd_sig, macd_str),
            "bollinger": (bb_sig, bb_str),
            "ema_crossover": (ema_sig, ema_str),
        }

        # Weighted score per class
        scores = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}
        for src, (sig, strength) in votes.items():
            weight = self.WEIGHTS.get(src, 0)
            scores[sig] += weight * strength

        # ML contribution
        feature_vector = self._build_feature_vector(last)
        ml_sig, ml_str = self._ml_predict(feature_vector)
        scores[ml_sig] += self.WEIGHTS["ml"] * ml_str

        # Final signal
        final_signal = max(scores, key=scores.get)
        final_strength = scores[final_signal]

        return {
            "signal": final_signal,
            "strength": round(final_strength, 4),
            "details": {
                src: {"signal": s, "strength": round(st, 4)}
                for src, (s, st) in votes.items()
            },
            "ml": {"signal": ml_sig, "strength": round(ml_str, 4)},
            "indicators": {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in last.to_dict().items()
                if k in {"rsi", "macd", "macd_signal", "macd_diff", "bb_upper", "bb_middle",
                         "bb_lower", "ema_fast", "ema_slow", "ema_diff", "ema_crossover", "atr"}
            },
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_feature_vector(row: pd.Series) -> np.ndarray:
        """Construct the feature vector expected by the XGBoost model."""
        features = [
            row.get("rsi", 50),
            row.get("macd", 0),
            row.get("macd_signal", 0),
            row.get("macd_diff", 0),
            row.get("bb_upper", row["close"]),
            row.get("bb_middle", row["close"]),
            row.get("bb_lower", row["close"]),
            row.get("ema_fast", row["close"]),
            row.get("ema_slow", row["close"]),
            row.get("ema_diff", 0),
            row.get("ema_crossover", 0),
            row.get("atr", 0),
            row.get("volume", 0),
        ]
        return np.array(features, dtype=np.float32)

    def generate_all(
        self, data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, Any]]:
        """Generate signals for multiple symbols at once."""
        return {sym: self.generate(df) for sym, df in data.items()}
