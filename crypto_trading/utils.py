"""Utility functions for the Crypto Trading Bot."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict


def to_decimal(value: Any, precision: int = 8) -> Decimal:
    """Convert a value to Decimal with given precision."""
    return Decimal(str(value)).quantize(Decimal(f"0.{'0' * precision}"))


def now_iso() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def load_json(path: str) -> Dict[str, Any]:
    """Load JSON file."""
    with open(path) as f:
        return dict(json.load(f))


def save_json(path: str, data: Dict[str, Any]) -> None:
    """Save JSON file, creating parent directories if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def clamp(value: float, low: float, high: float) -> float:
    """Clamp value to [low, high]."""
    return max(low, min(high, value))


def calculate_sharpe(pnls, risk_free: float = 0.0, periods_per_year: int = 365):
    """Calculate Sharpe ratio from a list of PnL values."""
    import numpy as np
    if len(pnls) < 2:
        return 0.0
    returns = np.array(pnls)
    excess = returns - risk_free
    if np.std(excess) == 0:
        return 0.0
    return float(np.mean(excess) / np.std(excess) * np.sqrt(periods_per_year))


def format_symbol(binance_symbol: str) -> str:
    """Convert Binance symbol (e.g. BTCUSDT) to human-friendly (BTC/USDT)."""
    if binance_symbol.endswith("USDT"):
        return f"{binance_symbol[:-4]}/USDT"
    return binance_symbol
