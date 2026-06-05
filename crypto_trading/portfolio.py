"""Portfolio tracker.

Maintains a real-time view of the portfolio:
- Current capital / balance
- Open positions with unrealised PnL
- Historical closed trade PnL
- Daily performance aggregation
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from crypto_trading.config import settings
from crypto_trading.database import Database
from crypto_trading.order_executor import OrderExecutor


class Portfolio:
    """Tracks portfolio state and computes performance metrics."""

    def __init__(
        self,
        db: Optional[Database] = None,
        executor: Optional[OrderExecutor] = None,
    ):
        self.db = db or Database()
        self.executor = executor

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def get_capital(self) -> float:
        """Return available USDT balance from exchange or 10 000 default."""
        if self.executor:
            try:
                return self.executor.get_balance("USDT")
            except Exception:
                pass
        return 10_000.0

    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Return open trades with unrealised PnL."""
        trades = self.db.get_open_trades()
        for t in trades:
            t["unrealised_pnl"] = 0.0  # requires current price — filled by bot loop
            t["pnl_pct"] = 0.0
        return trades

    def get_trade_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        return self.db.get_trade_history(limit=limit)

    # ------------------------------------------------------------------
    # Performance
    # ------------------------------------------------------------------

    def compute_daily_performance(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Aggregate today's performance for all or a single symbol."""
        today = date.today().isoformat()
        symbols = [symbol] if symbol else settings.symbol_list

        results = {}
        for sym in symbols:
            trades = self.db.get_trade_history(symbol=sym, limit=1000)
            today_trades = [t for t in trades if t["closed_at"] and t["closed_at"].startswith(today)]
            closed = [t for t in today_trades if t["pnl"] is not None]
            pnls = [t["pnl"] for t in closed]
            wins = [p for p in pnls if p > 0]

            total_trades = len(closed)
            win_rate = len(wins) / total_trades if total_trades > 0 else 0.0
            profit_loss = sum(pnls) if pnls else 0.0

            # Max drawdown for today
            if pnls:
                cum = np.cumsum(pnls)
                peak = np.maximum.accumulate(cum)
                dd = np.max((peak - cum) / (peak + 1e-9))
                max_dd = float(dd)
            else:
                max_dd = 0.0

            # Sharpe (daily)
            if len(pnls) > 1 and np.std(pnls) > 0:
                sharpe = float(np.mean(pnls) / np.std(pnls) * np.sqrt(365))
            else:
                sharpe = 0.0

            self.db.upsert_performance(
                date=today,
                symbol=sym,
                total_trades=total_trades,
                win_rate=win_rate,
                profit_loss=profit_loss,
                max_drawdown=max_dd,
                sharpe_ratio=sharpe,
            )
            results[sym] = {
                "total_trades": total_trades,
                "win_rate": round(win_rate, 4),
                "profit_loss": round(profit_loss, 2),
                "max_drawdown": round(max_dd, 4),
                "sharpe_ratio": round(sharpe, 4),
            }

        return results
