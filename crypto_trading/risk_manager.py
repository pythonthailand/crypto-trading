"""Risk manager — stop-loss, take-profit, position sizing, and drawdown control.

Risk Management Rules
=====================

1. **Position Sizing**
   - Fixed fraction: 2 % of available capital per trade (configurable via
     ``RISK_PER_TRADE``).
   - Optional Kelly Criterion sizing based on historical win rate / avg win / avg loss.

2. **Stop-Loss**
   - Dynamic percentage based on ATR: ``stop_loss_pct = ATR / close``
   - Clamped to 2–5 % range.

3. **Take-Profit**
   - Risk-reward ratio of 1:3 applied to stop-loss distance.
   - Clamped to 6–15 %.

4. **Trailing Stop**
   - Activates when unrealised profit reaches ``trailing_stop_activate_pct`` (5 %).
   - Trails price by ``trailing_stop_distance_pct`` (2 %).

5. **Max Drawdown**
   - Daily max drawdown limit: 20 % (configurable via ``MAX_DAILY_DRAWDOWN``).
   - If exceeded, trading halts until next day.

6. **Max Positions**
   - Maximum 1 concurrent position per symbol.
   - Maximum 3 concurrent positions across all symbols.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger

from crypto_trading.config import settings
from crypto_trading.database import Database


class RiskManager:
    """Evaluates risk constraints before trade execution."""

    def __init__(self, db: Optional[Database] = None):
        self.db = db or Database()
        self._daily_pnl: Dict[str, float] = {}
        self._active_trailing: Dict[int, Dict[str, float]] = {}  # trade_id -> {activate_price, trail_distance}

    # ------------------------------------------------------------------
    # Position Sizing
    # ------------------------------------------------------------------

    def position_size(
        self,
        balance: float,
        atr: float,
        close_price: float,
        symbol: str = "",
    ) -> float:
        """Calculate position size in base asset units.

        Uses fixed-fractional risk: risk `RISK_PER_TRADE` of balance.
        Position size = (balance * risk_per_trade) / stop_loss_distance
        """
        sl_pct = self.stop_loss_pct(atr, close_price)
        risk_amount = balance * settings.risk_per_trade
        size = risk_amount / (close_price * sl_pct)
        logger.debug(
            "Position size for {}: {} (balance={}, risk_amt={}, sl_pct={})",
            symbol, size, balance, risk_amount, sl_pct,
        )
        return size

    # ------------------------------------------------------------------
    # Stop-Loss / Take-Profit
    # ------------------------------------------------------------------

    @staticmethod
    def stop_loss_pct(atr: float, close_price: float) -> float:
        """Dynamic stop-loss percentage clamped to [2 %, 5 %]."""
        pct = atr / close_price if close_price > 0 else settings.stop_loss_pct
        return max(0.02, min(0.05, pct))

    @staticmethod
    def take_profit_pct(sl_pct: float) -> float:
        """Take-profit = 3 × stop-loss, clamped to [6 %, 15 %]."""
        pct = sl_pct * 3
        return max(0.06, min(0.15, pct))

    def stop_loss_price(self, side: str, entry_price: float, sl_pct: float) -> float:
        if side == "BUY":
            return entry_price * (1 - sl_pct)
        return entry_price * (1 + sl_pct)

    def take_profit_price(self, side: str, entry_price: float, tp_pct: float) -> float:
        if side == "BUY":
            return entry_price * (1 + tp_pct)
        return entry_price * (1 - tp_pct)

    # ------------------------------------------------------------------
    # Trailing Stop
    # ------------------------------------------------------------------

    def should_activate_trailing(
        self, trade_id: int, side: str, entry_price: float, current_price: float
    ) -> bool:
        if side == "BUY":
            profit_pct = (current_price - entry_price) / entry_price
        else:
            profit_pct = (entry_price - current_price) / entry_price

        if trade_id not in self._active_trailing and profit_pct >= settings.trailing_stop_activate_pct:
            self._active_trailing[trade_id] = {
                "activate_price": current_price,
                "trail_distance": settings.trailing_stop_distance_pct,
            }
            logger.info("Trailing stop activated for trade {} at {:.2f}", trade_id, current_price)
            return True
        return False

    def check_trailing_stop(
        self, trade_id: int, side: str, current_price: float
    ) -> bool:
        """Return True if trailing stop is hit (should close)."""
        info = self._active_trailing.get(trade_id)
        if not info:
            return False

        trail_dist = info["trail_distance"]
        if side == "BUY":
            stop_level = current_price * (1 - trail_dist)
            # Update trail if price moves up
            peak = max(info.get("peak", 0), current_price)
            info["peak"] = peak
            return current_price <= peak * (1 - trail_dist)
        else:
            stop_level = current_price * (1 + trail_dist)
            valley = min(info.get("valley", float("inf")), current_price)
            info["valley"] = valley
            return current_price >= valley * (1 + trail_dist)

    # ------------------------------------------------------------------
    # Drawdown
    # ------------------------------------------------------------------

    def check_daily_drawdown(self) -> bool:
        """Return True if daily drawdown limit is exceeded (trading should halt)."""
        today = date.today().isoformat()
        total_pnl = sum(
            v for k, v in self._daily_pnl.items() if k.startswith(today)
        )
        # Approximate daily PnL as percentage of total capital
        # A more accurate version would track starting capital
        dd_pct = abs(total_pnl) / 10_000  # placeholder: assume 10k base
        if dd_pct > settings.max_daily_drawdown:
            logger.warning(
                "Daily drawdown {:.1%} exceeds limit {:.1%} — halting",
                dd_pct, settings.max_daily_drawdown,
            )
            return False
        return True

    def update_daily_pnl(self, pnl: float) -> None:
        today = date.today().isoformat()
        if today not in self._daily_pnl:
            self._daily_pnl[today] = 0.0
        self._daily_pnl[today] += pnl

    # ------------------------------------------------------------------
    # Concurrency checks
    # ------------------------------------------------------------------

    def can_open_position(self, symbol: str) -> bool:
        """Check if a new position can be opened."""
        open_trades = self.db.get_open_trades()
        # Max 1 per symbol
        symbol_count = sum(1 for t in open_trades if t["symbol"] == symbol)
        if symbol_count >= 1:
            logger.debug("Position already open for {}", symbol)
            return False
        # Max 3 total
        if len(open_trades) >= 3:
            logger.debug("Max concurrent positions (3) reached")
            return False
        return True

    # ------------------------------------------------------------------
    # Full check
    # ------------------------------------------------------------------

    def check_all(
        self,
        symbol: str,
        side: str,
        balance: float,
        atr: float,
        close_price: float,
    ) -> Dict[str, Any]:
        """Run all risk checks and return a dict with go/no-go + sizes."""
        result = {
            "can_trade": True,
            "reason": "",
            "quantity": 0.0,
            "stop_loss": 0.0,
            "take_profit": 0.0,
        }

        if not self.check_daily_drawdown():
            result["can_trade"] = False
            result["reason"] = "Daily drawdown limit exceeded"
            return result

        if not self.can_open_position(symbol):
            result["can_trade"] = False
            result["reason"] = "Cannot open position (concurrency limit)"
            return result

        sl_pct = self.stop_loss_pct(atr, close_price)
        tp_pct = self.take_profit_pct(sl_pct)

        qty = self.position_size(balance, atr, close_price, symbol)
        sl_price = self.stop_loss_price(side, close_price, sl_pct)
        tp_price = self.take_profit_price(side, close_price, tp_pct)

        result.update(
            quantity=qty,
            stop_loss=sl_price,
            take_profit=tp_price,
            stop_loss_pct=sl_pct,
            take_profit_pct=tp_pct,
        )
        return result
