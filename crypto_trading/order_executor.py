"""Binance order executor.

Handles placement, cancellation, and status-checking of market/limit orders
via the `python-binance` library.  Supports both live and testnet endpoints.

Order Types
-----------
- **MARKET** — executed immediately at current best price.
- **LIMIT** — placed with a specific price; filled when the market reaches it.

Risk checks are delegated to `RiskManager`; this module only executes.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, Optional

from loguru import logger

from crypto_trading.config import settings
from crypto_trading.database import Database

try:
    from binance.client import Client as BinanceClient
    from binance.exceptions import BinanceAPIException
except ImportError:
    BinanceClient = None  # type: ignore
    BinanceAPIException = None  # type: ignore


class OrderExecutor:
    """Executes BUY / SELL orders on Binance."""

    def __init__(self, db: Optional[Database] = None):
        self.db = db or Database()
        self._client: Optional[BinanceClient] = None
        self._connect()

    def _connect(self) -> None:
        if BinanceClient is None:
            logger.error("python-binance not installed — orders disabled")
            return
        testnet = settings.binance_testnet
        self._client = BinanceClient(
            api_key=settings.binance_api_key,
            api_secret=settings.binance_api_secret,
            testnet=testnet,
        )
        logger.info(
            "Binance client connected (testnet={})", testnet
        )

    @property
    def client(self) -> BinanceClient:
        if self._client is None:
            raise RuntimeError("Binance client not initialised")
        return self._client

    # ------------------------------------------------------------------
    # Balance
    # ------------------------------------------------------------------

    def get_balance(self, asset: str = "USDT") -> float:
        """Return free balance for a given asset."""
        info = self.client.get_asset_balance(asset=asset)
        return float(info["free"])

    def get_all_balances(self) -> Dict[str, float]:
        """Return all non-zero balances."""
        acc = self.client.get_account()
        return {b["asset"]: float(b["free"]) for b in acc["balances"] if float(b["free"]) > 0}

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        strategy: str = "",
        signal_strength: float = 0.0,
    ) -> Optional[Dict[str, Any]]:
        """Place a MARKET order and record it in the database.

        Returns the Binance order response dict, or None on failure.
        """
        try:
            order = self.client.order_market(
                symbol=symbol.replace("/", ""),
                side=side,
                quantity=quantity,
            )
        except BinanceAPIException as e:
            logger.error("Market order failed: {}", e)
            return None

        self._record_order(order, symbol, side, strategy, signal_strength)
        logger.info("MARKET {} {} {} @ {}", side, quantity, symbol, order.get("price", "market"))
        return order

    def limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        strategy: str = "",
        signal_strength: float = 0.0,
    ) -> Optional[Dict[str, Any]]:
        """Place a LIMIT order and record it."""
        try:
            order = self.client.order_limit(
                symbol=symbol.replace("/", ""),
                side=side,
                quantity=quantity,
                price=str(Decimal(str(price))),
            )
        except BinanceAPIException as e:
            logger.error("Limit order failed: {}", e)
            return None

        self._record_order(order, symbol, side, strategy, signal_strength)
        logger.info("LIMIT {} {} {} @ {}", side, quantity, symbol, price)
        return order

    # ------------------------------------------------------------------
    # Position helpers
    # ------------------------------------------------------------------

    def get_open_orders(self, symbol: Optional[str] = None) -> list:
        bsym = symbol.replace("/", "") if symbol else None
        return self.client.get_open_orders(symbol=bsym)

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        try:
            self.client.cancel_order(
                symbol=symbol.replace("/", ""), orderId=order_id
            )
            return True
        except BinanceAPIException as e:
            logger.error("Cancel order failed: {}", e)
            return False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _record_order(
        self,
        order: Dict[str, Any],
        symbol: str,
        side: str,
        strategy: str,
        signal_strength: float,
    ) -> None:
        fills = order.get("fills", [])
        executed_qty = float(order.get("executedQty", 0))
        if fills:
            avg_price = sum(float(f["price"]) * float(f["qty"]) for f in fills) / executed_qty
            total_fee = sum(float(f.get("commission", 0)) for f in fills)
        else:
            avg_price = float(order.get("price", 0))
            total_fee = float(order.get("commission", 0))

        self.db.insert_trade(
            symbol=symbol,
            side=side,
            quantity=executed_qty,
            price=avg_price,
            amount=executed_qty * avg_price,
            fee=total_fee,
            strategy=strategy,
            signal_strength=signal_strength,
        )

    # ------------------------------------------------------------------
    # Lot size helper
    # ------------------------------------------------------------------

    def get_lot_size_precision(self, symbol: str) -> Dict[str, Any]:
        """Return step size, min/max Qty for a symbol."""
        info = self.client.get_symbol_info(symbol.replace("/", ""))
        for f in info["filters"]:
            if f["filterType"] == "LOT_SIZE":
                return {
                    "minQty": f["minQty"],
                    "maxQty": f["maxQty"],
                    "stepSize": f["stepSize"],
                }
        return {}
