"""Database module — SQLite models and connection management.

Database Schema
===============

trades
------
Stores every executed trade.

| Column          | Type      | Description                                  |
|-----------------|-----------|----------------------------------------------|
| id              | INTEGER   | Primary key, auto-increment                  |
| symbol          | TEXT      | Trading pair (e.g. BTC/USDT)                 |
| side            | TEXT      | BUY or SELL                                  |
| quantity        | REAL      | Asset quantity                               |
| price           | REAL      | Execution price                              |
| amount          | REAL      | Total amount (quantity * price)               |
| fee             | REAL      | Trading fee                                  |
| pnl             | REAL      | Realised profit/loss (NULL if open)           |
| status          | TEXT      | open / closed                                |
| strategy        | TEXT      | Strategy that generated the signal            |
| signal_strength | REAL      | Signal confidence 0–1                        |
| timestamp       | TEXT      | ISO-8601 open time                           |
| closed_at       | TEXT      | ISO-8601 close time (NULL if open)           |

signals
-------
Logs every signal generated, regardless of execution.

| Column              | Type      | Description                          |
|---------------------|-----------|--------------------------------------|
| id                  | INTEGER   | Primary key, auto-increment          |
| symbol              | TEXT      | Trading pair                         |
| signal              | TEXT      | BUY / SELL / HOLD                    |
| strength            | REAL      | Signal confidence 0–1                |
| strategy            | TEXT      | Strategy name                        |
| indicators_snapshot | TEXT      | JSON blob of TA values at generation |
| created_at          | TEXT      | ISO-8601 timestamp                   |

performance
-----------
Daily aggregated performance metrics per symbol.

| Column       | Type      | Description                    |
|--------------|-----------|--------------------------------|
| id           | INTEGER   | Primary key, auto-increment    |
| date         | TEXT      | Date (YYYY-MM-DD)              |
| symbol       | TEXT      | Trading pair                   |
| total_trades | INTEGER   | Number of trades that day      |
| win_rate     | REAL      | Winning trades / total trades  |
| profit_loss  | REAL      | Net PnL for the day            |
| max_drawdown | REAL      | Maximum drawdown observed      |
| sharpe_ratio | REAL      | Daily Sharpe ratio             |
| updated_at   | TEXT      | ISO-8601 last update           |

config
------
Key-value store for runtime configuration overrides.

| Column     | Type      | Description            |
|------------|-----------|------------------------|
| id         | INTEGER   | Primary key            |
| key        | TEXT      | Configuration key      |
| value      | TEXT      | Configuration value    |
| updated_at | TEXT      | ISO-8601 timestamp     |
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from crypto_trading.config import settings

try:
    import sqlite3
except ImportError:
    sqlite3 = None  # type: ignore


class Database:
    """Thin wrapper around SQLite for the trading bot."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or settings.database_url.replace("sqlite:///", "")
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    @property
    def conn(self) -> "sqlite3.Connection":
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        cur = self.conn.cursor()
        cur.executescript(SCHEMA_SQL)
        self.conn.commit()
        logger.info("Database initialised at {}", self.db_path)

    # ------------------------------------------------------------------
    # Trades
    # ------------------------------------------------------------------

    def insert_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        amount: float,
        fee: float = 0.0,
        strategy: str = "",
        signal_strength: float = 0.0,
    ) -> int:
        cur = self.conn.cursor()
        now = datetime.now(timezone.utc).isoformat()
        cur.execute(
            """INSERT INTO trades
               (symbol,side,quantity,price,amount,fee,status,strategy,signal_strength,timestamp)
               VALUES (?,?,?,?,?,?,'open',?,?,?)""",
            (symbol, side, quantity, price, amount, fee, strategy, signal_strength, now),
        )
        self.conn.commit()
        return cur.lastrowid

    def close_trade(self, trade_id: int, close_price: float, fee: float = 0.0) -> None:
        cur = self.conn.cursor()
        row = cur.execute("SELECT * FROM trades WHERE id=?", (trade_id,)).fetchone()
        if not row:
            logger.warning("Trade {} not found", trade_id)
            return
        entry_price = row["price"]
        side = row["side"]
        quantity = row["quantity"]
        if side == "BUY":
            pnl = (close_price - entry_price) * quantity - row["fee"] - fee
        else:
            pnl = (entry_price - close_price) * quantity - row["fee"] - fee
        now = datetime.now(timezone.utc).isoformat()
        cur.execute(
            """UPDATE trades SET status='closed', pnl=?, fee=fee+?, closed_at=?
               WHERE id=?""",
            (pnl, fee, now, trade_id),
        )
        self.conn.commit()

    def get_open_trades(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        cur = self.conn.cursor()
        if symbol:
            rows = cur.execute(
                "SELECT * FROM trades WHERE status='open' AND symbol=?", (symbol,)
            ).fetchall()
        else:
            rows = cur.execute(
                "SELECT * FROM trades WHERE status='open'"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_trade_history(
        self, symbol: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        cur = self.conn.cursor()
        if symbol:
            rows = cur.execute(
                "SELECT * FROM trades WHERE symbol=? ORDER BY timestamp DESC LIMIT ?",
                (symbol, limit),
            ).fetchall()
        else:
            rows = cur.execute(
                "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    def insert_signal(
        self,
        symbol: str,
        signal: str,
        strength: float,
        strategy: str,
        indicators: Optional[Dict[str, Any]] = None,
    ) -> int:
        cur = self.conn.cursor()
        now = datetime.now(timezone.utc).isoformat()
        indicators_json = json.dumps(indicators or {})
        cur.execute(
            """INSERT INTO signals
               (symbol,signal,strength,strategy,indicators_snapshot,created_at)
               VALUES (?,?,?,?,?,?)""",
            (symbol, signal, strength, strategy, indicators_json, now),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_recent_signals(
        self, symbol: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        cur = self.conn.cursor()
        if symbol:
            rows = cur.execute(
                "SELECT * FROM signals WHERE symbol=? ORDER BY created_at DESC LIMIT ?",
                (symbol, limit),
            ).fetchall()
        else:
            rows = cur.execute(
                "SELECT * FROM signals ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Performance
    # ------------------------------------------------------------------

    def upsert_performance(
        self,
        date: str,
        symbol: str,
        total_trades: int,
        win_rate: float,
        profit_loss: float,
        max_drawdown: float,
        sharpe_ratio: float,
    ) -> None:
        cur = self.conn.cursor()
        now = datetime.now(timezone.utc).isoformat()
        cur.execute(
            """INSERT INTO performance
               (date,symbol,total_trades,win_rate,profit_loss,max_drawdown,sharpe_ratio,updated_at)
               VALUES (?,?,?,?,?,?,?,?)
               ON CONFLICT(date,symbol) DO UPDATE SET
               total_trades=excluded.total_trades,
               win_rate=excluded.win_rate,
               profit_loss=excluded.profit_loss,
               max_drawdown=excluded.max_drawdown,
               sharpe_ratio=excluded.sharpe_ratio,
               updated_at=excluded.updated_at""",
            (date, symbol, total_trades, win_rate, profit_loss, max_drawdown, sharpe_ratio, now),
        )
        self.conn.commit()

    def get_performance(
        self, symbol: Optional[str] = None, limit: int = 30
    ) -> List[Dict[str, Any]]:
        cur = self.conn.cursor()
        if symbol:
            rows = cur.execute(
                "SELECT * FROM performance WHERE symbol=? ORDER BY date DESC LIMIT ?",
                (symbol, limit),
            ).fetchall()
        else:
            rows = cur.execute(
                "SELECT * FROM performance ORDER BY date DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def get_config(self, key: str, default: Optional[str] = None) -> Optional[str]:
        cur = self.conn.cursor()
        row = cur.execute(
            "SELECT value FROM config WHERE key=?", (key,)
        ).fetchone()
        return row["value"] if row else default

    def set_config(self, key: str, value: str) -> None:
        cur = self.conn.cursor()
        now = datetime.now(timezone.utc).isoformat()
        cur.execute(
            """INSERT INTO config (key,value,updated_at) VALUES (?,?,?)
               ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at""",
            (key, value, now),
        )
        self.conn.commit()

    def get_all_config(self) -> Dict[str, str]:
        cur = self.conn.cursor()
        rows = cur.execute("SELECT key,value FROM config").fetchall()
        return {r["key"]: r["value"] for r in rows}


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT    NOT NULL,
    side            TEXT    NOT NULL CHECK(side IN ('BUY','SELL')),
    quantity        REAL    NOT NULL,
    price           REAL    NOT NULL,
    amount          REAL    NOT NULL,
    fee             REAL    DEFAULT 0.0,
    pnl             REAL,
    status          TEXT    NOT NULL DEFAULT 'open' CHECK(status IN ('open','closed')),
    strategy        TEXT    DEFAULT '',
    signal_strength REAL    DEFAULT 0.0,
    timestamp       TEXT    NOT NULL,
    closed_at       TEXT
);

CREATE TABLE IF NOT EXISTS signals (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol              TEXT    NOT NULL,
    signal              TEXT    NOT NULL CHECK(signal IN ('BUY','SELL','HOLD')),
    strength            REAL    NOT NULL DEFAULT 0.0,
    strategy            TEXT    DEFAULT '',
    indicators_snapshot TEXT    DEFAULT '{}',
    created_at          TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS performance (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    date        TEXT    NOT NULL,
    symbol      TEXT    NOT NULL,
    total_trades INTEGER DEFAULT 0,
    win_rate    REAL    DEFAULT 0.0,
    profit_loss REAL    DEFAULT 0.0,
    max_drawdown REAL   DEFAULT 0.0,
    sharpe_ratio REAL   DEFAULT 0.0,
    updated_at  TEXT    NOT NULL,
    UNIQUE(date, symbol)
);

CREATE TABLE IF NOT EXISTS config (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    key         TEXT    NOT NULL UNIQUE,
    value       TEXT    NOT NULL DEFAULT '',
    updated_at  TEXT    NOT NULL
);
"""
