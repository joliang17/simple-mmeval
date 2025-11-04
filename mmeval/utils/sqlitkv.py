from __future__ import annotations
import sqlite3, json, time
from contextlib import contextmanager
from typing import Any, Iterable, Tuple, Optional, Callable

JSONType = Any  # dict | list | str | int | float | bool | None

class SQLiteKVStore:
    """
    A multi-process friendly local JSON KV store on SQLite.

    - Values are ALWAYS JSON (validated on write; parsed on read).
    - WAL mode; transactional; BEGIN IMMEDIATE to reduce write collisions.
    - Per-operation fresh connections (safe across processes).
    """

    def __init__(
        self,
        db_path: str,
        busy_timeout_ms: int = 3000,
        synchronous: str = "NORMAL",      # "FULL" for stronger durability
        retries: int = 5,
        base_sleep: float = 0.02,
        json_indent: Optional[int] = None # pretty print if you like
    ) -> None:
        self.db_path = db_path
        self.busy_timeout_ms = busy_timeout_ms
        self.synchronous = synchronous
        self.retries = retries
        self.base_sleep = base_sleep
        self.json_indent = json_indent
        self._init_db()

    # ---------- internals ----------
    @contextmanager
    def _conn(self) -> Iterable[sqlite3.Connection]:
        # New connection each time: safer for multiprocess.
        conn = sqlite3.connect(self.db_path, isolation_level=None)  # autocommit
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(f"PRAGMA synchronous={self.synchronous};")
            conn.execute("PRAGMA foreign_keys=ON;")
            conn.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms};")
            yield conn
        finally:
            conn.close()

    def _with_txn(self, fn: Callable[[sqlite3.Connection], Any]) -> Any:
        # BEGIN IMMEDIATE + exponential backoff when DB is busy/locked.
        attempt = 0
        while True:
            try:
                with self._conn() as conn:
                    conn.execute("BEGIN IMMEDIATE;")
                    out = fn(conn)
                    conn.execute("COMMIT;")
                    return out
            except sqlite3.OperationalError as e:
                msg = str(e).lower()
                if ("locked" in msg or "busy" in msg) and attempt < self.retries:
                    attempt += 1
                    time.sleep(self.base_sleep * (2 ** (attempt - 1)))
                    continue
                raise

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS kv (
                k TEXT PRIMARY KEY,
                v TEXT NOT NULL,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            """)

    def _dumps(self, obj: JSONType) -> str:
        # Validate JSON-compatibility; disallow NaN/Inf which aren't valid JSON.
        try:
            return json.dumps(
                obj,
                ensure_ascii=False,
                allow_nan=False,
                indent=self.json_indent,
                separators=None if self.json_indent is not None else (",", ":")
            )
        except (TypeError, ValueError) as e:
            raise ValueError(f"value is not JSON-serializable: {e}")

    def _loads(self, s: str) -> JSONType:
        return json.loads(s)

    # ---------- JSON-first API ----------
    def put(self, key: str, value: JSONType) -> None:
        """Upsert a JSON value under key."""
        payload = self._dumps(value)
        def _op(conn: sqlite3.Connection):
            conn.execute("""
            INSERT INTO kv(k, v) VALUES (?, ?)
            ON CONFLICT(k) DO UPDATE SET v=excluded.v, updated_at=CURRENT_TIMESTAMP;
            """, (key, payload))
        self._with_txn(_op)

    def get(self, key: str, default: Optional[JSONType] = None) -> Optional[JSONType]:
        """Get JSON value or default if missing."""
        with self._conn() as conn:
            row = conn.execute("SELECT v FROM kv WHERE k=?", (key,)).fetchone()
            if not row:
                return default
            return self._loads(row["v"])

    def delete(self, key: str) -> None:
        def _op(conn: sqlite3.Connection):
            conn.execute("DELETE FROM kv WHERE k=?", (key,))
        self._with_txn(_op)

    def exists(self, key: str) -> bool:
        with self._conn() as conn:
            row = conn.execute("SELECT 1 FROM kv WHERE k=? LIMIT 1", (key,)).fetchone()
            return bool(row)

    def keys(self, prefix: str = "") -> list[str]:
        with self._conn() as conn:
            if prefix:
                rows = conn.execute("SELECT k FROM kv WHERE k LIKE ? ORDER BY k", (prefix + "%",)).fetchall()
            else:
                rows = conn.execute("SELECT k FROM kv ORDER BY k").fetchall()
            return [r["k"] for r in rows]

    def items(self, prefix: str = "") -> list[tuple[str, JSONType]]:
        with self._conn() as conn:
            if prefix:
                rows = conn.execute("SELECT k, v FROM kv WHERE k LIKE ? ORDER BY k", (prefix + "%",)).fetchall()
            else:
                rows = conn.execute("SELECT k, v FROM kv ORDER BY k").fetchall()
            return [(r["k"], self._loads(r["v"])) for r in rows]

    def batch_put(self, items: Iterable[Tuple[str, JSONType]]) -> None:
        """Batch upsert of JSON values."""
        items = list(items)
        if not items:
            return
        rows = [(k, self._dumps(v)) for k, v in items]
        def _op(conn: sqlite3.Connection):
            conn.executemany("""
            INSERT INTO kv(k, v) VALUES(?, ?)
            ON CONFLICT(k) DO UPDATE SET v=excluded.v, updated_at=CURRENT_TIMESTAMP;
            """, rows)
        self._with_txn(_op)

    def update(self, key: str, updater: Callable[[Optional[JSONType]], JSONType]) -> JSONType:
        """
        Atomic read-modify-write with JSON:
          - Passes current value (or None) to updater
          - Validates JSON and writes new value
          - Returns the new JSON value
        """
        def _op(conn: sqlite3.Connection):
            row = conn.execute("SELECT v FROM kv WHERE k=?", (key,)).fetchone()
            current = self._loads(row["v"]) if row else None
            new_val = updater(current)
            payload = self._dumps(new_val)
            conn.execute("""
            INSERT INTO kv(k, v) VALUES (?, ?)
            ON CONFLICT(k) DO UPDATE SET v=excluded.v, updated_at=CURRENT_TIMESTAMP;
            """, (key, payload))
            return new_val
        return self._with_txn(_op)

    # ---------- dump & load ----------
    def dump_dict(self, prefix: str = "") -> dict[str, JSONType]:
        """Return a dict of all key->JSON objects (optionally filtered by prefix)."""
        return dict(self.items(prefix=prefix))

    def load_dict(self, data: dict[str, JSONType]) -> None:
        """Replace or insert keys from a Python dict of JSON values."""
        self.batch_put(data.items())