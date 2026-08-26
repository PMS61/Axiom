"""Case storage.

PostgreSQL is the production target; the POC uses SQLite so it runs with zero
setup. Everything goes through the `CaseStore` interface, so swapping the
backend later means writing one class, not touching the app.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from .config import DB_PATH


@dataclass
class Case:
    """One traced wallet, however it arrived (investigator or complaint feed)."""

    case_id: str
    created_at: str
    address: str
    chain: str
    status: str
    source: str = "manual"           # "manual" | "complaint"
    entity: str | None = None
    confidence: float = 0.0
    hops: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    trace: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CaseStore(Protocol):
    def save(self, case: Case) -> Case: ...
    def get(self, case_id: str) -> Case | None: ...
    def list(self, limit: int = 25) -> list[Case]: ...


class SQLiteCaseStore:
    """File-backed store. No server, no setup, same interface as the real one."""

    def __init__(self, path: Path | str = DB_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._create_table()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _create_table(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cases (
                    case_id    TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    address    TEXT NOT NULL,
                    chain      TEXT,
                    status     TEXT,
                    source     TEXT,
                    entity     TEXT,
                    confidence REAL,
                    hops       INTEGER,
                    metadata   TEXT,
                    trace      TEXT
                )
                """
            )

    def save(self, case: Case) -> Case:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO cases
                    (case_id, created_at, address, chain, status, source,
                     entity, confidence, hops, metadata, trace)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    case.case_id,
                    case.created_at,
                    case.address,
                    case.chain,
                    case.status,
                    case.source,
                    case.entity,
                    case.confidence,
                    case.hops,
                    json.dumps(case.metadata, default=str),
                    json.dumps(case.trace, default=str),
                ),
            )
        return case

    def _row_to_case(self, row: sqlite3.Row) -> Case:
        return Case(
            case_id=row["case_id"],
            created_at=row["created_at"],
            address=row["address"],
            chain=row["chain"],
            status=row["status"],
            source=row["source"],
            entity=row["entity"],
            confidence=row["confidence"] or 0.0,
            hops=row["hops"],
            metadata=json.loads(row["metadata"] or "{}"),
            trace=json.loads(row["trace"] or "{}"),
        )

    def get(self, case_id: str) -> Case | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM cases WHERE case_id = ?", (case_id,)
            ).fetchone()
        return self._row_to_case(row) if row else None

    def list(self, limit: int = 25) -> list[Case]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM cases ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_case(row) for row in rows]


_STORE: CaseStore | None = None


def get_case_store() -> CaseStore:
    """Process-wide store. Point this at PostgreSQL to move off SQLite."""
    global _STORE
    if _STORE is None:
        _STORE = SQLiteCaseStore()
    return _STORE


def new_case_id() -> str:
    return f"CT-{uuid.uuid4().hex[:8].upper()}"


def case_from_trace(result: Any, source: str = "manual", metadata: dict | None = None) -> Case:
    """Turn a TraceResult into a storable Case record."""
    return Case(
        case_id=new_case_id(),
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        address=result.address,
        chain=result.chain,
        status=result.status,
        source=source,
        entity=result.attributed_entity,
        confidence=result.confidence,
        hops=result.hops,
        metadata=metadata or {},
        trace=result.to_dict(),
    )
