"""Normalized data types shared by every module.

The fetchers turn Bitcoin/Ethereum/sample data into `Tx` objects, and everything
downstream (graph, trace, scoring, UI) only ever sees `Tx` - so adding a new
chain later means writing one fetcher, not touching the engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Tx:
    """One value transfer, normalized across chains.

    inputs/outputs are address lists; `amounts` maps an output address to the
    amount it received (satoshis for Bitcoin, ETH for Ethereum).
    """

    txid: str
    timestamp: str          # ISO-8601 UTC string, e.g. "2026-08-10T14:22:00Z"
    chain: str              # "bitcoin" | "ethereum"
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    amounts: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Tx":
        """Build a Tx from the bundled sample-data JSON shape."""
        return cls(
            txid=str(raw.get("txid", "")),
            timestamp=str(raw.get("timestamp", "")),
            chain=str(raw.get("chain", "unknown")),
            inputs=list(raw.get("inputs", [])),
            outputs=list(raw.get("outputs", [])),
            amounts={k: float(v) for k, v in (raw.get("amounts") or {}).items()},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def amount_to(self, address: str) -> float:
        """How much this transaction sent to `address` (0 if it sent nothing)."""
        return float(self.amounts.get(address, 0.0))

    @property
    def total_out(self) -> float:
        return float(sum(self.amounts.values()))


@dataclass
class Label:
    """An entry from the labeled-address database."""

    address: str
    entity: str
    type: str  # "exchange" | "mixer" | "bridge"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FetchResult:
    """Transactions plus where they came from - the UI shows this to the user."""

    address: str
    chain: str
    transactions: list[Tx] = field(default_factory=list)
    source: str = "none"   # "sample" | "esplora" | "mempool" | "etherscan" | "none"
    note: str = ""
    truncated: bool = False   # history was longer than the per-address cap

    def __len__(self) -> int:
        return len(self.transactions)
