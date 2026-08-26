"""Bitcoin fetcher - Blockstream Esplora, with mempool.space as a backup.

Both are free and need no API key. Any failure raises FetchError so the caller
can fall back to the bundled sample data instead of crashing.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import requests

from ..config import ESPLORA_API, HTTP_TIMEOUT, MEMPOOL_API
from ..models import Tx
from .base import FetchError


def _iso(block_time: int | None) -> str:
    """Unix seconds -> ISO-8601 UTC. Unconfirmed txs have no time yet."""
    if not block_time:
        return ""
    return datetime.fromtimestamp(int(block_time), tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize(raw: dict[str, Any]) -> Tx:
    """Esplora/mempool.space tx JSON -> our Tx dataclass.

    Bitcoin has no "from"/"to": inputs are the addresses of the outputs being
    spent, outputs are the addresses being paid. Amounts are in satoshis.
    """
    inputs, outputs, amounts = [], [], {}

    for vin in raw.get("vin", []):
        prevout = vin.get("prevout") or {}
        addr = prevout.get("scriptpubkey_address")
        if addr and addr not in inputs:
            inputs.append(addr)

    for vout in raw.get("vout", []):
        addr = vout.get("scriptpubkey_address")
        if not addr:
            continue  # OP_RETURN and other non-address outputs
        if addr not in outputs:
            outputs.append(addr)
        amounts[addr] = amounts.get(addr, 0.0) + float(vout.get("value", 0))

    return Tx(
        txid=str(raw.get("txid", "")),
        timestamp=_iso((raw.get("status") or {}).get("block_time")),
        chain="bitcoin",
        inputs=inputs,
        outputs=outputs,
        amounts=amounts,
    )


class BitcoinFetcher:
    name = "esplora"

    def __init__(self, esplora: str = ESPLORA_API, mempool: str = MEMPOOL_API) -> None:
        self.endpoints = [("esplora", esplora), ("mempool", mempool)]
        self.source = self.name

    def get_transactions(self, address: str) -> list[Tx]:
        """Recent transactions for `address` (both directions; caller filters)."""
        last_error: Exception | None = None
        for source, base in self.endpoints:
            try:
                response = requests.get(
                    f"{base.rstrip('/')}/address/{address}/txs", timeout=HTTP_TIMEOUT
                )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, list):
                    raise FetchError(f"{source} returned an unexpected payload")
                self.source = source
                return [_normalize(item) for item in payload]
            except Exception as exc:  # network down, 404, rate limit, bad JSON...
                last_error = exc
        raise FetchError(f"no Bitcoin API reachable for {address}: {last_error}")
