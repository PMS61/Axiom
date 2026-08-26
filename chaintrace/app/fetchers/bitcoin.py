"""Bitcoin fetcher - Blockstream Esplora, with mempool.space as a backup.

Both are free and need no API key. Any failure raises FetchError so the caller
can fall back to the bundled sample data instead of crashing.

Esplora paginates: `/address/:addr/txs/chain` returns 25 confirmed transactions
at a time and you ask for the next page by passing the last txid you saw. The
old code read one page and stopped, which silently truncated busy addresses -
and a trace built on a truncated history is wrong, not merely incomplete.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..config import ESPLORA_API, ESPLORA_RATE, MAX_TXS_PER_ADDRESS, MEMPOOL_API
from ..httpclient import RateLimiter, get_json
from ..models import Tx
from .base import FetchError

# Esplora's page size for confirmed history. A short page means the last page.
PAGE_SIZE = 25

# One limiter per host, shared by every trace in the process.
_LIMITERS: dict[str, RateLimiter] = {}


def _limiter_for(base_url: str) -> RateLimiter:
    return _LIMITERS.setdefault(base_url, RateLimiter(ESPLORA_RATE))


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
        self.truncated = False

    def _fetch_all(self, base_url: str, address: str) -> list[dict[str, Any]]:
        """Walk every page of this address's history, up to the cap."""
        root = base_url.rstrip("/")
        limiter = _limiter_for(root)
        collected: list[dict[str, Any]] = []

        # Unconfirmed transactions first - they are the freshest lead. Not every
        # Esplora mirror serves this route, so a failure here is not fatal.
        try:
            mempool_txs = get_json(f"{root}/address/{address}/txs/mempool", limiter=limiter)
            if isinstance(mempool_txs, list):
                collected.extend(mempool_txs)
        except Exception:
            pass

        last_seen: str | None = None
        while len(collected) < MAX_TXS_PER_ADDRESS:
            url = f"{root}/address/{address}/txs/chain"
            if last_seen:
                url = f"{url}/{last_seen}"

            page = get_json(url, limiter=limiter)
            if not isinstance(page, list):
                raise FetchError(f"{root} returned an unexpected payload for {address}")
            collected.extend(page)

            if len(page) < PAGE_SIZE:
                return collected            # short page = end of history
            last_seen = page[-1].get("txid")
            if not last_seen:
                return collected

        # We stopped because of the cap, not because history ran out.
        self.truncated = True
        return collected[:MAX_TXS_PER_ADDRESS]

    def get_transactions(self, address: str) -> list[Tx]:
        """Full transaction history for `address` (both directions; caller filters)."""
        last_error: Exception | None = None
        for source, base in self.endpoints:
            try:
                raw = self._fetch_all(base, address)
                self.source = source
                return [_normalize(item) for item in raw]
            except Exception as exc:  # network down, 404, rate limit, bad JSON...
                last_error = exc
        raise FetchError(f"no Bitcoin API reachable for {address}: {last_error}")
