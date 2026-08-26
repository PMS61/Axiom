"""Ethereum fetcher - Etherscan `txlist` plus `txlistinternal` when reachable.

Needs a free API key in ETHERSCAN_API_KEY. If the key is missing or the call
fails, this raises FetchError and the caller falls back to sample data, so the
app still runs.

Etherscan pages with `page` and `offset`; a short page means the end of the
history. The free tier allows about five calls a second, so requests go through
the shared rate limiter.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..config import (
    ETHERSCAN_API,
    ETHERSCAN_API_KEY,
    ETHERSCAN_RATE,
    MAX_TXS_PER_ADDRESS,
)
from ..httpclient import RateLimiter, get_json
from ..models import Tx
from .base import FetchError

WEI_PER_ETH = 10**18

# Rows per request. Etherscan allows more, but a smaller page keeps each call
# quick and makes the rate limiter do the pacing rather than one huge response.
PAGE_SIZE = 100

# One limiter for the whole process - the free tier's budget is per API key.
_LIMITER = RateLimiter(ETHERSCAN_RATE)


def _iso(unix_seconds: str | int | None) -> str:
    if not unix_seconds:
        return ""
    return (
        datetime.fromtimestamp(int(unix_seconds), tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _normalize(raw: dict[str, Any]) -> Tx:
    """Etherscan row -> Tx. Ethereum is simple: one sender, one recipient."""
    sender = (raw.get("from") or "").lower()
    recipient = (raw.get("to") or "").lower()
    value_eth = float(raw.get("value", 0)) / WEI_PER_ETH
    return Tx(
        txid=str(raw.get("hash", "")),
        timestamp=_iso(raw.get("timeStamp")),
        chain="ethereum",
        inputs=[sender] if sender else [],
        outputs=[recipient] if recipient else [],
        amounts={recipient: value_eth} if recipient else {},
    )


class EthereumFetcher:
    name = "etherscan"

    def __init__(self, api_key: str = ETHERSCAN_API_KEY, base_url: str = ETHERSCAN_API) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.source = self.name
        self.truncated = False

    def _page(self, address: str, action: str, page: int) -> list[dict[str, Any]]:
        payload = get_json(
            self.base_url,
            params={
                "module": "account",
                "action": action,
                "address": address,
                "startblock": 0,
                "endblock": 99999999,
                "sort": "asc",
                "page": page,
                "offset": PAGE_SIZE,
                "apikey": self.api_key,
            },
            limiter=_LIMITER,
        )
        result = payload.get("result")
        if not isinstance(result, list):
            # Etherscan reports "NOTOK" (bad key, rate limit) with a string result.
            raise FetchError(f"etherscan {action}: {payload.get('message')} {result}")
        return result

    def _call(self, address: str, action: str) -> list[dict[str, Any]]:
        """Every page of one endpoint's history for this address, up to the cap."""
        rows: list[dict[str, Any]] = []
        page = 1
        while len(rows) < MAX_TXS_PER_ADDRESS:
            result = self._page(address, action, page)
            rows.extend(result)
            if len(result) < PAGE_SIZE:
                return rows              # short page = end of history
            page += 1

        # We stopped because of the cap, not because history ran out.
        self.truncated = True
        return rows[:MAX_TXS_PER_ADDRESS]

    def get_transactions(self, address: str) -> list[Tx]:
        if not self.api_key:
            raise FetchError("ETHERSCAN_API_KEY is not set")

        rows = self._call(address, "txlist")

        # Internal transfers (contract-to-contract) matter for tracing through
        # bridges and mixers, but they are a bonus - never fail the whole fetch.
        try:
            rows += self._call(address, "txlistinternal")
        except Exception:
            pass

        return [_normalize(row) for row in rows]
