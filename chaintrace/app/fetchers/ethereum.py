"""Ethereum fetcher - Etherscan `txlist` plus `txlistinternal` when reachable.

Needs a free API key in ETHERSCAN_API_KEY. If the key is missing or the call
fails, this raises FetchError and the caller falls back to sample data, so the
app still runs.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import requests

from ..config import ETHERSCAN_API, ETHERSCAN_API_KEY, HTTP_TIMEOUT
from ..models import Tx
from .base import FetchError

WEI_PER_ETH = 10**18


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

    def _call(self, address: str, action: str) -> list[dict[str, Any]]:
        params = {
            "module": "account",
            "action": action,
            "address": address,
            "startblock": 0,
            "endblock": 99999999,
            "sort": "asc",
            "page": 1,
            "offset": 100,
            "apikey": self.api_key,
        }
        response = requests.get(self.base_url, params=params, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
        result = payload.get("result")
        if not isinstance(result, list):
            # Etherscan reports "NOTOK" (bad key, rate limit) with a string result.
            raise FetchError(f"etherscan {action}: {payload.get('message')} {result}")
        return result

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
