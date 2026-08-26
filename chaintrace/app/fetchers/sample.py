"""Offline fetcher backed by data/sample_transactions.json.

This is what makes the demo work with no network and no API keys. The JSON is
an adjacency map: address -> transactions in which that address is an input
(i.e. money leaving it), which is exactly what a forward trace needs.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from ..config import SAMPLE_TRANSACTIONS_FILE
from ..models import Tx


class SampleFetcher:
    name = "sample"

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path or SAMPLE_TRANSACTIONS_FILE)
        self._by_address: dict[str, list[Tx]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        raw = json.loads(self.path.read_text())
        for address, txs in raw.items():
            if address.startswith("_") or not isinstance(txs, list):
                continue  # skip the "_comment" key
            self._by_address[address.lower()] = [Tx.from_dict(t) for t in txs]

    def get_transactions(self, address: str) -> list[Tx]:
        return list(self._by_address.get((address or "").strip().lower(), []))

    def knows(self, address: str) -> bool:
        """True if the sample file has any outgoing transaction for this address."""
        return (address or "").strip().lower() in self._by_address

    def demo_wallets(self) -> list[str]:
        """Addresses meant as demo starting points, for the UI's quick buttons."""
        wallets = [
            tx.inputs[0]
            for txs in self._by_address.values()
            for tx in txs
            if tx.inputs and ("suspect" in tx.inputs[0].lower())
        ]
        return sorted(set(wallets))


@lru_cache(maxsize=1)
def get_sample_fetcher() -> SampleFetcher:
    return SampleFetcher()
