"""Fetcher interface shared by every data source.

A fetcher takes an address and returns normalized `Tx` objects. That is the
only contract; whether the data came from Esplora, Etherscan or the bundled
sample file is invisible to the trace engine.
"""

from __future__ import annotations

from typing import Protocol

from ..models import Tx


class FetchError(RuntimeError):
    """Raised when a live data source is unreachable or returns junk."""


class TransactionFetcher(Protocol):
    name: str

    def get_transactions(self, address: str) -> list[Tx]: ...
