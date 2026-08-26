"""Fetcher dispatcher: one function the rest of the app calls.

    get_transactions(address) -> list[Tx]

Rules, in order:
  1. Cached result? Use it.
  2. Address is in the bundled sample data? Use that (keeps the demo instant
     and deterministic).
  3. Offline mode, or the address is not a real on-chain address? Sample data.
  4. Otherwise call the live API for the detected chain; on ANY failure fall
     back to sample data rather than crashing.
"""

from __future__ import annotations

from ..cache import get_cache
from ..chains import BITCOIN, ETHEREUM, detect_chain, is_valid_live_address
from ..config import DEFAULT_CACHE_TTL, offline_mode
from ..models import FetchResult, Tx
from .base import FetchError, TransactionFetcher
from .bitcoin import BitcoinFetcher
from .ethereum import EthereumFetcher
from .sample import SampleFetcher, get_sample_fetcher

__all__ = [
    "fetch",
    "get_transactions",
    "FetchError",
    "TransactionFetcher",
    "BitcoinFetcher",
    "EthereumFetcher",
    "SampleFetcher",
    "get_sample_fetcher",
]


def _sample_result(address: str, chain: str, note: str) -> FetchResult:
    sample = get_sample_fetcher()
    txs = sample.get_transactions(address)
    return FetchResult(
        address=address,
        chain=chain,
        transactions=txs,
        source="sample" if txs else "none",
        note=note if txs else f"{note} (no sample data for this address either)",
    )


def fetch(address: str, allow_network: bool = True) -> FetchResult:
    """Fetch transactions for one address, with source and fallback notes."""
    address = (address or "").strip()
    chain = detect_chain(address)
    cache = get_cache()

    cache_key = f"txs:{chain}:{address.lower()}:{int(allow_network)}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    sample = get_sample_fetcher()

    if sample.knows(address):
        result = _sample_result(address, chain, "bundled sample data")
    elif not allow_network or offline_mode():
        result = _sample_result(address, chain, "offline mode")
    elif not is_valid_live_address(address):
        result = _sample_result(address, chain, "not a live-resolvable address")
    else:
        fetcher: TransactionFetcher | None = None
        if chain == BITCOIN:
            fetcher = BitcoinFetcher()
        elif chain == ETHEREUM:
            fetcher = EthereumFetcher()

        if fetcher is None:
            result = _sample_result(address, chain, "unrecognised address format")
        else:
            try:
                txs = fetcher.get_transactions(address)
                truncated = bool(getattr(fetcher, "truncated", False))
                result = FetchResult(
                    address=address,
                    chain=chain,
                    transactions=txs,
                    source=getattr(fetcher, "source", fetcher.name),
                    note=(
                        f"history truncated at {len(txs)} transactions for {address}"
                        if truncated
                        else "live API"
                    ),
                    truncated=truncated,
                )
            except Exception as exc:  # unreachable API, missing key, rate limit
                result = _sample_result(address, chain, f"live fetch failed: {exc}")

    cache.set(cache_key, result, ttl=DEFAULT_CACHE_TTL)
    return result


def get_transactions(address: str) -> list[Tx]:
    """The clean interface from the spec: address in, normalized Tx list out."""
    return fetch(address).transactions
