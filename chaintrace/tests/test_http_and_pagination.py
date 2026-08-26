"""Pagination, rate limiting and retries - all against a stubbed HTTP layer."""

from __future__ import annotations

import pytest

from app.config import MAX_TXS_PER_ADDRESS
from app.fetchers.bitcoin import PAGE_SIZE as BTC_PAGE, BitcoinFetcher
from app.fetchers.ethereum import PAGE_SIZE as ETH_PAGE, EthereumFetcher
from app.httpclient import HttpError, RateLimiter, get_json


class FakeResponse:
    def __init__(self, payload=None, status_code=200, headers=None):
        self._payload = payload if payload is not None else {}
        self.status_code = status_code
        self.headers = headers or {}

    @property
    def ok(self) -> bool:
        return self.status_code < 400

    def json(self):
        return self._payload


# --------------------------------------------------------------------------- #
# Rate limiter
# --------------------------------------------------------------------------- #
def test_limiter_spaces_calls_without_sleeping_on_the_first_one():
    now, slept = [0.0], []

    def sleeper(seconds):
        slept.append(seconds)
        now[0] += seconds

    limiter = RateLimiter(4, clock=lambda: now[0], sleeper=sleeper)
    for _ in range(4):
        limiter.wait()

    assert slept == [0.25, 0.25, 0.25]      # first call goes straight through


def test_limiter_does_not_delay_calls_that_are_already_spread_out():
    now = [0.0]
    limiter = RateLimiter(4, clock=lambda: now[0], sleeper=lambda s: None)
    assert limiter.wait() == 0.0
    now[0] = 10.0                            # ten seconds later
    assert limiter.wait() == 0.0


# --------------------------------------------------------------------------- #
# Retries
# --------------------------------------------------------------------------- #
def test_throttling_is_retried_then_succeeds(monkeypatch):
    responses = [
        FakeResponse(status_code=429, headers={"Retry-After": "0"}),
        FakeResponse(status_code=503),
        FakeResponse({"ok": True}),
    ]
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(url)
        return responses.pop(0)

    monkeypatch.setattr("app.httpclient.requests.get", fake_get)
    payload = get_json("https://example.test/x", sleeper=lambda s: None)

    assert payload == {"ok": True}
    assert len(calls) == 3


def test_retries_are_not_infinite(monkeypatch):
    monkeypatch.setattr(
        "app.httpclient.requests.get",
        lambda url, params=None, timeout=None: FakeResponse(status_code=429),
    )
    with pytest.raises(HttpError):
        get_json("https://example.test/x", max_retries=2, sleeper=lambda s: None)


def test_client_errors_are_not_retried(monkeypatch):
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(url)
        return FakeResponse(status_code=404)

    monkeypatch.setattr("app.httpclient.requests.get", fake_get)
    with pytest.raises(HttpError):
        get_json("https://example.test/missing", sleeper=lambda s: None)
    assert len(calls) == 1                   # a 404 will still be a 404


# --------------------------------------------------------------------------- #
# Bitcoin pagination
# --------------------------------------------------------------------------- #
def _btc_tx(index: int) -> dict:
    return {
        "txid": f"btc{index:04d}",
        "status": {"block_time": 1_760_000_000 + index},
        "vin": [{"prevout": {"scriptpubkey_address": "1Source"}}],
        "vout": [{"scriptpubkey_address": "1Dest", "value": 1000 + index}],
    }


def test_bitcoin_follows_every_page(monkeypatch):
    """Three full pages then a short one - the old code stopped after page one."""
    requested = []

    def fake_get_json(url, params=None, limiter=None, **kwargs):
        requested.append(url)
        if url.endswith("/txs/mempool"):
            return []
        page = url.rsplit("/chain", 1)[1].strip("/")
        start = 0 if not page else int(page.replace("btc", "")) + 1
        if start >= 3 * BTC_PAGE:
            return [_btc_tx(start)]                       # short page = the end
        return [_btc_tx(start + offset) for offset in range(BTC_PAGE)]

    monkeypatch.setattr("app.fetchers.bitcoin.get_json", fake_get_json)
    txs = BitcoinFetcher().get_transactions("1SomeAddress")

    assert len(txs) == 3 * BTC_PAGE + 1
    assert [t.txid for t in txs][:2] == ["btc0000", "btc0001"]
    assert sum(1 for url in requested if "/txs/chain" in url) == 4
    assert txs[0].chain == "bitcoin"
    assert txs[0].amount_to("1Dest") == 1000


def test_bitcoin_marks_a_truncated_history(monkeypatch):
    def endless(url, params=None, limiter=None, **kwargs):
        if url.endswith("/txs/mempool"):
            return []
        return [_btc_tx(index) for index in range(BTC_PAGE)]

    monkeypatch.setattr("app.fetchers.bitcoin.get_json", endless)
    fetcher = BitcoinFetcher()
    txs = fetcher.get_transactions("1BusyAddress")

    assert fetcher.truncated is True
    assert len(txs) == MAX_TXS_PER_ADDRESS


def test_bitcoin_falls_back_to_the_second_endpoint(monkeypatch):
    seen = []

    def fake_get_json(url, params=None, limiter=None, **kwargs):
        seen.append(url)
        if "blockstream" in url:
            raise RuntimeError("esplora is down")
        if url.endswith("/txs/mempool"):
            return []
        return [_btc_tx(1)]

    monkeypatch.setattr("app.fetchers.bitcoin.get_json", fake_get_json)
    fetcher = BitcoinFetcher()
    assert len(fetcher.get_transactions("1SomeAddress")) == 1
    assert fetcher.source == "mempool"


# --------------------------------------------------------------------------- #
# Ethereum pagination
# --------------------------------------------------------------------------- #
def _eth_row(index: int) -> dict:
    return {
        "hash": f"0xeth{index:04d}",
        "timeStamp": str(1_760_000_000 + index),
        "from": "0xAAA",
        "to": "0xBBB",
        "value": str(10**18),
    }


def test_ethereum_pages_until_a_short_page(monkeypatch):
    pages = []

    def fake_get_json(url, params=None, limiter=None, **kwargs):
        pages.append((params["action"], params["page"]))
        if params["action"] == "txlistinternal":
            return {"result": []}
        if params["page"] == 1:
            return {"result": [_eth_row(i) for i in range(ETH_PAGE)]}
        return {"result": [_eth_row(999)]}               # short page = the end

    monkeypatch.setattr("app.fetchers.ethereum.get_json", fake_get_json)
    txs = EthereumFetcher(api_key="demo").get_transactions("0xAAA")

    assert len(txs) == ETH_PAGE + 1
    assert ("txlist", 2) in pages
    assert txs[0].amount_to("0xbbb") == 1.0              # wei converted to ETH


def test_ethereum_without_a_key_fails_fast(monkeypatch):
    called = []
    monkeypatch.setattr(
        "app.fetchers.ethereum.get_json",
        lambda *a, **k: called.append(1) or {"result": []},
    )
    with pytest.raises(Exception):
        EthereumFetcher(api_key="").get_transactions("0xAAA")
    assert called == []                                   # no pointless request


# --------------------------------------------------------------------------- #
# The whole live path, end to end, against a stubbed API
# --------------------------------------------------------------------------- #
def test_live_trace_end_to_end_with_a_stubbed_esplora(monkeypatch):
    """A two-hop trace served entirely by a fake Esplora, ending at a label.

    This is the path that cannot be exercised against the real API from a
    sandbox with no outbound network, so it is worth pinning down here.
    """
    from app.cache import get_cache
    from app.labels import get_label_store
    from app.trace import STATUS_ATTRIBUTED, trace

    # Base58 has no lowercase "l" (or "I", "O", "0"), and the fetcher only
    # makes a network call for an address that could be real.
    suspect = "1SuspectAddressAAAAAAAAAAAAAAAAAAA"
    middle = "1MidwayAddressBBBBBBBBBBBBBBBBBBBB"
    exchange = "1ExchangeDepositAAA111111111111111111"   # already in the label file
    assert get_label_store().is_exchange(exchange)

    chain = {
        suspect: (middle, 500_000),
        middle: (exchange, 495_000),
    }

    def fake_get_json(url, params=None, limiter=None, **kwargs):
        if url.endswith("/txs/mempool"):
            return []
        address = url.split("/address/")[1].split("/")[0]
        if address not in chain:
            return []                                     # nothing spent onward
        destination, amount = chain[address]
        return [
            {
                "txid": f"tx-{address[:12]}",
                "status": {"block_time": 1_760_000_000},
                "vin": [{"prevout": {"scriptpubkey_address": address}}],
                "vout": [{"scriptpubkey_address": destination, "value": amount}],
            }
        ]

    monkeypatch.setattr("app.fetchers.bitcoin.get_json", fake_get_json)
    get_cache().clear()
    result = trace(suspect, allow_network=True)
    get_cache().clear()

    assert result.status == STATUS_ATTRIBUTED
    assert result.attributed_entity == "SampleExchange Alpha"
    assert result.hops == 2
    assert "esplora" in result.data_sources
    assert result.best_path.edges[0].amount == 500_000
