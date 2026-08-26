from app.fetchers import fetch, get_transactions, get_sample_fetcher


def test_sample_data_is_returned_offline():
    result = fetch("SUSPECT_WALLET_DEMO_1", allow_network=False)
    assert result.source == "sample"
    assert len(result.transactions) == 1
    tx = result.transactions[0]
    assert tx.chain == "bitcoin"
    assert "1HopA111111111111111111111111111111" in tx.outputs
    assert tx.amount_to("1HopA111111111111111111111111111111") == 42000


def test_clean_interface_returns_tx_list():
    txs = get_transactions("1HopC333333333333333333333333333333")
    assert [t.txid for t in txs] == ["tx0004"]


def test_unknown_address_degrades_to_empty_not_crash():
    result = fetch("SOME_ADDRESS_WE_HAVE_NEVER_SEEN", allow_network=False)
    assert result.transactions == []
    assert result.source == "none"


def test_demo_wallets_are_discoverable():
    wallets = get_sample_fetcher().demo_wallets()
    assert "SUSPECT_WALLET_DEMO_1" in wallets
    assert len(wallets) >= 4
