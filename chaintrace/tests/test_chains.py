from app.chains import detect_chain, is_valid_live_address


def test_detects_bitcoin_formats():
    assert detect_chain("1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa") == "bitcoin"
    assert detect_chain("3J98t1WpEZ73CNmQviecrnyiWrnqRhWNLy") == "bitcoin"
    assert detect_chain("bc1qar0srrr7xfkvy5l643lydnw9re59gtzzwf5mdq") == "bitcoin"


def test_detects_ethereum():
    assert detect_chain("0x742d35Cc6634C0532925a3b844Bc454e4438f44e") == "ethereum"
    # Placeholder demo addresses still resolve, so the sample data traces.
    assert detect_chain("0xHopE1111111111111111111111111111111111111") == "ethereum"


def test_demo_wallet_and_unknown():
    assert detect_chain("SUSPECT_WALLET_DEMO_1") == "bitcoin"
    assert detect_chain("") == "unknown"
    assert detect_chain("not an address") == "unknown"


def test_only_real_addresses_are_live_resolvable():
    assert is_valid_live_address("1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa")
    assert not is_valid_live_address("SUSPECT_WALLET_DEMO_1")
