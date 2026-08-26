"""Chain detection from an address string.

Deliberately forgiving: the POC's bundled sample data uses readable placeholder
addresses ("SUSPECT_WALLET_DEMO_1", "0xHopE111...") that would fail a strict
checksum test, and we still want those to trace.
"""

from __future__ import annotations

import re

BITCOIN = "bitcoin"
ETHEREUM = "ethereum"
UNKNOWN = "unknown"

# Real-world formats we accept for Bitcoin.
_BTC_P2PKH = re.compile(r"^[13][a-km-zA-HJ-NP-Z1-9]{25,60}$")   # legacy 1.. / P2SH 3..
_BTC_BECH32 = re.compile(r"^(bc1|tb1)[a-z0-9]{6,80}$", re.I)  # loose: placeholders too
_ETH = re.compile(r"^0x[0-9a-fA-F]{40}$")


def detect_chain(address: str) -> str:
    """Return "bitcoin", "ethereum" or "unknown" for an address string."""
    if not address:
        return UNKNOWN

    addr = address.strip()

    # Ethereum-style: anything starting with 0x. Strict hex form is the real
    # thing; the loose branch keeps the demo's placeholder 0x... addresses working.
    if _ETH.match(addr) or addr.lower().startswith("0x"):
        return ETHEREUM

    if _BTC_P2PKH.match(addr) or _BTC_BECH32.match(addr):
        return BITCOIN

    # Demo wallets from the bundled sample file.
    if addr.upper().startswith("SUSPECT_WALLET"):
        return BITCOIN

    return UNKNOWN


def is_valid_live_address(address: str) -> bool:
    """True only for addresses a real blockchain API could actually resolve.

    Used to decide whether it is worth making a network call at all.
    """
    addr = (address or "").strip()
    return bool(_ETH.match(addr) or _BTC_P2PKH.match(addr) or _BTC_BECH32.match(addr))
