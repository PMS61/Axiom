"""Central configuration and file paths for the ChainTrace POC.

Everything here has a safe default so the app runs with zero setup.
Environment variables let you turn on the "live" behaviour later:

    ETHERSCAN_API_KEY   Etherscan key. Missing -> Ethereum falls back to sample data.
    CHAINTRACE_OFFLINE  Set to 1 to never touch the network at all.
    CHAINTRACE_DB       Path to the SQLite case database.
"""

from __future__ import annotations

import os
from pathlib import Path

# chaintrace/app/config.py -> chaintrace/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

SAMPLE_TRANSACTIONS_FILE = DATA_DIR / "sample_transactions.json"
LABELED_ADDRESSES_FILE = DATA_DIR / "labeled_addresses.json"
TRAINING_SAMPLE_FILE = DATA_DIR / "training_sample.csv"

# Free, no-key Bitcoin APIs. Esplora first, mempool.space as a backup.
ESPLORA_API = os.environ.get("ESPLORA_API", "https://blockstream.info/api")
MEMPOOL_API = os.environ.get("MEMPOOL_API", "https://mempool.space/api")
ETHERSCAN_API = os.environ.get("ETHERSCAN_API", "https://api.etherscan.io/api")
ETHERSCAN_API_KEY = os.environ.get("ETHERSCAN_API_KEY", "").strip()

# Seconds to wait on any live API call before giving up and using sample data.
HTTP_TIMEOUT = float(os.environ.get("CHAINTRACE_HTTP_TIMEOUT", "6"))

# Trace defaults
DEFAULT_MAX_DEPTH = 5
DEFAULT_CACHE_TTL = 300  # seconds

# SQLite by default so the POC needs no database server. Swap the URL/driver in
# app/storage.py to move to PostgreSQL without touching the rest of the code.
DB_PATH = Path(os.environ.get("CHAINTRACE_DB", PROJECT_ROOT / "chaintrace.db"))


def offline_mode() -> bool:
    """True when the app must not make any network calls."""
    return os.environ.get("CHAINTRACE_OFFLINE", "").strip().lower() in {"1", "true", "yes"}
