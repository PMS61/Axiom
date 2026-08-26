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

SAMPLE_TRANSACTIONS_FILE = Path(
    os.environ.get("CHAINTRACE_SAMPLE_DATA", DATA_DIR / "sample_transactions.json")
)
LABELED_ADDRESSES_FILE = DATA_DIR / "labeled_addresses.json"
TRAINING_SAMPLE_FILE = DATA_DIR / "training_sample.csv"

# Free, no-key Bitcoin APIs. Esplora first, mempool.space as a backup.
ESPLORA_API = os.environ.get("ESPLORA_API", "https://blockstream.info/api")
MEMPOOL_API = os.environ.get("MEMPOOL_API", "https://mempool.space/api")
ETHERSCAN_API = os.environ.get("ETHERSCAN_API", "https://api.etherscan.io/api")
ETHERSCAN_API_KEY = os.environ.get("ETHERSCAN_API_KEY", "").strip()

# Seconds to wait on any live API call before giving up and using sample data.
HTTP_TIMEOUT = float(os.environ.get("CHAINTRACE_HTTP_TIMEOUT", "6"))
HTTP_MAX_RETRIES = int(os.environ.get("CHAINTRACE_HTTP_RETRIES", "3"))

# Requests per second per API. Esplora publishes no official limit, so this is
# just politeness; Etherscan's free tier allows 5/sec.
ESPLORA_RATE = float(os.environ.get("CHAINTRACE_ESPLORA_RATE", "4"))
ETHERSCAN_RATE = float(os.environ.get("CHAINTRACE_ETHERSCAN_RATE", "4"))

# How many transactions to pull for one address before stopping. A trace that
# stops early is reported as truncated rather than silently returning a partial
# (and therefore wrong) picture.
MAX_TXS_PER_ADDRESS = int(os.environ.get("CHAINTRACE_MAX_TXS", "200"))

# Trace defaults
DEFAULT_MAX_DEPTH = 5
DEFAULT_CACHE_TTL = 300  # seconds

# Safety rails. Real addresses fan out hard - a five-hop walk from a busy
# wallet reaches thousands of nodes and yields hundreds of paths, which is slow
# to fetch and impossible to read. These bound both.
MAX_NODES = int(os.environ.get("CHAINTRACE_MAX_NODES", "400"))
MAX_PATHS = int(os.environ.get("CHAINTRACE_MAX_PATHS", "50"))

# SQLite by default so the POC needs no database server. Swap the URL/driver in
# app/storage.py to move to PostgreSQL without touching the rest of the code.
DB_PATH = Path(os.environ.get("CHAINTRACE_DB", PROJECT_ROOT / "chaintrace.db"))


def offline_mode() -> bool:
    """True when the app must not make any network calls."""
    return os.environ.get("CHAINTRACE_OFFLINE", "").strip().lower() in {"1", "true", "yes"}
