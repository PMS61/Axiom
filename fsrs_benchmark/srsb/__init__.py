"""Comparison with the public srs-benchmark (open-spaced-repetition), Phase 2.

run_gbdt.py  : pooled GBDT driver that writes srs-benchmark style result jsonl lines.
compare.py   : reads published result jsonl files and our jsonl, reproduces the README means and
               computes paired per-user comparisons on a user subset.

srs-benchmark is imported from external/srs-benchmark (its files are never modified).
"""
import sys
from pathlib import Path

SRSB_DIR = Path(__file__).resolve().parents[2] / "external" / "srs-benchmark"


def ensure_srsb_on_path() -> Path:
    """Put the srs-benchmark checkout first on sys.path so its top-level modules (config, utils, ...) resolve."""
    s = str(SRSB_DIR)
    if s not in sys.path:
        sys.path.insert(0, s)
    return SRSB_DIR
