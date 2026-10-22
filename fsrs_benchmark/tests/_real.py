"""Shared, cached real-data fixtures for the model tests.

Why a shared module: loading the two real users and fitting the py-fsrs Optimizer on
them takes about 11 s. Several test files need the same fit, so it is computed once
per session here and reused. Tests must not mutate the returned event table.
"""
import functools
from pathlib import Path

import pandas as pd
import pytest

from fsrs_benchmark import data, split

REAL_ROOT = Path("/home/amal/fyp/Axiom/data/anki10k/revlogs")
REAL_USERS = (6370, 8506)


@functools.lru_cache(maxsize=None)
def _real_events_cached():
    paths = [REAL_ROOT / f"user_id={u}" / "data.parquet" for u in REAL_USERS]
    if not all(p.exists() for p in paths):
        pytest.skip("real revlogs for users 6370 and 8506 are not downloaded")
    return split.make_split(pd.concat([data.load_user(p) for p in paths]), max_cards=300, seed=0)


def real_events():
    """A fresh copy of the cached real event table, so callers may mutate it safely."""
    return _real_events_cached().copy()


@functools.lru_cache(maxsize=None)
def fitted_fsrs(optimize):
    from fsrs_benchmark.models.fsrs_model import FSRSModel
    return FSRSModel(optimize=optimize).fit(real_events())


@functools.lru_cache(maxsize=None)
def fitted_hlr(hlwt):
    from fsrs_benchmark.models.hlr import HLRModel
    return HLRModel(hlwt=hlwt).fit(real_events())
