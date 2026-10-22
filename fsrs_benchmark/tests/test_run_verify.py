"""Verifier checks for the Task 3.1 runner and the parallel FSRS optimiser.

These tests run on the cached real revlogs. They must not be edited to pass a buggy
run; a failing test here is a finding for the builder.
"""
import concurrent.futures
import multiprocessing
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from fsrs_benchmark import data, run as run_mod, split
from fsrs_benchmark.models.fsrs_model import FSRSModel, _worker_init
from fsrs_benchmark.tests._real import REAL_ROOT

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _paths(users):
    paths = [REAL_ROOT / f"user_id={u}" / "data.parquet" for u in users]
    if not all(p.exists() for p in paths):
        pytest.skip(f"real revlogs for users {users} are not downloaded")
    return paths


def _events(users, max_cards=300, seed=0):
    df = pd.concat([data.load_user(p) for p in _paths(users)], ignore_index=True)
    return split.make_split(df, max_cards=max_cards, seed=seed)


def _argv(out_dir, run_name, users_paths_n, *extra):
    return ["--n_users", str(users_paths_n), "--max_cards", "300", "--seed", "0",
            "--iters", "20", "--burn", "10", "--run_name", run_name,
            "--out_dir", str(out_dir), *extra]


# ---- Fit leakage: scrambling TEST labels must not change any model's fit ----

@pytest.mark.parametrize("key", run_mod.ALL_MODELS)
def test_fit_does_not_see_test_labels(key):
    """Fit on the real table, and on a copy whose TEST y are flipped. Predicting on the
    real table must give identical numbers, so no model's parameters depend on test labels."""
    events = _events((1, 6370))
    flipped = events.copy()
    test = ~flipped["is_train"].astype(bool)
    flipped.loc[test, "y"] = 1 - flipped.loc[test, "y"]

    class Args:  # the iters/burn/seed/workers that make_model reads
        iters, burn, seed, workers = 20, 10, 0, 1

    p_true = run_mod.make_model(key, Args()).fit(events).predict(events)
    p_flip = run_mod.make_model(key, Args()).fit(flipped).predict(events)
    pd.testing.assert_series_equal(p_true.sort_index(), p_flip.sort_index(), check_names=False)


# ---- Parallel FSRS: identical to serial, and the pool really runs ----

def test_parallel_optimiser_matches_serial_and_runs_in_pool():
    events = _events((1, 2698, 5111), max_cards=300)
    serial = FSRSModel(optimize=True, n_workers=1).fit(events)
    parallel = FSRSModel(optimize=True, n_workers=3).fit(events)
    optimised = [u for u in serial.params_ if u not in serial.fallback_]
    assert len(optimised) >= 2, "pool was not exercised: too few users were optimised"
    assert serial.params_ == parallel.params_
    assert serial.fallback_ == parallel.fallback_
    pd.testing.assert_series_equal(serial.predict(events), parallel.predict(events))


def _torch_threads(_):
    import torch
    return torch.get_num_threads()


def test_worker_init_limits_torch_to_one_thread():
    """Spawn workers must not each open a full torch thread pool (oversubscription)."""
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=2, mp_context=context,
                                                initializer=_worker_init) as pool:
        counts = list(pool.map(_torch_threads, range(2)))
    assert counts == [1, 1]


# ---- Resume: must skip finished models and must not reuse predictions from another config ----

def test_resume_reproduces_identical_predictions(tmp_path):
    paths = _paths((6370, 8506))
    with mock.patch.object(data, "download_users", lambda n, seed, out_dir: paths):
        run_mod.run(run_mod.parse_args(_argv(tmp_path, "r", 2, "--models", "hlr,ours")))
        first = pd.read_parquet(tmp_path / "r" / "predictions.parquet")
        run_mod.run(run_mod.parse_args(_argv(tmp_path, "r", 2, "--models", "hlr,ours", "--resume")))
        second = pd.read_parquet(tmp_path / "r" / "predictions.parquet")
    pd.testing.assert_frame_equal(first, second)


def test_resume_refuses_predictions_from_a_different_config(tmp_path):
    """Changing --iters between runs and resuming must not silently reuse the old `ours`
    predictions. Expected: a clear error (or a refit). Currently the stale file is reused."""
    paths = _paths((6370, 8506))
    with mock.patch.object(data, "download_users", lambda n, seed, out_dir: paths):
        run_mod.run(run_mod.parse_args(_argv(tmp_path, "c", 2, "--models", "ours")))
        changed = _argv(tmp_path, "c", 2, "--models", "ours", "--resume")
        changed[changed.index("20")] = "30"  # --iters 20 -> 30
        with pytest.raises((AssertionError, ValueError), match="config|args|iters|differ"):
            run_mod.run(run_mod.parse_args(changed))


def test_resume_with_different_max_cards_gives_a_clear_message(tmp_path):
    """A stale pred file from another --max_cards changes the event set. The error should say
    the config changed, not just that the event set differs."""
    paths = _paths((6370, 8506))
    with mock.patch.object(data, "download_users", lambda n, seed, out_dir: paths):
        run_mod.run(run_mod.parse_args(_argv(tmp_path, "m", 2, "--models", "hlr")))
        argv = _argv(tmp_path, "m", 2, "--models", "hlr", "--resume")
        argv[argv.index("300")] = "100"  # --max_cards 300 -> 100
        with pytest.raises(AssertionError) as err:
            run_mod.run(run_mod.parse_args(argv))
    assert "config" in str(err.value) or "max_cards" in str(err.value)


# ---- Messages and validation ----

def test_missing_hf_token_gives_clear_message(tmp_path):
    with mock.patch.object(data, "get_token", lambda: None):
        with pytest.raises(RuntimeError, match="Hugging Face token"):
            data.download_users(2, 0, tmp_path)


def test_unknown_and_invalid_flags_are_rejected():
    for bad in (["--models", "hlr,ours_typo"], ["--models", ""], ["--workers", "0"],
                ["--burn", "20", "--iters", "20"], ["--max_cards", "0"]):
        with pytest.raises(SystemExit):
            run_mod.parse_args(["--run_name", "x", *bad])


def test_make_model_rejects_unknown_key():
    class Args:
        iters, burn, seed, workers = 10, 5, 0, 1
    with pytest.raises(KeyError):
        run_mod.make_model("nope", Args())
