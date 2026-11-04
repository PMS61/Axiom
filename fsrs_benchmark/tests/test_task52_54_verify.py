"""Verifier tests for Phase 5 Tasks 5.2 and 5.4: laglogit continuation, FSRS parallel predict,
resume config (incl. existing Anki sidecars), --n_users 0, ours_gpu wiring, report wording.

Read-only on results/; reports are written into tmp_path.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fsrs_benchmark import run as run_mod
from fsrs_benchmark.models.fsrs_model import FSRSModel
from fsrs_benchmark.models.laglogit import LagLogitModel
from fsrs_benchmark.report import build_report
from fsrs_benchmark.tests._synthetic_events import synthetic_events, write_duolingo_csv

REPO = Path(__file__).resolve().parents[2]
ANKI_RUN = REPO / "results" / "full_u100_c1000"
SMOKE_RUN = REPO / "results" / "smoke_duo"


def _args(*extra):
    return run_mod.parse_args(["--run_name", "x", *extra])


# ---------------------------------------------------------------- laglogit continuation

def test_laglogit_history_is_the_train_prefix_of_each_card():
    """Test-time continuation: the frozen history for a card must be its TRAIN rows, so the
    first TEST review's lag is the last TRAIN outcome and its counts start from the TRAIN counts."""
    events = synthetic_events(seed=5)
    model = LagLogitModel().fit(events)
    train = events[events["is_train"].astype(bool)].sort_values(["user_id", "card_id", "review_idx"])
    checked = 0
    for (u, c), grp in train.groupby(["user_id", "card_id"]):
        key = (model._users[u], model._items[(u, c)])
        h = model.model_.hist[key]
        assert h["n"] == len(grp)
        assert h["right"] == int(grp["y"].sum())
        assert h["last_r"] == int(grp["y"].iloc[-1])
        checked += 1
    assert checked > 0


def test_laglogit_predict_of_scored_rows_is_not_changed_by_train_only_edit_after_fit():
    """The fitted history is frozen at fit(); editing TRAIN outcomes afterwards must not move
    the test predictions (no train data is re-read at predict time)."""
    events = synthetic_events(seed=2)
    model = LagLogitModel().fit(events)
    base = model.predict(events)
    edited = events.copy()
    tr = edited.index[edited["is_train"].astype(bool)]
    edited.loc[tr, "y"] = 1 - edited.loc[tr, "y"]
    assert np.allclose(model.predict(edited).to_numpy(), base.to_numpy(), rtol=0, atol=1e-12)


# ---------------------------------------------------------------- FSRS parallel predict

def test_fsrs_optimize_fallback_parallel_predict_matches_serial():
    """optimize=True with too little data: every user keeps defaults (fallback), so the
    parallel path (fit and predict) must still equal the serial one."""
    events = synthetic_events(seed=4, n_users=3)
    serial = FSRSModel(optimize=True, n_workers=1).fit(events).predict(events)
    par_model = FSRSModel(optimize=True, n_workers=2).fit(events)
    assert len(par_model.fallback_) == 3
    parallel = par_model.predict(events)
    assert serial.index.equals(parallel.index)
    assert np.allclose(serial.to_numpy(), parallel.to_numpy(), rtol=0, atol=1e-12)


# ---------------------------------------------------------------- resume config

@pytest.mark.skipif(not (ANKI_RUN / "run_info.json").exists(), reason="no Anki run in results/")
def test_existing_anki_sidecars_pass_the_new_resume_config():
    info = json.loads((ANKI_RUN / "run_info.json").read_text())
    args = argparse.Namespace(**info["args"])  # the args the Anki run was made with
    events = pd.read_parquet(ANKI_RUN / "events.parquet")
    scored = np.sort(events.loc[events["scored"].astype(bool), "event_id"].to_numpy(np.int64))
    checked = []
    for key in ("fsrs_default", "fsrs_opt", "hlr", "ours", "ours_kappa"):
        pred = ANKI_RUN / f"pred_{key}.parquet"
        if not pred.exists():
            continue
        run_mod.check_resume_config(pred, run_mod.model_config(key, args, scored))
        checked.append(key)
    assert {"ours", "hlr"} <= set(checked)


def test_anki_config_has_no_dataset_or_backend_keys():
    scored = np.array([1, 2], dtype=np.int64)
    for key in ("fsrs_default", "fsrs_opt", "hlr", "ours", "ours_kappa"):
        cfg = run_mod.model_config(key, _args(), scored)
        assert "dataset" not in cfg and "backend" not in cfg and "min_user_rows" not in cfg


def test_resume_config_records_thin_for_gpu_draws():
    """FINDING candidate: --thin changes which draws ours_gpu keeps, so it must be part of the
    resume config; otherwise --resume silently reuses predictions made with another thin."""
    scored = np.array([1, 2], dtype=np.int64)
    a = run_mod.model_config("ours_gpu", _args("--thin", "4"), scored)
    b = run_mod.model_config("ours_gpu", _args("--thin", "2"), scored)
    assert a != b


# ---------------------------------------------------------------- --n_users 0 and loader

def test_n_users_zero_loads_every_eligible_duolingo_user(tmp_path):
    csv = write_duolingo_csv(tmp_path / "traces.csv.gz", n_users=3, reviews=12)
    args = _args("--dataset", "duolingo", "--duolingo_path", str(csv), "--min_user_rows", "10",
                 "--n_users", "0")
    df = run_mod.load_events_source(args)
    assert df["user_id"].nunique() == 3


def test_n_users_sample_is_a_subset_of_eligible_users(tmp_path):
    csv = write_duolingo_csv(tmp_path / "traces.csv.gz", n_users=3, reviews=12)
    args = _args("--dataset", "duolingo", "--duolingo_path", str(csv), "--min_user_rows", "10",
                 "--n_users", "2")
    df = run_mod.load_events_source(args)
    assert df["user_id"].nunique() == 2


# ---------------------------------------------------------------- ours_gpu wiring

def test_make_model_ours_gpu_passes_thin_device_and_float32_store(tmp_path):
    csv = write_duolingo_csv(tmp_path / "traces.csv.gz", n_users=2, reviews=8)
    args = _args("--dataset", "duolingo", "--duolingo_path", str(csv), "--min_user_rows", "5",
                 "--iters", "6", "--burn", "2", "--thin", "2", "--device", "cpu")
    events = synthetic_events(seed=6, n_users=2)
    model = run_mod.make_model("ours_gpu", args).fit(events)
    assert model.thin == 2 and model.device == "cpu"
    assert model.model_.store_dtype == np.float32
    assert model.model_.samples["alpha"].dtype == np.float32
    assert len(model.model_.samples["lam"]) == len(range(2, 6, 2))


def test_run_details_record_device_for_gpu_keys():
    assert run_mod.model_details("ours_gpu", type("M", (), {"model_": type("S", (), {"device": "cpu"})()})()) \
        == {"device": "cpu"}


# ---------------------------------------------------------------- report wording

def test_duolingo_report_names_the_dataset(tmp_path):
    report_dir = build_report(SMOKE_RUN, out_dir=tmp_path / "duo")
    text = (report_dir / "report.md").read_text()
    assert text.splitlines()[0].endswith("(Duolingo dataset)")
    assert "Dataset: Duolingo learning traces" in text


@pytest.mark.skipif(not (ANKI_RUN / "run_info.json").exists(), reason="no Anki run in results/")
def test_anki_report_keeps_its_wording(tmp_path):
    report_dir = build_report(ANKI_RUN, out_dir=tmp_path / "anki")
    text = (report_dir / "report.md").read_text()
    assert "Duolingo" not in text
    assert text.splitlines()[0] == "# Results: full_u100_c1000"
