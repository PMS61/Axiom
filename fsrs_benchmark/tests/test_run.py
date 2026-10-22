"""Tests for the runner CLI. Uses the two cached real users; the download is mocked so no network is needed."""
import json
import re
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from fsrs_benchmark import data
from fsrs_benchmark import run as run_mod
from fsrs_benchmark.tests._real import REAL_ROOT, REAL_USERS

ITERS, BURN = "20", "10"


def _cached_paths():
    paths = [REAL_ROOT / f"user_id={u}" / "data.parquet" for u in REAL_USERS]
    if not all(p.exists() for p in paths):
        pytest.skip("real revlogs for users 6370 and 8506 are not downloaded")
    return paths


@pytest.fixture
def mocked_download():
    """Make download_users return the two cached files without touching the network."""
    paths = _cached_paths()
    with mock.patch.object(data, "download_users", lambda n, seed, out_dir: paths):
        yield paths


def _argv(out_dir, run_name, *extra):
    return ["--n_users", "2", "--max_cards", "300", "--seed", "0", "--iters", ITERS, "--burn", BURN,
            "--run_name", run_name, "--out_dir", str(out_dir), *extra]


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    """One full run with all five models, shared by the smoke tests."""
    out_dir = tmp_path_factory.mktemp("results")
    paths = _cached_paths()
    with mock.patch.object(data, "download_users", lambda n, seed, out_dir: paths):
        info = run_mod.run(run_mod.parse_args(_argv(out_dir, "smoke")))
    return out_dir / "smoke", info


def test_smoke_writes_all_files(smoke_run):
    run_dir, _ = smoke_run
    expected = {"events.parquet", "predictions.parquet", "run_info.json"}
    expected |= {f"pred_{m}.parquet" for m in run_mod.ALL_MODELS}
    assert {p.name for p in run_dir.iterdir()} >= expected


def test_predictions_columns_and_models(smoke_run):
    run_dir, _ = smoke_run
    preds = pd.read_parquet(run_dir / "predictions.parquet")
    assert list(preds.columns) == ["event_id", "user_id", "card_id", "y", "elapsed_days",
                                   "day_offset", *run_mod.ALL_MODELS]
    assert preds["event_id"].is_unique
    assert not preds.isna().any().any()
    for m in run_mod.ALL_MODELS:
        assert preds[m].between(0, 1).all(), m


def test_every_model_predicted_the_same_events(smoke_run):
    run_dir, _ = smoke_run
    preds = pd.read_parquet(run_dir / "predictions.parquet")
    events = pd.read_parquet(run_dir / "events.parquet")
    scored = set(events.loc[events["scored"], "event_id"].tolist())
    assert set(preds["event_id"].tolist()) == scored
    for m in run_mod.ALL_MODELS:
        pred = pd.read_parquet(run_dir / f"pred_{m}.parquet")
        assert set(pred["event_id"].tolist()) == scored, m


def test_run_info_contents(smoke_run):
    run_dir, _ = smoke_run
    info = json.loads((run_dir / "run_info.json").read_text())
    assert info["args"]["iters"] == int(ITERS) and info["args"]["burn"] == int(BURN)
    assert re.fullmatch(r"[0-9a-f]{40}", info["git"]["commit"] or "") is not None
    for pkg in ("fsrs", "numpy", "torch", "numba"):
        assert info["packages"][pkg]
    counts = info["counts"]
    assert counts["users"] == 2
    assert counts["events"] == counts["train"] + counts["test"]
    assert counts["scored"] > 0
    assert set(info["models"]) == set(run_mod.ALL_MODELS)
    for m in run_mod.ALL_MODELS:
        assert info["models"][m]["fit_seconds"] >= 0
        assert info["models"][m]["predict_seconds"] >= 0
    assert info["models"]["hlr"]["epochs"] >= 1
    assert info["models"]["fsrs_opt"]["fallback_count"] == len(info["models"]["fsrs_opt"]["fallback_users"])


def test_resume_skips_finished_models(smoke_run, mocked_download):
    run_dir, _ = smoke_run
    before = {m: (run_dir / f"pred_{m}.parquet").stat().st_mtime_ns for m in run_mod.ALL_MODELS}
    run_mod.run(run_mod.parse_args(_argv(run_dir.parent, "smoke", "--resume")))
    after = {m: (run_dir / f"pred_{m}.parquet").stat().st_mtime_ns for m in run_mod.ALL_MODELS}
    assert before == after
    info = json.loads((run_dir / "run_info.json").read_text())
    assert all(info["models"][m]["resumed"] for m in run_mod.ALL_MODELS)


def test_resume_refits_only_the_missing_model(smoke_run, mocked_download, tmp_path):
    run_dir, _ = smoke_run
    # Work on a copy so the shared fixture stays intact for the other tests.
    copy_dir = tmp_path / "smoke"
    copy_dir.mkdir()
    for p in run_dir.iterdir():
        (copy_dir / p.name).write_bytes(p.read_bytes())
    (copy_dir / "pred_hlr.parquet").unlink()
    before = {m: (copy_dir / f"pred_{m}.parquet").stat().st_mtime_ns
              for m in run_mod.ALL_MODELS if m != "hlr"}
    run_mod.run(run_mod.parse_args(_argv(tmp_path, "smoke", "--resume")))
    assert (copy_dir / "pred_hlr.parquet").exists()
    for m, mtime in before.items():
        assert (copy_dir / f"pred_{m}.parquet").stat().st_mtime_ns == mtime, m
    info = json.loads((copy_dir / "run_info.json").read_text())
    assert info["models"]["hlr"]["resumed"] is False
    assert info["models"]["ours"]["resumed"] is True


def test_mismatched_event_set_is_rejected(mocked_download, tmp_path):
    """A model that drops a scored event must make the run fail, not produce a partial table."""

    class DropsOne:
        def fit(self, events):
            return self

        def predict(self, events):
            scored = events.loc[events["scored"], "event_id"].to_numpy()
            return pd.Series(np.full(len(scored) - 1, 0.5), index=pd.Index(scored[:-1], name="event_id"))

    with mock.patch.object(run_mod, "make_model", lambda key, args: DropsOne()):
        with pytest.raises(AssertionError, match="predicted event set differs"):
            run_mod.run(run_mod.parse_args(_argv(tmp_path, "bad", "--models", "hlr")))


def test_parse_args_rejects_bad_input():
    with pytest.raises(SystemExit):
        run_mod.parse_args(["--run_name", "x", "--models", "hlr,nope"])
    with pytest.raises(SystemExit):
        run_mod.parse_args(["--run_name", "x", "--iters", "10", "--burn", "10"])


def test_git_info_returns_commit():
    info = run_mod.git_info()
    assert info["commit"] is None or re.fullmatch(r"[0-9a-f]{40}", info["commit"])


def test_diagnostics_written_for_ours_variants(smoke_run):
    run_dir, _ = smoke_run
    for key in ("ours", "ours_kappa"):
        diag = json.loads((run_dir / f"diagnostics_{key}.json").read_text())
        assert diag["n_samples"] > 0
        # 20 iterations keep only a few draws, so ESS / R-hat may be None here (not finite);
        # the keys must still be present.
        assert {"mean", "lo", "hi", "ess", "rhat"} <= set(diag["lam"])
        assert "mh_acc_rate" in diag
        assert ("kappa" in diag) == (key == "ours_kappa")
    assert not (run_dir / "diagnostics_hlr.json").exists()


def test_resume_takes_run_info_from_sidecar_details(tmp_path, mocked_download):
    """After a resume, the per-model timings come from the sidecar's details, not run_info.json."""
    run_mod.run(run_mod.parse_args(_argv(tmp_path, "d", "--models", "hlr")))
    sidecar = tmp_path / "d" / "pred_hlr.json"
    stored = json.loads(sidecar.read_text())
    assert set(stored["details"]) >= {"fit_seconds", "predict_seconds", "epochs"}
    stored["details"]["fit_seconds"] = 12345.0
    sidecar.write_text(json.dumps(stored))
    # run_info.json is lost (as after an interrupted run): the sidecar must be the only source.
    (tmp_path / "d" / "run_info.json").unlink()
    info = run_mod.run(run_mod.parse_args(_argv(tmp_path, "d", "--models", "hlr", "--resume")))
    assert info["models"]["hlr"]["resumed"] is True
    assert info["models"]["hlr"]["fit_seconds"] == 12345.0
    assert info["models"]["hlr"]["epochs"] == stored["details"]["epochs"]


def test_resume_accepts_old_sidecar_without_details(tmp_path, mocked_download):
    """Sidecars written before the details key existed must still resume, with run_info as before."""
    run_mod.run(run_mod.parse_args(_argv(tmp_path, "o", "--models", "hlr")))
    sidecar = tmp_path / "o" / "pred_hlr.json"
    stored = json.loads(sidecar.read_text())
    stored.pop("details")
    sidecar.write_text(json.dumps(stored))
    before = json.loads((tmp_path / "o" / "run_info.json").read_text())["models"]["hlr"]
    info = run_mod.run(run_mod.parse_args(_argv(tmp_path, "o", "--models", "hlr", "--resume")))
    assert info["models"]["hlr"]["resumed"] is True
    assert info["models"]["hlr"]["fit_seconds"] == before["fit_seconds"]
