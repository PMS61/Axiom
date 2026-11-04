"""Runner tests for Phase 5: dataset flag, opt-in keys, resume config and a synthetic Duolingo run."""
import json

import numpy as np
import pandas as pd
import pytest

from fsrs_benchmark import run as run_mod
from fsrs_benchmark.models.fsrs_model import FSRSModel
from fsrs_benchmark.models.laglogit import LagLogitModel
from fsrs_benchmark.models.ours import OursModel
from fsrs_benchmark.report import build_report
from fsrs_benchmark.tests._synthetic_events import write_duolingo_csv


def _args(*extra):
    return run_mod.parse_args(["--run_name", "x", *extra])


def test_default_models_are_unchanged_and_opt_in_keys_are_allowed():
    assert _args().models == run_mod.ALL_MODELS
    assert _args("--models", "ours_gpu,ours_kappa_gpu,laglogit").models == [
        "ours_gpu", "ours_kappa_gpu", "laglogit"]


def test_unknown_model_key_is_rejected():
    with pytest.raises(SystemExit):
        _args("--models", "ours_tpu")


def test_dataset_defaults_to_anki_and_zero_users_needs_duolingo():
    assert _args().dataset == "anki"
    with pytest.raises(SystemExit):
        _args("--n_users", "0")
    assert _args("--dataset", "duolingo", "--n_users", "0").n_users == 0


def test_device_and_thin_flags():
    args = _args("--device", "cpu", "--thin", "2")
    assert run_mod.device_arg(args) == "cpu"
    assert args.thin == 2
    assert run_mod.device_arg(_args()) is None  # auto


def test_make_model_builds_the_requested_backend():
    gpu = run_mod.make_model("ours_kappa_gpu", _args("--device", "cpu", "--iters", "5", "--burn", "1"))
    assert isinstance(gpu, OursModel) and gpu.backend == "gpu" and gpu.use_kappa and gpu.device == "cpu"
    numba_model = run_mod.make_model("ours", _args("--iters", "5", "--burn", "1"))
    assert numba_model.backend == "numba" and not numba_model.use_kappa
    assert isinstance(run_mod.make_model("laglogit", _args()), LagLogitModel)
    assert isinstance(run_mod.make_model("fsrs_default", _args("--workers", "2")), FSRSModel)


def test_model_config_adds_backend_and_dataset_only_when_needed():
    scored = np.array([1, 2, 3], dtype=np.int64)
    anki = run_mod.model_config("ours", _args("--iters", "5", "--burn", "1"), scored)
    assert "backend" not in anki and "dataset" not in anki
    gpu = run_mod.model_config("ours_gpu", _args("--iters", "5", "--burn", "1"), scored)
    assert gpu["backend"] == "gpu" and gpu["iters"] == 5 and gpu["burn"] == 1
    duo = run_mod.model_config("hlr", _args("--dataset", "duolingo", "--min_user_rows", "50"), scored)
    assert duo["dataset"] == "duolingo" and duo["min_user_rows"] == 50
    assert "backend" not in duo  # numba keys keep their old config


def test_gpu_keys_are_not_in_the_default_run_so_resume_configs_do_not_change():
    default_keys = run_mod.ALL_MODELS
    assert not any(k.endswith("_gpu") or k == "laglogit" for k in default_keys)


def test_end_to_end_duolingo_run_on_synthetic_csv(tmp_path):
    csv = write_duolingo_csv(tmp_path / "traces.csv.gz")
    out_dir = tmp_path / "results"
    argv = ["--dataset", "duolingo", "--duolingo_path", str(csv), "--n_users", "0",
            "--min_user_rows", "10", "--models", "fsrs_default,hlr,laglogit,ours_gpu",
            "--iters", "6", "--burn", "2", "--thin", "1", "--device", "cpu", "--max_cards", "300",
            "--seed", "0", "--run_name", "duo", "--out_dir", str(out_dir)]
    run_mod.main(argv)
    run_dir = out_dir / "duo"

    info = json.loads((run_dir / "run_info.json").read_text())
    assert info["dataset"] == "duolingo"
    assert info["counts"]["users"] == 3
    assert info["models"]["ours_gpu"]["device"] == "cpu"
    assert (run_dir / "pred_ours_gpu.json").exists()
    assert (run_dir / "diagnostics_ours_gpu.json").exists()

    cols = set(pd.read_parquet(run_dir / "predictions.parquet").columns)
    assert {"fsrs_default", "hlr", "laglogit", "ours_gpu"} <= cols

    report_dir = build_report(run_dir)
    assert "Duolingo" in (report_dir / "report.md").read_text()
