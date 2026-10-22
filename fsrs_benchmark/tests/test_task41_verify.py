"""Verifier checks for Phase 4 Task 4.1 (progress logging, details in the pred sidecar).

Tiny fits only (iters=60, burn=30). The committed-results check only reads files.
"""
import json
import re
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fsrs_benchmark import run as run_mod
from fsrs_benchmark.models.ours import OursModel
from fsrs_benchmark.tests.test_ours import SMOKE, _real_events
from fsrs_benchmark.tests.test_run import _argv, _cached_paths, mocked_download  # noqa: F401

REPO = Path("/home/amal/fyp/Axiom")
LINE = re.compile(r"^\[\d{2}:\d{2}:\d{2}\] \[(?P<label>[^\]]+)\] sweep (?P<done>\d+)/(?P<total>\d+), elapsed (?P<el>[\d.]+)s, "
                  r"eta (?P<eta>[\d.]+)s, lam=(?P<lam>[-\d.]+)(, kappa=(?P<kappa>[-\d.]+))?$")


@pytest.fixture(scope="module")
def events():
    return _real_events()


def _progress_lines(err):
    return [line for line in err.splitlines() if "[ours" in line]


def test_progress_final_sweep_logged_when_every_exceeds_iters(events, capsys):
    OursModel(progress_every=1000, **SMOKE).fit(events)
    lines = _progress_lines(capsys.readouterr().err)
    assert len(lines) == 1
    assert lines[0].split()[3] == "60/60,"


def test_progress_eta_and_elapsed_are_sane(events, capsys):
    OursModel(progress_every=20, **SMOKE).fit(events)
    lines = _progress_lines(capsys.readouterr().err)
    parsed = [LINE.match(line) for line in lines]
    assert all(parsed), lines
    elapsed = [float(m["el"]) for m in parsed]
    eta = [float(m["eta"]) for m in parsed]
    assert elapsed == sorted(elapsed)
    assert all(e >= 0 for e in eta)
    assert [int(m["done"]) for m in parsed] == [20, 40, 60]
    assert all(int(m["total"]) == 60 for m in parsed)
    assert eta[-1] == 0.0


def test_progress_kappa_label_and_field(events, capsys):
    OursModel(use_kappa=True, progress_every=60, **SMOKE).fit(events)
    lines = _progress_lines(capsys.readouterr().err)
    assert len(lines) == 1
    m = LINE.match(lines[0])
    assert m and m["label"] == "ours_kappa" and m["kappa"] is not None


def test_progress_not_logged_with_progress_every_zero_even_with_iters_multiple(events, capsys):
    OursModel(progress_every=0, **SMOKE).fit(events)
    assert _progress_lines(capsys.readouterr().err) == []


def _hlr_sidecar(tmp_path, mocked):
    run_mod.run(run_mod.parse_args(_argv(tmp_path, "x", "--models", "hlr")))
    return tmp_path / "x" / "pred_hlr.json"


def test_resume_ignores_extra_details_for_config_check(tmp_path, mocked_download):
    sidecar = _hlr_sidecar(tmp_path, mocked_download)
    stored = json.loads(sidecar.read_text())
    stored["details"] = {"anything": [1, 2]}
    sidecar.write_text(json.dumps(stored))
    info = run_mod.run(run_mod.parse_args(_argv(tmp_path, "x", "--models", "hlr", "--resume")))
    assert info["models"]["hlr"]["resumed"] is True
    assert info["models"]["hlr"]["anything"] == [1, 2]


def test_resume_partial_details_does_not_crash(tmp_path, mocked_download):
    sidecar = _hlr_sidecar(tmp_path, mocked_download)
    stored = json.loads(sidecar.read_text())
    stored["details"] = {"fit_seconds": 2.5}
    sidecar.write_text(json.dumps(stored))
    info = run_mod.run(run_mod.parse_args(_argv(tmp_path, "x", "--models", "hlr", "--resume")))
    assert info["models"]["hlr"]["fit_seconds"] == 2.5
    assert info["models"]["hlr"]["resumed"] is True


@pytest.mark.parametrize("bad", [None, "oops", [1, 2]])
def test_resume_corrupt_details_does_not_crash(tmp_path, mocked_download, bad):
    """A sidecar whose details is not a dict (hand-edited or truncated) should not abort --resume."""
    sidecar = _hlr_sidecar(tmp_path, mocked_download)
    stored = json.loads(sidecar.read_text())
    stored["details"] = bad
    sidecar.write_text(json.dumps(stored))
    info = run_mod.run(run_mod.parse_args(_argv(tmp_path, "x", "--models", "hlr", "--resume")))
    assert info["models"]["hlr"]["resumed"] is True


def test_check_resume_config_accepts_committed_full_run_sidecars():
    """The committed full run's sidecars (no details key) must still pass the resume config check.

    Read-only: the expected config is rebuilt from the run's own run_info.json and events.parquet.
    """
    run_dir = REPO / "results" / "full_u100_c1000"
    if not (run_dir / "events.parquet").exists():
        pytest.skip("committed full run not present")
    info = json.loads((run_dir / "run_info.json").read_text())
    args = Namespace(**{k: info["args"][k] for k in ("n_users", "max_cards", "seed", "iters", "burn")})
    events = pd.read_parquet(run_dir / "events.parquet")
    scored_ids = np.sort(events.loc[events["scored"], "event_id"].to_numpy(np.int64))
    checked = []
    for key in ("fsrs_default", "fsrs_opt", "hlr", "ours", "ours_kappa"):
        pred_path = run_dir / f"pred_{key}.parquet"
        if not pred_path.exists():
            continue
        run_mod.check_resume_config(pred_path, run_mod.model_config(key, args, scored_ids))
        checked.append(key)
    assert {"ours", "ours_kappa"} <= set(checked)
