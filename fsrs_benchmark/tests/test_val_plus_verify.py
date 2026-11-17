"""Verifier checks for Phase 6.2 (srsb/val_plus.py): the 1-1000 and overlap guards (only --final bypasses them,
and the guard runs before any download), the train pool rule of SRS-Beat (never 8001-8300), feature rows and
test mask equal to run_gbdt.user_table, holdout users drawn from the train users only, the published
comparison on common users with equal sizes, resumability (truncated line, stop and continue, model reuse),
max_train_rows, and run_gbdt defaults unchanged. Local users above 1000 only."""
import json
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from fsrs_benchmark import data as D
from fsrs_benchmark.srsb import compare as C
from fsrs_benchmark.srsb import run_gbdt as R
from fsrs_benchmark.srsb import val_plus as V

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "anki10k"
UID = 1236
needs_data = pytest.mark.skipif(not (DATA / "revlogs" / f"user_id={UID}").exists(),
                                reason="local anki10k data not present")


def _local_users_above_1000(n):
    ids = sorted(int(p.name.split("=")[1]) for p in (DATA / "revlogs").iterdir() if p.name.startswith("user_id="))
    ids = [u for u in ids if u > 1000 and (DATA / "revlogs" / f"user_id={u}" / "data.parquet").exists()]
    return ids[:n]


# ---- guards ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("train, ev", [([1000], [2001]), ([2001], [1]), ([2001], [500]), ([1], [2002])])
def test_test_set_users_refused_on_either_side(train, ev):
    with pytest.raises(ValueError, match="test set"):
        V.check_users(train, ev)


def test_main_refuses_1_1000_before_download(tmp_path):
    tr, ev = tmp_path / "t.txt", tmp_path / "e.txt"
    tr.write_text("2001\n2002\n")
    ev.write_text("1000\n")
    with mock.patch.object(D, "download_revlogs") as dr, mock.patch.object(D, "download_side_tables") as ds:
        with pytest.raises(ValueError, match="test set"):
            V.main(["--train_users", str(tr), "--eval_users", str(ev), "--out", str(tmp_path / "o.jsonl"),
                    "--download", "--data", str(tmp_path)])
    dr.assert_not_called()
    ds.assert_not_called()
    assert not (tmp_path / "o.jsonl").exists()


def test_final_flag_is_the_only_exemption(tmp_path):
    V.check_users([2001], [1000], final=True)
    with pytest.raises(ValueError):
        V.check_users([2001], [1000], final=False)


def test_validation_users_8001_8300_may_not_be_train_users():
    """SRS-Beat: the train pool is 2001-8000 and 9001-9999, never 8001-8300 (validation users)."""
    with pytest.raises(ValueError):
        V.check_users([8001, 2001], [2002])


def test_overlap_refused_even_with_final():
    with pytest.raises(ValueError, match="overlap"):
        V.check_users([2001, 2002], [2002], final=True)


# ---- feature rows equal run_gbdt ------------------------------------------------------------------

@needs_data
def test_feature_rows_and_mask_equal_run_gbdt_user_table():
    S = R._srs_modules()
    cfg = S["Config"](S["create_parser"]().parse_args(["--algo", "FSRS-6", "--short", "--data", str(DATA)]))
    loader = S["UserDataLoader"](cfg)
    ut = R.user_table(UID, loader, cfg, DATA)
    vt = V.user_table(UID, loader, cfg, DATA, R.FEATURE_COLS)
    assert np.array_equal(vt["X"], ut["X"], equal_nan=True)
    assert np.array_equal(vt["y"], ut["y"])
    assert np.array_equal(vt["test"], ut["test"])
    assert vt["test"].any() and not vt["test"].all()


# ---- holdout and defaults -----------------------------------------------------------------------

def test_holdout_users_come_from_train_only():
    train = list(range(2001, 2041))
    ev = [2100, 2101]
    hold = R.holdout_users(train)
    assert hold and set(hold) <= set(train)
    assert not set(hold) & set(ev)


def test_trainpool_routes_holdout_and_never_eval_users():
    train = [2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010]
    pool = R.TrainPool(hold=R.holdout_users(train), n_feat=3)
    for u in train:
        pool.add(u, np.zeros((2, 3), np.float32), np.array([0, 1]))
    assert set(pool.uids["val"]) <= set(R.holdout_users(train))
    assert set(pool.uids["fit"]) | set(pool.uids["val"]) == set(train)


def test_run_gbdt_defaults_unchanged():
    assert R.FEATURE_COLS == [c for g in R.FEATURE_GROUPS for c in R.G.GROUP_FEATURES[g]]
    assert R.LGB_PARAMS["num_leaves"] == 15 and R.LGB_PARAMS["learning_rate"] == 0.02
    assert R.TrainPool().n_feat == len(R.FEATURE_COLS)


def test_finalize_caps_rows_exactly():
    pool = R.TrainPool(n_feat=2)
    for u in range(5):
        pool.add(2000 + u, np.ones((100, 2), np.float32), np.ones(100, np.int64))
    X, y = pool.finalize("fit", 123)
    assert X.shape == (123, 2) and y.shape == (123,)


# ---- published comparison: common users, equal sizes ------------------------------------------------

def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_compare_uses_only_users_present_everywhere_with_equal_size(tmp_path):
    met = lambda ll: {"LogLoss": ll, "RMSE": 0.1, "RMSE(bins)": 0.1, "ICI": 0.1, "AUC": 0.5, "MBE": 0.0}
    ours = tmp_path / "ours.jsonl"
    _write(ours, [dict(user=5, size=10, metrics=met(0.3)), dict(user=6, size=20, metrics=met(0.4)),
                  dict(user=7, size=30, metrics=met(0.5))])
    res = tmp_path / "res"
    res.mkdir()
    for f in V.PUBLISHED.values():
        rows = [dict(user=5, size=10, metrics=met(0.2)), dict(user=6, size=21, metrics=met(0.2))]
        if f == "FSRS-6-short":
            rows.append(dict(user=7, size=30, metrics=met(0.2)))
        _write(res / f"{f}.jsonl", rows)
    cmp = V.compare_published(ours, res, n_boot=50)
    assert cmp["n_users"] == 1       # user 6 size differs, user 7 is missing from RWKV-P and RWKV
    assert cmp["n_rows"] == 10


def test_compare_returns_none_without_published_files(tmp_path):
    ours = tmp_path / "o.jsonl"
    _write(ours, [dict(user=5, size=10, metrics={"LogLoss": 0.3})])
    assert V.compare_published(ours, tmp_path / "empty") is None


# ---- end to end: resume, stop and continue, model reuse, max_train_rows -----------------------------

@pytest.fixture(scope="module")
def tiny_users():
    if not (DATA / "revlogs").exists():
        pytest.skip("no local revlogs")
    users = _local_users_above_1000(6)
    if len(users) < 6:
        pytest.skip("need six local users above 1000")
    return users[:4], users[4:]


def test_resume_after_stop_and_truncated_line(tmp_path, tiny_users):
    train, ev = tiny_users
    out = tmp_path / "r.jsonl"
    first = V.run(train, ev, DATA, out, groups=["gap", "card"], max_rounds=15, result_dir=tmp_path,
                  stop_after=2)
    assert first["stopped"] and first["new_lines"] == 2
    with open(out, "a", encoding="utf-8") as f:
        f.write('{"metrics": {"LogLo')                  # a line cut off mid-write
    second = V.run(train, ev, DATA, out, groups=["gap", "card"], max_rounds=15, result_dir=tmp_path)
    assert second["trained_now"] is False                # the saved model is reused
    lines = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert [r["user"] for r in lines] == sorted(ev)      # no duplicates, no truncated line left
    assert second["eval_lines"] == len(ev)


def test_max_train_rows_runs_and_is_recorded(tmp_path, tiny_users):
    train, ev = tiny_users
    out = tmp_path / "cap.jsonl"
    info = V.run(train, ev, DATA, out, groups=["gap", "card"], max_rounds=15, max_train_rows=300,
                 result_dir=tmp_path)
    assert info["max_train_rows"] == 300
    assert info["info"]["fit_rows"] <= 300
    assert info["eval_lines"] == len(ev)
