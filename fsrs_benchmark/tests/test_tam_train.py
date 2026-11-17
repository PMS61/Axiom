"""Tests for fsrs_benchmark/srsb/tam_train.py (Phase 6, task 6.4, step 4): the user guards, a tiny end-to-end
CPU run on four local users (3 train, 1 eval) that writes valid jsonl and parquet, and resume.
Local users only (> 1000)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fsrs_benchmark.srsb import tam_train as TT

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "anki10k"
needs_data = pytest.mark.skipif(not (DATA / "revlogs").exists(), reason="local anki10k data not present")


def _smallest_local(n: int) -> list[int]:
    """The n local users above 1000 with the fewest revlog rows (empty users skipped)."""
    import pyarrow.dataset as pds
    rev = DATA / "revlogs"
    sized = []
    for p in rev.iterdir():
        if not p.name.startswith("user_id="):
            continue
        u = int(p.name.split("=")[1])
        rows = pds.dataset(p).count_rows() if u > 1000 else 0
        if rows > 0:
            sized.append((rows, u))
    return [u for _, u in sorted(sized)[:n]]


# ---- guards ---------------------------------------------------------------------------------------

def test_guard_refuses_test_set_users_unless_final():
    with pytest.raises(ValueError, match="test set"):
        TT.check_users([5, 2001], [2002])
    with pytest.raises(ValueError, match="test set"):
        TT.check_users([2001], [1000])
    TT.check_users([5, 2001], [1000], final=True)          # final lifts it


def test_guard_refuses_validation_users_as_train_only():
    with pytest.raises(ValueError, match="validation"):
        TT.check_users([2001, 8001], [9001])
    with pytest.raises(ValueError, match="validation"):
        TT.check_users([8300], [9001])
    TT.check_users([2001], [8001, 8300])                   # validation users may be eval users
    TT.check_users([8001], [9001], final=True)


def test_guard_refuses_overlap_and_empty_sides():
    with pytest.raises(ValueError, match="overlap"):
        TT.check_users([2001, 2002], [2002])
    with pytest.raises(ValueError, match="no train"):
        TT.check_users([], [2002])
    with pytest.raises(ValueError, match="no eval"):
        TT.check_users([2001], [])


def test_subsample_targets_is_seeded_and_keeps_tokens():
    u = dict(scored=np.arange(100) % 2 == 0, tok=np.zeros((100, 3)))
    a = TT.subsample_targets(u, 10, np.random.default_rng(0))
    b = TT.subsample_targets(u, 10, np.random.default_rng(0))
    assert a["scored"].sum() == 10 and np.array_equal(a["scored"], b["scored"])
    assert a["tok"] is u["tok"]
    assert TT.subsample_targets(u, None, np.random.default_rng(0)) is u


# ---- end to end ----------------------------------------------------------------------------------

def _args(tmp_path, train, ev, out, epochs=2):
    return ["--data", str(DATA), "--train_users", ",".join(map(str, train)),
            "--eval_users", str(ev), "--out", str(out), "--cache", str(tmp_path / "cache"),
            "--epochs", str(epochs), "--batch", "32", "--device", "cpu",
            "--max_targets_per_user", "40", "--patience", "5"]


@needs_data
def test_end_to_end_cpu_writes_valid_jsonl_and_parquet_and_resumes(tmp_path):
    uids = _smallest_local(4)
    train, ev = uids[:3], uids[3]
    out = tmp_path / "tam.jsonl"
    info = TT.main(_args(tmp_path, train, ev, out))

    lines = [json.loads(x) for x in out.read_text().splitlines() if x.strip()]
    assert [x["user"] for x in lines] == [ev]
    m = lines[0]["metrics"]
    assert set(m) >= {"RMSE", "LogLoss", "RMSE(bins)", "ICI", "AUC", "MBE"}
    assert 0 < m["LogLoss"] < 5 and lines[0]["size"] > 0

    preds = pd.read_parquet(Path(str(out) + ".preds.parquet"))
    assert list(preds.columns) == ["uid", "review_th", "p"]
    assert (preds["uid"] == ev).all() and len(preds) == lines[0]["size"]
    assert preds["p"].between(0, 1).all()

    run_info = json.loads(Path(str(out) + ".run_info.json").read_text())
    assert run_info["train"]["params"] > 0 and run_info["train"]["best_epoch"] >= 1
    assert len(run_info["train"]["history"]) >= 1
    assert Path(str(out) + ".ckpt.pt").exists()

    # resume 1: nothing left to score, the trained model is reloaded and nothing is rewritten
    before = out.read_text()
    TT.main(_args(tmp_path, train, ev, out))
    assert out.read_text() == before

    # resume 2: drop the eval line and rerun; it is rescored from the saved checkpoint, same metrics
    out.write_text("")
    TT.main(_args(tmp_path, train, ev, out))
    again = [json.loads(x) for x in out.read_text().splitlines() if x.strip()]
    assert [x["user"] for x in again] == [ev]
    assert again[0]["metrics"]["LogLoss"] == pytest.approx(m["LogLoss"], abs=1e-6)
