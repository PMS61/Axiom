"""Tests for fsrs_benchmark/srsb/val_plus.py (Phase 6, task 6.2): the 1-1000 and overlap guards, feature group
selection, and a tiny end-to-end run on local users above 1000 (never the test set)."""
import json
import os
from pathlib import Path

import pytest

from fsrs_benchmark.models import gbdt as G
from fsrs_benchmark.srsb import run_gbdt as R
from fsrs_benchmark.srsb import val_plus as V
from fsrs_benchmark.srsb.features_plus import PLUS_COLS, PLUS_GROUPS

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "anki10k"


# ---- guards -------------------------------------------------------------------------------------

def test_guard_rejects_test_set_users_in_train_or_eval():
    with pytest.raises(ValueError, match="test set"):
        V.check_users([5, 2001], [2002])                      # user 5 is in 1-1000
    with pytest.raises(ValueError, match="test set"):
        V.check_users([2001], [1000])                         # boundary, eval side
    with pytest.raises(ValueError, match="test set"):
        V.check_users([2001], [1])                            # boundary, lower end


def test_guard_allows_users_just_outside_test_set():
    V.check_users([1001, 2001], [1002])                       # 1001 and 1002 are outside 1-1000
    V.check_users([2001], [1000], final=True)                 # final=True is the only exemption


def test_guard_rejects_overlap_and_empty_sides():
    with pytest.raises(ValueError, match="overlap"):
        V.check_users([2001, 2002], [2002], final=True)
    with pytest.raises(ValueError, match="no eval users"):
        V.check_users([2001], [])
    with pytest.raises(ValueError, match="no train users"):
        V.check_users([], [2001])


def test_run_refuses_before_any_work(tmp_path):
    with pytest.raises(ValueError, match="test set"):
        V.run([5], [2001], DATA, tmp_path / "x.jsonl")
    assert not (tmp_path / "x.jsonl").exists()


# ---- feature groups -------------------------------------------------------------------------------

def test_default_groups_are_all_features_once():
    cols = V.group_columns(None)
    assert cols == R.FEATURE_COLS + PLUS_COLS
    assert len(cols) == len(set(cols))


def test_group_selection_keeps_the_fixed_order_and_skips_others():
    cols = V.group_columns(["sib", "gap"])
    assert cols == G.GROUP_FEATURES["gap"] + PLUS_GROUPS["sib"]
    assert "card_n" not in cols and "deck_n" not in cols


def test_unknown_or_empty_groups_raise():
    with pytest.raises(ValueError, match="unknown"):
        V.group_columns(["gap", "lexeme"])
    with pytest.raises(ValueError, match="no feature groups"):
        V.group_columns([])


# ---- end to end on six local users above 1000 -------------------------------------------------

def _local_users_above_1000(n):
    ids = sorted(int(p.name.split("=")[1]) for p in (DATA / "revlogs").iterdir() if p.name.startswith("user_id="))
    ids = [u for u in ids if u > 1000 and (DATA / "revlogs" / f"user_id={u}" / "data.parquet").exists()]
    return ids[:n]


def test_end_to_end_tiny_run_writes_valid_jsonl(tmp_path):
    if not (DATA / "revlogs").exists():
        pytest.skip("no local revlogs")
    users = _local_users_above_1000(6)
    if len(users) < 6:
        pytest.skip("need six local users above 1000")
    train, ev = users[:4], users[4:]
    out = tmp_path / "tiny.jsonl"
    info = V.run(train, ev, DATA, out, groups=["gap", "card", "sib"], max_rounds=20, workers=1,
                 result_dir=tmp_path)                       # no published files here: comparison is None
    lines = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert [r["user"] for r in lines] == sorted(ev)
    for r in lines:
        assert set(r) == {"metrics", "user", "size"}
        assert r["size"] > 0
        assert {"RMSE", "LogLoss", "RMSE(bins)", "ICI", "AUC", "MBE"} <= set(r["metrics"])
        assert 0 < r["metrics"]["LogLoss"] < 5
    assert info["features"] == V.group_columns(["gap", "card", "sib"])
    assert Path(str(out) + ".model.txt").exists() and Path(str(out) + ".model.json").exists()
    saved = json.loads(Path(str(out) + ".run_info.json").read_text(encoding="utf-8"))
    assert saved["features"] == info["features"] and saved["best_iter"] >= 1
    assert saved["comparison"] is None
