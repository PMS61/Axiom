"""Tests for the frozen final mode of online_adapt (SRS-Beat): the users 1-1000 guard, one fixed setting,
and cache-vs-featurized equality. Local data only for the last test (validation users 8002 and 8019)."""
import json
from pathlib import Path

import numpy as np
import pytest

from fsrs_benchmark.srsb import online_adapt as O
from fsrs_benchmark.srsb import val_plus as V

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "results/srsb_beat/sweep_a_l63_side.jsonl.model.txt"
RUN_INFO = ROOT / "results/srsb_beat/sweep_a_l63_side.jsonl.run_info.json"
GBDT = ROOT / "results/srsb_beat/sweep_a_l63_side.jsonl"
DATA = ROOT / "data/anki10k"


def test_users_1_to_1000_need_final():
    with pytest.raises(ValueError):
        O.check_users([1, 2, 8001], final=False)
    with pytest.raises(ValueError):
        O.check_users([1000], final=False)
    O.check_users([1, 1000], final=True)                  # allowed only with the frozen final run
    O.check_users([1001, 8001], final=False)               # outside the test set: no flag needed


def test_cli_refuses_test_users_without_final_before_any_work(tmp_path):
    with pytest.raises(ValueError):
        O.main(["--users", "5,6", "--out", str(tmp_path / "x.jsonl")])
    assert not (tmp_path / "x.jsonl").exists()


def test_final_mode_is_one_fixed_setting():
    assert O.final_settings() == [("B", 0.003)]
    assert len(O.final_settings("A", 0.01)) == 1
    with pytest.raises(ValueError):
        O.final_settings("C", 0.003)
    with pytest.raises(ValueError):
        O.final_settings("B", 0.0)


@pytest.fixture(scope="module")
def local():
    if not (MODEL.exists() and RUN_INFO.exists() and DATA.exists()):
        pytest.skip("local data or model not available")
    return True


ONLINE_VAL = ROOT / "results/srsb_beat/online_adapt_side.jsonl"


def test_final_mode_reproduces_the_validation_online_layer(local, tmp_path, monkeypatch):
    """The fixed final setting (B, lr 0.003) on users 8002 and 8019 reproduces the saved validation run of the
    online layer (chosen setting B_lr0.003), and runs exactly one setting per user."""
    calls = []
    real = O.online_kernel

    def counting(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(O, "online_kernel", counting)
    users = [8002, 8019]
    res = O.run_final(MODEL, RUN_INFO, DATA, tmp_path / "online.jsonl", users, workers=1,
                      gbdt_jsonl=GBDT, final=False)
    assert len(calls) == len(users)                        # exactly one setting per user, no selection
    ours = O.C.load_rows(tmp_path / "online.jsonl")
    ref = O.C.load_rows(ONLINE_VAL)
    for u in users:
        assert ours[u]["size"] == ref[u]["size"]
        assert abs(ours[u]["metrics"]["LogLoss"] - ref[u]["metrics"]["LogLoss"]) < 1e-5
    assert res["gbdt_checked_users"] == 2
    assert res["setting"] == ["B", 0.003]
    assert json.loads((tmp_path / "online.jsonl").read_text().splitlines()[0])["setting"] == ["B", 0.003]
