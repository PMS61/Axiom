"""Verifier tests for ours_v3 (Phase 7 task 7.2): leakage on the FULL table, loader shape, frozen choice."""
import numpy as np
import pandas as pd
import pytest

from fsrs_benchmark import duolingo, split
from fsrs_benchmark.data import LOAD_COLUMNS, _empty_frame
from fsrs_benchmark.experiments.validation import make_validation_split
from fsrs_benchmark.models.ours_v3 import FROZEN_KWARGS, OursV3
from fsrs_benchmark.tests.test_ours_v2 import _raw

ALL_ON = dict(lex_effects=("s0", "d0", "growth", "bias"), lex_lr=0.1, lag=True, card_bias=True,
              use_dt=True, tau_pen=300.0)


@pytest.fixture(scope="module")
def full():
    """FULL event table (train AND test rows), lexeme shared across users."""
    raw = _raw(n_users=5)
    raw["lexeme"] = raw["card_id"] % 7
    ev = split.make_split(raw, max_cards=1000, seed=0)
    assert ((~ev["is_train"]) & ev["scored"]).sum() > 20, "need scored test rows"
    return ev


@pytest.mark.parametrize("kwargs", [ALL_ON, dict(lag=True, card_bias=True, tau_pen=300.0)])
def test_flipping_test_label_changes_no_other_prediction(full, kwargs):
    """Flip one TEST row's rating and y. Refit on the flipped table. Every scored prediction must be
    unchanged except the later reviews of the SAME card (which legitimately use this review as history).
    Rows of other users sharing the lexeme, and earlier rows of the same card, must not move."""
    base = OursV3(epochs=1, seed=0, **kwargs).fit(full).predict(full)
    test_scored = full.loc[(~full["is_train"]) & full["scored"], "event_id"].to_numpy()
    for eid in test_scored[:: max(1, len(test_scored) // 6)][:6]:
        pos = full.index[full["event_id"] == eid][0]
        r = full.loc[pos]
        mod = full.copy()
        new_rating = 1 if int(r["rating"]) != 1 else 3
        mod.loc[pos, "rating"] = new_rating
        mod.loc[pos, "y"] = int(new_rating > 1)
        after = OursV3(epochs=1, seed=0, **kwargs).fit(mod).predict(mod)
        later_same_card = set(full.loc[(full["user_id"] == r["user_id"]) & (full["card_id"] == r["card_id"])
                                       & (full["review_idx"] > r["review_idx"]), "event_id"])
        check = [e for e in base.index if e in after.index and e not in later_same_card]
        assert len(check) > 0
        np.testing.assert_allclose(after.loc[check].to_numpy(), base.loc[check].to_numpy(), atol=1e-6,
                                   err_msg=f"prediction moved after flipping test event {eid}")


def test_test_labels_never_change_fitted_parameters(full):
    """Fit reads only is_train rows: wiping every test label must leave all fitted parameters identical."""
    a = OursV3(epochs=2, seed=0, **ALL_ON).fit(full)
    mod = full.copy()
    late = ~mod["is_train"].astype(bool)
    mod.loc[late, "rating"] = 1
    mod.loc[late, "y"] = 0
    b = OursV3(epochs=2, seed=0, **ALL_ON).fit(mod)
    for name in ("theta", "z", "zl", "zb", "beta_lag", "tau_raw", "taul_raw"):
        np.testing.assert_allclose(getattr(a, name).detach().numpy(), getattr(b, name).detach().numpy(),
                                   atol=1e-7, err_msg=name)


def test_validation_table_contains_no_test_event(full):
    val = make_validation_split(full)
    test_ids = set(full.loc[~full["is_train"].astype(bool), "event_id"])
    assert test_ids.isdisjoint(set(val["event_id"]))


def test_validation_predictions_ignore_test_labels(full):
    """Validation table is built without test rows, so flipping test labels in the full table cannot move it."""
    val_a = make_validation_split(full)
    mod = full.copy()
    late = ~mod["is_train"].astype(bool)
    mod.loc[late, "y"] = 1 - mod.loc[late, "y"]
    mod.loc[late, "rating"] = np.where(mod.loc[late, "rating"] == 1, 4, 1)
    val_b = make_validation_split(mod)
    pd.testing.assert_frame_equal(val_a, val_b)


# ----- Duolingo loader ------------------------------------------------------------------------
RAW_CSV = """user_id,lexeme_id,timestamp,delta,p_recall
u1,lexA,1000,0,0.9
u1,lexA,90000,89000,0.3
u1,lexB,2000,0,0.6
u2,lexA,1500,0,0.2
u2,lexA,200000,198500,0.8
u2,lexC,3000,0,0.5
"""


@pytest.fixture()
def dl_file(tmp_path):
    p = tmp_path / "traces.csv"
    p.write_text(RAW_CSV)
    counts, t0 = duolingo.scan(p, tmp_path / "counts.parquet")
    return p, counts, t0


def test_default_loader_is_unchanged(dl_file):
    p, counts, t0 = dl_file
    df = duolingo.load_users(p, ["u1", "u2"], counts, t0)
    assert list(df.columns) == LOAD_COLUMNS
    assert "lexeme" not in df.columns
    assert df.dtypes.to_dict() == _empty_frame().dtypes.to_dict()
    assert len(df) == 6


def test_with_lexeme_adds_only_one_column(dl_file):
    p, counts, t0 = dl_file
    base = duolingo.load_users(p, ["u1", "u2"], counts, t0)
    lex = duolingo.load_users(p, ["u1", "u2"], counts, t0, with_lexeme=True)
    assert list(lex.columns) == LOAD_COLUMNS + ["lexeme"]
    assert lex["lexeme"].dtype == np.int64
    pd.testing.assert_frame_equal(lex[LOAD_COLUMNS].reset_index(drop=True), base.reset_index(drop=True))
    # one code per lexeme string, shared across users: lexA (4 rows, 2 users), lexB, lexC (1 row each)
    assert sorted(lex["lexeme"].value_counts().tolist()) == [1, 1, 4]
    assert lex.groupby("lexeme")["user_id"].nunique().max() == 2


# ----- frozen choice --------------------------------------------------------------------------
def test_frozen_kwargs_is_the_documented_q2():
    assert FROZEN_KWARGS == dict(epochs=20, tau_pen=300.0, lag=True, card_bias=True, card_tau=0.2)


def test_run_validation_q2_matches_frozen():
    from fsrs_benchmark.experiments.run_validation_v3 import VARIANTS
    assert VARIANTS["q2"][1] == FROZEN_KWARGS


def test_run_py_uses_frozen_kwargs():
    import argparse
    from fsrs_benchmark import run as run_mod
    args = argparse.Namespace(seed=0, n_users=2, max_cards=300, iters=5, burn=2, dataset="duolingo",
                              min_user_rows=1, models=["ours_v3"])
    m = run_mod.make_model("ours_v3", args)
    hp = m.hyperparams()
    for k, v in FROZEN_KWARGS.items():
        assert hp[k] == v, k
    assert hp["lex_effects"] == [] and hp["use_dt"] is False
