"""Tests for OursV3: one-step-ahead, fit uses only is_train rows, shapes, probabilities, lexeme data, registration."""
import numpy as np
import pandas as pd
import pytest

from fsrs_benchmark import split
from fsrs_benchmark.experiments.validation import make_validation_split
from fsrs_benchmark.models.ours_v3 import FROZEN_KWARGS, OursV3
from fsrs_benchmark.tests.test_ours_v2 import _raw

ALL_ON = dict(lex_effects=("s0", "d0", "growth", "bias"), lag=True, card_bias=True, use_dt=True)


@pytest.fixture(scope="module")
def events():
    raw = _raw()
    raw["lexeme"] = raw["card_id"] % 7          # shared across users
    return split.make_split(raw, max_cards=1000, seed=0)


@pytest.fixture(scope="module")
def val(events):
    return make_validation_split(events)


def test_lexeme_column_survives_split(events, val):
    assert "lexeme" in events.columns and "lexeme" in val.columns


@pytest.mark.parametrize("kwargs", [dict(), ALL_ON])
def test_shapes_and_probabilities(val, kwargs):
    p = OursV3(epochs=2, **kwargs).fit(val).predict(val)
    assert set(p.index) == set(val.loc[val["scored"], "event_id"]) and len(p) == int(val["scored"].sum())
    assert p.index.is_unique and np.isfinite(p).all() and p.between(1e-6, 1 - 1e-6).all() and (p > 0).all() and (p < 1).all()


def test_fit_uses_only_train_rows(val):
    a = OursV3(epochs=2, **ALL_ON).fit(val)
    mod = val.copy()
    late = ~mod["is_train"]
    mod.loc[late, "rating"] = 1
    mod.loc[late, "y"] = 0
    b = OursV3(epochs=2, **ALL_ON).fit(mod)
    for name in ("theta", "z", "zl", "zb", "beta_lag"):
        np.testing.assert_allclose(getattr(a, name).detach().numpy(), getattr(b, name).detach().numpy())


@pytest.mark.parametrize("kwargs", [dict(), ALL_ON, dict(lag=True, process_noise=True)])
def test_one_step_ahead_label_perturbation(val, kwargs):
    m = OursV3(epochs=1, **kwargs).fit(val)
    base = m.predict(val)
    rng = np.random.default_rng(1)
    for eid in rng.choice(base.index.to_numpy(), size=6, replace=False):
        row = val[val["event_id"] == eid].iloc[0]
        mod = val.copy()
        same = (mod["user_id"] == row["user_id"]) & (mod["card_id"] == row["card_id"]) & (mod["review_idx"] >= row["review_idx"])
        mod.loc[same, "rating"] = np.where(mod.loc[same, "rating"] == 1, 4, 1)   # flip this label and all later ones
        mod.loc[same, "y"] = (mod.loc[same, "rating"] > 1).astype(int)
        after = m.predict(mod)
        assert after.loc[eid] == pytest.approx(base.loc[eid], abs=1e-6)
    # but flipping the PREVIOUS label must change a lag prediction (the term is really used)
    m.beta_lag = m.beta_lag.detach() + 1.0
    base = m.predict(val)
    eid = base.index[len(base) // 2]
    row = val[val["event_id"] == eid].iloc[0]
    mod = val.copy()
    prev = (mod["user_id"] == row["user_id"]) & (mod["card_id"] == row["card_id"]) & (mod["review_idx"] < row["review_idx"])
    last = mod[prev].sort_values("review_idx").index[-1]
    mod.loc[last, "y"] = 1 - mod.loc[last, "y"]
    if kwargs.get("lag"):
        assert abs(m.predict(mod).loc[eid] - base.loc[eid]) > 1e-4


def test_flags_off_matches_v2(val):
    from fsrs_benchmark.models.ours_v2 import OursV2
    a = OursV2(epochs=2).fit(val).predict(val)
    b = OursV3(epochs=2).fit(val).predict(val)
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), atol=1e-5)


def test_lexeme_effect_changes_predictions_and_unseen_lexeme_ok(val):
    m = OursV3(epochs=2, lex_effects=("s0", "bias"), lex_lr=0.2).fit(val)
    assert float(m.zl.detach().abs().max()) > 0
    other = val.copy()
    other["lexeme"] = 10_000                       # lexeme never seen in fit
    p = m.predict(other)
    assert len(p) == int(val["scored"].sum()) and p.between(1e-6, 1 - 1e-6).all()


def test_no_lexeme_column_is_fine(val):
    p = OursV3(epochs=1, lex_effects=("s0",)).fit(val.drop(columns="lexeme")).predict(val.drop(columns="lexeme"))
    assert len(p) == int(val["scored"].sum())


def test_registered_in_run_and_report():
    from fsrs_benchmark import report, run as run_mod
    assert "ours_v3" in run_mod.OPT_IN_MODELS and "ours_v3" not in run_mod.ALL_MODELS
    assert all(report.DISPLAY["ours_v3"] and report.COLOURS["ours_v3"] and report.MARKERS["ours_v3"] for _ in [0])
    import argparse
    args = argparse.Namespace(seed=0, n_users=2, max_cards=300, iters=5, burn=2, dataset="anki")
    m = run_mod.make_model("ours_v3", args)
    assert isinstance(m, OursV3) and m.epochs == FROZEN_KWARGS["epochs"]
    cfg = run_mod.model_config("ours_v3", args, np.array([1, 2], np.int64))
    assert cfg["hyperparams"]["lag"] == FROZEN_KWARGS.get("lag", False)
