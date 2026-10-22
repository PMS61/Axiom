"""Tests for fsrs_benchmark.models.ours: no leakage in prediction, cold users/cards, valid output.

Why these tests: the adapter has to hand the model the right sequences. A test row's
own outcome, or a later one, must never reach the prediction for it, and unseen users
or cards must take the model's cold path instead of crashing or reading random state.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fsrs_benchmark import data, split
from fsrs_benchmark.models.ours import OursModel

REAL_ROOT = Path("/home/amal/fyp/Axiom/data/anki10k/revlogs")
REAL_USERS = (6370, 8506)
SMOKE = dict(iters=60, burn=30, thin=4, seed=0)


def _real_events(users=REAL_USERS):
    paths = [REAL_ROOT / f"user_id={u}" / "data.parquet" for u in users]
    if not all(p.exists() for p in paths):
        pytest.skip("real revlogs for users 6370 and 8506 are not downloaded")
    return split.make_split(pd.concat([data.load_user(p) for p in paths]), max_cards=300, seed=0)


@pytest.fixture(scope="module")
def smoke_events():
    return _real_events()


@pytest.fixture(scope="module")
def smoke_model(smoke_events):
    return OursModel(**SMOKE).fit(smoke_events)


def test_predicts_every_scored_event_in_unit_interval(smoke_events, smoke_model):
    got = smoke_model.predict(smoke_events)
    expected = set(smoke_events.loc[smoke_events["scored"], "event_id"].astype(np.int64))
    assert set(got.index) == expected
    assert got.index.name == "event_id"
    assert np.isfinite(got.to_numpy()).all()
    assert ((got > 0) & (got < 1)).all()


def test_prediction_ignores_own_and_later_outcomes(smoke_events, smoke_model):
    """Flipping a scored event's outcome, and every later outcome on its card, must not move it."""
    base = smoke_model.predict(smoke_events)
    scored = smoke_events[smoke_events["scored"].astype(bool)]
    # A scored event that has later test rows on the same card gives the strongest check.
    target = None
    for row in scored.itertuples():
        later = smoke_events[(smoke_events["user_id"] == row.user_id)
                             & (smoke_events["card_id"] == row.card_id)
                             & (smoke_events["review_idx"] >= row.review_idx)]
        if len(later) >= 2:
            target = row
            break
    assert target is not None, "no scored event with later reviews on its card"

    mutated = smoke_events.copy()
    same_card_from_target = ((mutated["user_id"] == target.user_id)
                             & (mutated["card_id"] == target.card_id)
                             & (mutated["review_idx"] >= target.review_idx))
    assert not mutated.loc[same_card_from_target, "is_train"].any()
    mutated.loc[same_card_from_target, "y"] = 1 - mutated.loc[same_card_from_target, "y"]

    after = smoke_model.predict(mutated)
    assert after.loc[target.event_id] == base.loc[target.event_id]


def test_earlier_outcome_on_same_card_does_move_prediction(smoke_events, smoke_model):
    """Sensitivity: the outcome of an earlier test review on the same card does reach the prediction."""
    base = smoke_model.predict(smoke_events)
    moved = False
    scored = smoke_events[smoke_events["scored"].astype(bool) & ~smoke_events["is_train"].astype(bool)]
    for row in scored.itertuples():
        earlier = smoke_events[(smoke_events["user_id"] == row.user_id)
                               & (smoke_events["card_id"] == row.card_id)
                               & (smoke_events["review_idx"] < row.review_idx)
                               & (~smoke_events["is_train"].astype(bool))]
        if len(earlier) == 0:
            continue
        tweaked = smoke_events.copy()
        tweaked.loc[earlier.index[-1], "y"] = 1 - tweaked.loc[earlier.index[-1], "y"]
        after = smoke_model.predict(tweaked)
        if not np.isclose(after.loc[row.event_id], base.loc[row.event_id], atol=1e-12):
            moved = True
            break
    assert moved, "no earlier test outcome changed any prediction"


def test_unseen_user_and_card_take_the_cold_path(smoke_events, smoke_model):
    """Rows of a user and a card unseen in TRAIN must still give valid probabilities."""
    new = smoke_events.copy()
    card_test = new[(~new["is_train"].astype(bool))]
    pick = card_test["card_id"].iloc[0]
    mask = (new["card_id"] == pick)
    new.loc[mask, "user_id"] = 999999          # unseen user, and therefore an unseen card key
    got = smoke_model.predict(new)
    expected = set(new.loc[new["scored"].astype(bool), "event_id"].astype(np.int64))
    assert set(got.index) == expected
    assert ((got > 0) & (got < 1)).all()


def test_predict_before_fit_raises():
    events = _real_events(users=(6370,))
    with pytest.raises(RuntimeError):
        OursModel(**SMOKE).predict(events)


def test_progress_logging_does_not_change_samples(smoke_events, capsys):
    """Same seed with and without progress lines must give identical posterior draws.

    The logging wrapper only reads sampler state, so it must not consume random numbers.
    """
    quiet = OursModel(**SMOKE).fit(smoke_events)
    capsys.readouterr()
    loud = OursModel(progress_every=20, **SMOKE).fit(smoke_events)
    for name in ("lam", "kappa", "sigma2", "b0", "g0"):
        np.testing.assert_array_equal(np.asarray(quiet.model_.samples[name]),
                                      np.asarray(loud.model_.samples[name]))
    err = capsys.readouterr().err
    lines = [line for line in err.splitlines() if "[ours] sweep" in line]
    assert [line.split()[3].rstrip(",") for line in lines] == ["20/60", "40/60", "60/60"]
    assert "lam=" in lines[0] and "eta" in lines[0] and "kappa=" not in lines[0]


def test_progress_is_silent_by_default(smoke_events, capsys):
    OursModel(**SMOKE).fit(smoke_events)
    assert "[ours]" not in capsys.readouterr().err
