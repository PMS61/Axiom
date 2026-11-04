"""Tests for the LagLogit+dt adapter: every scored event predicted, and no leakage."""
import numpy as np
import pytest

from fsrs_benchmark.models.laglogit import LagLogitModel
from fsrs_benchmark.tests._synthetic_events import synthetic_events


def _scored_ids(events):
    return set(events.loc[events["scored"], "event_id"].astype(int))


def test_every_scored_event_is_predicted_once():
    events = synthetic_events()
    p = LagLogitModel().fit(events).predict(events)
    assert p.index.is_unique
    assert set(p.index) == _scored_ids(events)
    assert ((p >= 1e-6) & (p <= 1 - 1e-6)).all()


def test_cards_without_train_rows_are_still_predicted():
    events = synthetic_events()
    cold = events.copy()
    cold.loc[cold["card_id"] == 0, "is_train"] = False  # card 0 of every user is test-only
    p = LagLogitModel().fit(cold).predict(cold)
    assert set(p.index) == _scored_ids(cold)


def test_prediction_ignores_own_and_later_outcomes():
    events = synthetic_events(seed=3)
    model = LagLogitModel().fit(events)
    base = model.predict(events)

    scored = events[events["scored"]].sort_values(["user_id", "card_id", "review_idx"])
    target = scored.iloc[len(scored) // 2]
    later = events[(events["user_id"] == target["user_id"]) & (events["card_id"] == target["card_id"])
                   & (events["review_idx"] >= target["review_idx"])]
    changed = events.copy()
    changed.loc[later.index, "y"] = 1 - changed.loc[later.index, "y"]  # flip the target and all later

    after = model.predict(changed)
    earlier = events[(events["user_id"] == target["user_id"]) & (events["card_id"] == target["card_id"])
                     & (events["review_idx"] < target["review_idx"]) & events["scored"]]
    assert np.isclose(after[target["event_id"]], base[target["event_id"]], rtol=0, atol=1e-12)
    for eid in earlier["event_id"]:
        assert np.isclose(after[eid], base[eid], rtol=0, atol=1e-12)


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        LagLogitModel().predict(synthetic_events())
