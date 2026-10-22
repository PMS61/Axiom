"""Verifier tests for fsrs_benchmark.models.ours (Task 2.3).

Checks the adapter against the model's contract: what the sampler receives (TRAIN only),
how test sequences start (warm from the last TRAIN state, cold otherwise), and that
predictions use only earlier outcomes and the scored events are predicted exactly once.
"""
import numpy as np
import pandas as pd
import pytest

from fsrs_benchmark.models import ours
from fsrs_benchmark.models.ours import OursModel
from fsrs_benchmark.tests._real import real_events

SMOKE = dict(iters=60, burn=30, thin=4, seed=0)


@pytest.fixture(scope="module")
def events():
    return real_events()


@pytest.fixture(scope="module")
def model(events):
    return OursModel(**SMOKE).fit(events)


def _train_test_keys(events):
    train = events[events["is_train"].astype(bool)]
    test = events[~events["is_train"].astype(bool)]
    train_keys = set(zip(train["user_id"], train["card_id"]))
    test_only = [k for k in set(zip(test["user_id"], test["card_id"])) if k not in train_keys]
    return train_keys, test_only


def test_sampler_receives_only_train_rows(events, monkeypatch):
    """The DynamicRetentionModelDT constructor sees TRAIN outcomes and gaps and nothing from TEST."""
    captured = {}

    class Recorder:
        def __init__(self, sequences, **kw):
            captured["seqs"] = sequences
            self.n_users = max(s["user"] for s in sequences) + 1
            self.n_items = max(s["item"] for s in sequences) + 1

        def run(self, **kw):
            captured["run"] = kw

        def summary(self):
            return {}

    monkeypatch.setattr(ours, "DynamicRetentionModelDT", Recorder)
    OursModel(iters=5, burn=2).fit(events)
    seqs = captured["seqs"]

    ids = np.concatenate([s["event_id"] for s in seqs])
    train_ids = events.loc[events["is_train"].astype(bool), "event_id"].to_numpy(np.int64)
    assert np.array_equal(np.sort(ids), np.sort(train_ids))
    assert not np.isin(events.loc[~events["is_train"].astype(bool), "event_id"].to_numpy(np.int64), ids).any()


def test_train_sequence_fields_follow_the_spec(events, monkeypatch):
    captured = {}

    class Recorder:
        def __init__(self, sequences, **kw):
            captured["seqs"] = sequences
            self.n_users = max(s["user"] for s in sequences) + 1
            self.n_items = max(s["item"] for s in sequences) + 1

        def run(self, **kw):
            pass

        def summary(self):
            return {}

    monkeypatch.setattr(ours, "DynamicRetentionModelDT", Recorder)
    OursModel(iters=1, burn=0).fit(events)
    by_event = events.set_index("event_id")
    for s in captured["seqs"]:
        assert np.array_equal(s["r"], by_event.loc[s["event_id"], "y"].to_numpy(float))
        assert s["x"][0] == 0.0
        assert np.array_equal(s["x"][1:], s["r"][:-1])
        assert s["dt"][0] == 1.0                          # first review: NaN gap -> 1.0
        gaps = by_event.loc[s["event_id"][1:], "dt_days"].to_numpy(float)
        assert np.allclose(s["dt"][1:], gaps)
        assert bool(by_event.loc[s["event_id"][0], "is_first"])  # a train card starts at its first review
        assert list(by_event.loc[s["event_id"], "review_idx"]) == sorted(by_event.loc[s["event_id"], "review_idx"])


def test_warm_test_sequence_starts_with_gap_from_last_train_review(events, model, monkeypatch):
    captured = {}
    original = model.model_.predict

    def spy(test_seqs, **kw):
        captured["seqs"] = test_seqs
        return original(test_seqs, **kw)

    monkeypatch.setattr(model.model_, "predict", spy)
    model.predict(events)
    by_event = events.set_index("event_id")
    train_keys, _ = _train_test_keys(events)
    checked = 0
    for s in captured["seqs"]:
        first_test = by_event.loc[s["event_id"][0]]
        if (first_test["user_id"], first_test["card_id"]) not in train_keys:
            continue  # cold card: no TRAIN history, so it should not be warm
        # The previous kept review of this card is a TRAIN row (the train period is a prefix).
        prev = events[(events["user_id"] == first_test["user_id"]) & (events["card_id"] == first_test["card_id"])
                      & (events["review_idx"] < first_test["review_idx"])]
        assert len(prev) and bool(prev.iloc[-1]["is_train"])
        # dt is dt_days (raw seconds since the last kept review, in days), not the whole-day count.
        assert s["dt"][0] == pytest.approx(float(first_test["dt_days"]), abs=1e-9)
        checked += 1
    assert checked > 0


def test_predictions_are_prefix_only_on_real_cards(events, model):
    """Dropping every test row after event k leaves p_k unchanged (uses the fitted state only)."""
    full = model.predict(events)
    scored = events[events["scored"].astype(bool)]
    probes = scored.iloc[np.linspace(0, len(scored) - 1, num=12, dtype=int)]
    for row in probes.itertuples():
        keep = events.drop(index=events[(events["user_id"] == row.user_id) & (events["card_id"] == row.card_id)
                                        & (events["review_idx"] > row.review_idx)].index)
        got = model.predict(keep)
        assert got.loc[row.event_id] == pytest.approx(full.loc[row.event_id], abs=1e-12)


def test_own_and_later_outcomes_do_not_move_p(events, model):
    base = model.predict(events)
    scored = events[events["scored"].astype(bool)]
    for row in scored.iloc[np.linspace(0, len(scored) - 1, num=8, dtype=int)].itertuples():
        mutated = events.copy()
        later = (mutated["user_id"] == row.user_id) & (mutated["card_id"] == row.card_id) \
            & (mutated["review_idx"] >= row.review_idx)
        mutated.loc[later, "y"] = 1 - mutated.loc[later, "y"]
        after = model.predict(mutated)
        assert after.loc[row.event_id] == pytest.approx(base.loc[row.event_id], abs=1e-12)


def test_unscored_test_rows_are_kept_as_history(events, model):
    """An unscored test row before a scored one must still move the scored prediction."""
    base = model.predict(events)
    unscored = events[(~events["is_train"].astype(bool)) & (~events["scored"].astype(bool))]
    moved = 0
    for row in unscored.itertuples():
        later_scored = events[(events["user_id"] == row.user_id) & (events["card_id"] == row.card_id)
                              & events["scored"].astype(bool) & (events["review_idx"] > row.review_idx)]
        if not len(later_scored):
            continue
        mutated = events.copy()
        mutated.loc[mutated["event_id"] == row.event_id, "y"] = 1 - row.y
        after = model.predict(mutated)
        target = int(later_scored.iloc[0]["event_id"])
        if not np.isclose(after.loc[target], base.loc[target], atol=1e-9):
            moved += 1
        if moved >= 3:
            break
    assert moved >= 1


def test_every_scored_event_predicted_exactly_once(events, model):
    got = model.predict(events)
    expected = events.loc[events["scored"].astype(bool), "event_id"].astype(np.int64)
    assert got.index.is_unique
    assert set(got.index) == set(expected)
    assert len(got) == len(expected)
    assert ((got > 0) & (got < 1)).all()


def test_fit_and_predict_ids_agree_for_train_cards(events, model):
    """Every TRAIN card is a warm key in the model, and its dense ids are the ones used at fit."""
    train_keys, _ = _train_test_keys(events)
    for key in train_keys:
        assert model._items[key] < model.model_.n_items
    dense = model._dense_ids(events)
    for key in train_keys:
        rows = dense[(dense["user_id"] == key[0]) & (dense["card_id"] == key[1])]
        assert (rows["item"] == model._items[key]).all()


def test_fit_is_deterministic_for_fixed_seed(events):
    a = OursModel(iters=30, burn=15, seed=3).fit(events).predict(events)
    b = OursModel(iters=30, burn=15, seed=3).fit(events).predict(events)
    assert np.array_equal(a.to_numpy(), b.to_numpy())


def test_test_only_cards_take_the_cold_path(events):
    m = OursModel(iters=30, burn=15, seed=1).fit(events)
    _, test_only = _train_test_keys(events)
    assert test_only, "expected test-only cards"
    dense = m._dense_ids(events)
    key_set = set(test_only)
    is_test_only = np.array([(u, c) in key_set for u, c in zip(events["user_id"], events["card_id"])])
    assert (dense.loc[is_test_only, "item"] == m._n_items).all(), "test-only cards must get the padding index"
    base = m.predict(events)
    m.model_.samples["beta"][:] += 5.0           # change every fitted item effect
    after = m.predict(events)
    cold_ids = [int(i) for i in events.loc[is_test_only, "event_id"] if int(i) in base.index]
    assert cold_ids
    assert np.allclose(after.loc[cold_ids].to_numpy(), base.loc[cold_ids].to_numpy(), atol=1e-12)


def test_predict_before_fit_raises(events):
    with pytest.raises(RuntimeError):
        OursModel(**SMOKE).predict(events)
