"""Tests for fsrs_benchmark.models.fsrs_model: no leakage, py-fsrs agreement, valid output.

Why these tests: the adapter rebuilds FSRS memory state by replaying cards. A bug in
the replay order or the timestamps would silently change every prediction, and a
leak from the future would make the benchmark look better than it is. So we check
the replay against py-fsrs directly, check that changing the future cannot change
a past prediction, and check that the fit never looks at test rows.
"""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from fsrs import Card, Rating, Scheduler
from fsrs.scheduler import DEFAULT_PARAMETERS

from fsrs_benchmark import data, split
from fsrs_benchmark.tests._real import REAL_USERS, fitted_fsrs, real_events
from fsrs_benchmark.models.fsrs_model import EPOCH, MIN_INFORMATIVE_TRAIN, FSRSModel


def _synthetic_frame(user_id, n_cards=60, reviews_per_card=20, seed=0):
    """Loader-shaped frame: each card has a new-card first review, then 0-5 day gaps and random ratings."""
    rng = np.random.default_rng(seed)
    rows = []
    for card in range(n_cards):
        day = int(rng.integers(0, 30))
        for k in range(reviews_per_card):
            gap = 0 if k == 0 else int(rng.integers(0, 6))
            day += gap
            rating = int(rng.integers(1, 5))
            rows.append(dict(
                user_id=user_id, card_id=card + 100 * user_id, k=k, review_idx=0,
                day_offset=day, rating=rating, state=0 if k == 0 else 2,
                elapsed_days=-1 if k == 0 else gap,
                dt_days=np.nan if k == 0 else float(gap),
                y=int(rating > 1), is_first=(k == 0),
            ))
    # Real revlogs are chronological across cards, so review_idx must follow the day
    # order (ties keep each card's own order). Otherwise train would not be a prefix.
    frame = pd.DataFrame(rows).sort_values(["day_offset", "card_id", "k"]).reset_index(drop=True)
    frame["review_idx"] = np.arange(len(frame), dtype=np.int64)
    return frame.drop(columns="k")


def _event_table(rows):
    """Build an event table by hand from (day_offset, rating, scored) tuples for one card."""
    out = []
    for i, (day, rating, scored) in enumerate(rows):
        out.append(dict(
            event_id=1000 + i, user_id=1, card_id=7, review_idx=i, day_offset=day,
            rating=rating, elapsed_days=-1 if i == 0 else day - rows[i - 1][0],
            is_first=(i == 0), y=int(rating > 1), is_train=True, scored=scored,
        ))
    return pd.DataFrame(out)


def test_replay_matches_py_fsrs_on_hand_built_card():
    """The replay gives the same retrievability as py-fsrs on the same card, reviewed at real datetimes."""
    sched = Scheduler(enable_fuzzing=False)
    history = [(0, Rating.Good), (2, Rating.Good), (5, Rating.Hard)]
    card = Card(card_id=7)
    for day, rating in history:
        card, _ = sched.review_card(card, rating, EPOCH + timedelta(days=day))
    when = EPOCH + timedelta(days=9)
    expected = sched.get_card_retrievability(card, when)

    # The same history as events, then the probe at day 9 (scored, the last row).
    events = _event_table([(0, 3, False), (2, 3, False), (5, 2, False), (9, 3, True)])
    model = FSRSModel(optimize=False).fit(events)
    got = model.predict(events)

    assert list(got.index) == [1003]
    assert got.loc[1003] == pytest.approx(expected, abs=1e-12)

    # Independent check of the FSRS-6 forgetting curve: R = (1 + F*t/S)^(-w20), with
    # F = 0.9^(1/D) - 1 and D = -w20, t = 4 elapsed days since the day-5 review.
    w20 = DEFAULT_PARAMETERS[20]
    factor = 0.9 ** (1 / -w20) - 1
    closed_form = (1 + factor * 4 / card.stability) ** (-w20)
    assert got.loc[1003] == pytest.approx(closed_form, abs=1e-12)


def test_prediction_is_a_probability():
    events = _event_table([(0, 1, False), (1, 1, False), (2, 4, False), (30, 3, True)])
    got = FSRSModel(optimize=False).fit(events).predict(events)
    assert 1e-6 <= got.loc[1003] <= 1 - 1e-6


@pytest.mark.parametrize("optimize", [False, True])
def test_prediction_ignores_own_and_later_ratings(optimize):
    """Changing a scored event's rating, or any later rating, must not move its prediction (or any earlier one)."""
    events = split.make_split(_synthetic_frame(user_id=1), max_cards=300, seed=0)
    model = FSRSModel(optimize=optimize).fit(events)
    assert not model.fallback_  # the optimised variant must really be optimised here
    base = model.predict(events)

    # Targets: scored events that have an earlier review on the same card, so their
    # memory state depends on history. Take the middle one.
    targets = []
    for row in events[events["scored"]].itertuples():
        earlier = events[(events["card_id"] == row.card_id) & (events["review_idx"] < row.review_idx)]
        if len(earlier):
            targets.append(row)
    target = targets[len(targets) // 2]

    # Flip the rating of the target and of every later review of the same user.
    later = events["review_idx"] >= target.review_idx
    assert not events.loc[later & (events["user_id"] == target.user_id), "is_train"].any()
    mutated = events.copy()
    mutated.loc[later, "rating"] = 5 - mutated.loc[later, "rating"]

    after = model.predict(mutated)
    unchanged = events[events["scored"] & (events["user_id"] == target.user_id)
                       & (events["review_idx"] <= target.review_idx)]["event_id"]
    assert np.array_equal(base.loc[unchanged].to_numpy(), after.loc[unchanged].to_numpy())

    # Sensitivity check: changing an EARLIER review on the target's card must move it,
    # otherwise the test above could pass trivially.
    earlier_same_card = events[(events["card_id"] == target.card_id)
                               & (events["review_idx"] < target.review_idx)].tail(1)
    tweaked = events.copy()
    tweaked.loc[earlier_same_card.index, "rating"] = 5 - tweaked.loc[earlier_same_card.index, "rating"]
    moved = model.predict(tweaked)
    assert not np.isclose(moved.loc[target.event_id], base.loc[target.event_id], atol=1e-12)


def test_fit_ignores_test_rows():
    """Parameters fitted with different test ratings must be identical: fit reads TRAIN rows only."""
    events = split.make_split(_synthetic_frame(user_id=1), max_cards=300, seed=0)
    changed = events.copy()
    test_rows = ~changed["is_train"]
    changed.loc[test_rows, "rating"] = 5 - changed.loc[test_rows, "rating"]

    a = FSRSModel(optimize=True).fit(events)
    b = FSRSModel(optimize=True).fit(changed)
    assert not a.fallback_ and a.params_[1] != tuple(DEFAULT_PARAMETERS)
    assert a.params_ == b.params_


def test_sparse_user_falls_back_to_defaults():
    """A user below the optimiser's batch size keeps the default weights, and is recorded."""
    events = split.make_split(_synthetic_frame(user_id=1, n_cards=3, reviews_per_card=5), max_cards=300, seed=0)
    informative = events["is_train"] & ~events["is_first"] & (events["elapsed_days"] >= 1)
    assert informative.sum() < MIN_INFORMATIVE_TRAIN
    model = FSRSModel(optimize=True).fit(events)
    assert model.params_[1] == tuple(DEFAULT_PARAMETERS)
    assert 1 in model.fallback_
    assert 1 not in model.fit_seconds_


def test_predict_requires_fit_and_known_users():
    events = _event_table([(0, 3, False), (3, 3, True)])
    with pytest.raises(RuntimeError):
        FSRSModel().predict(events)
    model = FSRSModel(optimize=False).fit(events)
    other = events.assign(user_id=2)
    with pytest.raises(KeyError):
        model.predict(other)






@pytest.mark.parametrize("optimize", [False, True])
def test_real_users_get_a_valid_prediction_for_every_scored_event(optimize):
    events = real_events()
    model = fitted_fsrs(optimize)
    got = model.predict(events)

    expected_ids = set(events.loc[events["scored"], "event_id"].astype(int))
    assert set(got.index) == expected_ids
    assert got.index.name == "event_id"
    assert np.isfinite(got.to_numpy()).all()
    assert ((got > 0) & (got < 1)).all()
    if optimize:
        print({uid: round(s, 2) for uid, s in model.fit_seconds_.items()}, "fallback:", model.fallback_)


def test_parallel_fit_matches_serial_on_real_users():
    """n_workers=2 runs the optimiser in a process pool; it must give the same parameters
    and the same predictions as the serial fit on the cached real users."""
    events = real_events()
    # User 6370 falls back to defaults, so only 8506 would be optimised. Add a copy of
    # 8506 under a new user id, so two users really go through the process pool.
    clone = events[events["user_id"] == 8506].copy()
    clone["user_id"] = 9000001
    clone["event_id"] = clone["event_id"] - 8506 * 10**9 + 9000001 * 10**9
    events = pd.concat([events, clone], ignore_index=True)

    serial = FSRSModel(optimize=True, n_workers=1).fit(events)
    parallel = FSRSModel(optimize=True, n_workers=2).fit(events)
    assert len(serial.fit_seconds_) >= 2

    assert parallel.params_ == serial.params_
    assert parallel.fallback_ == serial.fallback_
    pd.testing.assert_series_equal(parallel.predict(events), serial.predict(events))
