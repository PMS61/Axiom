"""Verifier tests for fsrs_benchmark.models.fsrs_model (Task 2.1).

These go beyond the builder's tests: an independent replay written here against
py-fsrs on real cards, a check that the optimiser only ever receives TRAIN logs,
a truncation test (dropping the future must not move past predictions), edge cases
(zero train rows, single-review cards, same-day repeats), and timing on real users.
"""
import time
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
from fsrs import Card, Rating, Scheduler
from fsrs.scheduler import DEFAULT_PARAMETERS

from fsrs_benchmark import data, split
from fsrs_benchmark.models import fsrs_model
from fsrs_benchmark.tests._real import REAL_USERS, fitted_fsrs, real_events
from fsrs_benchmark.models.fsrs_model import EPOCH, FSRSModel, _timestamp



def _real_events():
    return real_events()


def _fitted_optimised_real():
    return fitted_fsrs(True)


def _fresh_day_only_replay(card_rows):
    """Independent reference: one card, py-fsrs directly, timestamps = EPOCH + day_offset only.

    card_rows: list of (day_offset, rating, scored, event_id) in review order.
    Returns {event_id: retrievability before the review}. No seconds offset is used,
    so same-day reviews share a timestamp; py-fsrs only looks at whole-day gaps, so
    the result should be identical to the adapter's.
    """
    sched = Scheduler(enable_fuzzing=False)
    card = Card(card_id=1)
    out = {}
    for day, rating, scored, event_id in card_rows:
        when = EPOCH + timedelta(days=int(day))
        if scored:
            out[event_id] = sched.get_card_retrievability(card, when)
        card, _ = sched.review_card(card, Rating(int(rating)), when)
    return out


def _event_table(rows, user_id=1, card_id=7, first_id=1000):
    """Hand-built events from (day_offset, rating, scored, is_train) tuples for one card."""
    out = []
    for i, (day, rating, scored, is_train) in enumerate(rows):
        out.append(dict(
            event_id=first_id + i, user_id=user_id, card_id=card_id, review_idx=i,
            day_offset=day, rating=rating,
            elapsed_days=-1 if i == 0 else day - rows[i - 1][0],
            is_first=(i == 0), y=int(rating > 1), is_train=is_train, scored=scored,
        ))
    return pd.DataFrame(out)


# --- timestamps and rating mapping -------------------------------------------------

def test_timestamp_gap_days_equal_day_offset_difference():
    """For any two reviews in a card (positions strictly increasing), py-fsrs' .days equals the day difference."""
    rng = np.random.default_rng(1)
    days = np.sort(rng.integers(0, 3000, size=500))
    for i in range(len(days)):
        a, b = i, min(i + 1, len(days) - 1)
        t1 = _timestamp(days[a], a)
        t2 = _timestamp(days[b], b)
        assert t2 > t1 or a == b
        assert (t2 - t1).days == days[b] - days[a]


def test_timestamps_are_strictly_increasing_within_card_including_same_day():
    times = [_timestamp(5, p) for p in range(50)] + [_timestamp(6, p) for p in range(50, 60)]
    assert all(t2 > t1 for t1, t2 in zip(times, times[1:]))


def test_timestamp_rejects_too_many_reviews():
    with pytest.raises(ValueError):
        _timestamp(0, fsrs_model.MAX_REVIEWS_PER_CARD)


def test_rating_values_map_to_py_fsrs_enum():
    assert [Rating(r).name for r in (1, 2, 3, 4)] == ["Again", "Hard", "Good", "Easy"]
    # Again must be a lapse (lower stability than Easy from the same start).
    sched = Scheduler(enable_fuzzing=False)
    stab = {}
    for r in (1, 4):
        c, _ = sched.review_card(Card(card_id=1), Rating(r), EPOCH)
        stab[r] = c.stability
    assert stab[1] < stab[4]


# --- replay agrees with an independent reference -----------------------------------

def test_replay_matches_fresh_reference_on_real_cards():
    """Real cards, optimize=False: the adapter equals a separate py-fsrs replay with day-only timestamps."""
    events = _real_events()
    model = FSRSModel(optimize=False).fit(events)
    got = model.predict(events)

    checked = 0
    ordered = events.sort_values(["user_id", "card_id", "review_idx"])
    for (_, _), rows in ordered.groupby(["user_id", "card_id"], sort=True):
        if not rows["scored"].any():
            continue
        ref_rows = [(r.day_offset, r.rating, r.scored, r.event_id) for r in rows.itertuples()]
        ref = _fresh_day_only_replay(ref_rows)
        for event_id, expected in ref.items():
            assert got.loc[event_id] == pytest.approx(expected, abs=1e-12), event_id
            checked += 1
    assert checked == int(events["scored"].sum())


def test_same_day_repeats_do_not_change_result_vs_reference():
    """A card with same-day repeats: adapter (seconds offset) equals the day-only reference."""
    rows = [(0, 3, False, True), (0, 1, False, True), (0, 3, False, True), (2, 3, False, True),
            (2, 2, False, True), (5, 4, True, False), (5, 3, True, False), (9, 3, True, False)]
    events = _event_table(rows)
    got = FSRSModel(optimize=False).fit(events).predict(events)
    ref = _fresh_day_only_replay([(r.day_offset, r.rating, r.scored, r.event_id) for r in events.itertuples()])
    assert list(got.index) == sorted(ref)
    for event_id in ref:
        # predict() clips to [1e-6, 1-1e-6]; compare against the clipped reference.
        assert got.loc[event_id] == pytest.approx(min(max(ref[event_id], 1e-6), 1 - 1e-6), abs=1e-12)


def test_same_day_order_matters_and_is_respected():
    """Swapping two same-day ratings changes the stability, so the replay must follow review_idx order."""
    base = [(0, 3, False, True), (0, 1, False, True), (3, 3, True, False)]
    swapped = [(0, 1, False, True), (0, 3, False, True), (3, 3, True, False)]
    a = FSRSModel(optimize=False).fit(_event_table(base)).predict(_event_table(base))
    b = FSRSModel(optimize=False).fit(_event_table(swapped)).predict(_event_table(swapped))
    assert not np.isclose(a.iloc[0], b.iloc[0], atol=1e-9)


# --- no leakage ---------------------------------------------------------------------

def test_dropping_future_rows_does_not_change_past_predictions_real_users():
    """Truncating each user's history after event k leaves p_k unchanged (uses params fitted on full data)."""
    events = _real_events()
    model = _fitted_optimised_real()
    full = model.predict(events)
    for user_id in REAL_USERS:
        user = events[events["user_id"] == user_id]
        scored = user[user["scored"]]
        probes = scored.iloc[np.linspace(0, len(scored) - 1, num=min(6, len(scored)), dtype=int)]
        for row in probes.itertuples():
            trunc = user[user["review_idx"] <= row.review_idx]
            got = model.predict(trunc)
            assert got.loc[row.event_id] == pytest.approx(full.loc[row.event_id], abs=1e-12)


def test_optimizer_receives_only_train_logs(monkeypatch):
    """The Optimizer object gets exactly the TRAIN rows: no test review reaches the fit."""
    seen = {}
    monkeypatch.setattr(fsrs_model, "MIN_INFORMATIVE_TRAIN", 1)  # let the tiny user reach the optimiser

    class Recorder:
        def __init__(self, logs):
            seen["logs"] = list(logs)

        def compute_optimal_parameters(self):
            return list(DEFAULT_PARAMETERS)

    monkeypatch.setattr(fsrs_model, "Optimizer", Recorder)
    rows = [(d, r, False, d < 10) for d, r in [(0, 3), (2, 1), (4, 3), (6, 4), (12, 2), (20, 3)]]
    events = _event_table(rows)
    FSRSModel(optimize=True).fit(events)
    train_days = sorted(int(d) for d, _, _, t in rows if t)
    got_days = sorted(int(l.review_datetime.date().toordinal() - EPOCH.date().toordinal()) for l in seen["logs"])
    assert got_days == train_days
    assert len(seen["logs"]) == int(events["is_train"].sum())


def test_fit_on_real_users_sends_only_train_rows(monkeypatch):
    events = _real_events()
    sent = []
    model_ref = _fitted_optimised_real()  # which users reach the optimiser for real
    reached = {u for u in REAL_USERS if u not in model_ref.fallback_}

    class Recorder:
        def __init__(self, logs):
            sent.extend(logs)

        def compute_optimal_parameters(self):
            return list(DEFAULT_PARAMETERS)

    monkeypatch.setattr(fsrs_model, "Optimizer", Recorder)
    FSRSModel(optimize=True).fit(events)
    expected = int(events[events["is_train"] & events["user_id"].isin(reached)].shape[0])
    assert len(sent) == expected
    assert reached  # at least one real user reaches the optimiser


# --- edge cases ---------------------------------------------------------------------

def test_card_with_only_first_review_is_harmless():
    events = pd.concat([
        _event_table([(0, 3, False, False)], card_id=8, first_id=2000),
        _event_table([(0, 3, False, True), (4, 3, True, False)], card_id=9, first_id=3000),
    ], ignore_index=True)
    got = FSRSModel(optimize=False).fit(events).predict(events)
    assert list(got.index) == [3001]


def test_user_with_zero_train_rows_falls_back_and_predicts():
    events = _event_table([(0, 3, False, False), (3, 1, False, False), (9, 3, True, False)])
    model = FSRSModel(optimize=True).fit(events)
    assert model.params_[1] == tuple(DEFAULT_PARAMETERS)
    assert 1 in model.fallback_
    got = model.predict(events)
    assert list(got.index) == [1002]
    assert 0 < got.loc[1002] < 1


def test_fallback_reason_for_sparse_user_is_recorded():
    events = _event_table([(0, 3, False, True), (3, 3, True, False)])
    model = FSRSModel(optimize=True).fit(events)
    assert "informative" in model.fallback_[1]


def test_predict_on_real_users_is_fast_and_fallbacks_recorded():
    events = _real_events()
    model = _fitted_optimised_real()
    t1 = time.perf_counter()
    got = model.predict(events)
    t2 = time.perf_counter()
    print(f"predict {t2 - t1:.2f}s fallback={model.fallback_} "
          f"fit_seconds={ {k: round(v, 1) for k, v in model.fit_seconds_.items()} }")
    assert len(got) == int(events["scored"].sum())
    assert t2 - t1 < 10
    assert set(model.params_) == set(REAL_USERS)
