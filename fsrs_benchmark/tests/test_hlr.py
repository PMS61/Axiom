"""Tests for fsrs_benchmark.models.hlr: gradient, recovery, no-leakage features, valid output.

Why these tests: HLR is the simplest baseline, so a silent error here (a wrong
gradient, features that peek at the outcome they predict, a sign flip in the
half-life) would make the comparison meaningless. Each property is checked directly.
"""

import numpy as np
import pandas as pd
import pytest

from fsrs_benchmark import data, split
from fsrs_benchmark.tests._real import REAL_USERS, fitted_hlr, real_events
from fsrs_benchmark.models.hlr import HLRModel, features, observed_half_life, sgd_update



def _event_table(rows, user_id=1, card_id=7):
    """Event table for one card from (day_offset, y, scored, is_train) tuples; first row is new."""
    out = []
    for i, (day, y, scored) in enumerate(rows):
        gap = day - rows[i - 1][0] if i else -1
        out.append(dict(
            event_id=user_id * 10**9 + i, user_id=user_id, card_id=card_id, review_idx=i,
            day_offset=day, rating=3 if y else 1, y=int(y), dt_days=float(gap) if i else np.nan,
            elapsed_days=gap, is_first=(i == 0), is_train=True, scored=scored,
        ))
    return pd.DataFrame(out)


def _synthetic_events(theta_true, n_cards=150, n_reviews=12, seed=0):
    """Simulate reviews from the HLR model itself, so theta_true is the ground truth.

    Gaps are 1-20 days. The outcome is Bernoulli(2^(-dt/h)) with h = 2^(theta . x),
    where x uses the simulated history so far. Rows are then ordered by day, and the
    first 80% of days are TRAIN.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for card in range(n_cards):
        day = 0
        n_right = n_wrong = 0
        for k in range(n_reviews):
            if k == 0:
                y, dt = 1, np.nan
            else:
                dt = float(rng.integers(1, 21))
                day += int(dt)
                x = np.array([np.sqrt(1 + n_right), np.sqrt(1 + n_wrong), 1.0])
                h = 2.0 ** (x @ theta_true)
                y = int(rng.random() < 2.0 ** (-dt / h))
            n_right += y
            n_wrong += 1 - y
            rows.append(dict(user_id=1, card_id=card, day_offset=day if k else 0, y=y,
                             dt_days=dt, elapsed_days=int(dt) if k else -1,
                             is_first=(k == 0), rating=3 if y else 1))
    frame = pd.DataFrame(rows).sort_values(["day_offset", "card_id"]).reset_index(drop=True)
    frame["review_idx"] = np.arange(len(frame), dtype=np.int64)
    frame["state"] = 2
    frame["rating"] = frame["rating"].astype(np.int64)
    cutoff = np.quantile(frame["day_offset"], 0.8)
    frame["cutoff_day"] = cutoff
    frame["is_train"] = frame["day_offset"] < cutoff
    frame["scored"] = (~frame["is_train"]) & (~frame["is_first"]) & (frame["elapsed_days"] >= 1)
    frame["event_id"] = frame["user_id"] * 10**9 + frame["review_idx"]
    return frame


def test_features_match_hand_computed_counts():
    events = _event_table([(0, 1, False), (1, 0, False), (2, 1, False), (5, 1, True)])
    X = features(events)
    assert X.loc[(10**9)].tolist() == pytest.approx([1.0, 1.0, 1.0])            # no history
    assert X.loc[(10**9 + 1)].tolist() == pytest.approx([np.sqrt(2), 1.0, 1.0])    # one right
    assert X.loc[(10**9 + 2)].tolist() == pytest.approx([np.sqrt(2), np.sqrt(2), 1.0])  # right + wrong
    assert X.loc[(10**9 + 3)].tolist() == pytest.approx([np.sqrt(3), np.sqrt(2), 1.0])


def test_features_use_only_past_history():
    """A row's features must not change when its own outcome or any later outcome changes."""
    events = _event_table([(0, 1, False), (1, 0, False), (2, 1, False), (5, 1, True), (9, 0, False)])
    base = features(events)
    for target in (10**9 + 1, 10**9 + 2, 10**9 + 3):
        changed = events.copy()
        later = changed["event_id"] >= target
        changed.loc[later, "y"] = 1 - changed.loc[later, "y"]
        after = features(changed)
        assert np.array_equal(base.loc[target].to_numpy(), after.loc[target].to_numpy())


def test_features_are_per_card_per_user():
    a = _event_table([(0, 1, False), (1, 1, False)], user_id=1, card_id=5)
    b = _event_table([(0, 0, False), (1, 0, False)], user_id=2, card_id=5)
    X = features(pd.concat([a, b], ignore_index=True))
    # Same card id in another user must not share history.
    assert X.loc[2 * 10**9 + 1].tolist() == pytest.approx([1.0, np.sqrt(2), 1.0])


def test_one_sgd_step_matches_hand_computed_official_update():
    """One train_update step with w=0, x=[1, 0, 1], t=1, y=1, computed by hand from the official formulas."""
    ln2 = np.log(2.0)
    w, fcounts = [0.0, 0.0, 0.0], [0, 0, 0]
    x, t, y = [1.0, 0.0, 1.0], 1.0, 1.0
    h_obs = observed_half_life(np.array([t]), np.array([y]))[0]   # clipped to 274
    lrate, hlwt, l2wt = 0.001, 0.01, 0.1
    sgd_update(w, fcounts, x, t, y, h_obs, lrate, hlwt, l2wt)

    h = 1.0                                     # 2^0, inside the clip bounds
    p = 0.5                                     # 2^(-1/1)
    p_obs = 1 - 1e-4                            # pclip(1), as in the official code
    dlp = 2 * (p - p_obs) * ln2 ** 2 * p * (t / h)
    dlh = 2 * (h - h_obs) * ln2 * h
    rate0 = (1 / (1 + p_obs)) * lrate / np.sqrt(1 + 0)
    # The official update is sequential within a feature: the L2 term uses the weight
    # already changed by the loss terms (w_k -= rate*l2wt*w_k / sigma^2 after them).
    after_loss = 0.0 - rate0 * dlp * 1.0 - rate0 * hlwt * dlh * 1.0
    expected_w0 = after_loss - rate0 * 0.1 * after_loss
    expected_w2 = expected_w0                  # same features, same counts
    assert w[0] == pytest.approx(expected_w0, rel=1e-12)
    assert w[2] == pytest.approx(expected_w2, rel=1e-12)
    assert w[1] == 0.0                          # feature with x=0 gets no update
    assert fcounts == [1, 1, 1]


def test_observed_half_life_is_clipped():
    dt = np.array([1.0, 1.0, 1000.0])
    y = np.array([1.0, 0.0, 0.0])
    h = observed_half_life(dt, y)
    assert h[0] == pytest.approx(274.0)            # recall gives a huge h_obs, clipped to max
    assert h[1] == pytest.approx(1.0 / 13.287712, rel=1e-5)
    assert h[2] == pytest.approx(1000.0 / 13.287712, rel=1e-5)  # a miss with a long gap: in range


def test_recovers_true_theta_on_synthetic_data():
    theta_true = np.array([1.0, -0.6, 2.0])
    events = _synthetic_events(theta_true)
    # p-only loss (hlwt=0, no L2) and a larger lrate: with the official rate schedule the
    # bias converges slowly, so this checks recovery to a loose tolerance, not exact values.
    model = HLRModel(hlwt=0.0, l2wt=0.0, lrate=0.5).fit(events)
    assert np.all(np.abs(model.theta_ - theta_true) < 0.75), model.theta_
    assert model.theta_[1] < 0 < model.theta_[0]


def test_fit_ignores_test_rows():
    events = _synthetic_events(np.array([1.0, -0.6, 2.0]))
    changed = events.copy()
    test_rows = ~changed["is_train"].astype(bool)
    changed.loc[test_rows, "y"] = 1 - changed.loc[test_rows, "y"]
    a = HLRModel().fit(events)
    b = HLRModel().fit(changed)
    assert np.array_equal(a.theta_, b.theta_)
    assert a.epochs_ == b.epochs_


def test_prediction_matches_formula_and_is_clipped():
    events = _event_table([(0, 1, False), (2, 1, False), (5, 0, False), (12, 1, True)])
    model = HLRModel()
    model.theta_ = np.array([0.7, -0.4, 1.5])
    got = model.predict(events)
    x = features(events).loc[(10**9 + 3)].to_numpy()
    h = 2.0 ** (x @ model.theta_)
    expected = np.clip(2.0 ** (-7.0 / h), 1e-4, 1 - 1e-4)
    assert list(got.index) == [10**9 + 3]
    assert got.loc[10**9 + 3] == pytest.approx(expected, abs=1e-12)


def test_prediction_ignores_own_and_later_outcomes():
    events = _synthetic_events(np.array([1.0, -0.6, 2.0]))
    model = HLRModel().fit(events)
    base = model.predict(events)
    scored = events.loc[events["scored"].astype(bool)]
    target = scored.iloc[len(scored) // 2]
    mutated = events.copy()
    later = (mutated["user_id"] == target["user_id"]) & (mutated["card_id"] == target["card_id"]) \
        & (mutated["review_idx"] >= target["review_idx"])
    mutated.loc[later, "y"] = 1 - mutated.loc[later, "y"]
    after = model.predict(mutated)
    assert after.loc[target["event_id"]] == base.loc[target["event_id"]]


def test_predict_requires_fit():
    with pytest.raises(RuntimeError):
        HLRModel().predict(_event_table([(0, 1, False), (3, 1, True)]))


def _real_events():
    return real_events()


@pytest.mark.parametrize("hlwt", [0.01, 0.0])
def test_real_users_get_a_valid_prediction_for_every_scored_event(hlwt):
    events = _real_events()
    model = fitted_hlr(hlwt)
    got = model.predict(events)
    expected_ids = set(events.loc[events["scored"], "event_id"].astype(int))
    assert set(got.index) == expected_ids
    assert np.isfinite(got.to_numpy()).all()
    assert ((got > 0) & (got < 1)).all()


def test_fit_on_empty_train_set_emits_no_runtime_warning():
    """No train rows means no loss to average: the fit must not warn (the fix is in the code)."""
    import warnings

    events = real_events()
    events["is_train"] = False
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        model = HLRModel(max_epochs=5).fit(events)
    assert model.epochs_ == 0 and model.loss_history_ == []
