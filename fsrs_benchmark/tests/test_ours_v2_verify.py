"""Adversarial leakage and fairness checks for OursV2 and the Phase 6 validation protocol.

Written by the Phase 6 verifier. Each test tries to break one leakage path:
outcomes or grades of non-fit rows changing the fit, outcomes of the scored event or
of later rows changing its prediction, and FSRS and OursV2 scoring different events.
"""
import numpy as np
import pandas as pd
import pytest

from fsrs_benchmark import split
from fsrs_benchmark.experiments.validation import make_validation_split
from fsrs_benchmark.models import fsrs_model
from fsrs_benchmark.models.fsrs_model import FSRSModel
from fsrs_benchmark.models.ours_v2 import OursV2


def _raw(n_users=4, n_cards=25, reviews=14, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(1, n_users + 1):
        for c in range(n_cards):
            day = int(rng.integers(0, 40))
            for k in range(reviews):
                gap = 0 if k == 0 else int(rng.integers(0, 8))
                day += gap
                r = int(rng.integers(1, 5))
                rows.append(dict(user_id=u, card_id=c, day_offset=day, rating=r,
                                 state=0 if k == 0 else 2, elapsed_days=-1 if k == 0 else gap,
                                 dt_days=np.nan if k == 0 else float(gap), y=int(r > 1), is_first=k == 0))
    df = pd.DataFrame(rows).sort_values(["user_id", "day_offset"], kind="stable").reset_index(drop=True)
    df["review_idx"] = df.groupby("user_id").cumcount()
    return df


@pytest.fixture(scope="module")
def val():
    return make_validation_split(split.make_split(_raw(), max_cards=1000, seed=0))


def _fit_state(m):
    return (m.theta.detach().numpy().copy(), m.tau_raw.detach().numpy().copy(), m.z.detach().numpy().copy())


def _assert_same_fit(a, b):
    for x, y in zip(_fit_state(a), _fit_state(b)):
        np.testing.assert_array_equal(x, y)


def test_fit_invariant_to_shuffled_nonfit_outcomes(val):
    """Permute (rating, y) pairs among validation/test rows: fitted parameters must not move."""
    base = OursV2(epochs=2, seed=3).fit(val)
    rng = np.random.default_rng(7)
    mod = val.copy()
    idx = mod.index[~mod["is_train"]].to_numpy()
    assert len(idx) > 0
    shuffled = mod.loc[rng.permutation(idx), ["rating", "y"]].to_numpy()
    mod.loc[idx, ["rating", "y"]] = shuffled
    assert (mod.loc[idx, "rating"].to_numpy() != val.loc[idx, "rating"].to_numpy()).any()
    _assert_same_fit(base, OursV2(epochs=2, seed=3).fit(mod))


def test_fit_invariant_to_all_nonfit_y_set_to_one(val):
    base = OursV2(epochs=2, seed=3).fit(val)
    mod = val.copy()
    mod.loc[~mod["is_train"], "y"] = 1
    _assert_same_fit(base, OursV2(epochs=2, seed=3).fit(mod))


def test_fit_ignores_y_of_same_day_and_first_rows(val):
    """Loss mask is (not first) & (elapsed >= 1) & is_train, so y of other train rows is unused by the loss."""
    base = OursV2(epochs=2, seed=3).fit(val)
    mod = val.copy()
    excluded = mod["is_train"] & (mod["is_first"] | (mod["elapsed_days"] < 1))
    assert excluded.sum() > 0
    mod.loc[excluded, "y"] = 1 - mod.loc[excluded, "y"]
    _assert_same_fit(base, OursV2(epochs=2, seed=3).fit(mod))


def test_predictions_independent_of_every_y(val):
    m = OursV2(epochs=2, seed=3).fit(val)
    base = m.predict(val)
    mod = val.copy()
    mod["y"] = 1
    pd.testing.assert_series_equal(base, m.predict(mod))
    mod["y"] = 0
    pd.testing.assert_series_equal(base, m.predict(mod))


@pytest.mark.parametrize("kwargs", [dict(), dict(process_noise=True, calibrate=True)])
def test_prediction_independent_of_own_and_future_outcomes_and_gaps(val, kwargs):
    """p(e) may depend only on e's card, the earlier rows of that card, and e's own gap.
    Randomise ratings and gaps of EVERYTHING else (other cards, other users, later rows)."""
    m = OursV2(epochs=2, seed=3, **kwargs).fit(val)
    base = m.predict(val)
    rng = np.random.default_rng(11)
    scored_rows = val[val["scored"]].sample(8, random_state=5)
    for _, e in scored_rows.iterrows():
        same_card = (val["user_id"] == e["user_id"]) & (val["card_id"] == e["card_id"])
        history = same_card & (val["review_idx"] <= e["review_idx"])
        mod = val.copy()
        other = ~history
        r = rng.integers(1, 5, size=int(other.sum()))
        mod.loc[other, "rating"] = r
        mod.loc[other, "y"] = (r > 1).astype(int)
        mod.loc[other, "elapsed_days"] = rng.integers(1, 60, size=int(other.sum()))
        after = m.predict(mod)
        assert after.loc[e["event_id"]] == pytest.approx(base.loc[e["event_id"]], abs=1e-6)


def test_history_does_affect_prediction(val):
    """Sanity: the model must actually read earlier outcomes, or the invariance tests above prove nothing."""
    m = OursV2(epochs=2, seed=3).fit(val)
    base = m.predict(val)
    changed = []
    for _, e in val[val["scored"]].sample(10, random_state=2).iterrows():
        prev = val[(val["user_id"] == e["user_id"]) & (val["card_id"] == e["card_id"])
                   & (val["review_idx"] < e["review_idx"])].sort_values("review_idx").iloc[-1]
        mod = val.copy()
        new_rating = 1 if prev["rating"] != 1 else 4
        mod.loc[mod["event_id"] == prev["event_id"], "rating"] = new_rating
        mod.loc[mod["event_id"] == prev["event_id"], "y"] = int(new_rating > 1)
        changed.append(abs(m.predict(mod).loc[e["event_id"]] - base.loc[e["event_id"]]))
    assert max(changed) > 1e-4


def test_validation_split_matches_independent_recomputation(val):
    """Rebuild the split rule from the raw make_split table and compare sets exactly."""
    full = split.make_split(_raw(), max_cards=1000, seed=0)
    train = full[full["is_train"]]
    q = train.groupby("user_id")["day_offset"].quantile(0.8, interpolation="linear")
    expect_fit = set(train.loc[train["day_offset"] < train["user_id"].map(q), "event_id"])
    expect_val = set(train["event_id"]) - expect_fit
    assert set(val.loc[val["is_train"], "event_id"]) == expect_fit
    assert set(val.loc[~val["is_train"], "event_id"]) == expect_val
    assert set(val["event_id"]) == set(train["event_id"])            # no test row survives
    expect_scored = val["event_id"].isin(expect_val) & ~val["is_first"] & (val["elapsed_days"] >= 1)
    assert set(val.loc[expect_scored.astype(bool), "event_id"]) == set(val.loc[val["scored"], "event_id"])


def test_scored_sets_identical_for_ours_and_fsrs(val):
    scored = set(val.loc[val["scored"], "event_id"])
    assert scored
    ours = OursV2(epochs=1, seed=0).fit(val).predict(val)
    fsrs = FSRSModel(optimize=False).fit(val).predict(val)
    assert set(ours.index) == scored
    assert set(fsrs.index) == scored


def test_fsrs_optimizer_sees_only_fit_rows(val, monkeypatch):
    """Spy on the per-user Optimizer call: every row it gets must be a fit row of the validation split."""
    seen = []

    def spy(user_id, train):
        seen.append(train)
        return user_id, "ok", tuple(float(w) for w in fsrs_model.DEFAULT_PARAMETERS), 0.0

    monkeypatch.setattr(fsrs_model, "_optimize_user", spy)
    monkeypatch.setattr(fsrs_model, "MIN_INFORMATIVE_TRAIN", 1)
    FSRSModel(optimize=True, n_workers=1).fit(val)
    assert len(seen) == val["user_id"].nunique()
    fit_ids = set(val.loc[val["is_train"], "event_id"])
    for train in seen:
        assert train["is_train"].all()
        assert set(train["event_id"]) <= fit_ids


def test_fit_truncates_to_first_64_reviews_per_card(monkeypatch):
    """Documents the fit cap: outcomes of train reviews beyond the 64th per card never move the fit."""
    raw = _raw(n_users=2, n_cards=6, reviews=160, seed=4)
    v = make_validation_split(split.make_split(raw, max_cards=1000, seed=0))
    base = OursV2(epochs=1, seed=1).fit(v)
    mod = v.copy()
    deep = mod["is_train"] & (mod["review_idx"] >= 0) & (mod.groupby(["user_id", "card_id"]).cumcount() >= 64)
    assert deep.sum() > 0
    mod.loc[deep, "y"] = 1 - mod.loc[deep, "y"]
    _assert_same_fit(base, OursV2(epochs=1, seed=1).fit(mod))


def test_test_row_label_perturbation_leaves_own_and_earlier_predictions(val):
    """Phase 6 verifier, criterion 1: flip y (and rating) of one TEST row y_t in the full
    make_split table. The fit must not move, p_t and every earlier prediction on its card
    must stay identical; only later predictions on that card may change."""
    full = split.make_split(_raw(), max_cards=1000, seed=0)
    m = OursV2(epochs=2, seed=3).fit(full)
    base = m.predict(full)
    test_scored = full[(~full["is_train"]) & full["scored"]]
    assert len(test_scored) > 0
    for _, t in test_scored.sample(6, random_state=9).iterrows():
        mod = full.copy()
        row_mask = mod["event_id"] == t["event_id"]
        new_rating = 1 if int(t["rating"]) != 1 else 4
        mod.loc[row_mask, "rating"] = new_rating
        mod.loc[row_mask, "y"] = int(new_rating > 1)
        m2 = OursV2(epochs=2, seed=3).fit(mod)
        _assert_same_fit(m, m2)
        after = m2.predict(mod)
        same_card = (full["user_id"] == t["user_id"]) & (full["card_id"] == t["card_id"])
        earlier_ids = full.loc[same_card & (full["review_idx"] <= t["review_idx"]), "event_id"]
        earlier_scored = [e for e in earlier_ids if e in base.index]
        assert after.loc[t["event_id"]] == pytest.approx(base.loc[t["event_id"]], abs=1e-6)
        np.testing.assert_allclose(after.loc[earlier_scored].to_numpy(),
                                   base.loc[earlier_scored].to_numpy(), atol=1e-6)
