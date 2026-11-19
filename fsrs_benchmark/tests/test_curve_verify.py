"""Verifier tests for the Phase 8 Curve model (fsrs_benchmark/srsb/curve.py). Synthetic users only.

Stronger leak rules than test_curve.py: later reviews may change any day, elapsed time, duration or rating
(other cards or the same card, same day or later) and row i's TIME / CARD_I / FROM_J values and its online
prediction must not move; other cards' reviews may change at any position and TIME / CARD_I must not move;
split guards; and the previous-review index.
"""
import numpy as np
import pandas as pd
import pytest

from fsrs_benchmark.models import gbdt as G
from fsrs_benchmark.srsb import curve as K
from fsrs_benchmark.srsb import run_gbdt as R
from fsrs_benchmark.srsb import val_plus as V
from fsrs_benchmark.srsb.features_plus import plus_features


def _raw_user(seed, n_cards=12, reviews=10):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_cards):
        day = int(rng.integers(0, 20))
        for k in range(reviews):
            if k == 0:
                el_d, el_s = -1, -1
            else:
                el_d = int(rng.integers(1, 8))
                day += el_d
                el_s = el_d * 86400 + int(rng.integers(0, 86400))
            rows.append(dict(card_id=c, day_offset=day, rating=int(rng.integers(1, 5)),
                             state=0 if k == 0 else 2, duration=int(rng.integers(1000, 20000)),
                             elapsed_days=el_d, elapsed_seconds=el_s))
    return pd.DataFrame(rows).sort_values(["day_offset", "card_id"], kind="stable").reset_index(drop=True)


def _features(raw):
    st = R.review_stream(raw, 10 ** 6)
    base = G.base_features(st, span=20, user_group="row")[R.FEATURE_COLS]
    return st, pd.concat([base, plus_features(st, None, None)], axis=1)


def _scored(st):
    return st.index.to_numpy()[~st["is_first"].to_numpy()].astype(np.int64)


def _valid(raw):
    return raw["rating"].isin([1, 2, 3, 4]).to_numpy()


@pytest.mark.parametrize("seed", [11, 12, 13])
def test_later_reviews_change_nothing_of_row_i(seed):
    """Every review strictly after j (any card, any day field, any duration or rating), except row i itself,
    may change: row i's full curve vector and its online prediction must not move."""
    raw = _raw_user(seed)
    st, feats = _features(raw)
    keys = _scored(st)
    X, prev, th = K.curve_matrix(feats, st, keys)
    y = (st.loc[keys, "rating"].to_numpy() > 1).astype(float)
    p0 = K.curve_online(-0.5 * np.log1p(X[:, K.CURVE_COLS.index("el")]), X[:, K.EWM50_IX], X[:, K.RATE_IX],
                        y, th, prev)
    rng = np.random.default_rng(seed)
    ev_id = np.arange(1, len(raw) + 1)
    for k in np.linspace(2, len(keys) - 2, 6).astype(int):
        i, j = keys[k], prev[k]
        assert j < i
        r2 = raw.copy()
        m = (ev_id > j) & (ev_id != i) & _valid(raw)
        n = int(m.sum())
        r2.loc[m, "rating"] = rng.integers(1, 5, n)
        r2.loc[m, "duration"] = rng.integers(1000, 20000, n)
        r2.loc[m, "day_offset"] = r2.loc[m, "day_offset"] + rng.integers(0, 4, n)     # same day or later
        r2.loc[m, "elapsed_days"] = rng.integers(1, 9, n)
        r2.loc[m, "elapsed_seconds"] = rng.integers(0, 9 * 86400, n)
        st2, f2 = _features(r2)
        assert np.array_equal(_scored(st2), keys)
        X2, prev2, th2 = K.curve_matrix(f2, st2, keys)
        assert np.array_equal(prev2, prev)
        assert np.array_equal(X2[k], X[k], equal_nan=True), f"row {k}: a column of row i moved"
        y2 = (st2.loc[keys, "rating"].to_numpy() > 1).astype(float)
        p2 = K.curve_online(-0.5 * np.log1p(X2[:, K.CURVE_COLS.index("el")]), X2[:, K.EWM50_IX],
                            X2[:, K.RATE_IX], y2, th2, prev2)
        assert p2[k] == p0[k], f"row {k}: online prediction moved"


@pytest.mark.parametrize("seed", [21, 22])
def test_other_cards_change_nothing_of_time_or_card_columns(seed):
    """Reviews of OTHER cards, at any position (before or after j), must not move row i's TIME or CARD_I
    values (including since_first_j). FROM_J may move and is not checked here."""
    raw = _raw_user(seed)
    st, feats = _features(raw)
    keys = _scored(st)
    X, prev, _ = K.curve_matrix(feats, st, keys)
    cols = [K.CURVE_COLS.index(c) for c in K.TIME_COLS + K.CARD_I_COLS + K.CARD_I_DERIVED]
    rng = np.random.default_rng(seed)
    card = raw["card_id"].to_numpy()
    for k in (0, len(keys) // 3, len(keys) - 3):
        ci = int(st.loc[keys[k], "card_id"])
        r2 = raw.copy()
        m = (card != ci) & _valid(raw)
        n = int(m.sum())
        r2.loc[m, "rating"] = rng.integers(1, 5, n)
        r2.loc[m, "duration"] = rng.integers(1000, 20000, n)
        r2.loc[m, "day_offset"] = r2.loc[m, "day_offset"] + rng.integers(0, 4, n)
        st2, f2 = _features(r2)
        X2, _, _ = K.curve_matrix(f2, st2, keys)
        assert np.array_equal(X2[k, cols], X[k, cols], equal_nan=True), f"row {k}: other card moved TIME/CARD_I"


def test_prev_is_the_immediately_previous_review_of_the_same_card():
    raw = _raw_user(31, n_cards=8, reviews=9)
    st, feats = _features(raw)
    keys = _scored(st)
    _, prev, _ = K.curve_matrix(feats, st, keys)
    card = st["card_id"].to_numpy()
    pos = {r: p for p, r in enumerate(st.index.to_numpy())}
    for i, j in zip(keys, prev):
        assert j < i
        assert st.loc[j, "card_id"] == st.loc[i, "card_id"]
        between = st.index.to_numpy()[(st.index.to_numpy() > j) & (st.index.to_numpy() < i)]
        assert not np.any(card[[pos[b] for b in between]] == st.loc[i, "card_id"]), "a same-card review skipped"


@pytest.mark.xfail(strict=True, reason="LOW: since_first_j = since_first(i) - el(i) is exact only while "
                   "elapsed_days equals the day difference; it is checked on users 2001 and 8002 only")
def test_since_first_j_does_not_depend_on_elapsed_time_of_row_i():
    raw = _raw_user(41)
    st, feats = _features(raw)
    keys = _scored(st)
    X, prev, _ = K.curve_matrix(feats, st, keys)
    k = 5
    i = keys[k]
    r2 = raw.copy()
    pos_i = int(np.where(raw.index.to_numpy() == i - 1)[0][0])   # review_th = raw position + 1
    r2.loc[pos_i, "elapsed_days"] = r2.loc[pos_i, "elapsed_days"] + 3    # t changes, day_offset does not
    st2, f2 = _features(r2)
    X2, _, _ = K.curve_matrix(f2, st2, keys)
    c = K.CURVE_COLS.index(K.SINCE_FIRST_J)
    assert X2[k, c] == X[k, c]


def test_split_guards():
    with pytest.raises(ValueError, match="test set"):
        V.check_users([2001], [1], False)
    with pytest.raises(ValueError, match="validation"):
        V.check_users([8005], [8001], False)
    with pytest.raises(ValueError, match="overlap"):
        V.check_users([2001], [2001], False)
    V.check_users([2001, 2300], [8001, 8100], False)      # the intended split passes
    assert K.load_users("2001-2300")[0] == 2001 and K.load_users("2001-2300")[-1] == 2300
    assert K.DEFAULT_MAX_ROUNDS == 10000 and K.DEFAULT_MAX_TRAIN_ROWS == 12_000_000
    assert K.CURVE_PARAMS["num_leaves"] == 63 and K.CURVE_PARAMS["min_data_in_leaf"] == 200
    assert K.CURVE_PARAMS["learning_rate"] == 0.05 and K.CURVE_PARAMS["lambda_l2"] == 1.0
    assert K.ONLINE_LR == 0.003 and K.ONLINE_LAM == 1e-3
    assert np.array_equal(K.W0, np.array([1.0, 0.0, 0.0, 0.0]))
