"""Tests for fsrs_benchmark/srsb/curve.py (Phase 8, GBDT-Curve; spec paper/docs/CURVE_SPEC.md).

Synthetic users only (no data files): column classification, TIME helper vs the feature frame, the leak rule
(reviews strictly after j change nothing of row i's CARD_I / FROM_J values or its online prediction), w0 before
the first update, and the monotone constraint of a trained curve model.
"""
import numpy as np
import pandas as pd
import pytest

from fsrs_benchmark.models import gbdt as G
from fsrs_benchmark.srsb import curve as K
from fsrs_benchmark.srsb import run_gbdt as R
from fsrs_benchmark.srsb import val_plus as V
from fsrs_benchmark.srsb.features_plus import plus_features


def _raw_user(seed=0, n_cards=12, reviews=10):
    """A revlog-shaped frame in chronological file order. elapsed_days equals the day difference per card,
    as in the real data (checked on user 2001)."""
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
    df = pd.DataFrame(rows).sort_values(["day_offset", "card_id"], kind="stable").reset_index(drop=True)
    return df


def _features(raw):
    st = R.review_stream(raw, 10 ** 6)
    base = G.base_features(st, span=20, user_group="row")[R.FEATURE_COLS]
    feats = pd.concat([base, plus_features(st, None, None)], axis=1)
    return st, feats


def _scored_keys(st):
    return st.index.to_numpy()[~st["is_first"].to_numpy()].astype(np.int64)


def _ix(c):
    return K.CURVE_COLS.index(c)


def _toy_margin(X):
    """A fixed stand-in for the GBDT margin: decreasing in log gap, increasing in the card's rate."""
    return (-0.5 * np.nan_to_num(X[:, _ix("log_el_frac")], nan=0.0)
            + 0.3 * np.nan_to_num(X[:, _ix("card_rate")], nan=0.5))


# ---- classification ----------------------------------------------------------------------------------------

def test_classes_cover_all_78_columns_exactly_once():
    all_cols = V.ALL_COLS
    assert len(all_cols) == 78 and len(set(all_cols)) == 78
    classes = K.column_classes()
    flat = [c for cls in ("TIME", "CARD_I", "FROM_J", "DROP") for c in classes[cls]]
    counted = [c for c in flat if c != K.SINCE_FIRST_J]          # the derived column is not an ALL_COLS column
    assert sorted(counted) == sorted(all_cols), "classes must cover ALL_COLS exactly once"
    assert len(counted) == len(set(counted)), "a column is in two classes"
    assert K.SINCE_FIRST_J not in all_cols


def test_curve_columns_and_monotone_signs():
    assert K.CURVE_COLS == K.TIME_COLS + K.CARD_I_COLS + K.CARD_I_DERIVED + K.FROM_J_COLS
    assert len(K.CURVE_COLS) == len(set(K.CURVE_COLS)) == 74
    assert len(K.MONOTONE) == len(K.CURVE_COLS)
    mono = dict(zip(K.CURVE_COLS, K.MONOTONE))
    for c in ("el", "log_el", "el_frac", "log_el_frac", "f_el_over_S"):
        assert mono[c] == -1
    for c in ("f_R", "f_logitR", "f_R_secs"):
        assert mono[c] == 1
    assert all(mono[c] == 0 for c in K.CURVE_COLS if c not in K.TIME_COLS)
    for c in ("since_first", "same_day", "rows_since_card", "gap_bkt_rate", "gap_bkt_n"):
        assert c not in K.CURVE_COLS


# ---- TIME helper and the feature frame ---------------------------------------------------------------------

def test_time_helper_reproduces_the_feature_frame():
    st, feats = _features(_raw_user(seed=2))
    keys = _scored_keys(st)
    X, prev, _ = K.curve_matrix(feats, st, keys)
    tf = K.time_features(feats.loc[keys, "el"].to_numpy(float), feats.loc[keys, "el_frac"].to_numpy(float),
                         feats.loc[keys, "f_S"].to_numpy(float))
    for c in K.TIME_COLS:
        ref = feats.loc[keys, c].to_numpy(float)
        assert np.allclose(X[:, _ix(c)], ref, rtol=1e-5, atol=1e-6, equal_nan=True), c
        assert np.allclose(tf[c], ref, rtol=1e-5, atol=1e-6, equal_nan=True), c


def test_row_values_come_from_i_for_time_card_and_from_j_for_from_j():
    st, feats = _features(_raw_user(seed=3))
    keys = _scored_keys(st)
    X, prev, _ = K.curve_matrix(feats, st, keys)
    k = 17
    i, j = keys[k], prev[k]
    assert st.loc[i, "card_id"] == st.loc[j, "card_id"] and j < i
    assert X[k, _ix("el")] == np.float32(feats.loc[i, "el"])
    assert X[k, _ix("card_n")] == np.float32(feats.loc[i, "card_n"])
    assert X[k, _ix("u_rate")] == np.float32(feats.loc[j, "u_rate"])
    assert X[k, _ix("ewm50")] == np.float32(feats.loc[j, "ewm50"])
    assert X[k, _ix(K.SINCE_FIRST_J)] == np.float32(feats.loc[j, "since_first"])   # card age at j


# ---- leak rule ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [4, 5])
def test_reviews_after_j_do_not_change_row_i(seed):
    raw = _raw_user(seed=seed)
    st, feats = _features(raw)
    keys = _scored_keys(st)
    X, prev, th = K.curve_matrix(feats, st, keys)
    y = (st.loc[keys, "rating"].to_numpy() > 1).astype(float)
    ewm50, rate = X[:, K.EWM50_IX], X[:, K.RATE_IX]
    p0 = K.curve_online(_toy_margin(X), ewm50, rate, y, th, prev)
    fixed = [_ix(c) for c in K.CARD_I_COLS + K.CARD_I_DERIVED + K.FROM_J_COLS]
    event_id = np.arange(1, len(raw) + 1)
    for k in (3, len(keys) // 2, len(keys) - 5):
        j = prev[k]
        raw2 = raw.copy()
        later = event_id > j                                    # every review strictly after j, any card
        rng = np.random.default_rng(k)
        raw2.loc[later, "rating"] = rng.integers(1, 5, int(later.sum()))
        raw2.loc[later, "duration"] = rng.integers(1000, 20000, int(later.sum()))
        st2, feats2 = _features(raw2)
        assert np.array_equal(_scored_keys(st2), keys)
        X2, prev2, th2 = K.curve_matrix(feats2, st2, keys)
        assert np.array_equal(prev2, prev)
        assert not np.array_equal(X2, X), "the change after j must be visible somewhere"
        assert np.array_equal(X2[k, fixed], X[k, fixed], equal_nan=True), f"row {k}: CARD_I/FROM_J changed"
        y2 = (st2.loc[keys, "rating"].to_numpy() > 1).astype(float)
        p2 = K.curve_online(_toy_margin(X2), X2[:, K.EWM50_IX], X2[:, K.RATE_IX], y2, th2, prev2)
        assert np.isclose(p2[k], p0[k], rtol=0, atol=1e-12), f"row {k}: online prediction changed"


def test_online_uses_w0_before_any_update():
    z = np.array([0.4, -1.2, 0.9])
    ewm50 = np.array([0.7, 0.2, np.nan])
    rate = np.array([0.6, 0.5, 0.1])
    y = np.array([1.0, 0.0, 1.0])
    th = np.array([10, 20, 30], dtype=np.int64)
    prev = np.array([5, 15, 25], dtype=np.int64)        # row 0 has no earlier scored row; rows 1, 2 do
    p = K.curve_online(z, ewm50, rate, y, th, prev)
    assert np.isclose(p[0], 1 / (1 + np.exp(-z[0])), atol=1e-12)
    assert not np.isclose(p[1], 1 / (1 + np.exp(-z[1])), atol=1e-12)   # row 1 uses the update at th=10


def test_online_weight_zero_is_clipped():
    D = np.column_stack([np.ones(50), np.ones(50), np.zeros(50), np.zeros(50)])
    y = np.zeros(50)                                    # always wrong: pushes w[0] down
    W = K._online_kernel(D, y, K.W0.copy(), 0.5, 0.0)
    assert (W[:, 0] >= 0).all() and W[-1, 0] == 0.0


# ---- monotone constraint -----------------------------------------------------------------------------------

def _train_small(seed_users=(0, 1, 2, 3)):
    import lightgbm as lgb
    Xs, ys = [], []
    for u in seed_users:
        st, feats = _features(_raw_user(seed=10 + u, n_cards=12, reviews=12))
        keys = _scored_keys(st)
        X, _, _ = K.curve_matrix(feats, st, keys)
        Xs.append(X)
        ys.append((st.loc[keys, "rating"].to_numpy() > 1).astype(np.int64))
    X, y = np.vstack(Xs), np.concatenate(ys)
    params = dict(K.CURVE_PARAMS, min_data_in_leaf=20, num_threads=1)
    booster = lgb.train(params, lgb.Dataset(X, y, feature_name=K.CURVE_COLS, free_raw_data=False), 60)
    return booster, X, y


def _with_time(X, add_days):
    X2 = X.copy()
    tf = K.time_features(X[:, _ix("el")] + add_days, X[:, _ix("el_frac")] + add_days, X[:, _ix("f_S")])
    for c in K.TIME_COLS:
        X2[:, _ix(c)] = tf[c]
    return X2


def test_trained_curve_is_non_increasing_in_t():
    booster, X, y = _train_small()
    ok = np.isfinite(X[:, [_ix(c) for c in K.TIME_COLS]]).all(axis=1)
    Xs, ys = X[ok], y[ok]
    p1 = booster.predict(Xs)
    z1 = booster.predict(Xs, raw_score=True)
    # online weights fixed from the original margins (time-independent w0 and labels); th = row order, prev = row - 1
    th = np.arange(len(Xs), dtype=np.int64)
    W = K.online_weights(K.online_design(z1, Xs[:, K.EWM50_IX], Xs[:, K.RATE_IX]), ys, th, th - 1)
    for add in (1.0, 5.0, 40.0):
        X2 = _with_time(Xs, add)
        p2 = booster.predict(X2)
        assert np.all(p2 <= p1 + 1e-12), f"GBDT not non-increasing in t (+{add} days)"
        z2 = booster.predict(X2, raw_score=True)
        q1 = K.online_p(W, K.online_design(z1, Xs[:, K.EWM50_IX], Xs[:, K.RATE_IX]))
        q2 = K.online_p(W, K.online_design(z2, Xs[:, K.EWM50_IX], Xs[:, K.RATE_IX]))
        assert np.all(q2 <= q1 + 1e-12), f"curve+online not non-increasing in t (+{add} days)"


# ---- guards ------------------------------------------------------------------------------------------------

def test_cli_refuses_test_set_users(tmp_path):
    with pytest.raises(ValueError, match="test set"):
        K.main(["--train_users", "2001", "--eval_users", "5", "--out", str(tmp_path / "curve.jsonl")])
    assert not (tmp_path / "curve.jsonl").exists()


def test_load_users_accepts_ranges_and_lists():
    assert K.load_users("2001-2003,2010") == [2001, 2002, 2003, 2010]
    assert K.load_users("8001 8002") == [8001, 8002]
