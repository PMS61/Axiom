"""Tests for the GBDT model: past-only features, fit-row-only aggregates, shapes, probabilities, registration."""
import numpy as np
import pandas as pd
import pytest

from fsrs_benchmark import split
from fsrs_benchmark.experiments.validation import make_validation_split
from fsrs_benchmark.models.gbdt import ALL_GROUPS, FROZEN_KWARGS, GBDT, Aggregates, base_features
from fsrs_benchmark.tests.test_ours_v2 import _raw

FAST = dict(max_rounds=30, patience=10, min_data_in_leaf=5, num_leaves=7)


@pytest.fixture(scope="module")
def val():
    raw = _raw(n_users=6, n_cards=30, reviews=14)
    raw["lexeme"] = raw["card_id"] % 7          # shared across users
    return make_validation_split(split.make_split(raw, max_cards=1000, seed=0))


def _flip(df, row_pos):
    out = df.copy()
    i = out.index[row_pos]
    new_y = 1 - int(out.at[i, "y"])
    out.at[i, "y"] = new_y
    out.at[i, "rating"] = 3 if new_y == 1 else 1
    return out


def test_base_features_are_past_only(val):
    base = base_features(val)
    ev = val.sort_values("event_id")
    # pick a review in the middle of a user's stream and flip its label
    u = ev["user_id"].iloc[len(ev) // 2]
    ids = ev.index[ev["user_id"] == u]
    t = ids[len(ids) // 2]
    mod = _flip(val, val.index.get_loc(t))
    base2 = base_features(mod)
    same_user = val["user_id"] == u
    not_after = val["event_id"] <= val.at[t, "event_id"]
    # features of every review with event_id <= t are unchanged (all users)
    pd.testing.assert_frame_equal(base[not_after], base2[not_after])
    # and something later in the same user/card does change (the flip is actually used)
    assert not base[same_user & ~not_after].equals(base2[same_user & ~not_after])


def test_predictions_at_or_before_t_do_not_depend_on_label_t(val):
    m = GBDT(**FAST).fit(val)
    p = m.predict(val)
    ev = val[val["scored"]].sort_values("event_id")
    t_row = ev.index[5]
    t_id = val.at[t_row, "event_id"]
    p2 = m.predict(_flip(val, val.index.get_loc(t_row)))
    upto = p.index <= t_id
    np.testing.assert_array_equal(p[upto].to_numpy(), p2[upto].to_numpy())
    assert not np.array_equal(p.to_numpy(), p2.to_numpy())


def test_aggregates_use_fit_rows_only(val):
    base = base_features(val)
    a = Aggregates(val, base)
    mod = val.copy()
    other = ~(val["is_train"].to_numpy())          # validation rows are not fit rows
    mod.loc[other, "y"] = 1 - mod.loc[other, "y"]
    b = Aggregates(mod, base_features(mod))
    for x, y in [(a.u_tot, b.u_tot), (a.l_tot, b.l_tot), (a.u_fold, b.u_fold), (a.l_fold, b.l_fold)]:
        pd.testing.assert_frame_equal(x, y)
    assert a.prior == b.prior
    # fit rows that are first reviews / same-day are excluded too
    n_fit = int((val["is_train"] & ~val["is_first"] & (val["elapsed_days"] >= 1)).sum())
    assert int(a.u_tot["count"].sum()) == n_fit


def test_train_rows_get_out_of_fold_values(val):
    base = base_features(val)
    a = Aggregates(val, base)
    att = a.attach(val, base)
    tr = val["is_train"].to_numpy(bool)
    full_n = a.u_tot["count"].reindex(val["user_id"]).to_numpy()
    assert (att["up_n"][tr] <= full_n[tr]).all() and (att["up_n"][tr] < full_n[tr]).any()
    assert np.array_equal(att["up_n"][~tr], full_n[~tr])      # validation rows: full fit stats


@pytest.mark.parametrize("train_rows", ["scored_like", "nonfirst"])
def test_shapes_and_probabilities(val, train_rows):
    p = GBDT(train_rows=train_rows, **FAST).fit(val).predict(val)
    assert set(p.index) == set(val.loc[val["scored"], "event_id"]) and len(p) == int(val["scored"].sum())
    assert p.index.is_unique and p.index.name == "event_id"
    assert np.isfinite(p).all() and p.between(1e-6, 1 - 1e-6).all() and (p > 0).all() and (p < 1).all()


def test_no_lexeme_column_and_group_subsets(val):
    nolex = val.drop(columns=["lexeme"])
    m = GBDT(groups=("gap", "fsrs", "userprior", "lexeme"), **FAST).fit(nolex)
    assert not any(n.startswith("lex_") for n in m.names_)
    p = m.predict(nolex)
    assert len(p) == int(val["scored"].sum()) and p.between(1e-6, 1 - 1e-6).all()


def test_deterministic_and_fit_ignores_non_fit_labels(val):
    a = GBDT(**FAST).fit(val).predict(val)
    b = GBDT(**FAST).fit(val).predict(val)
    np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())
    # flipping labels of validation (non-fit) rows that no fit row can see leaves the fitted model's
    # training data alone: fit on the original and on a table whose LAST validation rows are flipped
    last = val.sort_values("event_id").groupby("user_id").tail(1).index
    mod = val.copy()
    mod.loc[last, "y"] = 1 - mod.loc[last, "y"]
    mod.loc[last, "rating"] = np.where(mod.loc[last, "y"] == 1, 3, 1)
    c = GBDT(**FAST).fit(mod).predict(mod)
    first_ids = c.index[~c.index.isin(val.loc[last, "event_id"])]
    # earlier rows of each user only change through their later history, which they cannot see
    assert len(first_ids) > 0
    np.testing.assert_allclose(a.loc[first_ids].to_numpy(), c.loc[first_ids].to_numpy())


def test_frozen_kwargs_construct():
    m = GBDT(seed=0, **FROZEN_KWARGS)
    assert set(m.groups) <= set(ALL_GROUPS)
    assert "num_leaves" in m.hyperparams()


def test_registered_in_run_and_report():
    from fsrs_benchmark import report, run
    assert "gbdt" in run.OPT_IN_MODELS and "gbdt" in report.ALL_MODELS and "gbdt" in report.DISPLAY
    args = type("A", (), dict(seed=0, workers=1))()
    assert isinstance(run.make_model("gbdt", args), GBDT)


def test_same_timestamp_rows_do_not_see_each_other(val):
    """Two rows of one user at the same ts (different lexemes): flipping one leaves the other's features alone."""
    ev = val.copy()
    ev["ts"] = ev["day_offset"].astype(np.int64) * 86400 + ev["review_idx"].astype(np.int64) * 10
    u = ev["user_id"].iloc[0]
    rows = ev.index[ev["user_id"] == u]
    k = next(i for i in range(len(rows) // 2, len(rows) - 1) if ev.at[rows[i], "card_id"] != ev.at[rows[i + 1], "card_id"])
    a, b = rows[k], rows[k + 1]       # consecutive rows of one user, different cards
    ev.loc[b, "ts"] = ev.loc[a, "ts"]
    ev.loc[b, "day_offset"] = ev.loc[a, "day_offset"]
    ev.loc[b, "lexeme"] = ev.loc[a, "lexeme"] + 1
    assert ev.loc[a, "card_id"] != ev.loc[b, "card_id"]
    base = base_features(ev)
    base2 = base_features(_flip(ev, ev.index.get_loc(a)))
    cols = [c for c in base.columns if c.startswith("u_")]
    pd.testing.assert_series_equal(base.loc[b, cols], base2.loc[b, cols])
    pd.testing.assert_series_equal(base.loc[a, cols], base2.loc[a, cols])
    # and without ts the same holds at day granularity (strictly earlier days only)
    nots = ev.drop(columns=["ts"])
    d1, d2 = base_features(nots), base_features(_flip(nots, nots.index.get_loc(a)))
    same_day = nots["user_id"].eq(u) & nots["day_offset"].eq(nots.at[a, "day_offset"])
    pd.testing.assert_frame_equal(d1.loc[same_day, cols], d2.loc[same_day, cols])


def test_anki_like_data_uses_per_row_user_updates(val):
    """No ts and no lexeme: user stats update per row (reviews are individually timed)."""
    ev = val.drop(columns=["lexeme"])
    b = base_features(ev, user_group="row")
    srt = ev.sort_values("event_id")
    first_user = srt[srt["user_id"] == srt["user_id"].iloc[0]]
    assert (b.loc[first_user.index, "u_n"].to_numpy() == np.arange(len(first_user))).all()
    # auto without ts is the safe day grouping
    assert not (base_features(ev).loc[first_user.index, "u_n"].to_numpy() == np.arange(len(first_user))).all()


def test_same_card_same_ts_rows_do_not_see_each_other(val):
    """A duplicate row (same card, same ts): flipping the first leaves the second's card features alone."""
    ev = val.copy()
    ev["ts"] = ev["day_offset"].astype(np.int64) * 86400 + ev["review_idx"].astype(np.int64) * 10
    srt = ev.sort_values("event_id")
    k = next(i for i in range(5, len(srt) - 1) if srt["card_id"].iloc[i] == srt["card_id"].iloc[i + 1]
             and srt["user_id"].iloc[i] == srt["user_id"].iloc[i + 1])
    a, b = srt.index[k], srt.index[k + 1]
    ev.loc[b, "ts"] = ev.loc[a, "ts"]
    ev.loc[b, "day_offset"] = ev.loc[a, "day_offset"]
    base, base2 = base_features(ev), base_features(_flip(ev, ev.index.get_loc(a)))
    cols = [c for c in base.columns if c.startswith(("card_", "g", "y", "gap", "f_", "since")) or c in ("el",)]
    pd.testing.assert_series_equal(base.loc[b, cols], base2.loc[b, cols])
    # without ts the second row DOES see the first (per-row behaviour kept)
    nots = ev.drop(columns=["ts"])
    c1, c2 = base_features(nots, user_group="row"), base_features(_flip(nots, nots.index.get_loc(a)), user_group="row")
    assert not c1.loc[b, ["card_rate", "y1", "g1"]].equals(c2.loc[b, ["card_rate", "y1", "g1"]])
