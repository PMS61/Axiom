"""Verifier tests for the GBDT model (Phase 7, task 7.3): leakage checks. Model code is not changed here.

Each test flips labels (and the matching rating) of rows that must NOT influence the checked value and
asserts that the value is unchanged. Tests that are expected to fail show a real leak (see the reason).
"""
import numpy as np
import pandas as pd
import pytest

from fsrs_benchmark import split
from fsrs_benchmark.experiments.validation import make_validation_split
from fsrs_benchmark.models import gbdt as G
from fsrs_benchmark.models.gbdt import ALL_GROUPS, Aggregates, GBDT, base_features
from fsrs_benchmark.tests.test_ours_v2 import _raw

FAST = dict(max_rounds=30, patience=10, min_data_in_leaf=5, num_leaves=7)


def _flip(df, mask):
    """Flip y (and the rating consistently) on the rows where mask is True."""
    out = df.copy()
    m = np.asarray(mask, dtype=bool)
    y = out["y"].to_numpy().copy()
    new_y = np.where(m, 1 - y, y)
    rating = out["rating"].to_numpy().copy()
    out["y"] = new_y
    out["rating"] = np.where(m, np.where(new_y == 1, 3, 1), rating)
    return out


def _add_ts(ev):
    """Duolingo-like timestamps: consecutive rows of one user on one day share a ts (a 'session'),
    a new session starts when the card repeats (a card is never reviewed twice at one ts)."""
    ev = ev.sort_values("event_id").copy()
    user, day, card = ev["user_id"].to_numpy(), ev["day_offset"].to_numpy(), ev["card_id"].to_numpy()
    ts = np.empty(len(ev), dtype=np.int64)
    sess, cur, prev_u, prev_d = 0, set(), None, None
    for i in range(len(ev)):
        if user[i] != prev_u or day[i] != prev_d:
            sess, cur = 0, {card[i]}
        elif card[i] in cur:
            sess, cur = sess + 1, {card[i]}
        else:
            cur.add(card[i])
        prev_u, prev_d = user[i], day[i]
        ts[i] = int(day[i]) * 86400 + sess * 60
    ev["ts"] = ts
    return ev


@pytest.fixture(scope="module")
def raw():
    r = _raw(n_users=8, n_cards=20, reviews=12, seed=1)
    r["lexeme"] = r["card_id"] % 7          # shared across users
    return r


@pytest.fixture(scope="module")
def duo(raw):
    """Duolingo-like: lexeme and ts."""
    return _add_ts(make_validation_split(split.make_split(raw, max_cards=1000, seed=0)))


@pytest.fixture(scope="module")
def anki(raw):
    """Anki-like: no lexeme, no ts."""
    return make_validation_split(split.make_split(raw.drop(columns=["lexeme"]), max_cards=1000, seed=0))


def _sample(rng, idx, k=25):
    idx = np.asarray(list(idx))
    return rng.choice(idx, size=min(k, len(idx)), replace=False)


# 1. User kernel and card kernel -------------------------------------------------------------

def test_ts_data_features_ignore_same_or_later_ts_labels(duo):
    """Duolingo-like (ts present): features of row t must not depend on labels of rows with ts >= ts_t
    (same session included, and the row itself)."""
    base = base_features(duo)
    rng = np.random.default_rng(0)
    for t in _sample(rng, duo.index):
        later = (duo["ts"] >= duo.at[t, "ts"]).to_numpy()
        b2 = base_features(_flip(duo, later))
        pd.testing.assert_series_equal(base.loc[t], b2.loc[t], check_names=False)


def test_ts_data_features_do_see_strictly_earlier_labels(duo):
    """Sensitivity check: the earlier-label flip must change some features, otherwise the test above is vacuous."""
    base = base_features(duo)
    rng = np.random.default_rng(1)
    changed = 0
    for t in _sample(rng, duo.index):
        earlier = ((duo["user_id"] == duo.at[t, "user_id"]) & (duo["ts"] < duo.at[t, "ts"])).to_numpy()
        if earlier.any():
            b2 = base_features(_flip(duo, earlier))
            changed += int(not base.loc[t].equals(b2.loc[t]))
    assert changed > 0


def test_card_history_ignores_same_or_later_labels_of_its_card(duo):
    """Card kernel: features of review t must not depend on y of the same card at or after t."""
    base = base_features(duo)
    rng = np.random.default_rng(2)
    for t in _sample(rng, duo.index):
        same_card_later = ((duo["user_id"] == duo.at[t, "user_id"]) & (duo["card_id"] == duo.at[t, "card_id"])
                           & (duo["event_id"] >= duo.at[t, "event_id"])).to_numpy()
        b2 = base_features(_flip(duo, same_card_later))
        pd.testing.assert_series_equal(base.loc[t], b2.loc[t], check_names=False)


def test_anki_row_kernel_ignores_same_or_later_rows(anki):
    """No ts, no lexeme: per-row user kernel. Row t must not see its own label or any later row's label,
    including later rows of the same day."""
    base = base_features(anki)
    rng = np.random.default_rng(3)
    same_day_candidates = []
    for u, g in anki.groupby("user_id"):
        d = g.sort_values("event_id")
        for pos in range(len(d) - 1):
            if d.iloc[pos + 1]["day_offset"] == d.iloc[pos]["day_offset"]:
                same_day_candidates.append(d.index[pos])
    assert same_day_candidates, "fixture has no same-day consecutive reviews"
    picks = list(_sample(rng, same_day_candidates, 15)) + list(_sample(rng, anki.index, 15))
    for t in picks:
        later = ((anki["user_id"] == anki.at[t, "user_id"]) & (anki["event_id"] >= anki.at[t, "event_id"])).to_numpy()
        b2 = base_features(_flip(anki, later))
        pd.testing.assert_series_equal(base.loc[t], b2.loc[t], check_names=False)


def test_day_mode_ignores_same_day_rows(duo):
    """Lexeme present, ts absent -> 'day' mode: a row sees only strictly earlier days."""
    ev = duo.drop(columns=["ts"])
    base = base_features(ev)
    rng = np.random.default_rng(4)
    for t in _sample(rng, ev.index, 20):
        # the user kernel must not see any same-day row; the CARD kernel legitimately sees an earlier review of
        # the same card on the same day, so those rows are excluded from the flip
        same_user_day = ((ev["user_id"] == ev.at[t, "user_id"]) & (ev["day_offset"] >= ev.at[t, "day_offset"])
                         & ~((ev["card_id"] == ev.at[t, "card_id"]) & (ev["event_id"] < ev.at[t, "event_id"]))).to_numpy()
        b2 = base_features(_flip(ev, same_user_day))
        pd.testing.assert_series_equal(base.loc[t], b2.loc[t], check_names=False)


def test_auto_rule_picks_the_right_mode(duo, anki):
    ts_ev = duo
    pd.testing.assert_frame_equal(base_features(ts_ev), base_features(ts_ev, user_group="ts"))
    lex_only = duo.drop(columns=["ts"])
    pd.testing.assert_frame_equal(base_features(lex_only), base_features(lex_only, user_group="day"))
    pd.testing.assert_frame_equal(base_features(anki), base_features(anki, user_group="day"))   # safe default without ts
    assert not base_features(anki, user_group="row")["u_n"].equals(base_features(anki)["u_n"])


def test_auto_fallback_is_safe_for_batched_data_without_ts(duo):
    ev = duo.drop(columns=["ts", "lexeme"])
    base = base_features(ev)
    rng = np.random.default_rng(5)
    for t in _sample(rng, ev.index, 20):
        same_day_earlier = ((ev["user_id"] == ev.at[t, "user_id"]) & (ev["day_offset"] == ev.at[t, "day_offset"])
                            & (ev["event_id"] < ev.at[t, "event_id"])).to_numpy()
        if not same_day_earlier.any():
            continue
        b2 = base_features(_flip(ev, same_day_earlier))
        ucols = [c for c in base.columns if c.startswith("u_")]   # card features may legitimately see the same card
        pd.testing.assert_series_equal(base.loc[t, ucols], b2.loc[t, ucols], check_names=False)


# 3. Aggregates -----------------------------------------------------------------------------

@pytest.mark.parametrize("which", ["duo", "anki"])
def test_aggregates_unchanged_when_every_non_fit_label_flipped(which, request):
    """Flip every label that is not a fit row (validation, test-side and non-fit train rows). The prior,
    all aggregate tables and every aggregate feature of every row must stay the same."""
    ev = request.getfixturevalue(which)
    fit = ev["is_train"].to_numpy(bool) & G._sl_mask(ev)
    base = base_features(ev)
    a1 = Aggregates(ev, base)
    o1 = a1.attach(ev, base)
    mod = _flip(ev, ~fit)
    base2 = base_features(mod)
    a2 = Aggregates(mod, base2)
    o2 = a2.attach(mod, base2)
    assert a1.prior == a2.prior
    pd.testing.assert_frame_equal(a1.u_tot, a2.u_tot)
    pd.testing.assert_frame_equal(a1.u_fold, a2.u_fold)
    if a1.has_lex:
        pd.testing.assert_frame_equal(a1.l_tot, a2.l_tot)
        pd.testing.assert_frame_equal(a1.l_fold, a2.l_fold)
    assert o1.keys() == o2.keys()
    for k in o1:
        np.testing.assert_array_equal(o1[k], o2[k], err_msg=k)


def test_fitted_booster_unchanged_when_validation_and_test_labels_flipped(duo):
    """Rows after the fit period (is_train False) must not influence the fitted model at all."""
    m1 = GBDT(**FAST).fit(duo)
    mod = _flip(duo, ~duo["is_train"].to_numpy(bool))
    m2 = GBDT(**FAST).fit(mod)
    assert m1.booster_.model_to_string() == m2.booster_.model_to_string()
    assert m1.best_iter_ == m2.best_iter_


# 4. Early stopping holdout -----------------------------------------------------------------

def test_early_stopping_holdout_and_train_are_fit_rows_only(duo, monkeypatch):
    seen = []
    orig = G.lgb.Dataset

    def rec(data, label=None, **kw):
        seen.append(pd.Index(data.index))
        return orig(data, label, **kw)

    monkeypatch.setattr(G.lgb, "Dataset", rec)
    big = make_validation_split(split.make_split(_raw(n_users=40, n_cards=20, reviews=14, seed=3),
                                                 max_cards=1000, seed=0))
    fit = big["is_train"].to_numpy(bool) & G._sl_mask(big)
    GBDT(**FAST).fit(big)
    assert len(seen) == 2, "expected a train and a holdout Dataset (holdout must exist for this check)"
    tr, ho = seen
    assert len(ho) > 0 and len(tr) > 0
    assert fit[big.index.get_indexer(tr)].all()
    assert fit[big.index.get_indexer(ho)].all()
    assert not set(tr) & set(ho)


# 5. Inputs -----------------------------------------------------------------------------------

def test_state_and_history_columns_are_not_inputs(duo):
    m = GBDT(**FAST).fit(duo)
    assert not {"state", "history_seen", "history_correct"} & set(m.names_)
    for g, cols in G.GROUP_FEATURES.items():
        assert not {"state", "history_seen", "history_correct"} & set(cols), g
    noisy = duo.copy()
    rng = np.random.default_rng(6)
    noisy["state"] = rng.integers(0, 4, len(noisy))
    noisy["history_seen"] = rng.integers(0, 50, len(noisy))
    noisy["history_correct"] = rng.integers(0, 50, len(noisy))
    pd.testing.assert_frame_equal(base_features(duo), base_features(noisy))


# 6. Known rule violation, expected to fail ---------------------------------------------------

@pytest.mark.xfail(reason="accepted: cross-user calendar time, same as global parameters under the per-user split")
def test_lexeme_aggregate_uses_no_labels_from_after_t(duo):
    """The lexeme rate pools fit rows of ALL users. Users have different cutoff days, so a fit row of
    another user can be later in calendar time than a scored row t. Its label must not feed lex_rate of t."""
    base = base_features(duo)
    att = Aggregates(duo, base).attach(duo, base)
    fit = duo["is_train"].to_numpy(bool) & G._sl_mask(duo)
    users = duo["user_id"].to_numpy()
    days = duo["day_offset"].to_numpy()
    lex = duo["lexeme"].to_numpy()
    scored = duo["scored"].to_numpy(bool)
    pair = None
    for r in np.flatnonzero(scored):
        js = np.flatnonzero(fit & (users != users[r]) & (lex == lex[r]) & (days > days[r]))
        if len(js):
            pair = (r, js[0])
            break
    assert pair is not None, "fixture has no cross-user future fit row for a shared lexeme"
    r, j = pair
    mod = _flip(duo, np.arange(len(duo)) == j)
    base2 = base_features(mod)
    att2 = Aggregates(mod, base2).attach(mod, base2)
    assert att2["lex_rate"][r] == att["lex_rate"][r], (
        f"lex_rate of row {duo.index[r]} changed when the label of later fit row {duo.index[j]} "
        f"(other user, day {days[j]} > {days[r]}) was flipped")


def _tiny_card_events(with_ts=True):
    """One user, one card: first review, then THREE same-card rows at one ts, then a later row."""
    rows = [  # (day, ts, rating)
        (0, 0, 3),
        (2, 200, 1), (2, 200, 3), (2, 200, 4),
        (5, 500, 3),
    ]
    ev = pd.DataFrame({
        "event_id": np.arange(len(rows), dtype=np.int64) + 1,
        "user_id": 1, "card_id": 0,
        "day_offset": [r[0] for r in rows], "rating": [r[2] for r in rows],
        "y": [int(r[2] > 1) for r in rows],
        "elapsed_days": [-1, 2, 2, 2, 3], "dt_days": [np.nan, 2.0, 2.0, 2.0, 3.0],
        "is_first": [True, False, False, False, False],
        "is_train": True, "scored": False, "state": 2, "ts": [r[1] for r in rows],
    })
    if not with_ts:
        ev = ev.drop(columns=["ts"])
    return ev


def test_three_same_card_same_ts_rows_group_semantics():
    """Rows e1..e3 share a card and a ts: none sees another. The later row e4 sees all three outcomes."""
    ev = _tiny_card_events()
    b = base_features(ev)
    cols = ["card_n", "card_lapses", "card_rate", "g1", "g2", "g3", "y1", "y2", "y3", "gap1", "gap2", "gap3"]
    for i in (1, 2, 3):
        assert b.iloc[i]["card_n"] == 1 and b.iloc[i]["g1"] == 3 and b.iloc[i]["y1"] == 1, i
        assert np.isnan(b.iloc[i]["g2"]) and b.iloc[i]["card_lapses"] == 0, i
    e4 = b.iloc[4]
    assert e4["card_n"] == 4
    assert e4["g1"] == 4 and e4["g2"] == 3 and e4["g3"] == 1      # e3, e2, e1 ratings, in order
    assert e4["y1"] == 1 and e4["y2"] == 1 and e4["y3"] == 0
    assert e4["card_lapses"] == 1 and e4["card_rate"] == pytest.approx(0.75)
    # flipping e1 (rating 1 -> 3) must not change e2, e3, but must change e4
    b2 = base_features(_flip(ev, [False, True, False, False, False]))
    for i in (2, 3):
        pd.testing.assert_series_equal(b.iloc[i][cols], b2.iloc[i][cols], check_names=False)
    assert not b.iloc[4][cols].equals(b2.iloc[4][cols])


def test_anki_without_ts_is_per_row_for_the_same_card():
    """No ts column: every same-card row is its own group, so e2 sees e1 and e3 sees e2."""
    ev = _tiny_card_events(with_ts=False)
    b = base_features(ev)
    assert b.iloc[2]["card_n"] == 2 and b.iloc[2]["g1"] == 1
    assert b.iloc[3]["card_n"] == 3 and b.iloc[3]["g1"] == 3
