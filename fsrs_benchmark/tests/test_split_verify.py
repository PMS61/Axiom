"""Verifier tests for fsrs_benchmark.split.make_split (Task 1.2 review)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fsrs_benchmark.split import load_events, make_split, save_events

REAL_ROOT = Path("/home/amal/fyp/Axiom/data/anki10k/revlogs")


def _user(user_id, cards, day_fn=None, first_elapsed=-1):
    """cards: list of lists of (day, elapsed_days, rating). Returns loader-shaped rows."""
    rows = []
    ridx = 0
    for c_i, reviews in enumerate(cards):
        for k, (day, elapsed, rating) in enumerate(reviews):
            rows.append(dict(
                user_id=user_id, card_id=1000 * user_id + c_i, review_idx=ridx,
                day_offset=day, rating=rating, state=0 if k == 0 else 2,
                elapsed_days=first_elapsed if k == 0 else elapsed,
                dt_days=np.nan if k == 0 else float(elapsed),
                y=int(rating > 1), is_first=(k == 0),
            ))
            ridx += 1
    return pd.DataFrame(rows)


def _spread_user(user_id, n_cards, seed, reviews=8):
    rng = np.random.default_rng(seed)
    cards = []
    for _ in range(n_cards):
        day = int(rng.integers(0, 200))
        seq = []
        for k in range(reviews):
            gap = 0 if k == 0 else int(rng.integers(0, 6))
            day += gap
            seq.append((day, gap, int(rng.integers(1, 5))))
        cards.append(seq)
    return _user(user_id, cards)


# ---- all reviews on one day --------------------------------------------------

def test_user_with_all_reviews_on_one_day_goes_entirely_to_test():
    df = _user(1, [[(5, -1, 3), (5, 0, 3)], [(5, -1, 1), (5, 0, 2)]])
    ev = make_split(df, max_cards=10, seed=0)
    assert not ev["is_train"].any()          # cutoff == 5, nothing strictly before it
    assert ev["cutoff_day"].eq(5.0).all()
    assert not ev["scored"].any()             # same-day reviews are never scored


# ---- card cap edge cases -----------------------------------------------------

def test_user_with_fewer_cards_than_cap_keeps_all_cards():
    df = _spread_user(3, 4, seed=1)
    ev = make_split(df, max_cards=300, seed=0)
    assert ev["card_id"].nunique() == 4
    assert len(ev) == len(df)
    expected = np.quantile(df["day_offset"], 0.8, method="linear")
    assert ev["cutoff_day"].iloc[0] == pytest.approx(expected)


def test_card_cap_is_independent_of_input_card_order():
    df = _spread_user(4, 20, seed=2)
    shuffled = df.sample(frac=1.0, random_state=9).reset_index(drop=True)
    a = make_split(df, max_cards=5, seed=3)
    b = make_split(shuffled, max_cards=5, seed=3)
    pd.testing.assert_frame_equal(a, b)


def test_event_id_stable_across_card_caps():
    df = _spread_user(5, 12, seed=4)
    small = make_split(df, max_cards=3, seed=0).set_index("event_id")
    big = make_split(df, max_cards=12, seed=0).set_index("event_id")
    common = small.index
    assert set(common) <= set(big.index)
    assert (small["review_idx"] == big.loc[common, "review_idx"]).all()


# ---- determinism and input-order invariance ----------------------------------

def test_deterministic_and_row_order_invariant():
    df = pd.concat([_spread_user(1, 15, 0), _spread_user(2, 15, 1)], ignore_index=True)
    shuffled = df.sample(frac=1.0, random_state=1).reset_index(drop=True)
    a = make_split(df, max_cards=6, seed=7)
    b = make_split(df, max_cards=6, seed=7)
    c = make_split(shuffled, max_cards=6, seed=7)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(a, c)


# ---- multi-user independence --------------------------------------------------

def test_one_users_split_does_not_depend_on_other_users():
    u1 = _spread_user(1, 10, 0)
    u2 = _spread_user(2, 10, 1)
    u2_alt = _spread_user(2, 10, 99)           # very different other user, same id
    alone = make_split(u1, max_cards=4, seed=0).set_index("event_id")
    with_a = make_split(pd.concat([u1, u2], ignore_index=True), max_cards=4, seed=0).set_index("event_id")
    with_b = make_split(pd.concat([u1, u2_alt], ignore_index=True), max_cards=4, seed=0).set_index("event_id")
    cols = ["cutoff_day", "is_train", "scored"]
    pd.testing.assert_frame_equal(alone[cols], with_a.loc[alone.index, cols])
    pd.testing.assert_frame_equal(alone[cols], with_b.loc[alone.index, cols])


# ---- event_id uniqueness across users ----------------------------------------

def test_event_id_unique_across_users_with_same_review_idx_range():
    u1 = _spread_user(1, 5, 0)
    u2 = _spread_user(2, 5, 1)
    ev = make_split(pd.concat([u1, u2], ignore_index=True), max_cards=10, seed=0)
    assert ev["event_id"].is_unique
    assert ev["event_id"].dtype == np.int64
    assert ev.loc[ev.user_id == 1, "event_id"].between(10**9 * 1, 10**9 * 1 + 10**8).all()


def test_duplicate_review_idx_within_user_is_rejected():
    df = _spread_user(1, 3, 0)
    df.loc[1, "review_idx"] = df.loc[0, "review_idx"]
    with pytest.raises(ValueError, match="duplicate"):
        make_split(df, max_cards=10, seed=0)


# ---- scoring rules on hand-built data ----------------------------------------

def test_scored_excludes_first_same_day_and_train_events():
    df = _user(1, [
        [(0, -1, 3), (1, 1, 3), (2, 1, 1), (2, 0, 3), (10, 8, 4)],
    ])
    ev = make_split(df, max_cards=5, seed=0)
    # 5 reviews: cutoff = 80th pct of [0,1,2,2,10] = 6.4 -> test = days 10 only
    assert ev["is_train"].tolist() == [True, True, True, True, False]
    assert ev["scored"].tolist() == [False, False, False, False, True]


def test_cutoff_day_column_is_float_and_per_user():
    df = pd.concat([_spread_user(1, 4, 0), _spread_user(2, 4, 1)], ignore_index=True)
    ev = make_split(df, max_cards=10, seed=0)
    assert ev["cutoff_day"].dtype == np.float64
    assert ev.groupby("user_id")["cutoff_day"].nunique().eq(1).all()


# ---- save/load round trip ----------------------------------------------------

def test_save_load_roundtrip_preserves_dtypes(tmp_path):
    df = _spread_user(1, 6, 0)
    ev = make_split(df, max_cards=10, seed=0)
    path = save_events(ev, tmp_path / "a" / "events.parquet")
    back = load_events(path)
    pd.testing.assert_frame_equal(back, ev)
    assert back["event_id"].dtype == np.int64
    assert back["is_train"].dtype == bool
    assert back["scored"].dtype == bool


# ---- real users --------------------------------------------------------------

@pytest.mark.skipif(not (REAL_ROOT / "user_id=6370" / "data.parquet").exists(),
                    reason="real revlogs not downloaded")
def test_real_users_scored_recall_and_counts():
    from fsrs_benchmark.data import load_user
    frames = [load_user(REAL_ROOT / f"user_id={u}" / "data.parquet") for u in (6370, 8506)]
    ev = make_split(pd.concat(frames, ignore_index=True), max_cards=300, seed=0)
    assert ev["scored"].sum() > 0
    rate = ev.loc[ev.scored, "y"].mean()
    assert 0.8 <= rate <= 0.95, rate
    assert ev.loc[ev.scored, "is_train"].sum() == 0
    assert (ev.loc[ev.is_train, "day_offset"] < ev.loc[ev.is_train, "cutoff_day"]).all()
