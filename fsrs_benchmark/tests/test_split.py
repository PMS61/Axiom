"""Tests for fsrs_benchmark.split: leakage-free cutoff, scoring rules, card cap, determinism."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fsrs_benchmark.split import load_events, make_split, save_events

REAL_ROOT = Path("/home/amal/fyp/Axiom/data/anki10k/revlogs")


def _synthetic_user(user_id, n_cards, reviews_per_card=10, seed=0):
    """A loader-shaped frame: each card has a first review then gaps of 1-3 days."""
    rng = np.random.default_rng(seed)
    rows = []
    review_idx = 0
    for card in range(n_cards):
        day = int(rng.integers(0, 30))
        for k in range(reviews_per_card):
            gap = 0 if k == 0 else int(rng.integers(0, 4))
            day += gap
            rows.append(dict(
                user_id=user_id, card_id=card + 100 * user_id, review_idx=review_idx,
                day_offset=day, rating=int(rng.integers(1, 5)),
                state=0 if k == 0 else 2,
                elapsed_days=-1 if k == 0 else gap,
                dt_days=np.nan if k == 0 else float(gap),
                y=int(rng.integers(0, 2)), is_first=(k == 0),
            ))
            review_idx += 1
    return pd.DataFrame(rows)


@pytest.fixture
def two_users():
    return pd.concat([_synthetic_user(1, 6), _synthetic_user(2, 5, seed=1)], ignore_index=True)


def test_train_is_strictly_before_cutoff_per_user(two_users):
    ev = make_split(two_users, max_cards=100, seed=0)
    assert (ev.loc[ev.is_train, "day_offset"] < ev.loc[ev.is_train, "cutoff_day"]).all()
    assert (ev.loc[~ev.is_train, "day_offset"] >= ev.loc[~ev.is_train, "cutoff_day"]).all()


def test_cutoff_is_80th_percentile_of_kept_reviews(two_users):
    ev = make_split(two_users, max_cards=100, seed=0)
    for uid, grp in two_users.groupby("user_id"):
        expected = np.quantile(grp["day_offset"], 0.8, method="linear")
        got = ev.loc[ev.user_id == uid, "cutoff_day"].unique()
        assert got.tolist() == [pytest.approx(expected)]


def test_scored_rules(two_users):
    ev = make_split(two_users, max_cards=100, seed=0)
    s = ev[ev.scored]
    assert len(s) > 0
    assert (~s.is_train).all()
    assert (~s.is_first).all()
    assert (s.elapsed_days >= 1).all()
    # every test event that satisfies the rules is scored, and nothing else is
    expected = ev[(~ev.is_train) & (~ev.is_first) & (ev.elapsed_days >= 1)]
    assert set(s.event_id) == set(expected.event_id)


def test_card_cap_keeps_all_rows_of_chosen_cards():
    df = _synthetic_user(7, 10)
    ev = make_split(df, max_cards=3, seed=0)
    assert ev["card_id"].nunique() == 3
    for card, grp in ev.groupby("card_id"):
        assert len(grp) == len(df[df.card_id == card])


def test_card_cap_applied_before_cutoff():
    df = _synthetic_user(7, 10)
    ev = make_split(df, max_cards=3, seed=0)
    kept = df[df.card_id.isin(ev.card_id.unique())]
    expected = np.quantile(kept["day_offset"], 0.8, method="linear")
    assert ev["cutoff_day"].iloc[0] == pytest.approx(expected)


def test_deterministic_for_fixed_seed_and_seed_changes_cards(two_users):
    a = make_split(two_users, max_cards=2, seed=5)
    b = make_split(two_users, max_cards=2, seed=5)
    pd.testing.assert_frame_equal(a, b)
    c = make_split(two_users, max_cards=2, seed=6)
    assert set(a.card_id) != set(c.card_id) or not a.equals(c)


def test_event_id_unique_int64_and_sorted(two_users):
    ev = make_split(two_users, max_cards=100, seed=0)
    assert ev["event_id"].dtype == np.int64
    assert ev["event_id"].is_unique
    assert ev["event_id"].is_monotonic_increasing


def test_no_card_cap_when_max_cards_large(two_users):
    ev = make_split(two_users, max_cards=10_000, seed=0)
    assert len(ev) == len(two_users)


def test_bad_max_cards_rejected(two_users):
    with pytest.raises(ValueError):
        make_split(two_users, max_cards=0, seed=0)


def test_save_and_load_roundtrip(two_users, tmp_path):
    ev = make_split(two_users, max_cards=100, seed=0)
    path = save_events(ev, tmp_path / "nested" / "events.parquet")
    assert path.exists()
    back = load_events(path)
    pd.testing.assert_frame_equal(back, ev)


@pytest.mark.skipif(not (REAL_ROOT / "user_id=6370" / "data.parquet").exists(),
                    reason="real revlogs not downloaded")
def test_real_users_split_is_consistent():
    from fsrs_benchmark.data import load_user
    frames = [load_user(REAL_ROOT / f"user_id={u}" / "data.parquet") for u in (6370, 8506)]
    ev = make_split(pd.concat(frames, ignore_index=True), max_cards=300, seed=0)
    assert set(ev.user_id) == {6370, 8506}
    assert (ev.loc[ev.is_train, "day_offset"] < ev.loc[ev.is_train, "cutoff_day"]).all()
    assert (ev.loc[ev.scored, "elapsed_days"] >= 1).all()
    assert ev["event_id"].is_unique
