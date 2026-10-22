"""Verifier tests for fsrs_benchmark.data.load_user (Task 1.1 review)."""
import time

import numpy as np
import pandas as pd
import pytest

from fsrs_benchmark.data import LOAD_COLUMNS, load_user

REAL_USERS = [6370, 8506]
REAL_ROOT = "/home/amal/fyp/Axiom/data/anki10k/revlogs"


def _row(card, day, rating, state, elapsed_days, elapsed_seconds):
    return dict(card_id=card, day_offset=day, rating=rating, state=state,
                duration=1000, elapsed_days=elapsed_days, elapsed_seconds=elapsed_seconds)


def _write(root, user_id, rows, index=None):
    frame = pd.DataFrame(rows)
    if index is not None:
        frame.index = index
    folder = root / "revlogs" / f"user_id={user_id}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "data.parquet"
    frame.to_parquet(path, index=False)
    return path


def test_card_whose_first_row_is_rating0_is_judged_on_next_row(tmp_path):
    # rating-0 row dropped first; the next kept row is state 2 -> card dropped entirely
    rows = [
        _row(1, 0, 0, 0, -1, -1),
        _row(1, 1, 3, 2, 1, 86400),
        _row(2, 0, 3, 0, -1, -1),
    ]
    df = load_user(_write(tmp_path, 1, rows))
    assert set(df["card_id"]) == {2}


def test_card_whose_first_kept_row_is_state4_is_dropped(tmp_path):
    # state 4 rows are kept now, so the first kept row is state 4 (not new) -> card dropped
    rows = [
        _row(1, 0, 3, 4, 0, 0),
        _row(1, 1, 3, 0, -1, -1),
        _row(1, 2, 2, 2, 1, 86400),
    ]
    df = load_user(_write(tmp_path, 1, rows))
    assert len(df) == 0


def test_state4_row_after_new_is_kept_with_day_gap(tmp_path):
    rows = [
        _row(1, 0, 3, 0, -1, -1),
        _row(1, 2, 3, 4, 2, 2 * 86400),
    ]
    df = load_user(_write(tmp_path, 1, rows))
    assert df["card_id"].tolist() == [1, 1]
    assert df.loc[1, "state"] == 4
    assert df.loc[1, "elapsed_days"] == 2


def test_interleaved_cards_keep_order_and_first_flags(tmp_path):
    rows = [
        _row(1, 0, 3, 0, -1, -1),
        _row(2, 0, 1, 0, -1, -1),
        _row(1, 1, 3, 2, 1, 86400),
        _row(2, 1, 4, 1, 1, 86400),
        _row(1, 2, 1, 2, 2, 2 * 86400),
    ]
    df = load_user(_write(tmp_path, 1, rows))
    assert df["card_id"].tolist() == [1, 2, 1, 2, 1]
    assert df["is_first"].tolist() == [True, True, False, False, False]
    assert df["review_idx"].tolist() == [0, 1, 2, 3, 4]
    assert df.loc[df.card_id == 1, "dt_days"].tolist()[1:] == pytest.approx([1.0, 2.0])


def test_elapsed_seconds_minus_one_on_non_first_falls_back_to_days(tmp_path):
    rows = [
        _row(1, 0, 3, 0, -1, -1),
        _row(1, 3, 3, 2, 3, -1),  # seconds missing (-1), days=3
    ]
    df = load_user(_write(tmp_path, 1, rows))
    assert df.loc[1, "dt_days"] == pytest.approx(3.0)


def test_elapsed_seconds_missing_falls_back_to_kept_day_gap(tmp_path):
    # raw seconds and raw days are both missing: dt falls back to the kept day difference
    rows = [
        _row(1, 0, 3, 0, -1, -1),
        _row(1, 3, 3, 2, -1, -1),
    ]
    df = load_user(_write(tmp_path, 1, rows))
    assert df.loc[1, "elapsed_days"] == 3
    assert df.loc[1, "dt_days"] == pytest.approx(3.0)


def test_all_cards_dropped_returns_typed_empty_frame(tmp_path):
    rows = [
        _row(1, 0, 3, 2, 0, 0),
        _row(2, 0, 0, 0, -1, -1),
        _row(2, 1, 1, 1, 0, 60),
    ]
    df = load_user(_write(tmp_path, 1, rows))
    assert len(df) == 0
    assert list(df.columns) == LOAD_COLUMNS


def test_zero_row_file_returns_empty_frame(tmp_path):
    # a zero-row file written with the real schema (pandas drops columns from an empty frame)
    schema_row = _row(1, 0, 3, 0, -1, -1)
    frame = pd.DataFrame([schema_row]).iloc[:0]
    folder = tmp_path / "revlogs" / "user_id=3"
    folder.mkdir(parents=True)
    path = folder / "data.parquet"
    frame.to_parquet(path, index=False)
    df = load_user(path)
    assert len(df) == 0
    assert list(df.columns) == LOAD_COLUMNS


def test_missing_required_column_raises_clear_error(tmp_path):
    frame = pd.DataFrame([_row(1, 0, 3, 0, -1, -1)]).drop(columns=["state"])
    folder = tmp_path / "revlogs" / "user_id=4"
    folder.mkdir(parents=True)
    path = folder / "data.parquet"
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="state"):
        load_user(path)


def test_non_default_index_is_handled(tmp_path):
    rows = [
        _row(1, 0, 3, 0, -1, -1),
        _row(1, 1, 3, 2, 1, 86400),
    ]
    path = _write(tmp_path, 1, rows, index=[100, 7])
    df = load_user(path)
    assert df["review_idx"].tolist() == [0, 1]
    assert df["dt_days"].iloc[1] == pytest.approx(1.0)


def test_user_id_from_nested_path(tmp_path):
    path = _write(tmp_path, 8506, [_row(1, 0, 3, 0, -1, -1)])
    assert (load_user(path)["user_id"] == 8506).all()


def test_load_is_idempotent(tmp_path):
    rows = [_row(1, 0, 3, 0, -1, -1), _row(1, 1, 1, 2, 1, 86400)]
    path = _write(tmp_path, 1, rows)
    pd.testing.assert_frame_equal(load_user(path), load_user(path))


def test_labels_are_int_and_match_rating_threshold(tmp_path):
    rows = [_row(1, 0, 3, 0, -1, -1)] + [
        _row(1, i + 1, r, 2, i + 1, 86400 * (i + 1)) for i, r in enumerate([1, 2, 3, 4])
    ]
    df = load_user(_write(tmp_path, 1, rows))
    assert df["y"].tolist() == [1, 0, 1, 1, 1]
    assert df["y"].dtype == np.int64


def test_performance_310k_rows_under_five_seconds(tmp_path):
    rng = np.random.default_rng(0)
    n_cards = 27000
    rows = []
    for c in range(n_cards):
        rows.append(_row(c, 0, int(rng.integers(1, 5)), 0, -1, -1))
        for k in range(10):
            rows.append(_row(c, k + 1, int(rng.integers(1, 5)), 2, k + 1, (k + 1) * 86400))
    path = _write(tmp_path, 9, rows)
    t = time.time()
    df = load_user(path)
    elapsed = time.time() - t
    assert len(df) == len(rows)
    assert elapsed < 5.0, f"load_user took {elapsed:.2f}s"


@pytest.mark.parametrize("uid", REAL_USERS)
def test_real_user_invariants(uid):
    path = f"{REAL_ROOT}/user_id={uid}/data.parquet"
    df = load_user(path)
    assert len(df) > 0
    assert (df["rating"] != 0).all()
    assert (df["y"] == (df["rating"] > 1).astype(int)).all()
    first = df.groupby("card_id", sort=False).head(1)
    assert (first["state"] == 0).all()
    assert first["is_first"].all()
    assert df["is_first"].sum() == df["card_id"].nunique()
    assert df["dt_days"].isna().eq(df["is_first"]).all()
    assert (df.loc[~df["is_first"], "dt_days"] >= 0).all()
    assert df["review_idx"].tolist() == list(range(len(df)))
    assert df["day_offset"].is_monotonic_increasing


@pytest.mark.parametrize("uid", REAL_USERS)
def test_real_user_day_gap_equals_elapsed_days(uid):
    # The invariant the FSRS replay relies on: between consecutive kept reviews of a
    # card, the day_offset difference is exactly elapsed_days (first reviews: -1).
    df = load_user(f"{REAL_ROOT}/user_id={uid}/data.parquet")
    gap = df.groupby("card_id", sort=False)["day_offset"].diff()
    later = ~df["is_first"]
    assert (gap[later] == df.loc[later, "elapsed_days"]).all()
    assert (df.loc[df["is_first"], "elapsed_days"] == -1).all()
    assert (df.loc[later, "elapsed_days"] >= 0).all()
