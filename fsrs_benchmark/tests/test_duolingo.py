"""Tests for fsrs_benchmark.duolingo: scan cache, sampling, event definition, chunking.

All data is synthetic and written into pytest's tmp_path with the real column names.
"""
import os

import numpy as np
import pandas as pd
import pytest

from fsrs_benchmark import duolingo
from fsrs_benchmark.data import LOAD_COLUMNS, _empty_frame
from fsrs_benchmark.duolingo import load_users, sample_users, scan
from fsrs_benchmark.split import make_split

CSV_COLUMNS = [
    "p_recall", "timestamp", "delta", "user_id", "learning_language", "ui_language",
    "lexeme_id", "lexeme_string", "history_seen", "history_correct", "session_seen",
    "session_correct",
]
T = 1_600_000_000  # base unix time for the synthetic rows


def _row(user, lexeme, timestamp, p=0.9, delta=0):
    return dict(
        p_recall=p, timestamp=timestamp, delta=delta, user_id=user, learning_language="de",
        ui_language="en", lexeme_id=lexeme, lexeme_string="w", history_seen=1,
        history_correct=1, session_seen=1, session_correct=1,
    )


def _write(path, rows):
    """Write rows as a gzipped csv with the real Duolingo column names."""
    pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(path, index=False, compression="gzip")
    return path


def _spy_read_csv(monkeypatch):
    """Count calls to pandas.read_csv made by the duolingo module."""
    calls = []
    real = pd.read_csv

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(duolingo.pd, "read_csv", spy)
    return calls


def _counts_dict(counts):
    return dict(zip(counts["user_id"], counts["n_rows"]))


def test_scan_counts_rows_and_t0_across_chunks(tmp_path):
    rows = [_row("10", "a", T + 5), _row("10", "b", T + 1), _row("10", "a", T + 9),
            _row("2", "a", T + 3)]
    path = _write(tmp_path / "traces.csv.gz", rows)
    counts, t0 = scan(path, tmp_path / "cache.parquet", chunksize=2)
    assert _counts_dict(counts) == {"10": 3, "2": 1}
    assert list(counts["user_id"]) == ["10", "2"]  # sorted by string
    assert t0 == T + 1


def test_scan_reuses_cache_when_source_unchanged(tmp_path, monkeypatch):
    path = _write(tmp_path / "traces.csv.gz", [_row("1", "a", T), _row("2", "a", T + 1)])
    cache = tmp_path / "cache.parquet"
    first = scan(path, cache, chunksize=1)
    calls = _spy_read_csv(monkeypatch)
    second = scan(path, cache, chunksize=1)
    assert calls == []
    assert _counts_dict(second[0]) == _counts_dict(first[0])
    assert second[1] == first[1] == T


def test_scan_recomputes_when_size_changes(tmp_path):
    path = _write(tmp_path / "traces.csv.gz", [_row("1", "a", T)])
    cache = tmp_path / "cache.parquet"
    scan(path, cache)
    _write(path, [_row("1", "a", T), _row("1", "b", T + 100), _row("3", "a", T - 50)])
    counts, t0 = scan(path, cache)
    assert _counts_dict(counts) == {"1": 2, "3": 1}
    assert t0 == T - 50


def test_scan_recomputes_when_only_mtime_changes(tmp_path, monkeypatch):
    path = _write(tmp_path / "traces.csv.gz", [_row("1", "a", T)])
    cache = tmp_path / "cache.parquet"
    scan(path, cache)
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000_000))
    calls = _spy_read_csv(monkeypatch)
    scan(path, cache)
    assert len(calls) == 1


def test_sample_users_is_sorted_eligible_and_seeded(tmp_path):
    counts = pd.DataFrame({"user_id": ["1", "2", "3", "10"], "n_rows": [200, 50, 150, 100]})
    picked = sample_users(counts, 2, seed=0, min_rows=100)
    assert len(picked) == 2 and picked == sorted(picked)
    assert set(picked) <= {"1", "3", "10"}  # "2" has too few rows
    assert sample_users(counts, 2, seed=0, min_rows=100) == picked


def test_sample_users_none_returns_all_eligible_sorted(tmp_path):
    counts = pd.DataFrame({"user_id": ["1", "2", "3", "10"], "n_rows": [200, 50, 150, 100]})
    assert sample_users(counts, None, seed=0, min_rows=100) == ["1", "10", "3"]
    assert sample_users(counts, None, seed=7, min_rows=100) == ["1", "10", "3"]


def test_sample_users_raises_when_not_enough_eligible(tmp_path):
    counts = pd.DataFrame({"user_id": ["1", "2", "3", "10"], "n_rows": [200, 50, 150, 100]})
    with pytest.raises(ValueError, match="only 3 users have at least 100 rows"):
        sample_users(counts, 4, seed=0, min_rows=100)


def test_output_columns_and_dtypes_match_load_user_shape(tmp_path):
    path = _write(tmp_path / "traces.csv.gz", [_row("1", "a", T), _row("1", "a", T + 60)])
    counts, t0 = scan(path, tmp_path / "cache.parquet")
    out = load_users(path, ["1"], counts, t0, chunksize=1)
    assert list(out.columns) == LOAD_COLUMNS
    assert out.dtypes.to_dict() == _empty_frame().dtypes.to_dict()


def test_order_is_timestamp_then_lexeme_and_review_idx_is_dense(tmp_path):
    rows = [_row("1", "b", T + 100), _row("1", "a", T + 200), _row("1", "a", T + 50)]
    path = _write(tmp_path / "traces.csv.gz", rows)
    counts, t0 = scan(path, tmp_path / "cache.parquet")
    out = load_users(path, ["1"], counts, t0)
    # Card ids: "a" -> 0, "b" -> 1. Event order: a@50, b@100, a@200.
    assert out["card_id"].tolist() == [0, 1, 0]
    assert out["review_idx"].tolist() == [0, 1, 2]
    assert out["is_first"].tolist() == [True, True, False]
    assert out["dt_days"].iloc[2] == pytest.approx(150 / 86400)


def test_ties_break_by_lexeme_then_file_row(tmp_path):
    rows = [
        _row("1", "b", T),
        _row("1", "a", T),              # same time, lexeme "a" sorts first
        _row("1", "a", T, p=0.1),       # same time and lexeme: earlier file row first
        _row("1", "a", T, p=0.9),
    ]
    path = _write(tmp_path / "traces.csv.gz", rows)
    counts, t0 = scan(path, tmp_path / "cache.parquet")
    out = load_users(path, ["1"], counts, t0)
    assert out["card_id"].tolist() == [0, 0, 0, 1]
    # Three "a" rows at T in file order have p = 0.9, 0.1, 0.9 -> y = 1, 0, 1; then "b".
    assert out["y"].tolist() == [1, 0, 1, 1]
    assert out["review_idx"].tolist() == [0, 1, 2, 3]


def test_y_and_rating_mapping(tmp_path):
    rows = [_row("1", "a", T, p=0.5), _row("1", "b", T, p=0.49), _row("1", "c", T, p=0.95)]
    path = _write(tmp_path / "traces.csv.gz", rows)
    counts, t0 = scan(path, tmp_path / "cache.parquet")
    out = load_users(path, ["1"], counts, t0)
    by_card = out.set_index("card_id")
    assert by_card.loc[0, "y"] == 1 and by_card.loc[0, "rating"] == 3   # 0.5 counts as recall
    assert by_card.loc[1, "y"] == 0 and by_card.loc[1, "rating"] == 1   # 0.49 does not
    assert by_card.loc[2, "y"] == 1 and by_card.loc[2, "rating"] == 3


def test_first_row_state_and_gaps(tmp_path):
    rows = [
        _row("1", "a", T, p=0.9, delta=0),
        _row("1", "a", T + 2 * 86400 + 3600, p=0.2, delta=3 * 86400 + 5),
    ]
    path = _write(tmp_path / "traces.csv.gz", rows)
    counts, t0 = scan(path, tmp_path / "cache.parquet")
    out = load_users(path, ["1"], counts, t0)
    assert out["is_first"].tolist() == [True, False]
    assert out["state"].tolist() == [0, 2]
    assert out["elapsed_days"].tolist() == [-1, 2]
    assert np.isnan(out["dt_days"].iloc[0])
    assert out["dt_days"].iloc[1] == pytest.approx((2 * 86400 + 3600) / 86400)
    assert out["raw_elapsed_days"].tolist() == [0, 3]
    assert out["day_offset"].tolist() == [0, 2]


def test_card_ids_are_dense_per_user(tmp_path):
    rows = [_row("1", "z", T), _row("1", "m", T), _row("2", "q", T), _row("2", "a", T)]
    path = _write(tmp_path / "traces.csv.gz", rows)
    counts, t0 = scan(path, tmp_path / "cache.parquet")
    out = load_users(path, ["1", "2"], counts, t0)
    user1 = out[out["user_id"] == 0].sort_values("review_idx")
    user2 = out[out["user_id"] == 1].sort_values("review_idx")
    # user "1": lexemes m < z -> m=0, z=1 (same timestamp, so lexeme order decides)
    assert user1["card_id"].tolist() == [0, 1]
    # user "2": a < q -> a=0, q=1; ids restart at 0 for every user
    assert user2["card_id"].tolist() == [0, 1]


def test_int_user_ids_do_not_depend_on_the_sample(tmp_path):
    rows = [_row(u, "a", T) for u in ["10", "2", "33", "4"]]
    path = _write(tmp_path / "traces.csv.gz", rows)
    counts, t0 = scan(path, tmp_path / "cache.parquet")
    # Sorted string order: "10" < "2" < "33" < "4" -> ids 0, 1, 2, 3.
    only_33 = load_users(path, ["33"], counts, t0)
    assert only_33["user_id"].unique().tolist() == [2]
    ends = load_users(path, ["10", "4"], counts, t0)
    assert sorted(ends["user_id"].unique().tolist()) == [0, 3]


def test_chunk_size_does_not_change_the_output(tmp_path):
    rows = []
    for u in ["1", "2", "3"]:
        for k in range(12):
            rows.append(_row(u, f"lex{k % 4}", T + (k * 7919) % 500, p=0.1 * (k % 10), delta=k * 3600))
    path = _write(tmp_path / "traces.csv.gz", rows)
    counts, t0 = scan(path, tmp_path / "cache.parquet", chunksize=5)
    small = load_users(path, ["1", "3"], counts, t0, chunksize=1)
    big = load_users(path, ["1", "3"], counts, t0, chunksize=1_000_000)
    pd.testing.assert_frame_equal(small, big)


def test_unknown_user_id_raises(tmp_path):
    path = _write(tmp_path / "traces.csv.gz", [_row("1", "a", T)])
    counts, t0 = scan(path, tmp_path / "cache.parquet")
    with pytest.raises(ValueError, match="not in the counts table"):
        load_users(path, ["999"], counts, t0)


def test_make_split_accepts_duolingo_output(tmp_path):
    rows = []
    for u in ["1", "2"]:
        for k in range(8):
            rows.append(_row(u, "a", T + k * 2 * 86400, p=0.8 if k % 2 else 0.3))
            rows.append(_row(u, "b", T + k * 2 * 86400 + 60, p=0.6))
    path = _write(tmp_path / "traces.csv.gz", rows)
    counts, t0 = scan(path, tmp_path / "cache.parquet")
    frame = pd.concat([load_users(path, ["1"], counts, t0), load_users(path, ["2"], counts, t0)],
                      ignore_index=True)
    events = make_split(frame, max_cards=300, seed=0)
    assert len(events) == len(frame)
    assert events["event_id"].is_unique
    assert events["scored"].any()


def test_load_users_optional_lexeme_column(tmp_path):
    from fsrs_benchmark import duolingo
    from fsrs_benchmark.data import LOAD_COLUMNS
    from fsrs_benchmark.tests._synthetic_events import write_duolingo_csv
    csv = write_duolingo_csv(tmp_path / "traces.csv.gz")
    counts, t0 = duolingo.scan(csv, tmp_path / "counts.parquet")
    users = duolingo.sample_users(counts, None, 0, 1)
    plain = duolingo.load_users(csv, users, counts, t0)
    lex = duolingo.load_users(csv, users, counts, t0, with_lexeme=True)
    assert list(plain.columns) == LOAD_COLUMNS and list(lex.columns) == LOAD_COLUMNS + ["lexeme"]
    pd.testing.assert_frame_equal(plain, lex[LOAD_COLUMNS])
    assert lex["lexeme"].dtype == np.int64 and lex["lexeme"].min() >= 0
    # card = (user, lexeme): one lexeme per card
    assert (lex.groupby(["user_id", "card_id"])["lexeme"].nunique() == 1).all()
