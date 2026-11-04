"""Verifier tests for fsrs_benchmark.duolingo (Task 5.1): gaps not covered by test_duolingo.py.

All data is synthetic, written into tmp_path with the real Duolingo column names.
"""
import numpy as np
import pandas as pd
import pytest

from fsrs_benchmark.data import LOAD_COLUMNS, _empty_frame
from fsrs_benchmark.duolingo import load_users, sample_users, scan
from fsrs_benchmark.split import make_split

CSV_COLUMNS = [
    "p_recall", "timestamp", "delta", "user_id", "learning_language", "ui_language",
    "lexeme_id", "lexeme_string", "history_seen", "history_correct", "session_seen",
    "session_correct",
]
T = 1_600_000_000


def _row(user, lexeme, timestamp, p=0.9, delta=0):
    return dict(
        p_recall=p, timestamp=timestamp, delta=delta, user_id=user, learning_language="de",
        ui_language="en", lexeme_id=lexeme, lexeme_string="w", history_seen=1,
        history_correct=1, session_seen=1, session_correct=1,
    )


def _write(path, rows):
    pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(path, index=False, compression="gzip")
    return path


def _load(tmp_path, rows, users, **kw):
    path = _write(tmp_path / "traces.csv.gz", rows)
    counts, t0 = scan(path, tmp_path / "cache.parquet")
    return load_users(path, users, counts, t0, **kw), counts, t0, path


# ---------- sample_users ----------

def test_sample_none_with_no_eligible_users_raises():
    counts = pd.DataFrame({"user_id": ["1", "2"], "n_rows": [3, 4]})
    with pytest.raises(ValueError, match="no users have at least 100 rows"):
        sample_users(counts, None, seed=0, min_rows=100)


def test_sample_zero_users_raises_clear_error():
    # Phase 5 spec says --n_users 0 means ALL users; the runner (5.2) must map 0 -> None.
    # sample_users itself rejects 0 with a message.
    counts = pd.DataFrame({"user_id": ["1"], "n_rows": [200]})
    with pytest.raises(ValueError, match="at least 1"):
        sample_users(counts, 0, seed=0, min_rows=100)


def test_sample_min_rows_boundary_is_inclusive_for_sampling():
    counts = pd.DataFrame({"user_id": ["a", "b"], "n_rows": [5, 4]})
    assert sample_users(counts, None, seed=0, min_rows=5) == ["a"]


def test_sample_is_independent_of_counts_row_order():
    c1 = pd.DataFrame({"user_id": ["1", "2", "3", "4", "5"], "n_rows": [9] * 5})
    c2 = c1.iloc[::-1].reset_index(drop=True)
    assert sample_users(c1, 3, seed=42, min_rows=1) == sample_users(c2, 3, seed=42, min_rows=1)


def test_sample_full_sample_equals_all_eligible():
    counts = pd.DataFrame({"user_id": ["1", "2", "3"], "n_rows": [9, 9, 9]})
    assert sample_users(counts, 3, seed=5, min_rows=1) == ["1", "2", "3"]


# ---------- scan ----------

def test_scan_header_only_file_raises(tmp_path):
    path = _write(tmp_path / "traces.csv.gz", [])
    with pytest.raises(ValueError, match="has no rows"):
        scan(path, tmp_path / "cache.parquet")


def test_scan_corrupt_cache_is_recomputed(tmp_path):
    path = _write(tmp_path / "traces.csv.gz", [_row("1", "a", T), _row("2", "a", T - 5)])
    cache = tmp_path / "cache.parquet"
    cache.write_bytes(b"not a parquet file")
    counts, t0 = scan(path, cache)
    assert dict(zip(counts["user_id"], counts["n_rows"])) == {"1": 1, "2": 1}
    assert t0 == T - 5


def test_scan_user_id_na_string_is_kept(tmp_path):
    # BUG candidate: pandas reads "NA"/"nan" user ids as NaN, and groupby drops NaN,
    # so such a user is missing from the counts table with no warning.
    rows = [_row("NA", "a", T), _row("nan", "a", T), _row("u1", "a", T)]
    path = _write(tmp_path / "traces.csv.gz", rows)
    counts, _ = scan(path, tmp_path / "cache.parquet")
    assert sorted(counts["user_id"].tolist()) == ["NA", "nan", "u1"]


# ---------- load_users: event definition edge cases ----------

def test_single_row_user(tmp_path):
    out, _, _, _ = _load(tmp_path, [_row("1", "a", T + 10)], ["1"])
    assert len(out) == 1
    assert out["is_first"].tolist() == [True]
    assert out["state"].tolist() == [0]
    assert out["elapsed_days"].tolist() == [-1]
    assert np.isnan(out["dt_days"].iloc[0])
    assert out["review_idx"].tolist() == [0]
    assert out["card_id"].tolist() == [0]


def test_identical_timestamps_same_card_give_zero_gap(tmp_path):
    rows = [_row("1", "a", T, p=0.9), _row("1", "a", T, p=0.2)]
    out, _, _, _ = _load(tmp_path, rows, ["1"])
    assert out["elapsed_days"].tolist() == [-1, 0]
    assert out["dt_days"].iloc[1] == 0.0
    assert out["raw_elapsed_days"].tolist() == [0, 0]


def test_day_offset_gap_and_dt_can_disagree(tmp_path):
    # Same card, 86399 s apart but in different day buckets (T0 = T, so day 0 and day 0 for
    # T+86399; then T+86400 is day 1). elapsed_days uses day_offset, dt_days uses seconds.
    rows = [_row("1", "a", T), _row("1", "a", T + 86399), _row("1", "a", T + 86400)]
    out, _, _, _ = _load(tmp_path, rows, ["1"])
    assert out["day_offset"].tolist() == [0, 0, 1]
    assert out["elapsed_days"].tolist() == [-1, 0, 1]
    assert out["dt_days"].iloc[1] == pytest.approx(86399 / 86400)


def test_t0_is_global_min_not_sampled_user_min(tmp_path):
    rows = [_row("1", "a", T + 3 * 86400), _row("2", "a", T)]
    path = _write(tmp_path / "traces.csv.gz", rows)
    counts, t0 = scan(path, tmp_path / "cache.parquet")
    assert t0 == T
    out = load_users(path, ["1"], counts, t0)
    assert out["day_offset"].tolist() == [3]


def test_user_int_ids_are_contiguous_over_all_users_and_grouped_in_order(tmp_path):
    rows = [_row("b", "x", T + 5), _row("a", "x", T + 1), _row("a", "y", T)]
    path = _write(tmp_path / "traces.csv.gz", rows)
    counts, t0 = scan(path, tmp_path / "cache.parquet")
    out = load_users(path, ["a", "b"], counts, t0)
    assert sorted(out["user_id"].unique().tolist()) == [0, 1]
    # sorted by user then review_idx
    assert out["user_id"].tolist() == sorted(out["user_id"].tolist())
    assert out["review_idx"].tolist() == [0, 1, 0]


def test_p_recall_nan_raises_with_row_count(tmp_path):
    # A missing p_recall cannot give a y, so the loader refuses instead of guessing.
    rows = [_row("1", "a", T, p=float("nan")), _row("1", "b", T, p=0.9)]
    with pytest.raises(ValueError, match="1 kept rows have a missing p_recall"):
        _load(tmp_path, rows, ["1"])


def test_empty_user_list_returns_empty_frame_with_same_dtypes(tmp_path):
    path = _write(tmp_path / "traces.csv.gz", [_row("1", "a", T)])
    counts, t0 = scan(path, tmp_path / "cache.parquet")
    out = load_users(path, [], counts, t0)
    assert len(out) == 0
    assert list(out.columns) == LOAD_COLUMNS
    assert out.dtypes.to_dict() == _empty_frame().dtypes.to_dict()


def test_negative_delta_floor_is_applied(tmp_path):
    out, _, _, _ = _load(tmp_path, [_row("1", "a", T, delta=-1)], ["1"])
    assert out["raw_elapsed_days"].tolist() == [-1]


def test_load_users_reads_cache_counts_not_file_for_validation(tmp_path):
    # A user present in the file but absent from a stale counts table must raise, not vanish.
    path = _write(tmp_path / "traces.csv.gz", [_row("1", "a", T), _row("2", "a", T)])
    counts = pd.DataFrame({"user_id": ["1"], "n_rows": [1]})
    with pytest.raises(ValueError, match="not in the counts table"):
        load_users(path, ["2"], counts, T)


# ---------- split integration ----------

def test_make_split_on_single_row_and_identical_timestamp_users(tmp_path):
    rows = [_row("1", "a", T), _row("2", "a", T), _row("2", "a", T)]
    path = _write(tmp_path / "traces.csv.gz", rows)
    counts, t0 = scan(path, tmp_path / "cache.parquet")
    frame = pd.concat([load_users(path, ["1"], counts, t0), load_users(path, ["2"], counts, t0)],
                      ignore_index=True)
    events = make_split(frame, max_cards=300, seed=0)
    assert len(events) == len(frame)
    assert events["event_id"].is_unique
