"""Tests for fsrs_benchmark.data: event definition, labels, gaps, first flags, download."""
import numpy as np
import pandas as pd
import pytest
from huggingface_hub.errors import HfHubHTTPError

from fsrs_benchmark import data
from fsrs_benchmark.data import LOAD_COLUMNS, download_users, load_user


def _write_user(root, user_id, rows, drop_seconds=False):
    """Write a synthetic revlog to root/revlogs/user_id=<N>/data.parquet."""
    frame = pd.DataFrame(rows)
    if drop_seconds:
        frame = frame.drop(columns=["elapsed_seconds"])
    folder = root / "revlogs" / f"user_id={user_id}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "data.parquet"
    frame.to_parquet(path, index=False)
    return path


def _row(card, day, rating, state, elapsed_days, elapsed_seconds):
    return dict(card_id=card, day_offset=day, rating=rating, state=state,
                duration=1000, elapsed_days=elapsed_days, elapsed_seconds=elapsed_seconds)


@pytest.fixture
def synthetic_path(tmp_path):
    rows = [
        # card 1: new, then Good, then Again, then Hard (learning -> review)
        _row(1, 0, 3, 0, -1, -1),
        _row(1, 1, 1, 2, 1, 86400 * 1),       # elapsed 1 day, Again -> y=0
        _row(1, 3, 4, 2, 2, 86400 * 2.5),     # Easy, 2.5 days -> y=1
        # card 2: first kept review is state 1 (learning) -> whole card dropped
        _row(2, 0, 3, 1, 0, 3600),
        _row(2, 1, 3, 2, 1, 86400),
        # card 3: manual reschedule (rating 0) first, then a real new review
        _row(3, 0, 0, 0, -1, -1),             # dropped: rating 0
        _row(3, 0, 2, 0, -1, -1),             # kept first review (state 0)
        _row(3, 0, 3, 1, 0, 0),              # same-day review, Good -> y=1
        # card 4: state 4 row (filtered deck, real rating) is KEPT as a recall event
        _row(4, 0, 1, 0, -1, -1),
        _row(4, 1, 2, 4, 1, 86400),           # state 4: kept, Hard -> y=1
        _row(4, 2, 1, 2, 2, 86400 * 1.0),     # Again -> y=0
    ]
    return _write_user(tmp_path, 42, rows)


def test_filters_rating0_and_non_new_cards_keeps_state4(synthetic_path):
    df = load_user(synthetic_path)
    assert list(df.columns) == LOAD_COLUMNS
    # card 2 gone (first kept row is learning), rating-0 row gone, state-4 row kept
    assert set(df["card_id"]) == {1, 3, 4}
    assert len(df) == 3 + 2 + 3
    assert (df["rating"] != 0).all()
    assert (df.loc[df["card_id"] == 4, "state"] == 4).sum() == 1


def test_user_id_from_folder_and_review_idx_is_contiguous(synthetic_path):
    df = load_user(synthetic_path)
    assert (df["user_id"] == 42).all()
    assert df["review_idx"].tolist() == list(range(len(df)))


def test_file_order_is_kept(synthetic_path):
    df = load_user(synthetic_path)
    # card 1 rows first, then card 3, then card 4 (file order)
    assert df["card_id"].tolist() == [1, 1, 1, 3, 3, 4, 4, 4]


def test_labels_rating_above_one_is_recall(synthetic_path):
    df = load_user(synthetic_path)
    by_rating = dict(zip(df["rating"], df["y"]))
    assert by_rating == {1: 0, 2: 1, 3: 1, 4: 1}
    assert df["y"].dtype.kind == "i"


def test_gap_from_seconds(synthetic_path):
    df = load_user(synthetic_path)
    card1 = df[df["card_id"] == 1].reset_index(drop=True)
    assert np.isnan(card1.loc[0, "dt_days"])               # first review: no gap
    assert card1.loc[1, "dt_days"] == pytest.approx(1.0)
    assert card1.loc[2, "dt_days"] == pytest.approx(2.5)


def test_same_day_gap_is_zero(synthetic_path):
    df = load_user(synthetic_path)
    card3 = df[df["card_id"] == 3].reset_index(drop=True)
    assert card3.loc[1, "dt_days"] == pytest.approx(0.0)


def test_first_flags(synthetic_path):
    df = load_user(synthetic_path)
    firsts = df.loc[df["is_first"], "card_id"].tolist()
    assert firsts == [1, 3, 4]
    assert df["is_first"].dtype == bool


def test_gap_falls_back_to_elapsed_days_when_seconds_missing(tmp_path):
    rows = [
        _row(7, 0, 3, 0, -1, -1),
        _row(7, 4, 3, 2, 4, 0),
    ]
    path = _write_user(tmp_path, 5, rows, drop_seconds=True)
    df = load_user(path)
    assert np.isnan(df.loc[0, "dt_days"])
    assert df.loc[1, "dt_days"] == pytest.approx(4.0)


def test_gap_spans_a_dropped_rating0_row(tmp_path):
    # first (day 0) -> rating-0 reschedule (day 1, dropped) -> kept review (day 2).
    # The kept gap is measured between kept rows: 2 days. The raw elapsed_days (1) refers
    # to the dropped row, and dt_days sums both raw seconds: 86400 + 86400 = 2 days.
    rows = [
        _row(1, 0, 3, 0, -1, -1),
        _row(1, 1, 0, 0, 1, 86400),
        _row(1, 2, 3, 2, 1, 86400),
    ]
    df = load_user(_write_user(tmp_path, 8, rows))
    assert df["card_id"].tolist() == [1, 1]
    assert df.loc[1, "elapsed_days"] == 2
    assert df.loc[1, "raw_elapsed_days"] == 1
    assert df.loc[1, "dt_days"] == pytest.approx(2.0)


def test_empty_after_filtering_returns_empty_frame(tmp_path):
    rows = [_row(9, 0, 3, 2, 0, 0)]  # only card starts in review state -> dropped
    path = _write_user(tmp_path, 6, rows)
    df = load_user(path)
    assert len(df) == 0
    assert list(df.columns) == LOAD_COLUMNS


def test_path_without_user_id_raises(tmp_path):
    path = _write_user(tmp_path, 1, [_row(1, 0, 3, 0, -1, -1)])
    bad = tmp_path / "data.parquet"
    path.rename(bad)
    with pytest.raises(ValueError, match="user_id"):
        load_user(bad)


# ---- download_users ---------------------------------------------------------

class _Forbidden(HfHubHTTPError):
    """Stands in for the 403 that HF returns without accepted dataset terms."""

    def __init__(self, message):
        Exception.__init__(self, message)


class _FakeApi:
    files = []

    def list_repo_files(self, repo_id, repo_type):
        return list(self.files)


def _fake_repo_files(n):
    files = [".gitattributes", "cards/user_id=1/data.parquet"]
    files += [f"revlogs/user_id={u}/data.parquet" for u in range(1, n + 1)]
    return files


@pytest.fixture
def fake_hub(monkeypatch, tmp_path):
    calls = []
    _FakeApi.files = _fake_repo_files(50)
    monkeypatch.setattr(data, "HfApi", _FakeApi)
    monkeypatch.setattr(data, "get_token", lambda: "hf_test")

    def fake_download(repo_id, filename, repo_type, local_dir):
        calls.append(filename)
        target = tmp_path / "out" / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")
        return str(target)

    monkeypatch.setattr(data, "hf_hub_download", fake_download)
    return calls


def test_download_is_deterministic_and_sized(fake_hub, tmp_path):
    out = tmp_path / "out"
    first = download_users(5, seed=0, out_dir=out)
    users_first = [int(p.parent.name.split("=")[1]) for p in first]
    assert len(first) == 5
    assert users_first == sorted(users_first)

    fake_hub.clear()
    again = download_users(5, seed=0, out_dir=out)
    assert again == first
    assert fake_hub == []  # all files already exist, nothing re-downloaded


def test_download_seed_changes_sample(fake_hub, tmp_path):
    a = download_users(5, seed=0, out_dir=tmp_path / "a")
    b = download_users(5, seed=1, out_dir=tmp_path / "b")
    assert a != b


def test_download_skips_existing_files(fake_hub, tmp_path):
    out = tmp_path / "out"
    paths = download_users(3, seed=0, out_dir=out)
    fake_hub.clear()
    download_users(3, seed=0, out_dir=out)
    assert fake_hub == []
    assert all(p.exists() for p in paths)


def test_download_without_token_fails_clearly(monkeypatch, tmp_path):
    monkeypatch.setattr(data, "get_token", lambda: None)
    with pytest.raises(RuntimeError, match="token"):
        download_users(2, seed=0, out_dir=tmp_path)


def test_download_licence_error_is_wrapped(monkeypatch, tmp_path):
    class DeniedApi:
        def list_repo_files(self, repo_id, repo_type):
            raise _Forbidden("403 Forbidden")

    monkeypatch.setattr(data, "get_token", lambda: "hf_test")
    monkeypatch.setattr(data, "HfApi", DeniedApi)
    with pytest.raises(RuntimeError, match="licence"):
        download_users(2, seed=0, out_dir=tmp_path)


def test_too_many_users_rejected(fake_hub, tmp_path):
    with pytest.raises(ValueError):
        download_users(10_000, seed=0, out_dir=tmp_path)


def test_user_id_cache_is_used_when_listing_fails(fake_hub, tmp_path, monkeypatch):
    """The first listing is cached in out_dir/user_ids.json. Later calls use the cache, so a
    failing listing (no network) still gives the same sample."""
    out = tmp_path / "out"
    first = download_users(5, seed=0, out_dir=out)
    assert (out / "user_ids.json").exists()

    class _Down:
        def list_repo_files(self, repo_id, repo_type):
            raise _Forbidden("network down")

    monkeypatch.setattr(data, "HfApi", _Down)
    again = download_users(5, seed=0, out_dir=out)
    assert again == first
