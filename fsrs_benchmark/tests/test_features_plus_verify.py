"""Verifier checks for Phase 6 task 6.1 (srsb/features_plus.py): permutation leakage on real local users,
stream order, a naive per-row reference implementation of every group on real users, the old driver's
features unchanged, and the download helper's skip/error behaviour. Uses the local data in data/anki10k."""
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from fsrs_benchmark.models import gbdt as G
from fsrs_benchmark.srsb import features_plus as P
from fsrs_benchmark.srsb import run_gbdt as R
from fsrs_benchmark import data as D

REPO = Path(__file__).resolve().parents[2]
ANKI = REPO / "data" / "anki10k"


def _load(uid):
    rev = ANKI / "revlogs" / f"user_id={uid}" / "data.parquet"
    cp = ANKI / "cards" / f"user_id={uid}" / "data.parquet"
    dp = ANKI / "decks" / f"user_id={uid}" / "data.parquet"
    if not (rev.exists() and cp.exists() and dp.exists()):
        pytest.skip(f"local data for user {uid} not present")
    return pd.read_parquet(rev), pd.read_parquet(cp), pd.read_parquet(dp)


def _same_rows(a, b, cols):
    x = a[cols].to_numpy(float)
    y = b[cols].to_numpy(float)
    return np.array_equal(x, y, equal_nan=True)


# ---- 1. leakage with real users ------------------------------------------------------------------

@pytest.mark.parametrize("uid", [1236, 1347])
def test_permuting_rating_and_duration_of_rows_ge_t_keeps_rows_le_t(uid):
    raw, cards, decks = _load(uid)
    stream = R.review_stream(raw, max_card_reviews=200)
    base = P.plus_features(stream, cards, decks)
    n = len(stream)
    rng = np.random.default_rng(uid)
    for t in sorted({1, 2, n // 3, n // 2, n - 3}):
        changed = stream.copy()
        idx = np.arange(t, n)
        perm = rng.permutation(idx)                        # shuffle rating and duration among rows >= t
        changed.iloc[idx, changed.columns.get_loc("rating")] = stream["rating"].to_numpy()[perm]
        changed.iloc[idx, changed.columns.get_loc("duration")] = stream["duration"].to_numpy()[perm]
        other = P.plus_features(changed, cards, decks)
        assert _same_rows(base.iloc[:t + 1], other.iloc[:t + 1], P.PLUS_COLS), f"t={t}: row <= t changed"
        if t <= n - 10:                                   # the change is visible after t (not vacuous)
            assert not _same_rows(base.iloc[t + 1:], other.iloc[t + 1:], P.PLUS_COLS), f"t={t}: test vacuous"


@pytest.mark.parametrize("uid", [1236, 1347])
def test_current_row_rating_and_duration_do_not_affect_its_own_features(uid):
    raw, cards, decks = _load(uid)
    stream = R.review_stream(raw, max_card_reviews=200)
    base = P.plus_features(stream, cards, decks)
    n = len(stream)
    for t in [1, n // 4, n // 2, n - 2]:
        changed = stream.copy()
        changed.iloc[t, changed.columns.get_loc("rating")] = 5 - stream["rating"].iloc[t]   # 1<->4, 2<->3
        changed.iloc[t, changed.columns.get_loc("duration")] = stream["duration"].iloc[t] * 7 + 123
        other = P.plus_features(changed, cards, decks)
        assert _same_rows(base.iloc[:t + 1], other.iloc[:t + 1], P.PLUS_COLS), f"row {t} sees its own answer"


# ---- 2. stream order -----------------------------------------------------------------------------

@pytest.mark.parametrize("uid", [1236, 1347])
def test_stream_is_review_th_order_and_days_non_decreasing(uid):
    raw, _, _ = _load(uid)
    stream = R.review_stream(raw, max_card_reviews=200)
    assert stream.index.is_monotonic_increasing
    assert (stream.index.to_numpy() == stream["event_id"].to_numpy()).all()
    assert (np.diff(stream["event_id"].to_numpy()) > 0).all()
    # same-day ties are kept in file order (stable sort), and day_offset never decreases in the file
    assert (np.diff(stream["day_offset"].to_numpy()) >= 0).all()


def test_same_day_ties_use_file_order():
    raw = pd.DataFrame({
        "card_id": [1, 2, 1], "day_offset": [5, 5, 5], "rating": [3, 3, 1], "state": [0, 0, 1],
        "duration": [1000, 2000, 3000], "elapsed_days": [-1, -1, 0], "elapsed_seconds": [-1, -1, 10],
    })
    stream = R.review_stream(raw, max_card_reviews=10)
    out = P.plus_features(stream, pd.DataFrame({"card_id": [1, 2], "note_id": [9, 9], "deck_id": [1, 1]}),
                          pd.DataFrame({"deck_id": [1], "parent_id": [0], "preset_id": [0]}))
    assert out["n_today"].tolist() == [0, 1, 2]           # the later same-day row sees the earlier one
    assert out["lapses_today"].tolist() == [0, 0, 0]
    assert out["sib_n"].tolist() == [0, 1, 1]             # row 2 and row 3 see the other card of note 9
    assert out["sib_last_y"].tolist()[2] == 1.0           # card 2 answered 3 (y=1)


# ---- 3. naive per-row reference (hand-computed with plain loops over the past) --------------------

def _reference(stream, cards, decks, rows):
    """Plain-python definition of each feature for the given row positions, using only earlier rows."""
    card = stream["card_id"].to_numpy(np.int64)
    day = stream["day_offset"].to_numpy(np.int64)
    rating = stream["rating"].to_numpy(np.int64)
    state = stream["state"].to_numpy(np.int64)
    dur = stream["duration"].to_numpy(float)
    secs = stream["elapsed_seconds"].to_numpy(float)
    gap = stream["elapsed_days"].to_numpy(np.int64)
    seen, first = set(), np.zeros(len(card), bool)
    for i, c in enumerate(card):
        first[i] = c not in seen
        seen.add(c)
    y = (rating > 1).astype(float)
    L = np.log1p(dur)
    note_of = dict(zip(cards["card_id"], cards["note_id"]))
    deck_of = dict(zip(cards["card_id"], cards["deck_id"]))
    preset_of = dict(zip(decks["deck_id"], decks["preset_id"]))
    parent_of = dict(zip(decks["deck_id"], decks["parent_id"]))
    note = np.array([note_of.get(c, np.nan) for c in card], float)
    deck = np.array([deck_of.get(c, np.nan) for c in card], float)
    preset = np.array([preset_of.get(d, np.nan) if d == d else np.nan for d in deck], float)
    parent = np.array([parent_of.get(d, np.nan) if d == d else np.nan for d in deck], float)
    nan = np.nan

    def mean(v):
        return float(np.mean(v)) if len(v) else nan

    out = {}
    for t in rows:
        p = np.arange(t)
        same_card = p[card[:t] == card[t]]
        same_day = p[day[:t] == day[t]]
        el = secs[t] / 86400.0 if (not first[t] and secs[t] >= 0) else nan
        out.setdefault("el_frac", {})[t] = el
        out.setdefault("log_el_frac", {})[t] = np.log1p(el)
        out.setdefault("n_today", {})[t] = len(same_day)
        out.setdefault("n_first_today", {})[t] = int(first[same_day].sum())
        out.setdefault("rate_today", {})[t] = mean(y[same_day[~first[same_day]]])
        out.setdefault("lapses_today", {})[t] = int((rating[same_day] == 1).sum())
        out.setdefault("u_last5", {})[t] = mean(y[max(0, t - 5):t]) if t > 0 else nan
        out.setdefault("rows_since_card", {})[t] = nan if first[t] else float(t - same_card[-1])
        out.setdefault("dur_card_last", {})[t] = L[same_card[-1]] if len(same_card) else nan
        out.setdefault("dur_card_mean", {})[t] = mean(L[same_card])
        out.setdefault("dur_user_mean", {})[t] = mean(L[:t])
        out.setdefault("dur_user_prev", {})[t] = L[t - 1] if t > 0 else nan
        for k in (1, 2, 3, 4):
            out.setdefault(f"rc{k}", {})[t] = int((rating[same_card] == k).sum())
        nf_card = same_card[~first[same_card]]
        out.setdefault("card_max_gap", {})[t] = float(gap[nf_card].max()) if len(nf_card) else nan
        out.setdefault("state", {})[t] = float(state[t])
        if np.isnan(note[t]):
            sib = np.array([], int)
        else:
            sib = p[(note[:t] == note[t]) & (card[:t] != card[t])]
        out.setdefault("sib_n", {})[t] = nan if np.isnan(note[t]) else len(sib)
        out.setdefault("sib_last_y", {})[t] = y[sib[-1]] if len(sib) else nan
        out.setdefault("sib_days_since", {})[t] = float(day[t] - day[sib[-1]]) if len(sib) else nan
        sib_nf = sib[~first[sib]]
        out.setdefault("sib_rate", {})[t] = mean(y[sib_nf]) if not np.isnan(note[t]) else nan
        for name, key in (("deck", deck), ("preset", preset), ("parent", parent)):
            if np.isnan(key[t]):
                out.setdefault(f"{name}_n", {})[t] = nan
                out.setdefault(f"{name}_rate", {})[t] = nan
                continue
            pk = p[key[:t] == key[t]]
            out.setdefault(f"{name}_n", {})[t] = len(pk)
            out.setdefault(f"{name}_rate", {})[t] = mean(y[pk[~first[pk]]])
    return out


def _check_against_reference(uid, rows):
    raw, cards, decks = _load(uid)
    stream = R.review_stream(raw, max_card_reviews=200)
    got = P.plus_features(stream, cards, decks)
    ref = _reference(stream, cards, decks, rows)
    for col, vals in ref.items():
        g = got[col].to_numpy(float)[rows]
        e = np.array([vals[t] for t in rows], float)
        assert np.allclose(g, e, rtol=1e-9, atol=1e-12, equal_nan=True), (
            f"user {uid} {col}: first mismatch at rows "
            f"{[int(rows[i]) for i in np.where(~np.isclose(g, e, rtol=1e-9, atol=1e-12, equal_nan=True))[0][:5]]}")
    return ref


def test_reference_matches_all_rows_of_small_real_user():
    raw, _, _ = _load(1236)
    n = len(R.review_stream(raw, max_card_reviews=200))
    _check_against_reference(1236, np.arange(n))


def test_reference_matches_sampled_rows_of_largest_local_user():
    raw, _, _ = _load(8060)
    n = len(R.review_stream(raw, max_card_reviews=200))
    rows = np.sort(np.random.default_rng(0).choice(n, size=150, replace=False))
    _check_against_reference(8060, rows)


# ---- 4. the old driver is unchanged --------------------------------------------------------------

def test_old_driver_features_unchanged_by_new_columns():
    """base_features reads only its own columns, so the extra review_stream columns must not change the
    past_only_features output: compare against base_features on the old column set."""
    raw, _, _ = _load(1347)
    ev = R.review_stream(raw, max_card_reviews=200)
    old = ev.drop(columns=["state", "duration", "elapsed_seconds"])
    a = G.base_features(old, span=20, user_group="row")[R.FEATURE_COLS]
    b = R.past_only_features(raw, 200)
    assert _same_rows(a, b, R.FEATURE_COLS)


# ---- 5. download_side_tables -----------------------------------------------------------------------

def test_download_skips_existing_and_never_prints_token(tmp_path, capsys):
    (tmp_path / "cards" / "user_id=5").mkdir(parents=True)
    (tmp_path / "cards" / "user_id=5" / "data.parquet").write_bytes(b"x")
    (tmp_path / "decks" / "user_id=5").mkdir(parents=True)
    (tmp_path / "decks" / "user_id=5" / "data.parquet").write_bytes(b"x")
    calls = []

    def fake(**kw):
        calls.append(kw["filename"])
        p = tmp_path / kw["filename"]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"y")
        return str(p)

    with mock.patch.object(D, "hf_hub_download", side_effect=fake):
        res = D.download_side_tables([5, 3], tmp_path)
    assert calls == ["cards/user_id=3/data.parquet", "decks/user_id=3/data.parquet"]
    assert res["cards"][0] == tmp_path / "cards/user_id=3/data.parquet"
    assert len(res["cards"]) == 2 and len(res["decks"]) == 2
    out = capsys.readouterr()
    assert "hf_" not in (out.out + out.err)


def test_download_wraps_http_error_without_token(tmp_path):
    from huggingface_hub.errors import HfHubHTTPError
    err = HfHubHTTPError("404 Client Error", response=mock.Mock(status_code=404))
    with mock.patch.object(D, "hf_hub_download", side_effect=err):
        with pytest.raises(RuntimeError, match="Could not access"):
            D.download_side_tables([7], tmp_path)
