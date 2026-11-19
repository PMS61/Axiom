"""Tests for fsrs_benchmark/srsb/features_plus.py (Phase 6, task 6.1): hand-computed toy values, the past-only
leakage check, the missing-cards/decks behaviour and one real local user."""
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fsrs_benchmark.models import gbdt as G
from fsrs_benchmark.srsb import features_plus as P
from fsrs_benchmark.srsb import run_gbdt as R

NAN = np.nan
REPO = Path(__file__).resolve().parents[2]
REAL_USER = 1236                       # a local user outside the test set (users 1-1000 are never touched)


def _raw_toy():
    """12 reviews of 3 cards. Columns as in an Anki revlog (durations in ms, elapsed_seconds -1 on a first review).
    Card A (1) note 10, deck 100; card B (2) note 10, deck 101; card C (3) note 20, deck 200."""
    rows = [
        # card, day, rating, state, duration, elapsed_seconds, elapsed_days
        (1, 0, 3, 0, 1000, -1, -1),        # row 1  A first
        (2, 0, 1, 0, 2000, -1, -1),        # row 2  B first
        (1, 1, 2, 1, 4000, 86400, 1),      # row 3  A
        (3, 1, 4, 0, 1000, -1, -1),        # row 4  C first
        (1, 1, 1, 1, 8000, 43200, 0),      # row 5  A (same day)
        (2, 3, 3, 2, 1000, 259200, 3),     # row 6  B
        (1, 3, 1, 1, 3000, 172800, 2),     # row 7  A
        (3, 3, 2, 0, 2000, 172800, 2),     # row 8  C
        (1, 3, 4, 0, 1000, 0, 0),          # row 9  A (same day, 0 s)
        (2, 4, 4, 3, 5000, 86400, 1),      # row 10 B
        (3, 4, 1, 0, 4000, 259200, 3),     # row 11 C
        (1, 5, 2, 1, 2000, 172800, 2),     # row 12 A
    ]
    return pd.DataFrame(rows, columns=["card_id", "day_offset", "rating", "state", "duration",
                                       "elapsed_seconds", "elapsed_days"])


def _toy_refs():
    cards = pd.DataFrame({"card_id": [1, 2, 3], "note_id": [10, 10, 20], "deck_id": [100, 101, 200]})
    decks = pd.DataFrame({"deck_id": [100, 101, 200], "parent_id": [50, 55, 50], "preset_id": [7, 7, 8]})
    return cards, decks


def _toy_stream():
    # review_stream stamps review_th = position + 1 and keeps the columns plus-features needs
    return R.review_stream(_raw_toy(), max_card_reviews=100)


def _f(arr):
    return np.asarray(arr, dtype=float)


def _assert_col(df, name, expected):
    got = df[name].to_numpy(float)
    exp = _f(expected)
    assert got.shape == exp.shape, name
    assert np.allclose(got, exp, rtol=1e-9, atol=1e-12, equal_nan=True), f"{name}: got {got}, want {exp}"


def _params():
    d = float(G.W[20])
    factor = 0.9 ** (-1.0 / d) - 1.0
    return d, factor


# ---- (a) hand-computed toy values --------------------------------------------------------------

def test_toy_values_match_hand_computation():
    stream = _toy_stream()
    cards, decks = _toy_refs()
    out = P.plus_features(stream, cards, decks)
    assert list(out.columns) == P.PLUS_COLS
    assert list(out.index) == list(stream.index)

    # secs: row 3 = 1 day, row 5 = 0.5 day, row 9 = 0 (same day, 0 s); first reviews NaN
    _assert_col(out, "el_frac", [NAN, NAN, 1.0, NAN, 0.5, 3.0, 2.0, 2.0, 0.0, 1.0, 3.0, 2.0])
    _assert_col(out, "log_el_frac", [NAN, NAN, math.log(2), NAN, math.log(1.5), math.log(4), math.log(3),
                                     math.log(3), 0.0, math.log(2), math.log(4), math.log(3)])
    # f_R_secs: FSRS-6 default state before the row (A after its first 3 -> S=w[2]; B after 1 -> w[0];
    # C after 4 -> w[3]), R evaluated at the fractional gap
    d, factor = _params()
    r = lambda t, s: (1.0 + factor * t / s) ** (-d)
    assert out.loc[3, "f_R_secs"] == pytest.approx(r(1.0, G.W[2]), rel=1e-9)
    assert out.loc[6, "f_R_secs"] == pytest.approx(r(3.0, G.W[0]), rel=1e-9)
    assert out.loc[8, "f_R_secs"] == pytest.approx(r(2.0, G.W[3]), rel=1e-9)
    assert np.isnan(out.loc[1, "f_R_secs"]) and np.isnan(out.loc[4, "f_R_secs"])

    # today
    _assert_col(out, "n_today", [0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 0])
    _assert_col(out, "n_first_today", [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    _assert_col(out, "rate_today", [NAN, NAN, NAN, 1.0, 1.0, NAN, 1.0, 0.5, 2 / 3, NAN, 1.0, NAN])
    _assert_col(out, "lapses_today", [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0])
    _assert_col(out, "u_last5", [NAN, 1.0, 0.5, 2 / 3, 0.75, 0.6, 0.6, 0.6, 0.6, 0.6, 0.8, 0.6])
    _assert_col(out, "rows_since_card", [NAN, NAN, 2, NAN, 2, 4, 2, 4, 2, 4, 3, 3])

    # dur (log1p of ms)
    L = math.log1p
    _assert_col(out, "dur_card_last", [NAN, NAN, L(1000), NAN, L(4000), L(2000), L(8000), L(1000),
                                       L(3000), L(1000), L(2000), L(1000)])
    _assert_col(out, "dur_user_prev", [NAN, L(1000), L(2000), L(4000), L(1000), L(8000), L(1000), L(3000),
                                       L(2000), L(1000), L(5000), L(4000)])
    assert out.loc[3, "dur_card_last_rel"] == pytest.approx(
        L(1000) - (L(1000) + L(2000)) / 2, rel=1e-9)
    assert out.loc[12, "dur_card_mean"] == pytest.approx(
        np.mean([L(1000), L(4000), L(8000), L(3000), L(1000)]), rel=1e-9)
    assert np.isnan(out.loc[1, "dur_user_mean"])                               # nothing before row 1
    assert out.loc[2, "dur_user_mean"] == pytest.approx(L(1000), rel=1e-9)

    # rcount
    _assert_col(out, "rc2", [0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1])
    _assert_col(out, "rc4", [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1])
    _assert_col(out, "card_max_gap", [NAN, NAN, NAN, NAN, 1, NAN, 1, NAN, 2, 3, 2, 2])

    # state: the current row's review type
    _assert_col(out, "state", [0, 0, 1, 0, 1, 2, 1, 0, 0, 3, 0, 1])

    # sib: other cards of the same note (A and B share note 10)
    _assert_col(out, "sib_n", [0, 1, 1, 0, 1, 3, 2, 0, 2, 5, 0, 3])
    _assert_col(out, "sib_last_y", [NAN, 1, 0, NAN, 0, 0, 1, NAN, 1, 1, NAN, 1])
    _assert_col(out, "sib_days_since", [NAN, 0, 1, NAN, 1, 2, 0, NAN, 0, 1, NAN, 1])
    _assert_col(out, "sib_rate", [NAN, NAN, NAN, NAN, NAN, 0.5, 1.0, NAN, 1.0, 0.5, NAN, 1.0])

    # deck: decks 100 (A), 101 (B), 200 (C); preset 7 = decks 100 and 101; parent 50 = decks 100 and 200
    _assert_col(out, "deck_n", [0, 0, 1, 0, 2, 1, 3, 1, 4, 2, 2, 5])
    _assert_col(out, "deck_rate", [NAN, NAN, NAN, NAN, 1.0, NAN, 0.5, NAN, 1 / 3, 1.0, 1.0, 0.5])
    _assert_col(out, "preset_n", [0, 1, 2, 0, 3, 4, 5, 1, 6, 7, 2, 8])
    _assert_col(out, "preset_rate", [NAN, NAN, NAN, NAN, 1.0, 0.5, 2 / 3, NAN, 0.5, 0.6, 1.0, 2 / 3])
    _assert_col(out, "parent_n", [0, 0, 1, 2, 3, 1, 4, 5, 6, 2, 7, 8])
    _assert_col(out, "parent_rate", [NAN, NAN, NAN, 1.0, 1.0, NAN, 0.5, 1 / 3, 0.5, 1.0, 0.6, 0.5])


def test_toy_groups_cover_all_columns():
    out = P.plus_features(_toy_stream(), *_toy_refs())
    assert sorted(P.PLUS_COLS) == sorted(c for cols in P.PLUS_GROUPS.values() for c in cols)
    assert set(out.columns) == set(P.PLUS_COLS)


# ---- (b) leakage ---------------------------------------------------------------------------------

def _synthetic_raw(n_cards=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(1, n_cards + 1):
        day = int(rng.integers(0, 30))
        for j in range(int(rng.integers(2, 12))):
            if j > 0:
                day += int(rng.integers(0, 6))
            rows.append(dict(card_id=c, day_offset=day, rating=int(rng.integers(1, 5)),
                             state=int(rng.integers(0, 4)), duration=int(rng.integers(500, 20000))))
    raw = pd.DataFrame(rows).sort_values(["day_offset", "card_id"], kind="stable").reset_index(drop=True)
    raw["elapsed_days"] = -1
    raw["elapsed_seconds"] = -1
    last = {}
    for i, row in raw.iterrows():
        c = row["card_id"]
        if c in last:
            days = int(row["day_offset"] - last[c])
            raw.at[i, "elapsed_days"] = days
            raw.at[i, "elapsed_seconds"] = days * 86400 + int(rng.integers(0, 86400))
        last[c] = row["day_offset"]
    return raw


def _synthetic_refs(n_cards=40):
    cards = pd.DataFrame({"card_id": np.arange(1, n_cards + 1), "note_id": (np.arange(n_cards) // 3) + 1,
                          "deck_id": np.array([100, 101, 102, 200])[np.arange(n_cards) % 4]})
    decks = pd.DataFrame({"deck_id": [100, 101, 102, 200], "parent_id": [50, 50, 55, 60],
                          "preset_id": [7, 7, 8, 8]})
    return cards, decks


def _same(a, b):
    a = a.to_numpy(float)
    b = b.to_numpy(float)
    return np.array_equal(a, b, equal_nan=True)


@pytest.mark.parametrize("seed", [0, 1])
def test_leakage_rating_and_duration_of_row_and_later(seed):
    raw = _synthetic_raw(seed=seed)
    stream = R.review_stream(raw, max_card_reviews=100)
    cards, decks = _synthetic_refs()
    base = P.plus_features(stream, cards, decks)
    n = len(stream)
    for t in sorted({0, 1, n // 4, n // 2, (3 * n) // 4, n - 2}):
        changed = stream.copy()
        later = changed.index[t:]                                  # row t and every row after it
        changed.loc[later, "rating"] = (changed.loc[later, "rating"] % 4) + 1
        changed.loc[later, "duration"] = changed.loc[later, "duration"] * 3 + 17
        other = P.plus_features(changed, cards, decks)
        earlier = base.index[:t + 1]                               # rows <= t
        for col in P.PLUS_COLS:
            assert _same(base.loc[earlier, col], other.loc[earlier, col]), f"row <= {t} changed in {col}"
        # the change is visible after t, so the test is not vacuous
        after = base.index[t + 1:]
        if len(after):
            assert any(not _same(base.loc[after, c], other.loc[after, c]) for c in P.PLUS_COLS)


# ---- (c) cards / decks missing ----------------------------------------------------------------

def test_no_cards_or_decks_gives_nan_sib_and_deck_without_crash():
    stream = _toy_stream()
    out = P.plus_features(stream, None, None)
    assert list(out.columns) == P.PLUS_COLS
    for c in P.PLUS_GROUPS["sib"] + P.PLUS_GROUPS["deck"]:
        assert out[c].isna().all(), c
    # the other groups are still computed
    assert out["n_today"].notna().all()
    assert out.loc[3, "el_frac"] == pytest.approx(1.0)


def test_cards_without_decks_gives_nan_deck_but_sib_is_computed():
    stream = _toy_stream()
    cards, _ = _toy_refs()
    out = P.plus_features(stream, cards, None)
    for c in P.PLUS_GROUPS["deck"]:
        assert out[c].isna().all(), c
    _assert_col(out, "sib_n", [0, 1, 1, 0, 1, 3, 2, 0, 2, 5, 0, 3])


# ---- (d) one real local user ----------------------------------------------------------------------

def test_real_local_user_runs_without_all_nan_columns():
    rev = REPO / "data" / "anki10k" / "revlogs" / f"user_id={REAL_USER}" / "data.parquet"
    cards_p = REPO / "data" / "anki10k" / "cards" / f"user_id={REAL_USER}" / "data.parquet"
    decks_p = REPO / "data" / "anki10k" / "decks" / f"user_id={REAL_USER}" / "data.parquet"
    if not (rev.exists() and cards_p.exists() and decks_p.exists()):
        pytest.skip("local revlog/cards/decks for the real test user are not present")
    raw = pd.read_parquet(rev)
    stream = R.review_stream(raw, max_card_reviews=200)
    out = P.plus_features(stream, pd.read_parquet(cards_p), pd.read_parquet(decks_p))
    assert len(out) == len(stream)
    assert list(out.columns) == P.PLUS_COLS
    assert not np.isinf(out.to_numpy(float)).any()
    all_nan = [c for c in out.columns if out[c].isna().all()]
    assert all_nan == [], f"columns that are NaN everywhere: {all_nan}"
    # the first review of every card has no card history: its gap features are NaN
    first = ~stream["card_id"].duplicated().to_numpy()
    assert out.loc[first, "rows_since_card"].isna().all()
    assert out.loc[first, "el_frac"].isna().all()


# ---- download helper and index order ------------------------------------------------------------

def test_side_table_download_skips_missing_files_and_goes_on(monkeypatch, tmp_path):
    from huggingface_hub.errors import EntryNotFoundError

    from fsrs_benchmark import data as D

    calls = []

    def fake_download(repo_id, filename, repo_type, local_dir):
        calls.append(filename)
        if filename == "cards/user_id=92/data.parquet":
            raise EntryNotFoundError(f"no {filename}")
        dest = Path(local_dir) / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"x")
        return str(dest)

    monkeypatch.setattr(D, "hf_hub_download", fake_download)
    res = D.download_side_tables([91, 92, 93], tmp_path)
    assert ("cards/user_id=93/data.parquet") in calls          # later users are still fetched
    assert res["missing"] == [(92, "cards")]
    assert [p.name for p in res["cards"]] == ["data.parquet", "data.parquet"]
    assert len(res["decks"]) == 3
    assert not (tmp_path / "cards" / "user_id=92").exists() or not any((tmp_path / "cards" / "user_id=92").iterdir())


def test_side_table_download_auth_error_still_raises(monkeypatch, tmp_path):
    from huggingface_hub.errors import HfHubHTTPError

    from fsrs_benchmark import data as D

    from unittest import mock

    def fake_download(**kwargs):
        raise HfHubHTTPError("401 unauthorized", response=mock.MagicMock(status_code=401, headers={}))

    monkeypatch.setattr(D, "hf_hub_download", fake_download)
    with pytest.raises(RuntimeError, match="Hugging Face"):
        D.download_side_tables([5], tmp_path)


def test_unsorted_stream_fails_loudly():
    stream = _toy_stream()
    shuffled = stream.iloc[::-1]
    with pytest.raises(ValueError, match="strictly increasing"):
        P.plus_features(shuffled, *_toy_refs())


# ---- umem (user and card memory) against a brute-force reference ----------------------------------

def _brute_umem(stream, base):
    """Direct, loop-based definitions of the umem features for the stream rows (test reference only)."""
    import math
    card = stream["card_id"].to_numpy()
    rating = stream["rating"].to_numpy()
    state = stream["state"].to_numpy()
    secs = stream["elapsed_seconds"].to_numpy(float)
    fR = base["f_R"].to_numpy(float)
    n = len(stream)
    seen = {}
    first = np.zeros(n, bool)
    j_in_card = np.zeros(n, int)
    for t in range(n):
        c = card[t]
        j_in_card[t] = seen.get(c, 0)
        first[t] = c not in seen
        seen[c] = seen.get(c, 0) + 1
    nf = ~first
    y = (rating > 1).astype(float)
    el = np.array([secs[t] / 86400.0 if nf[t] and secs[t] >= 0 else np.nan for t in range(n)])

    def gbk(e):
        if np.isnan(e):
            return None
        return 0 if e < 1 else min(8, 1 + int(math.floor(math.log2(e))))

    def cbk(k):
        return 0 if k == 1 else 1 if k == 2 else 2 if k == 3 else 3 if k <= 5 else 4 if k <= 10 else 5

    gb = [gbk(e) for e in el]
    cb = [cbk(j + 1) for j in j_in_card]
    g1 = [None] * n
    last = {}
    for t in range(n):
        g1[t] = last.get(card[t])
        last[card[t]] = rating[t]
    rv = [y[t] - fR[t] if nf[t] and np.isfinite(fR[t]) else None for t in range(n)]

    def ewm(seq, span):                       # weights over valid observations, newest weight 1
        obs = [v for v in seq if v is not None and not np.isnan(v)]
        if not obs:
            return np.nan
        a = 2.0 / (span + 1.0)
        w = [(1 - a) ** (len(obs) - 1 - i) for i in range(len(obs))]
        return sum(wi * v for wi, v in zip(w, obs)) / sum(w)

    out = {k: np.full(n, np.nan) for k in ("ewm10", "resid_mean", "resid_gap_mean", "gap_bkt_rate",
                                          "cnt_bkt_rate", "g1_rate", "state_rate", "card_ewm3",
                                          "card_mean_loggap", "card_r4", "card_gap5", "resid_ewm100")}
    for t in range(n):
        before = range(t)
        yv = [y[s] if nf[s] else None for s in before]
        out["ewm10"][t] = ewm(yv, 10)
        out["resid_ewm100"][t] = ewm([rv[s] for s in before], 100)
        valid = [rv[s] for s in before if rv[s] is not None]
        out["resid_mean"][t] = np.mean(valid) if valid else np.nan
        if gb[t] is not None:
            vg = [rv[s] for s in before if rv[s] is not None and gb[s] == gb[t]]
            out["resid_gap_mean"][t] = np.mean(vg) if vg else np.nan
        for name, keys in (("gap_bkt_rate", gb), ("cnt_bkt_rate", cb), ("g1_rate", g1),
                           ("state_rate", list(state))):
            if keys[t] is None:
                continue
            ys = [y[s] for s in before if nf[s] and keys[s] == keys[t]]
            out[name][t] = np.mean(ys) if ys else np.nan
        same = [s for s in before if card[s] == card[t]]
        out["card_ewm3"][t] = ewm([y[s] if nf[s] else None for s in same], 3)
        lgs = [math.log1p(el[s]) for s in same if nf[s] and not np.isnan(el[s])]
        out["card_mean_loggap"][t] = np.mean(lgs) if lgs else np.nan
        j = j_in_card[t]
        cardrows = [s for s in range(n) if card[s] == card[t]]
        out["card_r4"][t] = rating[cardrows[j - 4]] if j >= 4 else np.nan
        out["card_gap5"][t] = el[cardrows[j - 5]] if j >= 5 else np.nan
    return out


def test_umem_matches_brute_force_reference():
    stream = R.review_stream(_synthetic_raw(seed=3), max_card_reviews=100)
    cards, decks = _synthetic_refs()
    out = P.plus_features(stream, cards, decks)
    ev = stream.assign(is_first=~stream["card_id"].duplicated().to_numpy(),
                       y=(stream["rating"] > 1).astype(np.int64), user_id=0)
    base = G.base_features(ev, span=20, user_group="row")
    ref = _brute_umem(stream, base)
    for name, exp in ref.items():
        got = out[name].to_numpy(float)
        assert np.allclose(got, exp, rtol=1e-9, atol=1e-12, equal_nan=True), name
        assert np.isfinite(got).sum() > 20, f"{name} is almost always NaN; the check proves little"
