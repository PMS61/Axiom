"""Extra past-only features for the SRS-Beat GBDT (Phase 6, task 6.1). See paper/docs/SRSB_BEAT.md.

plus_features(stream, cards, decks) takes one user's review stream (the frame review_stream() builds,
rows in time order, indexed by review_th) and returns the feature groups secs, today, dur, rcount,
state, sib and deck, indexed like the stream. Feature groups are listed in PLUS_GROUPS.

Past-only rule: the features of row t use only rows strictly before t, plus the information shown
before the answer: elapsed_seconds, state and the card's note, deck and preset ids of row t. They never
use the rating or duration of row t or anything after it. Every statistic here is an exclusive cumulative
statistic (the current row is subtracted or shifted away), so the tests in test_features_plus.py check
this directly.

Missing reference data: sib needs cards (note_id); deck needs cards and decks (deck_id, preset_id,
parent_id). When they are missing, those columns are NaN.
"""
import numpy as np
import pandas as pd
from numba import njit

from fsrs_benchmark.models import gbdt as G

DECAY = float(G.W[20])                      # FSRS-6 default forgetting-curve decay
FACTOR = 0.9 ** (-1.0 / DECAY) - 1.0        # R(t, S) = (1 + FACTOR * t / S) ** -DECAY
SECS_PER_DAY = 86400.0

PLUS_GROUPS = {
    "secs": ["el_frac", "log_el_frac", "f_R_secs"],
    "today": ["n_today", "n_first_today", "rate_today", "lapses_today", "u_last5", "rows_since_card"],
    "dur": ["dur_card_last", "dur_card_mean", "dur_user_mean", "dur_card_last_rel", "dur_user_prev"],
    "rcount": ["rc1", "rc2", "rc3", "rc4", "card_max_gap"],
    "state": ["state"],
    "sib": ["sib_n", "sib_last_y", "sib_days_since", "sib_rate"],
    "deck": ["deck_n", "deck_rate", "preset_n", "preset_rate", "parent_n", "parent_rate"],
}
PLUS_COLS = [c for g in PLUS_GROUPS for c in PLUS_GROUPS[g]]

REQUIRED = ["card_id", "day_offset", "rating", "state", "duration", "elapsed_seconds", "elapsed_days"]


@njit(cache=True)
def _sib_last_kernel(newgrp, card, yv, day):
    """Rows sorted by (note, time). For each row, the y and day_offset of the most recent earlier row of the
    same note whose card differs from this row's card (NaN if none). Rows of one note form one group."""
    n = len(card)
    out_y = np.full(n, np.nan)
    out_d = np.full(n, np.nan)
    has_last = False
    last_card = 0
    last_y = 0.0
    last_day = 0.0
    has_diff = False            # most recent row with a card different from last_card
    diff_y = 0.0
    diff_day = 0.0
    for i in range(n):
        if newgrp[i]:
            has_last = False
            has_diff = False
        if has_last:
            if card[i] != last_card:
                out_y[i] = last_y
                out_d[i] = last_day
            elif has_diff:
                out_y[i] = diff_y
                out_d[i] = diff_day
        if has_last and card[i] != last_card:
            has_diff = True
            diff_y = last_y
            diff_day = last_day
        has_last = True
        last_card = card[i]
        last_y = yv[i]
        last_day = day[i]
    return out_y, out_d


def _keys(x):
    """Integer group keys; a missing key becomes -1 (its statistics are set to NaN)."""
    return pd.Series(x).fillna(-1).astype(np.int64).to_numpy()


def _excl_count(mask, key):
    """Number of earlier rows with the same key whose mask is True (current row excluded)."""
    m = pd.Series(mask.astype(float))
    return (m.groupby(key).cumsum() - m).to_numpy()


def _excl_sum(x, key):
    """Sum of x over earlier rows with the same key, NaN values counted as 0 (current row excluded).

    Shift first, then cumsum: the value at row t is then built from earlier values only. (cumsum minus x
    would leave float rounding that depends on x[t].)
    """
    s = pd.Series(np.nan_to_num(np.asarray(x, dtype=float)))
    return s.groupby(key).shift(1).fillna(0.0).groupby(key).cumsum().to_numpy()


def _ratio(num, den, missing):
    """num / den, NaN where den is 0 or the key is missing."""
    with np.errstate(divide="ignore", invalid="ignore"):
        r = num / den
    return np.where((den > 0) & ~missing, r, np.nan)


def _lookup(table, key_col, val_col, keys):
    """Map card-level keys to a table column, NaN where unknown."""
    if table is None:
        return np.full(len(keys), np.nan)
    m = table.drop_duplicates(key_col).set_index(key_col)[val_col]
    return m.reindex(keys).to_numpy(float)


def plus_features(stream, cards=None, decks=None):
    """Feature groups PLUS_COLS for one user's review stream, a DataFrame indexed like `stream`.

    stream : rows in time order with columns card_id, day_offset, rating, state, duration,
             elapsed_seconds, elapsed_days (review_stream() output has all of them). Only one user.
    cards  : optional DataFrame with card_id, note_id, deck_id (one row per card).
    decks  : optional DataFrame with deck_id, parent_id, preset_id.
    """
    missing = [c for c in REQUIRED if c not in stream.columns]
    if missing:
        raise KeyError(f"plus_features: stream is missing columns {missing}")
    if not stream.index.is_monotonic_increasing or not stream.index.is_unique:
        raise ValueError("plus_features: stream index must be strictly increasing (review_th order)")
    n = len(stream)
    idx = stream.index
    pos = np.arange(n, dtype=np.int64)
    card = stream["card_id"].to_numpy(np.int64)
    day = stream["day_offset"].to_numpy(np.int64)
    rating = stream["rating"].to_numpy(np.int64)
    state = stream["state"].to_numpy(np.int64)
    duration = stream["duration"].to_numpy(float)
    secs = stream["elapsed_seconds"].to_numpy(float)
    gap_days = stream["elapsed_days"].to_numpy(np.int64)

    first = ~pd.Series(card).duplicated().to_numpy()
    nf = ~first
    y = (rating > 1).astype(np.int64)
    yf = y.astype(float)
    zeros = np.zeros(n, dtype=np.int64)
    out = {}

    # secs: fractional gap (NaN on a first review, or when elapsed_seconds is the -1 sentinel)
    el_frac = np.where(nf & (secs >= 0), secs / SECS_PER_DAY, np.nan)
    out["el_frac"] = el_frac
    out["log_el_frac"] = np.log1p(el_frac)
    # FSRS-6 default state S before row t (base_features replays the card; its "f_S" uses past rows only)
    ev = pd.DataFrame({
        "event_id": pos + 1, "user_id": zeros, "card_id": card, "day_offset": day,
        "rating": rating, "y": y, "is_first": first, "elapsed_days": gap_days,
        "dt_days": pd.Series(day).diff().fillna(0).clip(lower=0).to_numpy(float),
    }, index=pos)
    f_S = G.base_features(ev, span=20, user_group="row")["f_S"].to_numpy(float)
    out["f_R_secs"] = (1.0 + FACTOR * np.maximum(el_frac, 0.0) / f_S) ** (-DECAY)

    # today: earlier rows of the same day (day_offset), and the user's recent history
    out["n_today"] = _excl_count(np.ones(n, bool), day)
    out["n_first_today"] = _excl_sum(first, day)
    out["rate_today"] = _ratio(_excl_sum(yf * nf, day), _excl_sum(nf, day), np.zeros(n, bool))
    out["lapses_today"] = _excl_sum(rating == 1, day)
    out["u_last5"] = pd.Series(yf).shift(1).rolling(5, min_periods=1).mean().to_numpy()
    prev_pos = pd.Series(pos).groupby(card).shift(1).to_numpy(float)
    out["rows_since_card"] = np.where(nf, pos - prev_pos, np.nan)

    # dur: log(1 + duration) of past reviews (duration is in ms; the current row's duration is never used)
    logd = pd.Series(np.log1p(duration))
    out["dur_card_last"] = logd.groupby(card).shift(1).to_numpy()
    dur_card_mean = _ratio(_excl_sum(logd.to_numpy(), card), _excl_count(logd.notna().to_numpy(), card),
                           np.zeros(n, bool))
    out["dur_card_mean"] = dur_card_mean
    dur_user_mean = _ratio(_excl_sum(logd.to_numpy(), zeros), _excl_count(logd.notna().to_numpy(), zeros),
                           np.zeros(n, bool))
    out["dur_user_mean"] = dur_user_mean
    out["dur_card_last_rel"] = out["dur_card_last"] - dur_user_mean
    out["dur_user_prev"] = logd.shift(1).to_numpy()

    # rcount: the card's past counts of each rating, and its longest past gap in days
    for k in (1, 2, 3, 4):
        out[f"rc{k}"] = _excl_count(rating == k, card)
    gap_nf = pd.Series(np.where(nf, gap_days.astype(float), np.nan))
    cummax = gap_nf.groupby(card).cummax()
    out["card_max_gap"] = cummax.groupby(card).shift(1).to_numpy()

    # state: the current row's review type (known before the answer)
    out["state"] = state.astype(float)

    # sib and deck need the cards (and decks) tables
    note = _lookup(cards, "card_id", "note_id", card)
    deck = _lookup(cards, "card_id", "deck_id", card)
    if cards is None:
        for c in PLUS_GROUPS["sib"]:
            out[c] = np.full(n, np.nan)
    else:
        nk = _keys(note)
        miss_note = np.isnan(note)
        out["sib_n"] = np.where(miss_note, np.nan,
                                _excl_count(np.ones(n, bool), nk) - _excl_count(np.ones(n, bool), card))
        nfy = yf * nf
        num = _excl_sum(nfy, nk) - _excl_sum(nfy, card)
        den = _excl_sum(nf, nk) - _excl_sum(nf, card)
        out["sib_rate"] = _ratio(num, den, miss_note)
        order = np.lexsort((pos, nk))
        newgrp = np.ones(n, bool)
        newgrp[1:] = nk[order][1:] != nk[order][:-1]
        last_y, last_d = _sib_last_kernel(newgrp, card[order], yf[order], day[order])
        ly = np.empty(n)
        ld = np.empty(n)
        ly[order] = last_y
        ld[order] = last_d
        ly[miss_note] = np.nan
        ld[miss_note] = np.nan
        out["sib_last_y"] = ly
        out["sib_days_since"] = day - ld

    if cards is None or decks is None:
        for c in PLUS_GROUPS["deck"]:
            out[c] = np.full(n, np.nan)
    else:
        dk = _keys(deck)
        miss_deck = np.isnan(deck)
        preset = _lookup(decks, "deck_id", "preset_id", deck)
        parent = _lookup(decks, "deck_id", "parent_id", deck)
        pk = _keys(preset)
        rk = _keys(parent)
        miss_preset = np.isnan(preset)
        miss_parent = np.isnan(parent)
        for name, key, miss in (("deck", dk, miss_deck), ("preset", pk, miss_preset),
                                ("parent", rk, miss_parent)):
            cnt = _excl_count(np.ones(n, bool), key)
            out[f"{name}_n"] = np.where(miss, np.nan, cnt)
            out[f"{name}_rate"] = _ratio(_excl_sum(yf * nf, key), _excl_sum(nf, key), miss)

    res = pd.DataFrame({c: np.asarray(out[c], dtype=float) for c in PLUS_COLS}, index=idx)
    return res
