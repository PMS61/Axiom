"""
duolingo.py -- Load Duolingo's half-life-regression learning traces (Settles & Meeder 2016).

Why this module exists: Phase 5 runs the same benchmark protocol on Duolingo's public data.
load_users() returns exactly the columns of data.load_user (LOAD_COLUMNS), so split, the
models, run and report are reused unchanged. The Duolingo event definition lives here, next
to the rules it implements (see docs/PLAN.md, Phase 5).

Why two passes: the raw file has about 13M rows and is too big to hold at once. scan() makes
one chunked pass that counts rows per user and finds T0 (the earliest timestamp, which
day_offset is measured from), and caches the result because every run needs the counts to
draw a sample. load_users() then reads the file again in chunks. Each chunk keeps only the
sampled users, and is turned into small integer arrays (user id, lexeme code, y, ...) right
away, so the strings and the unused columns never pile up in memory.

Event definition (one row = one practice session of one lexeme by one user = one review):
  - card = (user, lexeme_id); card_id is dense per user, from the sorted lexeme_id strings
  - user_id (int) = index of the user in the sorted list of ALL user_id strings in the file,
    so the id does not depend on which users were sampled
  - order inside a user: timestamp, then lexeme_id, then original file row (stable tie-break)
  - day_offset = floor((timestamp - T0) / 86400)
  - y = 1 if p_recall >= 0.5 (the convention of data_pipeline.py); rating = 3 (Good) if y
    else 1 (Again), because FSRS only sees Again/Good on this data
  - the first row of each card is state 0 and is never scored; later rows are state 2
  - elapsed_days = day_offset difference to the card's previous row (-1 on a first row)
  - dt_days = timestamp difference to the card's previous row / 86400 (NaN on a first row)
  - raw_elapsed_days = floor(delta / 86400), the file's own gap since the lexeme was last seen
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from fsrs_benchmark.data import LOAD_COLUMNS, _empty_frame

SECONDS_PER_DAY = 86400
DEFAULT_CHUNKSIZE = 1_000_000
SCAN_COLUMNS = ["user_id", "timestamp"]
RAW_COLUMNS = ["p_recall", "timestamp", "delta", "user_id", "lexeme_id"]
COUNTS_COLUMNS = ["user_id", "n_rows"]
# Missing-value markers for p_recall only. Id columns are read as literal strings, so an
# id such as "NA" or "nan" is kept as a user or lexeme instead of becoming NaN and vanishing.
P_RECALL_NA = ["", "nan", "NaN", "NA", "null"]

# Keys stored in the cache's parquet metadata. The cache is only reused when the source
# file still has the size and mtime it had when the cache was written.
_META_T0 = "fsrs_t0"
_META_SIZE = "fsrs_source_size"
_META_MTIME = "fsrs_source_mtime_ns"


def _read_cache(cache_path, source):
    """Return (counts, T0) from the cache, or None if it is missing, stale or unreadable."""
    if not cache_path.exists():
        return None
    try:
        meta = pq.read_metadata(cache_path).metadata or {}
        meta = {k.decode(): v.decode() for k, v in meta.items()}
        if meta.get(_META_SIZE) != source[_META_SIZE] or meta.get(_META_MTIME) != source[_META_MTIME]:
            return None
        counts = pq.read_table(cache_path).to_pandas()
        return counts[COUNTS_COLUMNS].astype({"user_id": str, "n_rows": np.int64}), int(meta[_META_T0])
    except (OSError, KeyError, ValueError, pa.ArrowException):
        return None


def _write_cache(cache_path, counts, t0, source):
    """Write the counts table to parquet with T0 and the source size/mtime in its metadata."""
    table = pa.Table.from_pandas(counts[COUNTS_COLUMNS], preserve_index=False)
    table = table.replace_schema_metadata({
        _META_T0: str(t0), _META_SIZE: source[_META_SIZE], _META_MTIME: source[_META_MTIME],
    })
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, cache_path)


def _csv_options(usecols):
    """read_csv options shared by both passes: literal id strings, explicit p_recall missing values."""
    return dict(
        usecols=usecols,
        dtype={"user_id": str, "lexeme_id": str, "timestamp": np.int64, "delta": np.int64},
        keep_default_na=False,
        na_values={"p_recall": P_RECALL_NA},
    )


def scan(path, cache_path, chunksize=DEFAULT_CHUNKSIZE):
    """Count rows per user and find T0 in one chunked pass over the raw file, with a cache.

    Returns (counts, t0): counts is a DataFrame with columns user_id (str) and n_rows (int64),
    sorted by user_id, covering ALL users in the file; t0 is the minimum timestamp (int).

    Why a cache: the counts are needed on every run to draw the sample, and one pass over
    13M rows is slow. The cache is reused only while the source file's size and mtime match
    the values stored with it, so a replaced download is never read through a stale cache.
    """
    path = Path(path)
    cache_path = Path(cache_path)
    stat = path.stat()
    source = {_META_SIZE: str(stat.st_size), _META_MTIME: str(stat.st_mtime_ns)}

    cached = _read_cache(cache_path, source)
    if cached is not None:
        return cached

    counts = None
    t0 = None
    total = 0
    for chunk in pd.read_csv(path, chunksize=chunksize, **_csv_options(SCAN_COLUMNS)):
        if chunk.empty:
            continue
        total += len(chunk)
        part = chunk.groupby("user_id").size()
        counts = part if counts is None else counts.add(part, fill_value=0)
        chunk_min = int(chunk["timestamp"].min())
        t0 = chunk_min if t0 is None else min(t0, chunk_min)

    if total == 0:
        raise ValueError(f"{path} has no rows")

    counts = counts.sort_index()
    table = pd.DataFrame({
        "user_id": counts.index.astype(str),
        "n_rows": counts.to_numpy().astype(np.int64),
    })
    _write_cache(cache_path, table, t0, source)
    return table, t0


def sample_users(counts, n_users, seed, min_rows):
    """Pick n_users user ids at random from the users with at least min_rows rows.

    Why a fixed seed: the sample must be the same on every run, so results can be reproduced.
    Why the eligible ids are sorted first: the draw then depends only on the seed and the
    data, not on the order rows happened to be read in.

    n_users=None means ALL eligible users (no random draw, so the seed is unused).

    Returns a sorted list of user_id strings. Raises ValueError if fewer eligible users exist
    than requested.
    """
    eligible = sorted(counts.loc[counts["n_rows"] >= min_rows, "user_id"].astype(str).tolist())
    if n_users is None:
        if not eligible:
            raise ValueError(f"no users have at least {min_rows} rows")
        return eligible
    if n_users < 1:
        raise ValueError(f"n_users must be at least 1, got {n_users}")
    if n_users > len(eligible):
        raise ValueError(
            f"only {len(eligible)} users have at least {min_rows} rows, cannot sample {n_users}"
        )
    rng = np.random.default_rng(seed)
    chosen = rng.choice(np.array(eligible, dtype=object), size=n_users, replace=False)
    return sorted(str(u) for u in chosen)


def _build_events(user, lex, timestamp, y, raw_days, row, t0, with_lexeme=False, with_ts=False):
    """Turn the kept rows of the sampled users into the LOAD_COLUMNS frame.

    Inputs are equal-length integer arrays: user (int id), lex (lexeme rank in sorted-string
    order), timestamp, y, raw_days (floor(delta/86400)) and row (original file row, the
    tie-break). Plain numpy keeps the copies small: the whole build is a few sorts and
    searchsorted calls, with no per-row Python loop and no pandas merge.
    """
    n = len(user)
    # Sort by user, then timestamp, then lexeme, then file row (np.lexsort: last key is primary).
    perm = np.lexsort((row, lex, timestamp, user))
    user, lex, timestamp, y, raw_days = user[perm], lex[perm], timestamp[perm], y[perm], raw_days[perm]
    ts_keep = timestamp.copy() if with_ts else None  # raw timestamp (s), for the optional `ts` column
    del perm
    lex_keep = lex.copy() if with_lexeme else None  # time-ordered lexeme rank, for the optional column

    # Card = (user, lexeme). One stable sort by card key does three jobs: it numbers the
    # cards, gives each card's reviews in time order (the input is already time-ordered
    # within a user), and marks each card's first review.
    n_lex = int(lex.max()) + 1
    card_key = user * n_lex + lex
    del lex
    by_card = np.argsort(card_key, kind="stable")
    sorted_key = card_key[by_card]
    new_card = np.ones(n, dtype=bool)
    new_card[1:] = sorted_key[1:] != sorted_key[:-1]
    group = np.cumsum(new_card) - 1  # global card number, in card-key order
    card_global = np.empty(n, dtype=np.int64)
    card_global[by_card] = group
    is_first = np.empty(n, dtype=bool)
    is_first[by_card] = new_card

    # card_id is dense per user: a card's number minus the first card number of its user.
    card_user = sorted_key[new_card] // n_lex
    card_of_group = np.arange(len(card_user), dtype=np.int64) - np.searchsorted(card_user, card_user, side="left")
    card_id = card_of_group[card_global]
    del card_key, card_global, group, card_of_group, card_user, sorted_key, new_card

    review_idx = np.arange(n, dtype=np.int64) - np.searchsorted(user, user, side="left")
    day_offset = ((timestamp - t0) // SECONDS_PER_DAY).astype(np.int64)

    # Previous review of the same card: the previous element of the card's time-ordered run.
    prev = np.full(n, -1, dtype=np.int64)
    prev[by_card[1:]] = np.where(is_first[by_card[1:]], -1, by_card[:-1])
    has_prev = prev >= 0
    safe_prev = np.where(has_prev, prev, 0)
    del by_card, prev

    elapsed_days = np.where(has_prev, day_offset - day_offset[safe_prev], -1)
    dt_days = np.where(has_prev, (timestamp - timestamp[safe_prev]) / SECONDS_PER_DAY, np.nan)
    rating = np.where(y == 1, 3, 1)
    state = np.where(is_first, 0, 2)
    del safe_prev, has_prev

    # One frame, built once with the final dtypes (no extra copies of the big columns).
    dtypes = _empty_frame().dtypes.to_dict()
    arrays = {
        "user_id": user, "card_id": card_id, "review_idx": review_idx, "day_offset": day_offset,
        "rating": rating, "state": state, "elapsed_days": elapsed_days,
        "raw_elapsed_days": raw_days, "dt_days": dt_days, "y": y, "is_first": is_first,
    }
    # copy=False: the arrays are fresh, so this avoids a second copy of the whole frame.
    frame = pd.DataFrame({c: np.asarray(arrays[c], dtype=dtypes[c]) for c in LOAD_COLUMNS}, copy=False)
    if with_lexeme:
        frame["lexeme"] = lex_keep.astype(np.int64)
    if with_ts:
        frame["ts"] = ts_keep.astype(np.int64)
    return frame


def load_users(path, user_strings, counts, t0, chunksize=DEFAULT_CHUNKSIZE, with_lexeme=False, with_ts=False):
    """Load the sampled users' review events from the raw file, in the data.load_user shape.

    user_strings: the sampled user_id strings (from sample_users).
    counts, t0: the output of scan(); counts defines the int user ids for every user.

    Why chunked and encoded per chunk: the file is read in chunks. Each chunk keeps only the
    sampled users' rows and turns them into integers straight away (user id via the sorted
    counts index, lexeme via a growing table of distinct lexemes, y from p_recall). Peak
    memory is therefore set by the sampled rows as small integer arrays, not by the strings
    of the whole file.

    with_lexeme=True appends one extra column `lexeme` (int64, the lexeme_id string mapped to a
    dense int by sorted-string rank among the loaded lexemes; shared across users). The default
    (False) returns exactly LOAD_COLUMNS, unchanged.

    with_ts=True appends the raw timestamp in seconds as `ts` (int64; a whole session shares one ts).
    Returns one DataFrame with exactly LOAD_COLUMNS (plus `lexeme` if asked), sorted by user and then by event order.
    Raises ValueError if a kept row has a missing p_recall.
    """
    all_ids = sorted(counts["user_id"].astype(str).tolist())
    user_index = pd.Index(all_ids)
    wanted = {str(u) for u in user_strings}
    unknown = sorted(wanted - set(all_ids))
    if unknown:
        raise ValueError(f"user ids not in the counts table: {unknown[:5]}")

    # The counts table says how many rows each sampled user has, so the buffers are allocated
    # once at full size. Filling them in place avoids a second copy from pd.concat.
    expected = int(counts.loc[counts["user_id"].astype(str).isin(wanted), "n_rows"].sum())
    buf_user = np.empty(expected, dtype=np.int64)
    buf_lex = np.empty(expected, dtype=np.int64)
    buf_ts = np.empty(expected, dtype=np.int64)
    buf_y = np.empty(expected, dtype=np.int64)
    buf_raw = np.empty(expected, dtype=np.int64)
    buf_row = np.empty(expected, dtype=np.int64)

    lex_uniq = pd.Index([], dtype=object)  # distinct sampled lexemes, in first-seen order
    missing_p = 0
    pos = 0
    start = 0
    for chunk in pd.read_csv(path, chunksize=chunksize, **_csv_options(RAW_COLUMNS)):
        mask = chunk["user_id"].isin(wanted).to_numpy()
        if mask.any():
            part = chunk.loc[mask]
            k = len(part)
            if pos + k > expected:
                raise ValueError(
                    "the file has more rows for the sampled users than the counts table; "
                    "delete the scan cache and rerun"
                )
            p = part["p_recall"].to_numpy(dtype=float)
            missing_p += int(np.isnan(p).sum())

            lex_values = part["lexeme_id"].to_numpy(dtype=object)
            new = pd.Index(pd.unique(lex_values)).difference(lex_uniq, sort=False)
            lex_uniq = lex_uniq.append(new)

            sl = slice(pos, pos + k)
            buf_user[sl] = user_index.get_indexer(part["user_id"].to_numpy(dtype=object))
            buf_lex[sl] = lex_uniq.get_indexer(lex_values)
            buf_ts[sl] = part["timestamp"].to_numpy(dtype=np.int64)
            buf_y[sl] = (p >= 0.5)
            buf_raw[sl] = np.floor(part["delta"].to_numpy(dtype=np.int64) / SECONDS_PER_DAY)
            buf_row[sl] = start + np.flatnonzero(mask)
            pos += k
        start += len(chunk)

    if missing_p:
        raise ValueError(
            f"{missing_p} kept rows have a missing p_recall; cannot compute y. Fix the source file."
        )
    if pos != expected:
        raise ValueError(
            f"read {pos} rows for the sampled users but the counts table says {expected}; "
            "delete the scan cache and rerun"
        )
    if expected == 0:
        return _empty_frame()

    # Lexeme codes -> ranks in sorted-string order, so card_id is dense per user over sorted ids.
    order = np.argsort(lex_uniq.to_numpy(dtype=object), kind="stable")
    rank = np.empty(len(lex_uniq), dtype=np.int64)
    rank[order] = np.arange(len(lex_uniq), dtype=np.int64)
    buf_lex = rank[buf_lex]
    return _build_events(buf_user, buf_lex, buf_ts, buf_y, buf_raw, buf_row, t0, with_lexeme, with_ts)
