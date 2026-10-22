"""
data.py -- Download and load Anki review logs (open-spaced-repetition/anki-revlogs-10k).

Why this module exists: every model in the benchmark must be scored on the same
review events, built the same way. Putting the event definition in one loader
means FSRS, HLR and our model cannot quietly disagree about what a "review" is.

Event definition (revised in Phase 2; see docs/PLAN.md):
  - rows keep their file order, which is chronological within a user
  - rating == 0 rows are manual reschedules, not reviews: dropped
  - state == 4 rows are KEPT: they have real ratings and durations (filtered decks)
  - a card is dropped entirely if its first kept review is not state == 0 (new),
    because its earlier history is truncated and its memory state cannot be rebuilt
  - y = 1 if rating > 1 (Hard, Good, Easy count as recalled)
  - elapsed_days = day_offset difference to the card's previous KEPT review (-1 on a
    first review); raw_elapsed_days keeps the file's own value
  - dt_days = raw elapsed seconds summed over the window since the previous kept
    review, / 86400 (falls back to elapsed_days if any seconds are missing); NaN on
    a first review
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, get_token, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

REPO_ID = "open-spaced-repetition/anki-revlogs-10k"
REVLOG_FILE_RE = re.compile(r"^revlogs/user_id=(\d+)/data\.parquet$")
USER_ID_RE = re.compile(r"user_id=(\d+)")

LOAD_COLUMNS = [
    "user_id", "card_id", "review_idx", "day_offset", "rating", "state",
    "elapsed_days", "raw_elapsed_days", "dt_days", "y", "is_first",
]


def _access_message(err):
    return (
        "Could not access the Hugging Face dataset "
        f"{REPO_ID}. Check that your HF token is valid (in ~/.cache/huggingface/token "
        "or HF_TOKEN) and that you accepted the dataset licence at "
        f"https://huggingface.co/datasets/{REPO_ID}. Original error: {err}"
    )


USER_IDS_CACHE = "user_ids.json"


def _user_ids(out_dir):
    """All user ids in the dataset. Read from <out_dir>/user_ids.json when it exists, so no
    network call is needed; otherwise listed from the Hub and cached there."""
    cache = Path(out_dir) / USER_IDS_CACHE
    if cache.exists():
        return [int(u) for u in json.loads(cache.read_text())]
    if not get_token():
        raise RuntimeError(
            "No Hugging Face token found. Log in with `huggingface-cli login` or set "
            f"HF_TOKEN, and accept the licence at https://huggingface.co/datasets/{REPO_ID}."
        )
    try:
        files = HfApi().list_repo_files(REPO_ID, repo_type="dataset")
    except HfHubHTTPError as err:
        raise RuntimeError(_access_message(err)) from err
    ids = sorted({int(m.group(1)) for f in files if (m := REVLOG_FILE_RE.match(f))})
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(ids))
    return ids


def download_users(n_users, seed, out_dir):
    """Download revlog parquet files for a deterministic random sample of users.

    Why a fixed seed: the sample must be the same on every run, so results can be
    reproduced and the 100-user Kaggle run uses the same users as the local smoke run.
    Files that already exist locally are skipped, so reruns are cheap.

    Returns a sorted list of local Path objects, one per sampled user.
    """
    ids = _user_ids(out_dir)
    if not 1 <= n_users <= len(ids):
        raise ValueError(f"n_users must be between 1 and {len(ids)}, got {n_users}")

    rng = np.random.default_rng(seed)
    chosen = sorted(int(u) for u in rng.choice(ids, size=n_users, replace=False))

    out_dir = Path(out_dir)
    paths = []
    for uid in chosen:
        filename = f"revlogs/user_id={uid}/data.parquet"
        local = out_dir / filename
        if not local.exists():
            try:
                hf_hub_download(
                    repo_id=REPO_ID, filename=filename, repo_type="dataset",
                    local_dir=str(out_dir),
                )
            except HfHubHTTPError as err:
                raise RuntimeError(_access_message(err)) from err
        paths.append(local)
    return paths


def _user_id_from_path(path):
    m = USER_ID_RE.search(Path(path).as_posix())
    if m is None:
        raise ValueError(f"Cannot find user_id=<N> in path: {path}")
    return int(m.group(1))


def load_user(path):
    """Load one user's revlog parquet and apply the event definition.

    The user id comes from the folder name (`user_id=<N>`), because the dataset
    files do not store it as a column.

    Returns a DataFrame with columns LOAD_COLUMNS. Rows are the kept reviews in
    file order. `review_idx` is the 0-based position among the kept reviews.

    Why gaps are recomputed from the kept rows: the models replay each card and
    need the gap between consecutive kept reviews. The raw elapsed_days can refer to
    a dropped row (for example a rating-0 reschedule), so it would not match the
    replayed history. Here elapsed_days is the day_offset difference between kept
    rows, and dt_days is the raw elapsed seconds summed over the same window
    (dropped rows included), so the time spent in that window is not lost.
    """
    user_id = _user_id_from_path(path)
    raw = pd.read_parquet(path).reset_index(drop=True)

    required = ["card_id", "day_offset", "rating", "state", "elapsed_days"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    # Only manual reschedules (rating 0) are dropped. state 4 rows (filtered decks)
    # carry real ratings and durations, so they are genuine recall events.
    kept = raw[raw["rating"] != 0]

    # A card whose first kept review is not a new-card review has truncated history,
    # so its memory state cannot be reconstructed. Drop the whole card.
    first_rows = kept.groupby("card_id", sort=False).head(1)
    keep_cards = first_rows.loc[first_rows["state"] == 0, "card_id"]
    kept = kept[kept["card_id"].isin(keep_cards)].copy()
    if kept.empty:
        return _empty_frame()

    # Per-card cumulative seconds over ALL raw rows of the kept cards (dropped rows
    # included). The seconds of a window (prev kept, current kept] are then a difference.
    seconds = raw["elapsed_seconds"].astype(float) if "elapsed_seconds" in raw.columns \
        else pd.Series(np.nan, index=raw.index)
    valid = seconds.notna() & (seconds >= 0)
    cards = raw[raw["card_id"].isin(keep_cards)].copy()
    cards["_sec"] = seconds.where(valid, 0.0)[cards.index]
    cards["_bad"] = (~valid[cards.index]).astype(np.int64)
    cards["_cum_sec"] = cards.groupby("card_id", sort=False)["_sec"].cumsum()
    cards["_cum_bad"] = cards.groupby("card_id", sort=False)["_bad"].cumsum()
    cum_sec = cards["_cum_sec"]
    cum_bad = cards["_cum_bad"]

    kept["is_first"] = kept.groupby("card_id", sort=False).cumcount().eq(0)
    kept["review_idx"] = np.arange(len(kept), dtype=np.int64)
    kept["y"] = (kept["rating"] > 1).astype(np.int64)
    kept["user_id"] = user_id
    kept["raw_elapsed_days"] = kept["elapsed_days"].astype(np.int64)

    grouped = kept.groupby("card_id", sort=False)
    prev_day = grouped["day_offset"].shift(1)
    kept["elapsed_days"] = (kept["day_offset"] - prev_day).fillna(-1).astype(np.int64)

    window_sec = cum_sec.loc[kept.index] - cum_sec.loc[kept.index].groupby(kept["card_id"]).shift(1).reindex(kept.index)
    window_bad = cum_bad.loc[kept.index] - cum_bad.loc[kept.index].groupby(kept["card_id"]).shift(1).reindex(kept.index)
    dt = (window_sec / 86400.0).where(window_bad.fillna(0) == 0, kept["elapsed_days"].astype(float))
    kept["dt_days"] = dt.where(~kept["is_first"], np.nan).astype(float)

    return kept[LOAD_COLUMNS].reset_index(drop=True)


def _empty_frame():
    return pd.DataFrame({c: pd.Series(dtype=float) for c in LOAD_COLUMNS}).astype({
        "user_id": np.int64, "card_id": np.int64, "review_idx": np.int64,
        "day_offset": np.int64, "rating": np.int64, "state": np.int64,
        "elapsed_days": np.int64, "raw_elapsed_days": np.int64, "dt_days": float,
        "y": np.int64, "is_first": bool,
    })
