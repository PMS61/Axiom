"""
split.py -- Per-user time split and the event table every model is scored on.

Why a per-user time cutoff: a random split would let a model see a card's later
reviews while predicting its earlier ones, which leaks the future. Here each user's
reviews are cut at one day: everything before it is TRAIN (model parameters are
fitted on it only), everything from it onward is TEST (parameters stay frozen).

Order of operations inside make_split (fixed in docs/PLAN.md):
  1. Card cap: per user, keep at most max_cards cards, sampled with a fixed seed.
     This runs BEFORE the cutoff, so the cutoff is the 80th percentile of the
     reviews the benchmark actually evaluates, not of cards it throws away.
  2. cutoff_day = 80th percentile of day_offset over that user's kept reviews
     (np.quantile, method="linear").
  3. is_train = day_offset < cutoff_day.
  4. scored = test event that is not a card's first review and has elapsed_days >= 1.
     Same-day reviews are still fed to the models as history, but are not scored,
     following the srs-benchmark convention. First reviews have no gap to predict.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_EVENTS_PATH = Path("data/processed/events.parquet")
TRAIN_QUANTILE = 0.8
EVENT_ID_STRIDE = 10**9  # event_id = user_id * stride + review_idx, unique for review_idx < 1e9


def _sample_cards(card_ids, max_cards, seed):
    """Return the card ids kept for one user: all of them if few enough, else a seeded sample."""
    unique = np.sort(np.unique(card_ids))
    if len(unique) <= max_cards:
        return unique
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(unique, size=max_cards, replace=False))


def make_split(df, max_cards=300, seed=0):
    """Build the event table from concatenated load_user() outputs.

    Returns one row per kept review with columns:
      event_id (int64, unique), user_id, card_id, review_idx, day_offset, rating,
      state, elapsed_days, dt_days, y, is_first, cutoff_day, is_train, scored.
    Rows are sorted by event_id.
    """
    if max_cards < 1:
        raise ValueError("max_cards must be at least 1")

    pieces = []
    for user_id, user in df.groupby("user_id", sort=True):
        keep = _sample_cards(user["card_id"].to_numpy(), max_cards, seed)
        user = user[user["card_id"].isin(keep)].copy()
        if user.empty:
            continue
        cutoff = float(np.quantile(user["day_offset"].to_numpy(), TRAIN_QUANTILE, method="linear"))
        user["cutoff_day"] = cutoff
        user["is_train"] = user["day_offset"] < cutoff
        pieces.append(user)

    if not pieces:
        raise ValueError("no reviews left after the card cap")

    events = pd.concat(pieces, ignore_index=True)
    events["event_id"] = (
        events["user_id"].astype(np.int64) * EVENT_ID_STRIDE + events["review_idx"].astype(np.int64)
    )
    if events["event_id"].duplicated().any():
        raise ValueError("duplicate event_id: review_idx must be unique within each user")

    events["scored"] = (~events["is_train"]) & (~events["is_first"].astype(bool)) & (events["elapsed_days"] >= 1)
    events["is_train"] = events["is_train"].astype(bool)
    events["scored"] = events["scored"].astype(bool)
    events["event_id"] = events["event_id"].astype(np.int64)
    events["cutoff_day"] = events["cutoff_day"].astype(float)

    columns = [
        "event_id", "user_id", "card_id", "review_idx", "day_offset", "rating", "state",
        "elapsed_days", "dt_days", "y", "is_first", "cutoff_day", "is_train", "scored",
    ]
    return events[columns].sort_values("event_id").reset_index(drop=True)


def save_events(events, path=DEFAULT_EVENTS_PATH):
    """Write the event table to parquet, creating the parent folder if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(path, index=False)
    return path


def load_events(path=DEFAULT_EVENTS_PATH):
    """Read an event table written by save_events."""
    return pd.read_parquet(path)
