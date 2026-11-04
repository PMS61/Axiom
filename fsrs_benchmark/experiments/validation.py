"""
validation.py -- Development split for Phase 6: model selection WITHOUT test events.

make_validation_split(events) takes the table from split.make_split and returns a new
table in which:
  - every test row (is_train == False) is DROPPED, so nothing downstream can touch it;
  - per user, val_cutoff = 0.8 quantile (linear) of the day_offset of that user's TRAIN
    rows; rows with day_offset < val_cutoff stay is_train=True (the FIT part), the
    other train rows become the VALIDATION part;
  - scored is recomputed: validation row, not a first review, elapsed_days >= 1.
Same-day / earlier rows remain in the table as history so models can replay cards,
exactly as in the real test protocol. Columns are the same as make_split's, so any
model's fit(events)/predict(events) works unchanged (fit uses is_train rows only).
"""
import numpy as np

from fsrs_benchmark.split import TRAIN_QUANTILE


def make_validation_split(events):
    train = events[events["is_train"].astype(bool)].copy()  # test rows are discarded here
    if train.empty:
        raise ValueError("no train rows")
    cut = train.groupby("user_id")["day_offset"].transform(
        lambda s: np.quantile(s.to_numpy(), TRAIN_QUANTILE, method="linear"))
    train["val_cutoff_day"] = cut.astype(float)
    is_fit = train["day_offset"] < train["val_cutoff_day"]
    train["is_train"] = is_fit.astype(bool)
    train["scored"] = ((~is_fit) & (~train["is_first"].astype(bool))
                       & (train["elapsed_days"] >= 1)).astype(bool)
    return train.sort_values("event_id").reset_index(drop=True)
