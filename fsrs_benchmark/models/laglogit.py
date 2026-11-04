"""
laglogit.py -- Adapter that runs the validated LagLogit+dt baseline on the event table.

Why this model: LaggedLogistic (baselines.py, use_dt=True) is the fair control for the
dynamic model: a logistic regression on the lagged outcome, the sqrt right/wrong counts,
log(dt) and item/user effects, with no latent state. Keeping it in the benchmark answers
"is the latent state doing the work, or just the lagged-outcome feature?". The baseline
file itself is not modified.

Sequences are built with the same code as ours.py (_sequences), so the baseline sees the
same reviews, gaps and split. Ids follow ours.py: dense ids come from TRAIN rows, and a
user or card with no TRAIN rows gets the padding index, which LaggedLogistic ignores.

No-leakage: fit() sees TRAIN rows only. predict() uses the TRAIN history of each pair as
the starting counts and lag, then TEST rows in order. Review t's features use only reviews
before t, so p_t never sees y_t or any later outcome.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.append(str(_REPO))

from baselines import LaggedLogistic  # noqa: E402

from fsrs_benchmark.models.ours import _sequences  # noqa: E402

P_CLIP = 1e-6


class LagLogitModel:
    """LaggedLogistic(use_dt=True) with the common interface: fit(events) and predict(events)."""

    def __init__(self, use_dt=True):
        self.use_dt = use_dt
        self.model_ = None

    def _dense_ids(self, events):
        """Dense int user and item (item = card) per row, from the ids stored at fit().

        Unknown users and cards get the padding index (n_users / n_items).
        """
        users = [self._users.get(u, self._n_users) for u in events["user_id"]]
        items = [self._items.get(k, self._n_items)
                 for k in zip(events["user_id"], events["card_id"])]
        out = events.copy()
        out["user"] = np.asarray(users, dtype=np.int64)
        out["item"] = np.asarray(items, dtype=np.int64)
        return out

    def fit(self, events):
        """Fit on TRAIN rows. Ids are assigned from TRAIN rows only."""
        train = events[events["is_train"].astype(bool)]
        self._users = {u: i for i, u in enumerate(sorted(train["user_id"].unique()))}
        self._items = {k: i for i, k in enumerate(sorted(set(zip(train["user_id"], train["card_id"]))))}
        self._n_users, self._n_items = len(self._users), len(self._items)
        ev = self._dense_ids(events)
        train_seqs = _sequences(ev, ev["is_train"].astype(bool).to_numpy())
        if not train_seqs:
            raise ValueError("no TRAIN rows to fit on")
        self.model_ = LaggedLogistic(use_dt=self.use_dt).fit(train_seqs)
        return self

    def predict(self, events):
        """Recall probabilities for scored rows, indexed by event_id, clipped to [1e-6, 1-1e-6]."""
        if self.model_ is None:
            raise RuntimeError("call fit() before predict()")
        ev = self._dense_ids(events)
        test_seqs = _sequences(ev, ~ev["is_train"].astype(bool).to_numpy())
        if not test_seqs:
            return pd.Series([], index=pd.Index([], name="event_id"), name="p", dtype=float)
        preds = self.model_.predict(test_seqs)
        scored = set(ev.loc[ev["scored"].astype(bool), "event_id"].astype(np.int64))
        ids, vals = [], []
        for seq, p in zip(test_seqs, preds):
            keep = np.array([int(e) in scored for e in seq["event_id"]], dtype=bool)
            ids.append(seq["event_id"][keep])
            vals.append(np.asarray(p, dtype=float)[keep])
        out = pd.Series(np.concatenate(vals), index=pd.Index(np.concatenate(ids), name="event_id"),
                        name="p", dtype=float)
        return out.sort_index().clip(P_CLIP, 1 - P_CLIP)
