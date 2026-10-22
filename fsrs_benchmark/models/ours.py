"""
ours.py -- Adapter that runs our Bayesian dynamic retention model on the event table.

Why an adapter: DynamicRetentionModelDT (AnkiFirstRun/dynamic_retention_dt.py) works on
sequences: one per item (here, one per card), with a per-review outcome r, a lagged
outcome x, a gap dt, and dense integer user and item ids. This module builds those
sequences from the event table, so the model sees exactly the same reviews, gaps and
split as FSRS and HLR. The model file itself is not modified.

No-leakage rules:
  - fit() sees TRAIN rows only: the sequences hold train reviews, and the fitted
    posterior draws are frozen afterwards.
  - predict() gives each card's TEST rows, in order, starting from the state left by
    its TRAIN history. The model's predict() emits p for review t from the state after
    review t-1 and only then applies outcome t. The gap dt_t is known before review t,
    and the lagged outcome x_t is review t-1's. So p_t never sees y_t or any later y.
  - Users and cards are remapped over the whole event table. A card that has no TRAIN
    rows (or a user the fit never saw) gets an index outside the fitted range, so the
    model's cold path is used for it.

Path setup: DynamicRetentionModelDT imports pg_sampler from the repository root, and
the model file lives in AnkiFirstRun/. Both folders are added to sys.path here, and
only here, at the end of the path so that existing modules are not shadowed.
"""
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
for _folder in (_REPO, _REPO / "AnkiFirstRun"):
    if str(_folder) not in sys.path:
        sys.path.append(str(_folder))

from dynamic_retention_dt import DynamicRetentionModelDT  # noqa: E402


def _finite_or_none(x):
    x = float(x)
    return x if np.isfinite(x) else None


def _record_lam_acceptance(sampler, total=0, progress_every=0, label="ours"):
    """Wrap the sampler's per-sweep lambda update so each sweep's acceptance rate is kept.

    The sampler overwrites its acc_rate on every sweep, so only the last sweep survives.
    The wrapper is an instance attribute, so the model file itself is not changed. The
    kappa acceptance count is a local variable inside the sampler and cannot be recorded
    this way, so it is not reported.

    If progress_every > 0, the wrapper also prints one line to stderr every progress_every
    sweeps (and at the last one): sweep number out of total, elapsed time, ETA, lambda and
    kappa. It only reads the sampler's state and never draws random numbers, so the
    samples are the same with logging on or off.
    """
    records = []
    if not hasattr(sampler, "_update_lam_sigma2"):
        return records
    inner = sampler._update_lam_sigma2
    started = time.perf_counter()
    last = {"done": 0, "time": started}

    def wrapped(psi, it, adapt):
        inner(psi, it, adapt)
        records.append((int(it), float(getattr(sampler, "acc_rate", np.nan))))
        done = int(it) + 1
        if progress_every and total and (done % progress_every == 0 or done == total):
            now = time.perf_counter()
            elapsed = now - started
            # The ETA uses the rate since the previous progress line, not the cumulative
            # average: the first sweeps include numba compilation and would overstate it.
            rate = (now - last["time"]) / (done - last["done"])
            eta = rate * (total - done)
            last["done"], last["time"] = done, now
            extra = f", kappa={sampler.kappa:.4f}" if sampler.use_kappa else ""
            stamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{stamp}] [{label}] sweep {done}/{total}, elapsed {elapsed:.1f}s, eta {eta:.1f}s, "
                  f"lam={sampler.lam:.4f}{extra}", file=sys.stderr, flush=True)

    sampler._update_lam_sigma2 = wrapped
    return records


def _diagnostics(sampler, acceptance=(), burn=0, use_kappa=False):
    """Plain, JSON-serialisable summary of one fitted chain.

    Per parameter: mean, 95% interval, ESS and split R-hat (from the model's summary()).
    Also: the number of kept draws, the lambda Metropolis-Hastings acceptance averaged over
    all sweeps (mh_acc_rate) and over sweeps after burn-in (mh_acc_rate_mean_postburn), and
    a notes string. acceptance is the list of (sweep, acc_rate) from _record_lam_acceptance.
    """
    out = {}
    if hasattr(sampler, "summary"):
        for name, d in sampler.summary().items():
            out[name] = {"mean": _finite_or_none(d["mean"]), "lo": _finite_or_none(d["lo"]),
                         "hi": _finite_or_none(d["hi"]), "ess": _finite_or_none(d["ess"]),
                         "rhat": _finite_or_none(d["rhat"])}
    samples = getattr(sampler, "samples", {})
    out["n_samples"] = int(len(samples["lam"])) if "lam" in samples else 0
    vals = [a for _, a in acceptance if np.isfinite(a)]
    post = [a for it, a in acceptance if it >= burn and np.isfinite(a)]
    out["mh_acc_rate"] = _finite_or_none(np.mean(vals)) if vals else None
    out["mh_acc_rate_mean_postburn"] = _finite_or_none(np.mean(post)) if post else None
    notes = ["lambda acceptance: mean over all Gibbs sweeps (mh_acc_rate) and over sweeps after burn-in "
             "(mh_acc_rate_mean_postburn)."]
    if use_kappa:
        notes.append("kappa acceptance is not stored by the sampler, so it is not reported.")
    out["notes"] = " ".join(notes)
    return out

def _sequences(events, part_mask):
    """One sequence per (user, card) over the rows selected by part_mask.

    Each dict has: user, item (dense ints), r (outcomes), x (lagged outcomes, x[0] = 0),
    dt (gaps, with NaN replaced by 1.0 as the model expects), event_id (per row).
    Rows keep review_idx order inside each card.
    """
    rows = events.loc[part_mask].sort_values(["user_id", "card_id", "review_idx"])
    seqs = []
    for (user, item), grp in rows.groupby(["user", "item"], sort=True):
        r = grp["y"].to_numpy(dtype=float)
        x = np.concatenate([[0.0], r[:-1]])
        dt = grp["dt_days"].to_numpy(dtype=float)
        dt = np.where(np.isfinite(dt), dt, 1.0)
        seqs.append(dict(user=int(user), item=int(item), r=r, x=x, dt=dt,
                         event_id=grp["event_id"].to_numpy(np.int64)))
    return seqs


class OursModel:
    """DynamicRetentionModelDT with the common interface: fit(events) and predict(events).

    fit() runs the Gibbs sampler on TRAIN sequences (iters sweeps, the first burn are
    discarded, every thin-th kept). predict() uses at most max_draws posterior draws.
    progress_every > 0 logs one stderr line every that many sweeps during fit().
    After fit(): `model_` is the fitted sampler and `fit_seconds_` is the wall time.
    """

    def __init__(self, use_kappa=False, iters=2000, burn=1000, thin=4, seed=0, max_draws=100,
                 progress_every=0):
        self.use_kappa = use_kappa
        self.iters = iters
        self.burn = burn
        self.thin = thin
        self.seed = seed
        self.max_draws = max_draws
        self.progress_every = progress_every
        self.model_ = None
        self.fit_seconds_ = None

    def _dense_ids(self, events):
        """Dense int user and item (item = card) for each row, using the ids stored at fit().

        Ids come from TRAIN rows only. A user or card with no TRAIN rows maps to the
        padding index (n_users / n_items), so the model's cold path applies to it.
        """
        users = [self._users.get(u, self._n_users) for u in events["user_id"]]
        items = [self._items.get(k, self._n_items)
                 for k in zip(events["user_id"], events["card_id"])]
        out = events.copy()
        out["user"] = np.asarray(users, dtype=np.int64)
        out["item"] = np.asarray(items, dtype=np.int64)
        return out

    def fit(self, events):
        """Fit on TRAIN rows. Ids are assigned from TRAIN rows only, so test-only cards get padding."""
        train = events[events["is_train"].astype(bool)]
        self._users = {u: i for i, u in enumerate(sorted(train["user_id"].unique()))}
        self._items = {k: i for i, k in enumerate(sorted(set(zip(train["user_id"], train["card_id"]))))}
        self._n_users, self._n_items = len(self._users), len(self._items)
        ev = self._dense_ids(events)
        train_seqs = _sequences(ev, ev["is_train"].astype(bool).to_numpy())
        if not train_seqs:
            raise ValueError("no TRAIN rows to fit on")
        started = time.perf_counter()
        self.model_ = DynamicRetentionModelDT(
            train_seqs, rng=np.random.default_rng(self.seed), use_kappa=self.use_kappa)
        label = "ours_kappa" if self.use_kappa else "ours"
        acceptance = _record_lam_acceptance(self.model_, total=self.iters,
                                            progress_every=self.progress_every, label=label)
        self.model_.run(n_iter=self.iters, burn_in=self.burn, thin=self.thin)
        self.fit_seconds_ = time.perf_counter() - started
        self.diagnostics_ = _diagnostics(self.model_, acceptance, burn=self.burn, use_kappa=self.use_kappa)
        return self

    def predict(self, events):
        """Recall probabilities for scored rows, indexed by event_id.

        Unknown users and cards get the model's padding index, so they take its cold path.
        """
        if self.model_ is None:
            raise RuntimeError("call fit() before predict()")
        ev = self._dense_ids(events)
        test_seqs = _sequences(ev, ~ev["is_train"].astype(bool).to_numpy())
        if not test_seqs:
            return pd.Series([], index=pd.Index([], name="event_id"), name="p", dtype=float)
        preds = self.model_.predict(test_seqs, max_draws=self.max_draws)
        scored = set(ev.loc[ev["scored"].astype(bool), "event_id"].astype(np.int64))
        ids, vals = [], []
        for seq, p in zip(test_seqs, preds):
            keep = np.array([int(e) in scored for e in seq["event_id"]], dtype=bool)
            ids.append(seq["event_id"][keep])
            vals.append(np.asarray(p, dtype=float)[keep])
        out = pd.Series(np.concatenate(vals), index=pd.Index(np.concatenate(ids), name="event_id"),
                        name="p", dtype=float)
        return out.sort_index().clip(1e-6, 1 - 1e-6)
