"""
fsrs_model.py -- Adapter that runs py-fsrs (official FSRS-6) on our event table.

Why a replay: FSRS keeps a per-card memory state (stability and difficulty) that
every review updates. To predict event k we must rebuild that state from the
card's history before k, then ask for retrievability at k's time. We do this by
running the card through py-fsrs' own Scheduler, so the memory update is the
library's, not ours. Predictions are one-step-ahead: p_k uses only events before k.

Why synthetic timestamps: py-fsrs works in datetimes, but our data has integer
day offsets. Each review maps to a fixed UTC epoch plus day_offset days, plus its
position within its card in seconds. The seconds only keep the order strict and
keep same-day reviews distinct. py-fsrs measures gaps in whole days
((t2 - t1).days), so the gap it sees equals the day_offset difference, which is how
Anki counts elapsed_days.

Why two variants:
  - optimize=False: py-fsrs default weights (FSRS-6, 21 parameters) for everyone.
  - optimize=True: py-fsrs' Optimizer fits the 21 weights per user on that user's
    TRAIN reviews only. Users with too little data keep the defaults, and we record
    who fell back, so the report can say so.
Parameters are frozen after fit, so the test period never influences them.
"""
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from fsrs import Card, Optimizer, Rating, ReviewLog, Scheduler
from fsrs.scheduler import DEFAULT_PARAMETERS

EPOCH = datetime(2000, 1, 1, tzinfo=timezone.utc)
# Within-card position is stored in seconds, so it must stay inside one day.
MAX_REVIEWS_PER_CARD = 86400
# py-fsrs' Optimizer returns DEFAULT_PARAMETERS without any warning when a user has
# fewer non-same-day train reviews than its mini_batch_size (512). We apply the same
# rule up front, so the fallback is recorded instead of silently hidden.
MIN_INFORMATIVE_TRAIN = 512
P_CLIP = 1e-6


def _timestamp(day_offset, position):
    """UTC datetime for a review: EPOCH + day_offset days + position seconds."""
    if position >= MAX_REVIEWS_PER_CARD:
        raise ValueError(f"a card has more than {MAX_REVIEWS_PER_CARD} reviews")
    return EPOCH + timedelta(days=int(day_offset), seconds=int(position))


def _informative(frame):
    """Rows whose gap to the previous review is at least one day (the scoring convention)."""
    return frame[(~frame["is_first"].astype(bool)) & (frame["elapsed_days"] >= 1)]


def _replay_card(scheduler, card_id, rows):
    """Replay one card's reviews in order. Return {event_id: p} for the scored rows.

    rows: DataFrame sorted by review_idx with event_id, day_offset, rating, scored.
    p is the retrievability just BEFORE the review is applied, at that review's time.
    """
    card = Card(card_id=int(card_id))
    out = {}
    previous_day = None
    for position, row in enumerate(rows.itertuples(index=False)):
        if previous_day is not None and row.day_offset < previous_day:
            raise ValueError(f"card {card_id}: day_offset goes backwards at review_idx {row.review_idx}")
        previous_day = row.day_offset
        when = _timestamp(row.day_offset, position)
        if row.scored:
            out[int(row.event_id)] = scheduler.get_card_retrievability(card, when)
        card, _ = scheduler.review_card(card, Rating(int(row.rating)), when)
    return out


def _review_logs(train_user):
    """ReviewLog list for the Optimizer, built from one user's TRAIN rows only."""
    logs = []
    for _, rows in train_user.sort_values(["card_id", "review_idx"]).groupby("card_id", sort=True):
        for position, row in enumerate(rows.itertuples(index=False)):
            logs.append(ReviewLog(
                card_id=int(row.card_id),
                rating=Rating(int(row.rating)),
                review_datetime=_timestamp(row.day_offset, position),
                review_duration=None,
            ))
    return logs


def _optimize_user(user_id, train):
    """Run py-fsrs' Optimizer for one user. Module-level so it can run in a worker process.

    Returns (user_id, status, value, seconds): status "ok" with the weight tuple, or
    "error" with the message. Users are independent, so the result does not depend on
    which process ran it.
    """
    started = time.perf_counter()
    try:
        params = Optimizer(_review_logs(train)).compute_optimal_parameters()
        Scheduler(parameters=params, enable_fuzzing=False)  # validates bounds
        value = tuple(float(w) for w in params)
        status = "ok"
    except Exception as err:  # the optimiser can fail on odd histories
        value, status = str(err), "error"
    return int(user_id), status, value, time.perf_counter() - started


def _worker_init():
    """One torch thread per worker, so N workers do not oversubscribe the CPU."""
    import torch
    torch.set_num_threads(1)


class FSRSModel:
    """py-fsrs predictor with the common interface: fit(events) and predict(events).

    fit() uses only rows where is_train is True. predict() replays each card's full
    history (train and test rows, in review_idx order) and returns p for the scored
    rows only, indexed by event_id.

    n_workers > 1 runs the per-user Optimizer fits in a process pool. Each user is
    independent and the fit is deterministic, so the results match n_workers=1.

    After fit(), these attributes describe what happened:
      params_       {user_id: tuple of 21 weights}
      fallback_     {user_id: reason} for users that kept the default weights
      fit_seconds_  {user_id: seconds spent in the Optimizer}
    """

    def __init__(self, optimize=True, n_workers=1):
        self.optimize = optimize
        self.n_workers = int(n_workers)
        if self.n_workers < 1:
            raise ValueError("n_workers must be at least 1")
        self.params_ = {}
        self.fallback_ = {}
        self.fit_seconds_ = {}

    def fit(self, events):
        """Fit per-user weights on TRAIN rows. With optimize=False, only defaults are set."""
        self.params_, self.fallback_, self.fit_seconds_ = {}, {}, {}
        jobs = []
        for user_id, user in events.groupby("user_id", sort=True):
            if not self.optimize:
                self.params_[int(user_id)] = tuple(DEFAULT_PARAMETERS)
                continue
            train = user[user["is_train"].astype(bool)]
            n_informative = len(_informative(train))
            if n_informative < MIN_INFORMATIVE_TRAIN:
                self._fall_back(user_id, f"only {n_informative} informative train reviews")
                continue
            jobs.append((int(user_id), train))

        for user_id, status, value, seconds in self._run_jobs(jobs):
            if status == "error":
                self._fall_back(user_id, f"optimizer error: {value}")
                continue
            self.fit_seconds_[user_id] = seconds
            if value == tuple(float(w) for w in DEFAULT_PARAMETERS):
                self._fall_back(user_id, "optimizer returned the default weights")
                continue
            self.params_[user_id] = value
        return self

    def _run_jobs(self, jobs):
        """Results in the same order as jobs, serially or in a spawn-context process pool."""
        if self.n_workers == 1 or len(jobs) <= 1:
            return [_optimize_user(uid, train) for uid, train in jobs]
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=self.n_workers, mp_context=context,
                                 initializer=_worker_init) as pool:
            futures = [pool.submit(_optimize_user, uid, train) for uid, train in jobs]
            return [f.result() for f in futures]

    def _fall_back(self, user_id, reason):
        self.params_[int(user_id)] = tuple(DEFAULT_PARAMETERS)
        self.fallback_[int(user_id)] = reason
        self.fit_seconds_.pop(int(user_id), None)

    def predict(self, events):
        """Return a Series of recall probabilities, indexed by event_id, for scored rows."""
        if not self.params_:
            raise RuntimeError("call fit() before predict()")
        values = {}
        for user_id, user in events.groupby("user_id", sort=True):
            if int(user_id) not in self.params_:
                raise KeyError(f"user {user_id} was not seen in fit()")
            scheduler = Scheduler(parameters=self.params_[int(user_id)], enable_fuzzing=False)
            ordered = user.sort_values(["card_id", "review_idx"])
            for card_id, rows in ordered.groupby("card_id", sort=True):
                values.update(_replay_card(scheduler, card_id, rows))
        series = pd.Series(values, dtype=float, name="p")
        series.index.name = "event_id"
        series = series.sort_index()
        return series.clip(P_CLIP, 1 - P_CLIP)
