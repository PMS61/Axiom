"""Verifier tests for fsrs_benchmark.models.hlr (Task 2.2).

The reference below is a transcription of duolingo/halflife-regression experiment.py
(MIT, Copyright 2016 Duolingo Inc.): the pclip/hclip helpers, predict (hlr branch),
and train_update (hlr branch). It is used only as a test oracle, to compare our
single SGD step against the official arithmetic on random states.
"""
import math
from collections import defaultdict

import numpy as np
import pandas as pd
import pytest

from fsrs_benchmark import data, split
from fsrs_benchmark.models import hlr
from fsrs_benchmark.tests._real import fitted_hlr, real_events
from fsrs_benchmark.models.hlr import HLRModel, features, sgd_update

LN2 = math.log(2.0)


# --- official transcription (oracle) -------------------------------------------------

def _official_pclip(p):
    return min(max(p, 0.0001), .9999)


def _official_hclip(h):
    return min(max(h, 15.0 / (24 * 60)), 274.)


class _OfficialHLR:
    """experiment.py SpacedRepetitionModel, hlr branch only (fv is a list of (name, value))."""

    def __init__(self, lrate=.001, hlwt=.01, l2wt=.1, sigma=1.):
        self.weights = defaultdict(float)
        self.fcounts = defaultdict(int)
        self.lrate, self.hlwt, self.l2wt, self.sigma = lrate, hlwt, l2wt, sigma

    def predict(self, inst, base=2.):
        dp = sum([self.weights[k] * x_k for (k, x_k) in inst["fv"]])
        h = _official_hclip(base ** dp)
        p = 2. ** (-inst["t"] / h)
        return _official_pclip(p), h

    def train_update(self, inst):
        p, h = self.predict(inst, 2.)
        dlp_dw = 2. * (p - inst["p"]) * (LN2 ** 2) * p * (inst["t"] / h)
        dlh_dw = 2. * (h - inst["h"]) * LN2 * h
        for (k, x_k) in inst["fv"]:
            rate = (1. / (1 + inst["p"])) * self.lrate / math.sqrt(1 + self.fcounts[k])
            self.weights[k] -= rate * dlp_dw * x_k
            self.weights[k] -= rate * self.hlwt * dlh_dw * x_k
            self.weights[k] -= rate * self.l2wt * self.weights[k] / self.sigma ** 2
            self.fcounts[k] += 1


def _random_states(n, seed):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        w = rng.normal(0, 1.5, size=3).tolist()
        fcounts = rng.integers(0, 50, size=3).tolist()
        x = [float(np.sqrt(1 + rng.integers(0, 30))), float(np.sqrt(1 + rng.integers(0, 30))), 1.0]
        t = float(rng.uniform(0.01, 400))
        y = float(rng.integers(0, 2))
        yield w, fcounts, x, t, y


def _run_official(w, fcounts, x, t, y, clip_obs=True):
    model = _OfficialHLR()
    for k in range(3):
        model.weights[k] = w[k]
        model.fcounts[k] = fcounts[k]
    p_obs = _official_pclip(y) if clip_obs else y
    inst = {"p": p_obs, "t": t, "fv": [(k, x[k]) for k in range(3)],
            "h": _official_hclip(-t / math.log(_official_pclip(y), 2))}
    model.train_update(inst)
    return [model.weights[k] for k in range(3)], [model.fcounts[k] for k in range(3)]


def _mine(w, fcounts, x, t, y):
    w = list(w)
    fcounts = list(fcounts)
    h_obs = hlr.observed_half_life(np.array([t]), np.array([y]))[0]
    sgd_update(w, fcounts, x, t, y, h_obs)
    return w, fcounts


# --- SGD step vs official arithmetic ------------------------------------------------

def test_sgd_step_matches_official_on_random_states():
    """Random states, binary outcomes, observed p = pclip(y) as in the official code."""
    for w, fcounts, x, t, y in _random_states(300, seed=3):
        got_w, got_c = _mine(w, fcounts, x, t, y)
        exp_w, exp_c = _run_official(w, fcounts, x, t, y, clip_obs=True)
        assert np.allclose(got_w, exp_w, rtol=1e-12, atol=1e-14), (w, x, t, y)
        assert got_c == exp_c


def test_sgd_step_uses_clipped_observed_p_like_official():
    for w, fcounts, x, t, y in _random_states(300, seed=4):
        got_w, _ = _mine(w, fcounts, x, t, y)
        exp_w, _ = _run_official(w, fcounts, x, t, y, clip_obs=True)
        assert np.allclose(got_w, exp_w, rtol=1e-12, atol=1e-14)


def test_observed_half_life_matches_official_read_data():
    """h_obs = hclip(-t / log2(pclip(p))) as in read_data."""
    for t, y in [(0.5, 1.0), (3.0, 0.0), (1000.0, 0.0), (0.001, 1.0), (10.0, 0.5)]:
        exp = _official_hclip(-t / math.log(_official_pclip(y), 2))
        got = hlr.observed_half_life(np.array([t]), np.array([y]))[0]
        assert got == pytest.approx(exp, rel=1e-12)


def test_clip_constants_match_official():
    assert hlr.MIN_HALF_LIFE == pytest.approx(15.0 / (24 * 60))
    assert hlr.MAX_HALF_LIFE == 274.0
    assert hlr.P_MIN == 1e-4 and hlr.P_MAX == pytest.approx(0.9999)


# --- features and leakage ------------------------------------------------------------

def _two_user_events():
    rows_a = [(0, 1), (1, 0), (4, 1), (9, 1)]
    rows_b = [(0, 0), (2, 1), (3, 0)]
    out = []
    for uid, rows, cid in [(1, rows_a, 5), (2, rows_b, 5)]:
        for i, (day, y) in enumerate(rows):
            gap = day - rows[i - 1][0] if i else -1
            out.append(dict(event_id=uid * 10**9 + i, user_id=uid, card_id=cid, review_idx=0,
                            day_offset=day, rating=3 if y else 1, y=y, dt_days=float(gap) if i else np.nan,
                            elapsed_days=gap, is_first=(i == 0), is_train=day < 4, scored=(day >= 4 and i > 0)))
    frame = pd.DataFrame(out).sort_values(["day_offset", "user_id"]).reset_index(drop=True)
    frame["review_idx"] = np.arange(len(frame))
    return frame


def test_features_are_strictly_prior_for_every_row():
    """Changing any outcome at or after row k must leave row k's features unchanged."""
    events = _two_user_events()
    base = features(events)
    for target in events["event_id"]:
        mutated = events.copy()
        later = (mutated["user_id"] == mutated.loc[mutated.event_id == target, "user_id"].iloc[0]) & \
                (mutated["card_id"] == 5) & (mutated["review_idx"] >= mutated.loc[mutated.event_id == target, "review_idx"].iloc[0])
        mutated.loc[later, "y"] = 1 - mutated.loc[later, "y"]
        after = features(mutated)
        assert np.array_equal(base.loc[target].to_numpy(), after.loc[target].to_numpy()), target


def test_features_equal_official_style_prior_counts_on_real_card():
    """On a real card, the counts equal a direct prior-count loop (per user and card)."""
    events = real_events()
    X = features(events)
    checked = 0
    for (uid, cid), rows in events.sort_values("review_idx").groupby(["user_id", "card_id"]):
        right = wrong = 0
        for r in rows.sort_values("review_idx").itertuples():
            assert X.loc[r.event_id].tolist() == pytest.approx(
                [np.sqrt(1 + right), np.sqrt(1 + wrong), 1.0], rel=1e-12)
            right += int(r.y)
            wrong += 1 - int(r.y)
            checked += 1
        if checked > 4000:
            break
    assert checked > 1000


def test_fit_ignores_test_rows_including_their_gaps():
    events = _two_user_events()
    events = events.assign(is_train=events["day_offset"] < 4)
    a = HLRModel(max_epochs=50).fit(events)
    changed = events.copy()
    test = ~changed["is_train"].astype(bool)
    changed.loc[test, "y"] = 1 - changed.loc[test, "y"]
    changed.loc[test, "dt_days"] = changed.loc[test, "dt_days"] * 3 + 7
    b = HLRModel(max_epochs=50).fit(changed)
    assert np.array_equal(a.theta_, b.theta_)


def test_fit_on_empty_train_set_does_not_crash():
    events = _two_user_events()
    events["is_train"] = False
    model = HLRModel(max_epochs=5).fit(events)
    assert np.array_equal(model.theta_, np.zeros(3))
    got = model.predict(events)
    assert len(got) == int(events["scored"].sum())


def test_predict_on_truncated_history_is_unchanged():
    """A scored row's prediction does not change when all later rows are removed."""
    events = _two_user_events()
    model = HLRModel(max_epochs=50).fit(events)
    full = model.predict(events)
    for row in events[events["scored"]].itertuples():
        trunc = events[(events["user_id"] == row.user_id) & (events["review_idx"] <= row.review_idx)]
        got = model.predict(trunc)
        assert got.loc[row.event_id] == pytest.approx(full.loc[row.event_id], abs=1e-15)


# --- determinism and convergence -----------------------------------------------------

def test_fit_is_deterministic_for_fixed_seed():
    events = _two_user_events()
    a = HLRModel(max_epochs=80, seed=7).fit(events)
    b = HLRModel(max_epochs=80, seed=7).fit(events)
    assert np.array_equal(a.theta_, b.theta_)
    assert a.epochs_ == b.epochs_


def _fitted_default_real():
    return real_events(), fitted_hlr(0.01)


def test_real_users_converge_and_predict_every_scored_event():
    events, model = _fitted_default_real()
    print(f"epochs={model.epochs_} theta={np.round(model.theta_, 4).tolist()}")
    assert model.epochs_ < model.max_epochs, "HLR did not converge within max_epochs"
    got = model.predict(events)
    assert set(got.index) == set(events.loc[events["scored"], "event_id"].astype(int))
    assert ((got > 0) & (got < 1)).all()
