"""Verifier cross-checks for fsrs_benchmark.metrics (Task 1.3 review)."""
import time

import numpy as np
import pandas as pd
import pytest
import torch

from fsrs_benchmark.metrics import (
    METRICS, auc, brier, ece, log_loss, mean_bias, paired_bootstrap, reliability_curve,
    rmse_bins, summarize,
)


def _brute_auc(y, p):
    """O(n^2) AUC: a pair counts 1 if pos > neg, 0.5 if tied."""
    y = np.asarray(y)
    p = np.asarray(p)
    pos = p[y == 1]
    neg = p[y == 0]
    total = 0.0
    for a in pos:
        for b in neg:
            total += 1.0 if a > b else (0.5 if a == b else 0.0)
    return total / (len(pos) * len(neg))


def _loop_ece(y, p, n_bins=10):
    """Independent ECE: explicit bin edges with np.histogram-style closed last bin."""
    edges = np.linspace(0, 1, n_bins + 1)
    n = len(y)
    total = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if b < n_bins - 1:
            m = (p >= lo) & (p < hi)
        else:
            m = (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        total += m.sum() * abs(p[m].mean() - y[m].mean())
    return total / n


# ---- AUC ------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(8))
def test_auc_matches_brute_force_with_ties(seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, 60)
    p = np.round(rng.uniform(size=60), 1)  # coarse values -> many ties
    if y.min() == y.max():
        y[0] = 1 - y[0]
    assert auc(y, p) == pytest.approx(_brute_auc(y, p), abs=1e-12)


def test_auc_invariant_to_monotone_transform():
    rng = np.random.default_rng(11)
    y = rng.integers(0, 2, 200)
    p = rng.uniform(size=200)
    assert auc(y, p) == pytest.approx(auc(y, np.sqrt(p)))


# ---- log loss and brier vs torch ------------------------------------------------

@pytest.mark.parametrize("seed", range(4))
def test_log_loss_matches_torch_bce(seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, 500).astype(float)
    p = rng.uniform(0.001, 0.999, 500)
    ref = torch.nn.functional.binary_cross_entropy(
        torch.tensor(p, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)).item()
    assert log_loss(y, p) == pytest.approx(ref, rel=1e-9)


def test_log_loss_extreme_values_clipped_like_eps():
    y = np.array([1.0, 0.0])
    p = np.array([1.0, 0.0])  # perfect predictions, clipped to 1-1e-6 / 1e-6
    assert log_loss(y, p) == pytest.approx(-np.log(1 - 1e-6), rel=1e-9)


def test_brier_matches_manual_random():
    rng = np.random.default_rng(5)
    y = rng.integers(0, 2, 300)
    p = rng.uniform(size=300)
    assert brier(y, p) == pytest.approx(np.mean([(pi - yi) ** 2 for pi, yi in zip(p, y)]))


# ---- ECE and reliability -------------------------------------------------------

@pytest.mark.parametrize("seed", range(5))
def test_ece_matches_independent_loop(seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, 400)
    p = rng.uniform(size=400)
    assert ece(y, p) == pytest.approx(_loop_ece(y, p), abs=1e-12)


def test_ece_p_one_and_zero_in_end_bins():
    y = np.array([1, 0])
    p = np.array([1.0, 0.0])
    rc = reliability_curve(y, p)
    assert rc.loc[9, "n"] == 1 and rc.loc[0, "n"] == 1
    assert ece(y, p) == pytest.approx(0.0)


def test_reliability_bin_edges_on_boundary_values():
    rc = reliability_curve([1, 1, 1], [0.1, 0.2, 0.3], n_bins=10)
    # 0.1 -> bin 1 (floor(10*0.1)=1), 0.2 -> bin 2, 0.3 -> bin 3 (0.3*10=3.0000000000000004)
    assert rc.loc[1, "n"] == 1 and rc.loc[2, "n"] == 1 and rc.loc[3, "n"] == 1


def test_rc_counts_sum_to_n_and_weighted_means_match_totals():
    rng = np.random.default_rng(6)
    y = rng.integers(0, 2, 1000)
    p = rng.uniform(size=1000)
    rc = reliability_curve(y, p)
    assert rc["n"].sum() == 1000
    assert np.sum(rc["n"] * rc["mean_p"].fillna(0)) == pytest.approx(p.sum())
    assert np.sum(rc["n"] * rc["mean_y"].fillna(0)) == pytest.approx(y.sum())


def test_rmse_bins_bounded_by_max_bin_gap_and_matches_hand():
    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, 500)
    p = rng.uniform(size=500)
    rc = reliability_curve(y, p, n_bins=20)
    gaps = (rc["mean_p"] - rc["mean_y"]).abs().dropna()
    r = rmse_bins(y, p, n_bins=20)
    assert r <= gaps.max() + 1e-12
    assert r >= ece(y, p, n_bins=20) - 1e-12  # RMSE >= mean abs gap (Jensen)


# ---- mean bias ----------------------------------------------------------------

def test_mean_bias_negative_when_pessimistic():
    assert mean_bias([1, 1, 1, 0], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5 - 0.75)


# ---- input validation -----------------------------------------------------------

def test_nan_prediction_is_not_silently_binned():
    # A NaN prediction is a model failure. It should raise, not be counted in bin 0.
    with pytest.raises(ValueError):
        reliability_curve([1, 0, 1], [0.5, np.nan, 0.9])


def test_out_of_range_prediction_is_rejected():
    with pytest.raises(ValueError):
        reliability_curve([1, 0], [1.2, 0.3])


def test_non_binary_label_is_rejected():
    with pytest.raises(ValueError):
        auc([0, 2, 1], [0.1, 0.5, 0.9])


# ---- summarize -----------------------------------------------------------------

def test_summarize_matches_individual_functions():
    rng = np.random.default_rng(8)
    y = rng.integers(0, 2, 200)
    p = rng.uniform(0.01, 0.99, 200)
    s = summarize(y, p)
    assert s["log_loss"] == pytest.approx(log_loss(y, p))
    assert s["brier"] == pytest.approx(brier(y, p))
    assert s["auc"] == pytest.approx(auc(y, p))
    assert s["ece"] == pytest.approx(ece(y, p))
    assert s["rmse_bins"] == pytest.approx(rmse_bins(y, p))
    assert s["mean_bias"] == pytest.approx(mean_bias(y, p))


# ---- paired bootstrap ---------------------------------------------------------

def _reference_bootstrap(uid, y, pa, pb, fn, n_boot, seed):
    users = np.unique(uid)
    groups = {u: np.where(uid == u)[0] for u in users}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(users), size=len(users))
        idx = np.concatenate([groups[users[i]] for i in pick])
        d = fn(y[idx], pa[idx]) - fn(y[idx], pb[idx])
        if np.isfinite(d):
            draws.append(d)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return lo, hi


def test_bootstrap_matches_reference_implementation():
    rng = np.random.default_rng(12)
    uid = np.repeat(np.arange(25), 12)
    y = rng.integers(0, 2, uid.size)
    pa = rng.uniform(size=uid.size)
    pb = rng.uniform(size=uid.size)
    out = paired_bootstrap(uid, y, pa, pb, "brier", n_boot=200, seed=3)
    lo, hi = _reference_bootstrap(uid, y, pa, pb, METRICS["brier"], 200, 3)
    assert out["lo"] == pytest.approx(lo)
    assert out["hi"] == pytest.approx(hi)
    assert out["diff"] == pytest.approx(brier(y, pa) - brier(y, pb))


def test_identical_models_auc_and_logloss_zero_interval():
    rng = np.random.default_rng(13)
    uid = np.repeat(np.arange(15), 10)
    y = rng.integers(0, 2, uid.size)
    p = rng.uniform(size=uid.size)
    for metric in ("auc", "log_loss", "ece"):
        out = paired_bootstrap(uid, y, p, p, metric, n_boot=50, seed=0)
        assert out == {"diff": 0.0, "lo": 0.0, "hi": 0.0}


def test_resample_with_one_class_is_dropped_not_crashing():
    # user 0 is all positives, user 1 all negatives: a resample of only user 0 has
    # no negatives, so AUC is NaN for that draw. The CI must still be finite.
    uid = np.array([0] * 10 + [1] * 10)
    y = np.array([1] * 10 + [0] * 10)
    rng = np.random.default_rng(0)
    pa = np.concatenate([rng.uniform(0.6, 1, 10), rng.uniform(0, 0.4, 10)])
    pb = rng.uniform(size=20)
    out = paired_bootstrap(uid, y, pa, pb, "auc", n_boot=200, seed=0)
    assert np.isfinite(out["lo"]) and np.isfinite(out["hi"])
    assert out["lo"] <= out["hi"]


def test_bootstrap_ci_is_ordered_and_contains_diff_for_clear_difference():
    rng = np.random.default_rng(14)
    uid = np.repeat(np.arange(60), 20)
    y = rng.integers(0, 2, uid.size)
    good = np.clip(0.5 + 0.35 * (2 * y - 1) + rng.normal(0, 0.1, y.size), 0.01, 0.99)
    bad = np.full(y.size, 0.5)
    out = paired_bootstrap(uid, y, good, bad, "log_loss", n_boot=400, seed=1)
    assert out["lo"] <= out["hi"]
    assert out["lo"] <= out["diff"] <= out["hi"]
    assert out["hi"] < 0


def test_bootstrap_user_clustering_widens_interval_vs_event_level():
    # Users with identical per-user behaviour: clustering should make the CI wider
    # than a naive event-level bootstrap. Checked loosely (not a strict theorem).
    rng = np.random.default_rng(15)
    n_users, per = 50, 20
    uid = np.repeat(np.arange(n_users), per)
    user_effect = np.repeat(rng.normal(0, 0.3, n_users), per)
    y = (rng.uniform(size=uid.size) < 1 / (1 + np.exp(-user_effect))).astype(int)
    pa = np.full(uid.size, 0.6)
    pb = np.full(uid.size, 0.5)
    clustered = paired_bootstrap(uid, y, pa, pb, "log_loss", n_boot=300, seed=0)
    event_uid = np.arange(uid.size)
    naive = paired_bootstrap(event_uid, y, pa, pb, "log_loss", n_boot=300, seed=0)
    assert (clustered["hi"] - clustered["lo"]) > (naive["hi"] - naive["lo"])


def test_bootstrap_rejects_mismatched_user_ids():
    with pytest.raises(ValueError):
        paired_bootstrap([0, 1], [0, 1, 1], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3], "brier", n_boot=5)


def test_bootstrap_30k_events_1000_boots_under_budget():
    rng = np.random.default_rng(16)
    uid = np.repeat(np.arange(300), 100)
    y = rng.integers(0, 2, uid.size)
    a = rng.uniform(size=uid.size)
    b = rng.uniform(size=uid.size)
    t = time.perf_counter()
    out = paired_bootstrap(uid, y, a, b, "log_loss", n_boot=1000, seed=0)
    elapsed = time.perf_counter() - t
    assert elapsed < 10.0, f"{elapsed:.1f}s"
    assert out["lo"] <= out["diff"] <= out["hi"]
