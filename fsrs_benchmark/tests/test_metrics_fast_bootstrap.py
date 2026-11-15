"""The fast paired bootstrap must agree with the reference (rebuild-and-recompute) to 1e-9.

Random data is built with ties (p rounded to 2 decimals), few users, and users whose events
are all one class, so resamples can be one-class and the NaN-dropping rule is exercised.
"""
import numpy as np
import pytest

from fsrs_benchmark import metrics

NAMED = list(metrics.FAST_METRICS)


def _data(seed, n_users=6, n_events=300, tie_decimals=2, one_class_users=2):
    rng = np.random.default_rng(seed)
    user = rng.integers(0, n_users, size=n_events)
    y = (rng.random(n_events) < 0.4).astype(int)
    for u in range(one_class_users):  # users whose outcomes are all 1 or all 0
        y[user == u] = u % 2
    p_a = np.round(rng.random(n_events), tie_decimals)
    p_b = np.round(np.clip(rng.random(n_events) * 0.8 + 0.1, 0, 1), tie_decimals)
    return user, y, p_a, p_b


def _assert_same(fast, ref):
    assert fast["diff"] == pytest.approx(ref["diff"], abs=1e-9)
    for key in ["lo", "hi"]:
        if np.isnan(ref[key]):
            assert np.isnan(fast[key])
        else:
            assert fast[key] == pytest.approx(ref[key], abs=1e-9), key


@pytest.mark.parametrize("metric", NAMED)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fast_matches_reference(metric, seed):
    user, y, p_a, p_b = _data(seed)
    fast = metrics.paired_bootstrap(user, y, p_a, p_b, metric, n_boot=150, seed=seed)
    ref = metrics._paired_bootstrap_reference(user, y, p_a, p_b, metric, n_boot=150, seed=seed)
    _assert_same(fast, ref)


@pytest.mark.parametrize("metric", ["auc", "log_loss"])
def test_few_users_with_one_class_samples(metric):
    # 3 users: many resamples are single-class, so AUC is NaN and those draws are dropped.
    user, y, p_a, p_b = _data(7, n_users=3, n_events=60, one_class_users=1)
    fast = metrics.paired_bootstrap(user, y, p_a, p_b, metric, n_boot=200, seed=3)
    ref = metrics._paired_bootstrap_reference(user, y, p_a, p_b, metric, n_boot=200, seed=3)
    _assert_same(fast, ref)


def test_heavy_ties_auc():
    # Only three distinct p values: the tie handling (0.5 per tied pair) carries the result.
    rng = np.random.default_rng(4)
    user = rng.integers(0, 8, size=200)
    y = rng.integers(0, 2, size=200)
    p_a = rng.choice([0.2, 0.5, 0.9], size=200)
    p_b = rng.choice([0.3, 0.6], size=200)
    fast = metrics.paired_bootstrap(user, y, p_a, p_b, "auc", n_boot=200, seed=5)
    ref = metrics._paired_bootstrap_reference(user, y, p_a, p_b, "auc", n_boot=200, seed=5)
    _assert_same(fast, ref)


def test_constant_predictions_auc_is_half_in_every_draw():
    user, y, _, _ = _data(8)
    p = np.full(len(y), 0.5)
    fast = metrics.paired_bootstrap(user, y, p, p, "auc", n_boot=50, seed=1)
    assert fast["diff"] == 0.0 and fast["lo"] == 0.0 and fast["hi"] == 0.0


def test_weighted_auc_matches_expanded_rows():
    # Weights = counts must give the same AUC as repeating each row count times.
    rng = np.random.default_rng(9)
    y = rng.integers(0, 2, size=40)
    p = np.round(rng.random(40), 1)
    counts = rng.integers(0, 4, size=40).astype(float)
    counts[y == 1] += 1  # make sure both classes keep some weight
    rep = np.repeat(np.arange(40), counts.astype(int))
    w_metric = metrics._WeightedMetric("auc", y.astype(float), p)
    expected = metrics.auc(y[rep], p[rep])
    assert w_metric.value(counts) == pytest.approx(expected, abs=1e-12)


def test_weighted_ece_and_rmse_match_expanded_rows():
    rng = np.random.default_rng(10)
    y = rng.integers(0, 2, size=50).astype(float)
    p = rng.random(50)
    counts = rng.integers(0, 3, size=50).astype(float)
    rep = np.repeat(np.arange(50), counts.astype(int))
    assert metrics._WeightedMetric("ece", y, p).value(counts) == pytest.approx(metrics.ece(y[rep], p[rep]), abs=1e-12)
    assert metrics._WeightedMetric("rmse_bins", y, p).value(counts) == pytest.approx(
        metrics.rmse_bins(y[rep], p[rep]), abs=1e-12)


def test_callable_metric_uses_reference_path():
    user, y, p_a, p_b = _data(11)
    def my_brier(yy, pp):
        return float(np.mean((pp - yy) ** 2))
    fast = metrics.paired_bootstrap(user, y, p_a, p_b, my_brier, n_boot=60, seed=2)
    ref = metrics._paired_bootstrap_reference(user, y, p_a, p_b, my_brier, n_boot=60, seed=2)
    _assert_same(fast, ref)


def test_shape_mismatch_still_raises():
    with pytest.raises(ValueError):
        metrics.paired_bootstrap(np.arange(3), np.zeros(4), np.full(4, 0.5), np.full(4, 0.5), "brier", n_boot=2)
