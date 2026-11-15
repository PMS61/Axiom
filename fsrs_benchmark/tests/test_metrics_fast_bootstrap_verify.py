"""Verifier adversarial tests for the weighted fast paired bootstrap (metrics.paired_bootstrap)."""
import numpy as np
import pytest

from fsrs_benchmark import metrics as M

ALL = list(M.FAST_METRICS)


def _ref(uid, y, pa, pb, metric, n_boot=60, seed=0):
    return M._paired_bootstrap_reference(uid, y, pa, pb, metric, n_boot=n_boot, seed=seed)


def _fast(uid, y, pa, pb, metric, n_boot=60, seed=0):
    return M.paired_bootstrap(uid, y, pa, pb, metric, n_boot=n_boot, seed=seed)


def _same(a, b):
    for k in ("diff", "lo", "hi"):
        if np.isnan(a[k]) or np.isnan(b[k]):
            assert np.isnan(a[k]) and np.isnan(b[k]), (k, a, b)
        else:
            assert a[k] == pytest.approx(b[k], abs=1e-9), (k, a, b)


@pytest.mark.parametrize("metric", ALL)
def test_all_equal_predictions_agree(metric):
    rng = np.random.default_rng(1)
    uid = rng.integers(0, 15, 300)
    y = rng.integers(0, 2, 300)
    p = np.full(300, 0.7)
    _same(_fast(uid, y, p, p * 0.9, metric), _ref(uid, y, p, p * 0.9, metric))


@pytest.mark.parametrize("metric", ALL)
def test_single_user_agrees_and_has_zero_width(metric):
    rng = np.random.default_rng(2)
    uid = np.zeros(80, int)
    y = rng.integers(0, 2, 80)
    pa = rng.uniform(0, 1, 80)
    pb = rng.uniform(0, 1, 80)
    f = _fast(uid, y, pa, pb, metric)
    _same(f, _ref(uid, y, pa, pb, metric))
    # one cluster: every resample is the same data, so the interval collapses to the point
    if np.isfinite(f["lo"]):
        assert f["lo"] == pytest.approx(f["diff"], abs=1e-9)
        assert f["hi"] == pytest.approx(f["diff"], abs=1e-9)


@pytest.mark.parametrize("metric", ALL)
def test_user_with_one_event_agrees(metric):
    rng = np.random.default_rng(3)
    uid = np.concatenate([np.zeros(50, int), np.array([7])])
    y = rng.integers(0, 2, 51)
    pa = rng.uniform(0, 1, 51)
    pb = rng.uniform(0, 1, 51)
    _same(_fast(uid, y, pa, pb, metric), _ref(uid, y, pa, pb, metric))


@pytest.mark.parametrize("metric", ALL)
def test_very_unequal_user_sizes_agree(metric):
    rng = np.random.default_rng(4)
    sizes = [1, 2, 3, 500, 1, 40, 900]
    uid = np.concatenate([np.full(s, i) for i, s in enumerate(sizes)])
    y = rng.integers(0, 2, uid.size)
    pa = rng.uniform(0, 1, uid.size)
    pb = np.clip(pa + rng.normal(0, 0.1, uid.size), 0, 1)
    _same(_fast(uid, y, pa, pb, metric, n_boot=80, seed=5), _ref(uid, y, pa, pb, metric, n_boot=80, seed=5))


def test_user_ids_are_not_required_to_be_contiguous_or_sorted():
    rng = np.random.default_rng(6)
    uid = rng.choice([1000003, 5, 77, 2], 400)
    y = rng.integers(0, 2, 400)
    pa = rng.uniform(0, 1, 400)
    pb = rng.uniform(0, 1, 400)
    for m in ALL:
        _same(_fast(uid, y, pa, pb, m), _ref(uid, y, pa, pb, m))


def test_boundary_probabilities_0_and_1_agree():
    rng = np.random.default_rng(7)
    uid = rng.integers(0, 10, 200)
    y = rng.integers(0, 2, 200)
    pa = rng.choice([0.0, 1.0, 0.1, 0.3, 0.5], 200)
    pb = rng.choice([0.0, 1.0, 0.9, 0.3], 200)
    for m in ["ece", "rmse_bins", "auc", "log_loss", "brier", "mean_bias"]:
        _same(_fast(uid, y, pa, pb, m), _ref(uid, y, pa, pb, m))


def test_all_one_class_gives_nan_interval_for_auc():
    uid = np.repeat(np.arange(5), 6)
    y = np.ones(30, int)
    pa = np.linspace(0.1, 0.9, 30)
    out = _fast(uid, y, pa, pa, "auc", n_boot=20)
    assert np.isnan(out["lo"]) and np.isnan(out["hi"])


def test_identical_models_zero_difference_for_every_metric():
    rng = np.random.default_rng(8)
    uid = rng.integers(0, 12, 240)
    y = rng.integers(0, 2, 240)
    p = rng.uniform(0.05, 0.95, 240)
    for m in ALL:
        out = _fast(uid, y, p, p, m, n_boot=30)
        assert out["diff"] == 0.0 and out["lo"] == 0.0 and out["hi"] == 0.0, m


def test_rng_stream_is_the_same_as_reference_for_same_seed():
    # The fast path must consume the same draws; a different seed must give a different interval.
    rng = np.random.default_rng(9)
    uid = rng.integers(0, 20, 500)
    y = rng.integers(0, 2, 500)
    pa = rng.uniform(0, 1, 500)
    pb = rng.uniform(0, 1, 500)
    a = _fast(uid, y, pa, pb, "log_loss", n_boot=100, seed=11)
    b = _fast(uid, y, pa, pb, "log_loss", n_boot=100, seed=12)
    assert (a["lo"], a["hi"]) != (b["lo"], b["hi"])
    _same(a, _ref(uid, y, pa, pb, "log_loss", n_boot=100, seed=11))


def test_rejects_nan_prediction_like_reference():
    uid = np.array([0, 0, 1, 1])
    y = np.array([0, 1, 0, 1])
    p = np.array([0.1, np.nan, 0.3, 0.4])
    with pytest.raises(ValueError):
        _fast(uid, y, p, p, "brier", n_boot=5)


def test_rejects_bad_label_like_reference():
    uid = np.array([0, 0, 1, 1])
    y = np.array([0, 2, 0, 1])
    p = np.array([0.1, 0.2, 0.3, 0.4])
    with pytest.raises(ValueError):
        _fast(uid, y, p, p, "brier", n_boot=5)
