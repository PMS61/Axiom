"""Known-answer tests for fsrs_benchmark.metrics."""
import time

import numpy as np
import pytest

from fsrs_benchmark.metrics import (
    auc, brier, ece, log_loss, mean_bias, paired_bootstrap, reliability_curve,
    rmse_bins, summarize,
)


def test_auc_textbook_075():
    # positives 0.8, 0.4; negatives 0.6, 0.2. Winning pairs: (0.8,0.6), (0.8,0.2), (0.4,0.2) = 3 of 4.
    assert auc([1, 1, 0, 0], [0.8, 0.4, 0.6, 0.2]) == pytest.approx(0.75)


def test_auc_ties_average_rank_and_constant_is_half():
    assert auc([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)
    # one positive tied with one negative, one clean win: (1 + 0.5) / 2 = 0.75
    assert auc([1, 0, 0], [0.7, 0.7, 0.1]) == pytest.approx(0.75)


def test_auc_single_class_is_nan():
    assert np.isnan(auc([1, 1, 1], [0.2, 0.5, 0.9]))
    assert np.isnan(auc([0, 0], [0.2, 0.5]))


def test_log_loss_hand_computed():
    expected = -(np.log(0.8) + np.log(0.8)) / 2  # y=1,p=.8 ; y=0,p=.2 -> -ln(.8)
    assert log_loss([1, 0], [0.8, 0.2]) == pytest.approx(expected)


def test_log_loss_clips_extremes():
    # p = 0 for a positive would be infinite without clipping
    val = log_loss([1], [0.0], eps=1e-6)
    assert np.isfinite(val)
    assert val == pytest.approx(-np.log(1e-6))


def test_brier_hand_computed():
    # ((1-0.8)^2 + (0-0.3)^2) / 2 = (0.04 + 0.09) / 2 = 0.065
    assert brier([1, 0], [0.8, 0.3]) == pytest.approx(0.065)


def test_ece_zero_for_perfect_calibration():
    # bin p=0.25: 4 events, 1 positive; bin p=0.75: 4 events, 3 positives
    y = [1, 0, 0, 0, 1, 1, 1, 0]
    p = [0.25] * 4 + [0.75] * 4
    assert ece(y, p, n_bins=10) == pytest.approx(0.0)


def test_ece_hand_computed():
    # p=0.9 with y=1 -> gap 0.1 ; p=0.1 with y=0 -> gap 0.1 ; weighted mean 0.1
    assert ece([1, 0], [0.9, 0.1], n_bins=10) == pytest.approx(0.1)


def test_ece_p_equal_one_lands_in_last_bin():
    rc = reliability_curve([1, 1], [1.0, 1.0], n_bins=10)
    assert rc.loc[9, "n"] == 2
    assert rc["n"].sum() == 2


def test_reliability_curve_columns_and_values():
    rc = reliability_curve([1, 0, 1, 1], [0.15, 0.12, 0.95, 0.85], n_bins=10)
    assert list(rc.columns) == ["bin_lo", "bin_hi", "n", "mean_p", "mean_y"]
    assert len(rc) == 10
    row1 = rc.loc[1]
    assert row1["n"] == 2
    assert row1["mean_p"] == pytest.approx(0.135)
    assert row1["mean_y"] == pytest.approx(0.5)
    assert np.isnan(rc.loc[5, "mean_p"])  # empty bin
    assert rc.loc[9, "n"] == 1 and rc.loc[8, "n"] == 1


def test_mean_bias_sign_and_value():
    # mean p = 0.6, mean y = 0.5 -> bias +0.1 (optimistic)
    assert mean_bias([1, 0, 1, 0], [0.7, 0.5, 0.6, 0.6]) == pytest.approx(0.1)


def test_rmse_bins_hand_computed():
    # one bin only with n_bins=1: mean_p=0.6, mean_y=0.5 -> rmse 0.1
    assert rmse_bins([1, 0, 1, 0], [0.7, 0.5, 0.6, 0.6], n_bins=1) == pytest.approx(0.1)


def test_summarize_keys_and_n():
    out = summarize([1, 0, 1, 0], [0.7, 0.4, 0.6, 0.3])
    assert set(out) == {"n", "log_loss", "brier", "auc", "ece", "rmse_bins", "mean_bias"}
    assert out["n"] == 4
    assert out["auc"] == pytest.approx(1.0)


def test_bad_inputs_rejected():
    with pytest.raises(ValueError):
        brier([1, 0], [0.5])
    with pytest.raises(ValueError):
        log_loss([], [])


def test_paired_bootstrap_identical_models_give_zero():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    p = rng.uniform(0.05, 0.95, 400)
    uid = np.repeat(np.arange(20), 20)
    out = paired_bootstrap(uid, y, p, p, "log_loss", n_boot=100, seed=0)
    assert out["diff"] == 0.0
    assert out["lo"] == pytest.approx(0.0) and out["hi"] == pytest.approx(0.0)


def test_paired_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(2)
    n_users, per = 40, 30
    uid = np.repeat(np.arange(n_users), per)
    y = rng.integers(0, 2, n_users * per)
    p_good = np.clip(0.5 + 0.3 * (2 * y - 1) + rng.normal(0, 0.1, y.size), 0.01, 0.99)
    p_bad = np.full(y.size, 0.5)
    out = paired_bootstrap(uid, y, p_good, p_bad, "brier", n_boot=300, seed=0)
    assert out["lo"] <= out["diff"] <= out["hi"]
    assert out["diff"] < 0  # good model has lower Brier
    assert out["hi"] < 0    # and the CI excludes zero


def test_paired_bootstrap_accepts_callable_and_is_seeded():
    rng = np.random.default_rng(3)
    uid = np.repeat(np.arange(10), 10)
    y = rng.integers(0, 2, 100)
    a = rng.uniform(size=100)
    b = rng.uniform(size=100)
    f = lambda yy, pp: float(np.mean(pp))
    r1 = paired_bootstrap(uid, y, a, b, f, n_boot=50, seed=7)
    r2 = paired_bootstrap(uid, y, a, b, f, n_boot=50, seed=7)
    assert r1 == r2


def test_paired_bootstrap_speed_30k_events():
    rng = np.random.default_rng(4)
    n_users, per = 300, 100  # 30k events
    uid = np.repeat(np.arange(n_users), per)
    y = rng.integers(0, 2, uid.size)
    a = rng.uniform(size=uid.size)
    b = rng.uniform(size=uid.size)
    start = time.perf_counter()
    out = paired_bootstrap(uid, y, a, b, "auc", n_boot=1000, seed=0)
    elapsed = time.perf_counter() - start
    assert elapsed < 20.0, f"bootstrap took {elapsed:.1f}s"
    assert out["lo"] <= out["diff"] <= out["hi"]
