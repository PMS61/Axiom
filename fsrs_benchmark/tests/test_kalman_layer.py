"""Tests for fsrs_benchmark/srsb/kalman_layer.py (Phase 6, task 6.11): the leak rule (synthetic and on one real
validation user), the Bernoulli-logit update, and the variant-2 bucket separation."""
import json
from pathlib import Path

import numpy as np
import pytest

from fsrs_benchmark.srsb import kalman_layer as K
from fsrs_benchmark.srsb import val_plus as V

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "results/srsb_beat/sweep_a_l63_side.jsonl.model.txt"
RUN_INFO = ROOT / "results/srsb_beat/sweep_a_l63_side.jsonl.run_info.json"
DATA = ROOT / "data/anki10k"


def _synthetic(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.normal(-0.3, 1.0, n)
    dt = np.where(rng.random(n) < 0.3, 0.0, rng.integers(1, 5, n).astype(float))
    elapsed = rng.choice([0.5, 3.0, 12.0, 60.0], n)
    bucket = K.bucket_of(elapsed)
    labelled = np.ones(n, bool)
    labelled[::7] = False
    # a user whose true bias drifts slowly: the labels depend on it
    drift = np.cumsum(rng.normal(0, 0.05, n))
    y = (rng.random(n) < 1 / (1 + np.exp(-(m + drift)))).astype(np.float64)
    return m, y, labelled, dt, bucket


@pytest.mark.parametrize("variant", ["K1", "K2"])
def test_labels_at_and_after_t_do_not_change_p_up_to_t(variant):
    m, y, labelled, dt, bucket = _synthetic()
    cfg = (variant, 1e-3, 1e-2, 1.0)
    arr = dict(m=m, y=y, labelled=labelled, dt=dt, bucket=bucket)
    base = K.predict_config(arr, cfg)
    for k in (100, 600, 1200):
        y2 = y.copy()
        y2[k:] = 1.0 - y2[k:]
        other = K.predict_config(dict(arr, y=y2), cfg)
        assert np.array_equal(base[:k + 1], other[:k + 1]), f"p up to row {k} changed"
        assert not np.array_equal(base[k + 1:], other[k + 1:])


def test_no_drift_no_learning_gives_the_gbdt():
    m, y, labelled, dt, bucket = _synthetic()
    p = K.predict_config(dict(m=m, y=y, labelled=labelled, dt=dt, bucket=bucket), ("K1", 0.0, 0.0, 1e-12))
    assert np.allclose(p, 1 / (1 + np.exp(-m)), rtol=1e-9)


def test_bias_moves_toward_the_labels():
    n = 200
    m = np.full(n, np.log(0.1 / 0.9))
    y = np.ones(n)
    labelled = np.ones(n, bool)
    z = np.zeros(n)
    bucket = np.zeros(n, np.int64)
    p = K.kalman_kernel(m, y, labelled, z, bucket, 0.0, 0.0, 1.0, 1)
    assert p[-1] > p[0] > 0.1


def test_update_formula_matches_the_bernoulli_logit_ekf():
    # one labelled row after a prior of P0 = 1, b = 0: P_new = 1/(1/P + p(1-p)), b_new = P_new (y - p)
    m = np.array([0.0, 0.0])
    y = np.array([1.0, 0.0])
    labelled = np.array([True, True])
    z = np.zeros(2)
    bucket = np.zeros(2, np.int64)
    p = K.kalman_kernel(m, y, labelled, z, bucket, 0.0, 0.0, 1.0, 1)
    pu = 0.5
    Pn = 1.0 / (1.0 / 1.0 + pu * (1 - pu))
    bn = Pn * (1.0 - pu)
    # the second prediction uses the updated state: corrected sigmoid(bn / sqrt(1 + pi Pn / 8))
    expect = 1.0 / (1.0 + np.exp(-bn / np.sqrt(1.0 + np.pi * Pn / 8.0)))
    assert np.isclose(p[1], expect)


def test_variant_two_keeps_gap_buckets_separate():
    n = 400
    rng = np.random.default_rng(1)
    m = np.zeros(n)
    bucket = np.where(np.arange(n) % 2 == 0, 0, 3)           # short gaps and long gaps alternate
    y = np.where(bucket == 0, 1.0, 0.0)                       # short gaps always recalled, long never
    labelled = np.ones(n, bool)
    dt = np.zeros(n)
    p = K.kalman_kernel(m, y, labelled, dt, bucket, 0.0, 0.0, 1.0, 4)
    assert p[bucket == 0][-1] > 0.6
    assert p[bucket == 3][-1] < 0.4
    single = K.kalman_kernel(m, y, labelled, dt, np.zeros(n, np.int64), 0.0, 0.0, 1.0, 1)
    assert abs(single[-1] - 0.5) < 0.2                        # one state cannot separate them


def test_bucket_boundaries():
    assert list(K.bucket_of([0.0, 0.99, 1.0, 6.9, 7.0, 29.9, 30.0, np.nan])) == [0, 0, 1, 1, 2, 2, 3, 0]


def test_grid_size():
    assert len(K.configs()) == 2 * 3 * 2 * 2


def test_real_user_leak_rule_on_labels():
    if not (MODEL.exists() and RUN_INFO.exists() and DATA.exists()):
        pytest.skip("local data or model not available")
    import lightgbm as lgb
    info = json.loads(RUN_INFO.read_text(encoding="utf-8"))
    names, best_iter = info["features"], info["best_iter"]
    V._init(str(DATA), names)
    booster = lgb.Booster(model_file=str(MODEL))
    arr = K.user_arrays(8019, booster, names, best_iter, V._W["loader"], V._W["cfg"], DATA)
    cfg = ("K2", 1e-3, 1e-2, 1.0)
    base = K.predict_config(arr, cfg)
    k = len(arr["y"]) // 2
    y2 = arr["y"].copy()
    y2[k:] = 1.0 - y2[k:]
    other = K.predict_config(dict(arr, y=y2), cfg)
    assert np.array_equal(base[:k + 1], other[:k + 1])
    assert np.any(base[k + 1:] != other[k + 1:])
