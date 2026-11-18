"""Tests for fsrs_benchmark/srsb/user_boost.py (Phase 6, task 6.8): refit schedule, the leak rule (synthetic and
on two real validation users), and the GBDT reproduction on those users."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fsrs_benchmark.srsb import user_boost as U
from fsrs_benchmark.srsb import val_plus as V

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "results/srsb_beat/sweep_a_l63_side.jsonl.model.txt"
RUN_INFO = ROOT / "results/srsb_beat/sweep_a_l63_side.jsonl.run_info.json"
GBDT = ROOT / "results/srsb_beat/sweep_a_l63_side.jsonl"
DATA = ROOT / "data/anki10k"
REAL_USERS = (8002, 8019)              # validation users with 1000+ scored rows (1795 and 2415 test rows)


def _synthetic(n=1600, seed=0):
    rng = np.random.default_rng(seed)
    day = np.arange(n) // 5                       # a new day every 5 rows
    labelled = np.ones(n, bool)
    labelled[::7] = False                         # some first reviews
    sub = rng.normal(0, 1, (n, len(U.SUBSET)))
    margin = rng.normal(-0.5, 1.0, n)
    logit = margin + 1.2 * sub[:, 0] - 0.8 * sub[:, 1]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(np.float64)
    return margin, sub, y, labelled, day


def test_refit_points_follow_day_boundaries_spacing_and_min_rows():
    n = 2000
    day = np.arange(n) // 3
    labelled = np.ones(n, bool)
    pts = U.refit_points(day, labelled, refit_rows=200, min_rows=100)
    assert pts, "expected refits"
    assert all(day[t] != day[t - 1] for t in pts)            # only at the start of a new day_offset
    assert all(b - a >= 200 for a, b in zip(pts, pts[1:]))   # at most one refit per 200 rows
    assert pts[0] >= 100                                     # MIN_ROWS labelled rows before the first refit
    # no labelled rows at all: never refit
    assert U.refit_points(day, np.zeros(n, bool)) == []
    # first-review flags count toward the min-rows rule only when labelled
    lab = np.zeros(n, bool)
    lab[:50] = True
    assert U.refit_points(day, lab, min_rows=100) == []


def test_before_first_refit_prediction_is_the_global_gbdt():
    margin, sub, y, labelled, day = _synthetic(n=150)
    labelled[:] = True
    delta, nref = U.boost_predict(margin, sub, y, labelled, day, rounds=20, leaves=4)
    first = U.refit_points(day, labelled)[0]
    assert nref >= 1
    assert np.all(delta[:first] == 0.0)


def test_labels_at_and_after_t_do_not_change_predictions_up_to_t():
    margin, sub, y, labelled, day = _synthetic()
    base, nref = U.boost_predict(margin, sub, y, labelled, day, rounds=20, leaves=4)
    assert nref >= 3                                         # the test is not vacuous
    for k in (300, 700, 1200):
        y2 = y.copy()
        y2[k:] = 1.0 - y2[k:]                                # flip every label from row k on
        other, _ = U.boost_predict(margin, sub, y2, labelled, day, rounds=20, leaves=4)
        assert np.array_equal(base[:k], other[:k]), f"prediction before row {k} changed"
        assert not np.array_equal(base[k:], other[k:])       # the change is visible later


def test_features_and_margin_of_later_rows_do_not_change_earlier_predictions():
    margin, sub, y, labelled, day = _synthetic()
    base, _ = U.boost_predict(margin, sub, y, labelled, day, rounds=20, leaves=4)
    k = 800
    m2, s2, y2 = margin.copy(), sub.copy(), y.copy()
    m2[k:] += 3.0
    s2[k:] += 5.0
    y2[k:] = 1.0 - y2[k:]
    other, _ = U.boost_predict(m2, s2, y2, labelled, day, rounds=20, leaves=4)
    assert np.array_equal(base[:k], other[:k])


def test_settings_grid_is_limited_to_three():
    assert len(U.settings_grid((10, 20, 40), 4)) == 3
    with pytest.raises(ValueError):
        U.settings_grid((10, 20, 40, 80), 4)


def _real_setup():
    import lightgbm as lgb
    info = json.loads(RUN_INFO.read_text(encoding="utf-8"))
    names, best_iter = info["features"], info["best_iter"]
    V._init(str(DATA), names)
    booster = lgb.Booster(model_file=str(MODEL))
    return booster, names, best_iter


@pytest.fixture(scope="module")
def real():
    if not (MODEL.exists() and RUN_INFO.exists() and DATA.exists()):
        pytest.skip("local data or model not available")
    return _real_setup()


@pytest.mark.parametrize("uid", REAL_USERS)
def test_real_user_gbdt_reproduces_saved_logloss(real, uid):
    booster, names, best_iter = real
    arr = U.user_arrays(uid, booster, names, best_iter, V._W["loader"], V._W["cfg"], DATA)
    res = U.summarise_user(arr, U.settings_grid((10,), 4))
    gb = U.C.load_rows(GBDT)
    assert abs(res["gbdt_ll"] - gb[uid]["metrics"]["LogLoss"]) < 1e-4
    assert res["size"] == gb[uid]["size"]


@pytest.mark.parametrize("uid", REAL_USERS)
def test_real_user_leak_rule_on_labels(real, uid):
    booster, names, best_iter = real
    arr = U.user_arrays(uid, booster, names, best_iter, V._W["loader"], V._W["cfg"], DATA)
    settings = U.settings_grid((20,), 4)
    base = U.predict_stream(arr, settings)["r20_l4"][0]
    n = len(arr["y"])
    k = n // 2
    arr2 = dict(arr, y=arr["y"].copy())
    arr2["y"][k:] = 1.0 - arr2["y"][k:]                      # labels of rows >= k flipped
    other = U.predict_stream(arr2, settings)["r20_l4"][0]
    assert np.array_equal(base[:k], other[:k])
    assert np.any(base[k:] != other[k:])


@pytest.mark.parametrize("uid", REAL_USERS)
def test_real_user_leak_rule_on_durations(real, uid):
    """Changing the durations of reviews after review_th K changes no prediction for review_th <= K."""
    booster, names, best_iter = real
    raw = pd.read_parquet(DATA / "revlogs" / f"user_id={uid}").reset_index(drop=True)
    k = len(raw) // 2
    raw2 = raw.copy()
    raw2.loc[k:, "duration"] = raw2.loc[k:, "duration"] + 60000
    loader, cfg = V._W["loader"], V._W["cfg"]
    settings = U.settings_grid((20,), 4)
    a = U.user_arrays(uid, booster, names, best_iter, loader, cfg, DATA, raw=raw)
    b = U.user_arrays(uid, booster, names, best_iter, loader, cfg, DATA, raw=raw2)
    pa = U.predict_stream(a, settings)["r20_l4"][0]
    pb = U.predict_stream(b, settings)["r20_l4"][0]
    before = a["rt"] <= k
    assert before.any() and (~before).any()
    assert np.array_equal(pa[before], pb[before])
    assert not np.array_equal(pa[~before], pb[~before])
