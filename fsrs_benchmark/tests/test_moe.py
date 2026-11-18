"""Tests for fsrs_benchmark/srsb/moe.py (Phase 6, task 6.7): fold plan by user, expert coverage checks, gate fit
improves on its offset, cross-fit never scores a user from its training half, per-user metrics."""
import numpy as np
import pandas as pd
import pytest

from fsrs_benchmark.srsb import moe as M


def _frame(n_per_user=30, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in list(range(8001, 8101)):
        for k in range(n_per_user):
            rows.append(dict(uid=u, review_th=u * 1000 + k, y=float(rng.random() < 0.6)))
    df = pd.DataFrame(rows)
    n = len(df)
    base_logit = rng.normal(0.4, 1.0, n)
    df["p_gbdt"] = M.sigmoid(base_logit)
    df["p_online"] = M.sigmoid(base_logit + rng.normal(0, 0.2, n))
    df["f_R"] = rng.uniform(0.2, 0.95, n)
    df["log_el_frac"] = rng.normal(0, 1, n)
    df["card_n"] = rng.integers(0, 20, n).astype(float)
    df["u_n"] = rng.integers(0, 500, n).astype(float)
    df["rate_today"] = np.where(rng.random(n) < 0.2, np.nan, rng.random(n))
    df["ewm50"] = rng.random(n)
    df["u_last5"] = rng.random(n)
    df["state"] = rng.integers(0, 4, n)
    return df


def test_fold_plan_is_by_user_and_disjoint():
    plan = M.fold_plan(range(8001, 8101))
    (tr1, sc1), (tr2, sc2) = plan
    assert set(tr1) == set(sc2) and set(sc1) == set(tr2)
    assert not set(tr1) & set(sc1)
    assert len(sc1) == 50 and len(sc2) == 50
    with pytest.raises(ValueError):
        M.fold_plan(range(8001, 8051))             # one half only


def test_logit_sigmoid_roundtrip():
    p = np.array([0.01, 0.3, 0.5, 0.9])
    assert np.allclose(M.sigmoid(M.logit(p)), p, atol=1e-9)


def test_assemble_requires_full_expert_coverage():
    df = _frame(n_per_user=3)
    ok = pd.DataFrame({"uid": df["uid"], "review_th": df["review_th"], "p": 0.5})
    out = M.assemble(df, {"p_tam": ok.rename(columns={"p": "p_tam"})})
    assert out["p_tam"].eq(0.5).all()
    short = ok.iloc[:-1]
    with pytest.raises(ValueError, match="does not cover"):
        M.assemble(df, {"p_tam": short.rename(columns={"p": "p_tam"})})


def test_logreg_gate_beats_its_offset_when_the_offset_is_wrong():
    rng = np.random.default_rng(1)
    n = 4000
    x = rng.normal(0, 1, (n, 1))
    offset = np.zeros(n)
    y = (rng.random(n) < M.sigmoid(1.5 * x[:, 0])).astype(float)   # the offset (0) misses the signal
    m = M.fit_logreg_gate(x, y, offset)
    p = M.predict_logreg_gate(m, x, offset)
    ll = lambda q: -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
    assert ll(p) < ll(M.sigmoid(offset)) - 0.1


def test_crossfit_scores_each_half_with_the_other_half_and_stays_in_range():
    df = _frame()
    df["p_fsrs"] = df["f_R"]
    p, info = M.crossfit(df, ["p_gbdt", "p_online", "p_fsrs"], kind="logreg", offset_col="p_online")
    assert np.isfinite(p).all() and ((p > 0) & (p < 1)).all()
    scored = [set(i["scored"]) for i in info]
    assert scored[0] == set(range(8051, 8101)) and scored[1] == set(range(8001, 8051))


def test_lgb_gate_runs_and_is_out_of_fold():
    df = _frame(n_per_user=10)
    df["p_fsrs"] = df["f_R"]
    p, info = M.crossfit(df, ["p_gbdt", "p_online", "p_fsrs"], kind="lgb", offset_col="p_online")
    assert np.isfinite(p).all()
    assert all("best_iter" in i and i["best_iter"] >= 1 for i in info)


def test_per_user_and_summary_on_known_values():
    df = pd.DataFrame({"uid": [8001] * 4 + [8002] * 4, "y": [1, 0, 1, 0] * 2, "p": [0.5] * 8})
    pu = M.per_user(df, "p")
    assert pu[8001][0] == pytest.approx(np.log(2))
    s = M.summarize(pu, [8001, 8002])
    assert s["n_users"] == 2 and s["LogLoss_u"] == pytest.approx(np.log(2), abs=1e-5)
