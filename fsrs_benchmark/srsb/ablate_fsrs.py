"""Ablation: GBDT + online layer WITH vs WITHOUT FSRS-derived features (validation only; Phase 6 task 6.12).

Both GBDTs are trained identically from the cached features (users 2001-2300, per-user row cap, early stopping on
their holdout users, the frozen sweep_a_l63_side LightGBM settings). The no-FSRS model drops every FSRS-derived
column: f_S, f_D, f_R, f_logitR, f_el_over_S, f_R_secs and the residual-vs-FSRS columns resid_*.
The same online layer (online_adapt's kernel and update rule, lr 0.003, L2 pull 1e-3) is applied to both; here it
updates on the SCORED rows of each user in order (the cache holds scored rows only), so absolute numbers differ a
little from the frozen pipeline (which also updates on unscored stream rows) but the comparison is like-for-like.
Variant "B" for the FSRS model uses (log_el_frac, ewm50, f_logitR, rate_today); variant "B-noFSRS" drops f_logitR.
Users 8001-8100 are scored; nothing is selected here (fixed lr 0.003 as frozen).
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from fsrs_benchmark.srsb import compare as C
from fsrs_benchmark.srsb import online_adapt as O
from fsrs_benchmark.srsb import val_plus as V
from fsrs_benchmark.srsb.type_experts import load_users, split_hold, stack

FSRS_COLS = ["f_S", "f_D", "f_R", "f_logitR", "f_el_over_S", "f_R_secs", "resid_mean", "resid_ewm100", "resid_gap_mean"]
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=4, seed=0)


def train(cols_idx, fit_u, hold_u):
    Xf, yf = stack(fit_u)
    Xh, yh = stack(hold_u)
    d = lgb.Dataset(Xf[:, cols_idx], yf, free_raw_data=False)
    h = lgb.Dataset(Xh[:, cols_idx], yh, reference=d)
    return lgb.train(PARAMS, d, 10000, valid_sets=[h], callbacks=[lgb.early_stopping(50, verbose=False)])


def online(margin, extra, y, use_fsrs, lr=O.FINAL_LR if hasattr(O, "FINAL_LR") else 0.003):
    cols = [margin, np.ones_like(margin),
            np.nan_to_num(extra["log_el_frac"]) / 3.0,
            (np.nan_to_num(extra["ewm50"], nan=0.5) - 0.5) * 4.0]
    if use_fsrs:
        cols.append(np.nan_to_num(extra["f_logitR"]) / 3.0)
    cols.append((np.nan_to_num(extra["rate_today"], nan=0.5) - 0.5) * 4.0)
    X = np.column_stack(cols).astype(np.float64)
    w0 = np.zeros(X.shape[1]); w0[0] = 1.0
    return O.online_kernel(X, y.astype(np.float64), np.ones(len(y), bool), w0, lr, O.LAMBDA)


def main():
    out = Path("results/srsb_beat/ablate_fsrs"); out.mkdir(parents=True, exist_ok=True)
    cache = V.cache_dir_for("data/anki10k")
    allc = list(V.ALL_COLS)
    idx_all = list(range(len(allc)))
    idx_nof = [i for i, c in enumerate(allc) if c not in FSRS_COLS]
    print("features: with FSRS", len(idx_all), "| without", len(idx_nof), flush=True)
    train_u = load_users(range(2001, 2301), cache, per_user_cap=25000)
    fit_u, hold_u = split_hold(train_u)
    models = {}
    for name, idx in (("with_fsrs", idx_all), ("no_fsrs", idx_nof)):
        models[name] = (train(idx, fit_u, hold_u), idx)
        print(name, "best_iter", models[name][0].best_iteration, flush=True)
    del train_u, fit_u, hold_u
    xi = {c: allc.index(c) for c in ("log_el_frac", "ewm50", "f_logitR", "rate_today")}
    rw = C.load_rows(C.DEFAULT_RESULT_DIR / "RWKV-P.jsonl")
    rows = []
    for u in range(8001, 8101):
        p = Path(cache) / f"u{u}.npz"
        if not p.is_file():
            continue
        with np.load(p) as z:
            X, y, test = z["X"], z["y"].astype(np.float64), z["test"].astype(bool)
        extra = {c: X[:, j].astype(np.float64) for c, j in xi.items()}
        rec = dict(uid=u, n=int(test.sum()), rwkv=rw[u]["metrics"]["LogLoss"] if u in rw else np.nan)
        yt = y[test]
        for name, (b, idx) in models.items():
            m = b.predict(X[:, idx], raw_score=True, num_iteration=b.best_iteration)
            pg = np.clip(1 / (1 + np.exp(-m)), 1e-6, 1 - 1e-6)
            po = np.clip(online(m, extra, y, use_fsrs=(name == "with_fsrs")), 1e-6, 1 - 1e-6)
            for tag, pp in (("gbdt", pg), ("online", po)):
                pt = pp[test]
                rec[f"{name}_{tag}"] = float(-np.mean(yt * np.log(pt) + (1 - yt) * np.log(1 - pt)))
        rows.append(rec)
    df = pd.DataFrame(rows).dropna()
    df.to_csv(out / "per_user.csv", index=False)
    w = df.n.to_numpy(float)
    rng = np.random.default_rng(0)
    res = {}
    for c in ["with_fsrs_gbdt", "with_fsrs_online", "no_fsrs_gbdt", "no_fsrs_online", "rwkv"]:
        res[c] = dict(LL_w=float(np.average(df[c], weights=w)), LL_u=float(df[c].mean()))
    def paired(a, b):
        d = df[a].to_numpy() - df[b].to_numpy(); bs = []
        for _ in range(1000):
            k = rng.integers(0, len(df), len(df)); bs.append(np.average(d[k], weights=w[k]))
        return dict(mean=float(np.average(d, weights=w)), lo=float(np.quantile(bs, .005)), hi=float(np.quantile(bs, .995)))
    res["noFSRS_minus_withFSRS_gbdt"] = paired("no_fsrs_gbdt", "with_fsrs_gbdt")
    res["noFSRS_minus_withFSRS_online"] = paired("no_fsrs_online", "with_fsrs_online")
    res["with_online_minus_RWKV"] = paired("with_fsrs_online", "rwkv")
    res["no_online_minus_RWKV"] = paired("no_fsrs_online", "rwkv")
    (out / "summary.json").write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print(k, {kk: round(vv, 5) for kk, vv in v.items()}, flush=True)


if __name__ == "__main__":
    main()
