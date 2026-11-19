"""MoE of GBDTs over a user-form x gap grid with a learned soft gate (Phase 6 task 6.13; validation only).

Experts: 9 LightGBM fine-tunes (init_score = global margin) for cells (u_rate tertile) x (gap: <3 d, 3-30 d,
>30 d). u_rate is the user's past-only running recall; gap is log_el_frac. Thresholds come from TRAINING rows.
Every expert predicts EVERY row (soft MoE), so the gate can mix all of them.
Gate: logistic regression on [global logit, 9 expert logits, context] with the global logit as offset,
2-fold cross-fit by user halves (8001-8050 <-> 8051-8100). Also reports hard routing (each row -> its own cell).
Online (Elo) layer is applied on top of the best variant for comparison with the frozen pipeline's style.
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from fsrs_benchmark.srsb import compare as C
from fsrs_benchmark.srsb import online_adapt as O
from fsrs_benchmark.srsb import val_plus as V
from fsrs_benchmark.srsb.type_experts import PARAMS, load_users, split_hold, stack

GAP_EDGES = np.log1p(np.array([3.0, 30.0]))


def sig(z):
    return 1 / (1 + np.exp(-z))


def main():
    out = Path("results/srsb_beat/moe_grid"); out.mkdir(parents=True, exist_ok=True)
    cache = V.cache_dir_for("data/anki10k")
    cols = list(V.ALL_COLS)
    ui, gi = cols.index("u_rate"), cols.index("log_el_frac")
    xi = {c: cols.index(c) for c in ("log_el_frac", "ewm50", "f_logitR", "rate_today", "card_n", "u_n")}
    train = load_users(range(2001, 2301), cache, per_user_cap=25000)
    fit_u, hold_u = split_hold(train)
    Xf, yf = stack(fit_u); Xh, yh = stack(hold_u)
    del train, fit_u, hold_u
    p = dict(PARAMS); p["num_threads"] = 6
    g = lgb.train(p, lgb.Dataset(Xf, yf), 10000, valid_sets=[lgb.Dataset(Xh, yh)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    print("global best_iter", g.best_iteration, flush=True)
    mf, mh = g.predict(Xf, raw_score=True), g.predict(Xh, raw_score=True)
    rq = np.nanquantile(Xf[:, ui], [1 / 3, 2 / 3])
    med = float(np.nanmedian(Xf[:, ui]))

    def cell(X):
        a = np.digitize(np.nan_to_num(X[:, ui], nan=med), rq)
        b = np.digitize(np.nan_to_num(X[:, gi], nan=0.0), GAP_EDGES)
        return a * 3 + b

    cf, ch = cell(Xf), cell(Xh)
    experts = {}
    for k in range(9):
        s, t = cf == k, ch == k
        if s.sum() < 5000 or t.sum() < 500:
            continue
        experts[k] = lgb.train(p, lgb.Dataset(Xf[s], yf[s], init_score=mf[s]), 2000,
                               valid_sets=[lgb.Dataset(Xh[t], yh[t], init_score=mh[t])],
                               callbacks=[lgb.early_stopping(50, verbose=False)])
        print(f"cell {k} (form {k // 3}, gap {k % 3}): fit rows {int(s.sum())}, iters {experts[k].best_iteration}", flush=True)
    del Xf, yf, Xh, yh
    rows = []
    for u in range(8001, 8101):
        f = Path(cache) / f"u{u}.npz"
        if not f.is_file():
            continue
        with np.load(f) as z:
            X, y, test = z["X"], z["y"].astype(np.float64), z["test"].astype(bool)
        m = g.predict(X, raw_score=True, num_iteration=g.best_iteration)
        E = np.column_stack([m + experts[k].predict(X, raw_score=True, num_iteration=experts[k].best_iteration)
                             if k in experts else m for k in range(9)])
        c = cell(X)
        hard = E[np.arange(len(m)), c]
        rows.append(dict(uid=u, X=X, y=y, test=test, m=m, E=E, hard=hard))
    print("val users", len(rows), flush=True)

    def ctx(r):
        X = r["X"]
        return np.column_stack([np.nan_to_num(X[:, xi["log_el_frac"]]) / 3, (np.nan_to_num(X[:, xi["ewm50"]], nan=.5) - .5) * 4,
                                np.log1p(np.nan_to_num(X[:, xi["card_n"]])) / 3, (np.nan_to_num(X[:, xi["rate_today"]], nan=.5) - .5) * 4])

    def feats(r):
        L = r["E"] - r["m"][:, None]
        Z = ctx(r)
        inter = np.column_stack([L * Z[:, [j]] for j in range(Z.shape[1])])
        return np.column_stack([L, Z, inter])

    halves = [set(range(8001, 8051)), set(range(8051, 8101))]
    soft = {}
    for tr, te in ((halves[0], halves[1]), (halves[1], halves[0])):
        R = [r for r in rows if r["uid"] in tr]
        F = np.vstack([feats(r)[r["test"]] for r in R]); Y = np.concatenate([r["y"][r["test"]] for r in R])
        OFF = np.concatenate([r["m"][r["test"]] for r in R])
        mu, sd = F.mean(0), F.std(0) + 1e-9
        Fz = (F - mu) / sd

        def loss(w):
            z = OFF + Fz @ w[1:] + w[0]
            pp = np.clip(sig(z), 1e-7, 1 - 1e-7)
            return -np.mean(Y * np.log(pp) + (1 - Y) * np.log(1 - pp)) + 1e-4 * np.sum(w[1:] ** 2)

        def grad(w):
            z = OFF + Fz @ w[1:] + w[0]
            gz = sig(z) - Y
            return np.concatenate([[gz.mean()], Fz.T @ gz / len(Y) + 2e-4 * w[1:]])

        w = minimize(loss, np.zeros(Fz.shape[1] + 1), jac=grad, method="L-BFGS-B", options=dict(maxiter=300)).x
        for r in rows:
            if r["uid"] in te:
                soft[r["uid"]] = r["m"] + w[0] + ((feats(r) - mu) / sd) @ w[1:]
    rw = C.load_rows(C.DEFAULT_RESULT_DIR / "RWKV-P.jsonl")
    recs = []
    for r in rows:
        u, y, t = r["uid"], r["y"], r["test"]
        ex = {c: r["X"][:, xi[c]].astype(np.float64) for c in ("log_el_frac", "ewm50", "f_logitR", "rate_today")}
        outs = {"global": r["m"], "hard_grid": r["hard"], "soft_gate": soft[u]}
        rec = dict(uid=u, n=int(t.sum()), rwkv=rw[u]["metrics"]["LogLoss"])
        for name, z in outs.items():
            for tag, pp in (("", sig(z)), ("+online", None)):
                if tag:
                    Xd = np.column_stack([z, np.ones_like(z), np.nan_to_num(ex["log_el_frac"]) / 3, (np.nan_to_num(ex["ewm50"], nan=.5) - .5) * 4,
                                          np.nan_to_num(ex["f_logitR"]) / 3, (np.nan_to_num(ex["rate_today"], nan=.5) - .5) * 4])
                    w0 = np.zeros(Xd.shape[1]); w0[0] = 1
                    pp = O.online_kernel(Xd, y, np.ones(len(y), bool), w0, 0.003, O.LAMBDA)
                pt = np.clip(pp[t], 1e-6, 1 - 1e-6); yt = y[t]
                rec[name + tag] = float(-np.mean(yt * np.log(pt) + (1 - yt) * np.log(1 - pt)))
        recs.append(rec)
    df = pd.DataFrame(recs); df.to_csv(out / "per_user.csv", index=False)
    w = df.n.to_numpy(float); rng = np.random.default_rng(0)

    def paired(a, b):
        d = (df[a] - df[b]).to_numpy(); bs = [np.average(d[k], weights=w[k]) for k in (rng.integers(0, len(d), len(d)) for _ in range(1000))]
        return f"{np.average(d, weights=w):+.5f} [{np.quantile(bs, .005):+.5f}, {np.quantile(bs, .995):+.5f}]"

    for c in [c for c in df.columns if c not in ("uid", "n")]:
        print(f"{c:18s} LL_w {np.average(df[c], weights=w):.5f}  LL_u {df[c].mean():.5f}", flush=True)
    for a, b in [("hard_grid", "global"), ("soft_gate", "global"), ("soft_gate+online", "global+online"),
                 ("hard_grid+online", "global+online"), ("soft_gate+online", "rwkv"), ("global+online", "rwkv")]:
        print(f"{a} minus {b}: {paired(a, b)}", flush=True)


if __name__ == "__main__":
    main()
