"""Elo-bucketed GBDT experts, hard and soft thresholding (Phase 6 task 6.14; validation only).

Elo per user stream (past-only, online): p = sigmoid(theta - d_card); after each labelled non-first review
theta += K_U (y - p) and d_card -= K_C(n) (y - p), K_C(n) = 0.4 / (1 + 0.05 n). The value used for row t is theta
BEFORE row t's answer. Buckets: tertiles of theta on TRAINING scored rows.
Buckets: 3 or 5 equal-count Elo ranges (quantiles of training rows). Experts trained from scratch or fine-tuned
from the global margin (init_score). Hard: each expert trains on its own range; each row uses its range's expert.
Soft: each expert's training range is widened by 25% of the neighbouring bucket width into ADJACENT buckets only;
rows inside an overlap zone are predicted by a linear blend of the two adjacent experts.
Online (Elo) layer (frozen settings) is applied on top of every variant for comparison. Users: train 2001-2300,
validation 8001-8100 (no selection here; fixed settings).
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from numba import njit

from fsrs_benchmark.srsb import compare as C
from fsrs_benchmark.srsb import online_adapt as O
from fsrs_benchmark.srsb import run_gbdt as R
from fsrs_benchmark.srsb import val_plus as V
from fsrs_benchmark.srsb.type_experts import PARAMS

K_U = 0.05
import os
FSRS_MODE = os.environ.get('ELO_FSRS', 'all')     # all | none | mixed (drop FSRS only in the upper half of buckets)
FSRS_COLS = ['f_S', 'f_D', 'f_R', 'f_logitR', 'f_el_over_S', 'f_R_secs', 'resid_mean', 'resid_ewm100', 'resid_gap_mean']
TRAIN_USERS = range(2001, 2011) if os.environ.get('ELO_QUICK') else range(2001, 2301)
VAL_USERS = range(8001, 8004) if os.environ.get('ELO_QUICK') else range(8001, 8101)


@njit(cache=True)
def elo_kernel(card_idx, y, first, n_cards):
    n = len(y)
    theta = 0.0
    d = np.zeros(n_cards)
    cnt = np.zeros(n_cards)
    out = np.empty(n)
    for i in range(n):
        out[i] = theta
        if first[i]:
            continue
        c = card_idx[i]
        p = 1.0 / (1.0 + np.exp(-(theta - d[c])))
        r = y[i] - p
        theta += K_U * r
        d[c] -= 0.4 / (1.0 + 0.05 * cnt[c]) * r
        cnt[c] += 1
    return out


def user_rows(uid, cache, loader, cfg, data_dir, cap=None, test_only=False, seed=0):
    f = Path(cache) / f"u{uid}.npz"
    if not f.is_file():
        return None
    with np.load(f) as z:
        X, y, test = z["X"], z["y"], z["test"].astype(bool)
    ds = loader.load_user_data(uid)
    raw = pd.read_parquet(Path(data_dir) / "revlogs" / f"user_id={uid}")
    st = R.review_stream(raw, cfg.max_seq_len * 2)
    cidx = pd.factorize(st["card_id"])[0]
    elo = pd.Series(elo_kernel(cidx.astype(np.int64), st["y"].to_numpy(np.float64), st["is_first"].to_numpy(),
                               int(cidx.max()) + 1), index=st.index)
    e = elo.reindex(ds["review_th"].to_numpy()).to_numpy()
    if test_only:
        return dict(uid=uid, X=X, y=y, test=test, elo=e)
    if cap and len(y) > cap:
        keep = np.sort(np.random.default_rng([seed, uid]).choice(len(y), cap, replace=False))
        X, y, e = X[keep], y[keep], e[keep]
    return dict(uid=uid, X=X, y=y, elo=e)


def sig(z):
    return 1 / (1 + np.exp(-z))


def main():
    out = Path("results/srsb_beat/elo_experts" + ("" if FSRS_MODE == "all" else "_fsrs_" + FSRS_MODE)); out.mkdir(parents=True, exist_ok=True)
    data = "data/anki10k"
    cache = V.cache_dir_for(data)
    V._init(data, V.ALL_COLS, None)
    loader, cfg = V._W["loader"], V._W["cfg"]
    cols = list(V.ALL_COLS)
    xi = {c: cols.index(c) for c in ("log_el_frac", "ewm50", "f_logitR", "rate_today")}
    train = [r for r in (user_rows(u, cache, loader, cfg, data, cap=25000) for u in TRAIN_USERS) if r]
    hold_ids = set(R.holdout_users([r["uid"] for r in train]))
    fit = [r for r in train if r["uid"] not in hold_ids]; hold = [r for r in train if r["uid"] in hold_ids]
    S = lambda rs, k: np.concatenate([r[k] for r in rs]) if k != "X" else np.vstack([r[k] for r in rs])
    Xf, yf, ef = S(fit, "X"), S(fit, "y"), S(fit, "elo"); Xh, yh, eh = S(hold, "X"), S(hold, "y"), S(hold, "elo")
    del train, fit, hold
    print(f"fit rows {len(yf)}, hold rows {len(yh)}; Elo on train rows: mean {np.nanmean(ef):.3f} sd {np.nanstd(ef):.3f}", flush=True)
    p = dict(PARAMS); p["num_threads"] = 6
    g = lgb.train(p, lgb.Dataset(Xf, yf), 10000, valid_sets=[lgb.Dataset(Xh, yh)], callbacks=[lgb.early_stopping(50, verbose=False)])
    mf, mh = g.predict(Xf, raw_score=True), g.predict(Xh, raw_score=True)
    MARGIN = 0.25                     # overlap = 25% of a bucket's width, adjacent buckets only
    med = float(np.nanmedian(ef))
    variants = {}                     # name -> (edges, experts per scheme)
    for nb in (3, 5):
        edges = np.nanquantile(ef, np.linspace(0, 1, nb + 1))
        edges[0], edges[-1] = -np.inf, np.inf
        inner = np.nanquantile(ef, np.linspace(0, 1, nb + 1))
        width = np.diff(inner)
        lo = inner[:-1] - MARGIN * np.r_[width[0], width[:-1]]
        hi = inner[1:] + MARGIN * np.r_[width[1:], width[-1]]
        lo[0], hi[-1] = -np.inf, np.inf
        print(f"{nb} buckets: edges {np.round(inner[1:-1], 3)}", flush=True)
        for scheme in ("scratch", "warm"):
            for mode in ("hard", "soft"):
                ex = {}
                for k in range(nb):
                    a_lo, a_hi = (edges[k], edges[k + 1]) if mode == "hard" else (lo[k], hi[k])
                    a = (np.nan_to_num(ef, nan=med) >= a_lo) & (np.nan_to_num(ef, nan=med) < a_hi)
                    b = (np.nan_to_num(eh, nan=med) >= a_lo) & (np.nan_to_num(eh, nan=med) < a_hi)
                    Xa = Xf[a]
                    Xb = Xh[b]
                    if FSRS_MODE == "none" or (FSRS_MODE == "mixed" and k >= nb // 2 + nb % 2):
                        fidx = [cols.index(c) for c in FSRS_COLS]
                        Xa = Xa.copy(); Xa[:, fidx] = np.nan
                        Xb = Xb.copy(); Xb[:, fidx] = np.nan
                    init_f = mf[a] if scheme == "warm" else None
                    init_h = mh[b] if scheme == "warm" else None
                    if b.sum() >= 200:
                        ex[k] = lgb.train(p, lgb.Dataset(Xa, yf[a], init_score=init_f), 3000,
                                          valid_sets=[lgb.Dataset(Xb, yh[b], init_score=init_h)],
                                          callbacks=[lgb.early_stopping(50, verbose=False)])
                    else:             # too few holdout rows in this range (tiny runs only): fixed rounds
                        ex[k] = lgb.train(p, lgb.Dataset(Xa, yf[a], init_score=init_f), 200)
                variants[f"elo{nb}_{scheme}_{mode}"] = (nb, inner, lo, hi, scheme, mode, ex)
                print(f"  {scheme}/{mode}: iters {[ex[k].best_iteration for k in range(nb)]}", flush=True)

    def predict_variant(v, X, e, m):
        nb, inner, lo, hi, scheme, mode, ex = v
        e = np.nan_to_num(e, nan=med)
        def Xk(k):
            if FSRS_MODE == "none" or (FSRS_MODE == "mixed" and k >= nb // 2 + nb % 2):
                Xc = X.copy(); Xc[:, [cols.index(c) for c in FSRS_COLS]] = np.nan; return Xc
            return X
        Z = np.column_stack([ex[k].predict(Xk(k), raw_score=True, num_iteration=ex[k].best_iteration) + (m if scheme == "warm" else 0)
                             for k in range(nb)])
        k0 = np.clip(np.searchsorted(inner[1:-1], e, side="right"), 0, nb - 1)
        z = Z[np.arange(len(e)), k0]
        if mode == "soft":            # inside an overlap zone: linear blend with the adjacent expert
            for k in range(nb - 1):
                b = inner[k + 1]; ml = b - lo[k + 1]; mr = hi[k] - b
                zone = (e > b - ml) & (e < b + mr)
                if zone.any():
                    t = np.clip((e[zone] - (b - ml)) / (ml + mr), 0, 1)
                    z[zone] = (1 - t) * Z[zone, k] + t * Z[zone, k + 1]
        return z

    rw = C.load_rows(C.DEFAULT_RESULT_DIR / "RWKV-P.jsonl")
    recs = []
    for u in VAL_USERS:
        r = user_rows(u, cache, loader, cfg, data, test_only=True)
        if r is None:
            continue
        X, y, t, e = r["X"], r["y"].astype(np.float64), r["test"], r["elo"]
        m = g.predict(X, raw_score=True, num_iteration=g.best_iteration)
        outs = {"global": m}
        for name, v in variants.items():
            outs[name] = predict_variant(v, X, e, m)
        ex = {c: X[:, j].astype(np.float64) for c, j in xi.items()}
        rec = dict(uid=u, n=int(t.sum()), rwkv=rw[u]["metrics"]["LogLoss"])
        for name, z in outs.items():
            Xd = np.column_stack([z, np.ones_like(z), np.nan_to_num(ex["log_el_frac"]) / 3, (np.nan_to_num(ex["ewm50"], nan=.5) - .5) * 4,
                                  np.nan_to_num(ex["f_logitR"]) / 3, (np.nan_to_num(ex["rate_today"], nan=.5) - .5) * 4])
            w0 = np.zeros(Xd.shape[1]); w0[0] = 1
            for tag, pp in (("", sig(z)), ("+online", O.online_kernel(Xd, y, np.ones(len(y), bool), w0, 0.003, O.LAMBDA))):
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
    for name in variants:
        print(f"{name} minus global: {paired(name, 'global')} | +online minus global+online: {paired(name + '+online', 'global+online')}", flush=True)
    print(f"global+online minus RWKV-Instant: {paired('global+online', 'rwkv')}", flush=True)


if __name__ == "__main__":
    main()
