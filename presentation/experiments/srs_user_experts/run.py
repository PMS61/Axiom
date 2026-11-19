"""User-type experts (Phase 6, task 6.9): route each review to an expert GBDT trained for its user type.

User type is PAST-ONLY: the user's running recall rate so far (feature `u_rate`, computed from earlier reviews
only), bucketed into forgetful / mid / advanced by thresholds fixed on the TRAINING users' rows.
Models (all on the same cached features, users 2001-2300 for training, 8001-8100 for validation):
  global : one LightGBM on all training rows (the reference, same settings as sweep_a_l63_side)
  hard   : one expert per bucket trained from scratch on that bucket's rows
  warm   : one expert per bucket trained on that bucket's rows with init_score = global margin (fine-tune)
Selection honesty: the variant is chosen on users 8001-8050; reported on 8051-8100 and on all 100.
Writes srs-benchmark style jsonl lines per variant and per-row test predictions (parquet).
"""

# Give direct file execution the same imports as invocation from the repository.
if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from fsrs_benchmark.srsb import compare as C
from fsrs_benchmark.srsb import run_gbdt as R
from fsrs_benchmark.srsb import val_plus as V

PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=6, seed=0)


def load_users(uids, cache_dir, per_user_cap=None, test_only=False, seed=0):
    out = []
    for u in uids:
        p = Path(cache_dir) / f"u{u}.npz"
        if not p.is_file():
            continue
        with np.load(p) as z:
            X, y, test = z["X"], z["y"], z["test"].astype(bool)
            meta = pd.DataFrame({k: z["meta_" + k] for k in V.META_COLS})
        if test_only:
            X, y, meta = X[test], y[test], meta[test].reset_index(drop=True)
        elif per_user_cap and len(y) > per_user_cap:
            keep = np.sort(np.random.default_rng([seed, u]).choice(len(y), per_user_cap, replace=False))
            X, y = X[keep], y[keep]
        out.append(dict(uid=u, X=X, y=y, meta=meta))
    return out


def fit(X, y, init=None, hold=None, rounds=3000):
    dtr = lgb.Dataset(X, y, init_score=init, free_raw_data=False)
    if hold is None:
        return lgb.train(PARAMS, dtr, rounds)
    Xh, yh, ih = hold
    dva = lgb.Dataset(Xh, yh, init_score=ih, reference=dtr)
    return lgb.train(PARAMS, dtr, rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])


def split_hold(users, frac=0.1):
    hold = set(R.holdout_users([u["uid"] for u in users]))
    return [u for u in users if u["uid"] not in hold], [u for u in users if u["uid"] in hold]


def stack(users):
    return (np.vstack([u["X"] for u in users]), np.concatenate([u["y"] for u in users]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/anki10k")
    ap.add_argument("--train_lo", type=int, default=2001)
    ap.add_argument("--train_hi", type=int, default=2300)
    ap.add_argument("--per_user_cap", type=int, default=25000)
    ap.add_argument("--out", default="results/srsb_beat/type_experts")
    a = ap.parse_args(argv)
    cache = V.cache_dir_for(a.data)
    cols = list(V.ALL_COLS)
    ui = cols.index("u_rate")
    train = load_users(range(a.train_lo, a.train_hi + 1), cache, a.per_user_cap)
    val = load_users(range(8001, 8101), cache, test_only=True)
    fit_u, hold_u = split_hold(train)
    Xf, yf = stack(fit_u)
    Xh, yh = stack(hold_u)
    print(f"train users {len(train)} (fit {len(fit_u)}, hold {len(hold_u)}), fit rows {len(yf)}, val users {len(val)}", flush=True)

    g = fit(Xf, yf, hold=(Xh, yh, None))
    print("global best_iter", g.best_iteration, flush=True)
    mf, mh = g.predict(Xf, raw_score=True), g.predict(Xh, raw_score=True)

    rate = Xf[:, ui]
    qs = np.nanquantile(rate, [1 / 3, 2 / 3])
    print("bucket thresholds on u_rate (train rows):", np.round(qs, 4), flush=True)

    def bucket(X):
        r = X[:, ui]
        b = np.digitize(np.nan_to_num(r, nan=float(np.nanmedian(rate))), qs)
        return b

    bf, bh = bucket(Xf), bucket(Xh)
    hard, warm = {}, {}
    for b in range(3):
        sf, sh = bf == b, bh == b
        hard[b] = fit(Xf[sf], yf[sf], hold=(Xh[sh], yh[sh], None))
        warm[b] = fit(Xf[sf], yf[sf], init=mf[sf], hold=(Xh[sh], yh[sh], mh[sh]))
        print(f"bucket {b}: fit rows {int(sf.sum())}, hard iter {hard[b].best_iteration}, warm iter {warm[b].best_iteration}", flush=True)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = {k: open(out / f"{k}.jsonl", "w") for k in ("global", "hard", "warm")}
    rows = []
    for u in val:
        X, y = u["X"], u["y"].astype(np.float64)
        m = g.predict(X, raw_score=True)
        b = bucket(X)
        ph, pw = np.empty(len(y)), np.empty(len(y))
        for k in range(3):
            s = b == k
            if s.any():
                ph[s] = hard[k].predict(X[s])
                pw[s] = 1 / (1 + np.exp(-(m[s] + warm[k].predict(X[s], raw_score=True))))
        preds = {"global": 1 / (1 + np.exp(-m)), "hard": ph, "warm": pw}
        for k, p in preds.items():
            p = np.clip(p, 1e-6, 1 - 1e-6)
            files[k].write(json.dumps({"metrics": R.user_metrics(y, p, u["meta"]), "user": int(u["uid"]),
                                       "size": int(len(y))}) + "\n")
        rows.append(pd.DataFrame({"uid": u["uid"], "y": y, "bucket": b, **{"p_" + k: v for k, v in preds.items()}}))
    for f in files.values():
        f.close()
    pd.concat(rows).to_parquet(out / "preds.parquet", index=False)

    rw = C.load_rows(C.DEFAULT_RESULT_DIR / "RWKV-P.jsonl")
    res = {}
    for k in files:
        mine = C.load_rows(out / f"{k}.jsonl")
        for name, us in (("8001-8050", range(8001, 8051)), ("8051-8100", range(8051, 8101)), ("all", range(8001, 8101))):
            us = [x for x in us if x in mine and x in rw]
            w = np.array([rw[x]["size"] for x in us], float)
            ll = np.array([mine[x]["metrics"]["LogLoss"] for x in us])
            lr = np.array([rw[x]["metrics"]["LogLoss"] for x in us])
            auc = np.array([mine[x]["metrics"].get("AUC", np.nan) for x in us], float)
            res[(k, name)] = dict(n=len(us), ll_w=float(np.average(ll, weights=w)), gap_w=float(np.average(ll - lr, weights=w)),
                                  ll_u=float(ll.mean()), auc_u=float(np.nanmean(auc)))
    for (k, name), r in res.items():
        print(f"{k:7s} {name:10s} users {r['n']:3d}  LL_w {r['ll_w']:.5f}  gap_vs_RWKV-Instant_w {r['gap_w']:+.5f}  LL_u {r['ll_u']:.5f}  AUC_u {r['auc_u']:.4f}", flush=True)
    (out / "summary.json").write_text(json.dumps({f"{k}|{n}": v for (k, n), v in res.items()}, indent=1))


if __name__ == "__main__":
    main()
