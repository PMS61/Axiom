"""Mixture-of-experts gate over the validation users (Phase 6, task 6.7; paper/docs/SRSB_BEAT.md).

Experts give a per-row probability for every test row of users 8001-8100:
  p_gbdt    the GBDT (sweep_a_l63_side model), from online_adapt.dump_expert_preds
  p_online  the GBDT + online layer (variant B, lr 0.003)
  p_fsrs    FSRS-6 default R (f_R)
  p_gru     GBDT + residual GRU, and p_tam (TAM), read from parquet files (uid, review_th, y, p)
The gate (a logistic regression with the online logit as offset, or a small LightGBM with the same init_score)
is trained on one half of the users and scored on the other half, both ways (2-fold cross-fit BY USER):
  fold 1: train on 8001-8050, score 8051-8100;  fold 2: train on 8051-8100, score 8001-8050.
Early stopping of the LightGBM gate uses an inner split of the TRAINING half's users only.
"""

# Give direct file execution the same imports as invocation from the repository.
if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import log_loss, roc_auc_score

from fsrs_benchmark.srsb import compare as C

P_CLIP = 1e-6
HALF_A = tuple(range(8001, 8051))
HALF_B = tuple(range(8051, 8101))
EXPERT_COLS = ("p_gbdt", "p_online", "p_fsrs", "p_gru", "p_tam")
CONTEXT = ("log_el_frac", "card_n", "u_n", "rate_today", "ewm50", "u_last5")
STATES = (0, 1, 2, 3, 4)


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), P_CLIP, 1 - P_CLIP)
    return np.log(p) - np.log1p(-p)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def fold_plan(users):
    """Two folds by user: each half is scored by the gate trained on the other half. Raises if a user lies in
    neither half, or a half is empty."""
    users = {int(u) for u in users}
    a, b = users & set(HALF_A), users & set(HALF_B)
    if not a or not b or users - (a | b):
        raise ValueError("need users from both halves 8001-8050 and 8051-8100, and no others")
    return [(sorted(b), sorted(a)), (sorted(a), sorted(b))]        # (train users, score users)


def load_preds(path, col, value_col="p"):
    """A parquet of per-row predictions (uid, review_th, p) renamed to `col`."""
    df = pd.read_parquet(path)
    need = {"uid", "review_th", value_col}
    if not need <= set(df.columns):
        raise ValueError(f"{path}: needs columns {sorted(need)}")
    out = df[["uid", "review_th", value_col]].rename(columns={value_col: col})
    if out.duplicated(["uid", "review_th"]).any():
        raise ValueError(f"{path}: duplicate (uid, review_th) rows")
    return out


def assemble(base, experts):
    """Attach each expert's probabilities to the base rows (uid, review_th, y, p_gbdt, p_online, context).
    Every expert must cover exactly the base rows. Returns the merged frame."""
    df = base.copy()
    for col, e in experts.items():
        m = df.merge(e, on=["uid", "review_th"], how="left", validate="one_to_one")
        if m[col].isna().any():
            raise ValueError(f"expert {col} does not cover {int(m[col].isna().sum())} base rows")
        df[col] = m[col].to_numpy()
    return df


def gate_matrix(df, experts):
    """Gate inputs: logits of the experts, plus context (with fixed fills for missing values) and state dummies."""
    cols, names = [], []
    for e in experts:
        cols.append(logit(df[e].to_numpy()))
        names.append("logit_" + e)
    lel = df["log_el_frac"].to_numpy(np.float64)
    cols.append(np.nan_to_num(lel, nan=0.0)); names.append("log_el_frac")
    cols.append(np.log1p(np.nan_to_num(df["card_n"].to_numpy(np.float64), nan=0.0))); names.append("log1p_card_n")
    cols.append(np.log1p(np.nan_to_num(df["u_n"].to_numpy(np.float64), nan=0.0))); names.append("log1p_u_n")
    rt = df["rate_today"].to_numpy(np.float64)
    cols.append(np.nan_to_num(rt, nan=0.5) - 0.5); names.append("rate_today")
    cols.append(np.isnan(rt).astype(float)); names.append("rate_today_missing")
    cols.append(np.nan_to_num(df["ewm50"].to_numpy(np.float64), nan=0.5) - 0.5); names.append("ewm50")
    cols.append(np.nan_to_num(df["u_last5"].to_numpy(np.float64), nan=0.5) - 0.5); names.append("u_last5")
    st = df["state"].to_numpy(np.int64)
    for s in STATES[:-1]:
        cols.append((st == s).astype(float)); names.append(f"state_{s}")
    return np.column_stack(cols), names


def fit_logreg_gate(X, y, offset, l2=1.0):
    """Logistic regression on X with a fixed offset (the online logit): p = sigmoid(offset + X w)."""
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd

    def f(w):
        z = offset + Z @ w[1:] + w[0]
        p = sigmoid(z)
        loss = -np.mean(y * np.log(np.clip(p, P_CLIP, 1)) + (1 - y) * np.log(np.clip(1 - p, P_CLIP, 1)))
        loss += l2 * np.sum(w[1:] ** 2) / len(y)
        g = (p - y) / len(y)
        grad = np.concatenate([[g.sum()], Z.T @ g + 2 * l2 * w[1:] / len(y)])
        return loss, grad

    res = minimize(f, np.zeros(Z.shape[1] + 1), jac=True, method="L-BFGS-B")
    return dict(kind="logreg", w=res.x, mu=mu, sd=sd)


def predict_logreg_gate(model, X, offset):
    Z = (X - model["mu"]) / model["sd"]
    return sigmoid(offset + Z @ model["w"][1:] + model["w"][0])


def fit_lgb_gate(X, y, offset, users, seed=0, max_rounds=500, patience=30):
    """Small LightGBM gate (15 leaves, lr 0.05) with init_score = offset; early stopping on an inner 20% split
    of the training users (never the scored users)."""
    import lightgbm as lgb

    users = np.asarray(users)
    uu = np.unique(users)
    rng = np.random.default_rng(seed)
    inner = set(rng.choice(uu, size=max(1, int(round(0.2 * len(uu)))), replace=False).tolist())
    va = np.array([u in inner for u in users])
    params = dict(objective="binary", metric="binary_logloss", num_leaves=15, learning_rate=0.05,
                  min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  lambda_l2=1.0, num_threads=2, seed=seed, deterministic=True, force_row_wise=True, verbose=-1)
    if va.any() and (~va).any():
        dtr = lgb.Dataset(X[~va], y[~va], init_score=offset[~va])
        dva = lgb.Dataset(X[va], y[va], init_score=offset[va], reference=dtr)
        booster = lgb.train(params, dtr, max_rounds, valid_sets=[dva],
                            callbacks=[lgb.early_stopping(patience, verbose=False)])
        it = booster.best_iteration or booster.current_iteration()
    else:
        dtr = lgb.Dataset(X, y, init_score=offset)
        booster = lgb.train(params, dtr, 200)
        it = booster.current_iteration()
    return dict(kind="lgb", booster=booster, it=int(it))


def predict_lgb_gate(model, X, offset):
    return sigmoid(model["booster"].predict(X, raw_score=True, num_iteration=model["it"]) + offset)


def crossfit(df, experts, kind="logreg", offset_col="p_online", seed=0):
    """Out-of-fold gate probabilities for every row: each user's rows are scored by the gate trained on the
    other half. Returns (p_gate per row, per-fold info)."""
    X, names = gate_matrix(df, experts)
    y = df["y"].to_numpy(np.float64)
    off = logit(df[offset_col].to_numpy())
    users = df["uid"].to_numpy()
    p = np.full(len(df), np.nan)
    info = []
    for train_users, score_users in fold_plan(np.unique(users)):
        tr = np.isin(users, train_users)
        te = np.isin(users, score_users)
        assert not (set(np.unique(users[tr])) & set(score_users)), "a scored user is in the training fold"
        if kind == "logreg":
            m = fit_logreg_gate(X[tr], y[tr], off[tr])
            p[te] = predict_logreg_gate(m, X[te], off[te])
        elif kind == "lgb":
            m = fit_lgb_gate(X[tr], y[tr], off[tr], users[tr], seed=seed)
            p[te] = predict_lgb_gate(m, X[te], off[te])
            info.append(dict(scored=[int(u) for u in score_users], best_iter=m["it"]))
            continue
        else:
            raise ValueError(kind)
        info.append(dict(scored=[int(u) for u in score_users], features=names))
    return np.clip(p, P_CLIP, 1 - P_CLIP), info


def per_user(df, col):
    """{uid: (LogLoss, AUC or None, rows)} for the probability column `col`."""
    out = {}
    for u, g in df.groupby("uid"):
        y = g["y"].to_numpy(np.float64)
        p = np.clip(g[col].to_numpy(np.float64), P_CLIP, 1 - P_CLIP)
        try:
            auc = float(roc_auc_score(y, p))
        except ValueError:
            auc = None
        out[int(u)] = (float(log_loss(y, p, labels=[0, 1])), auc, int(len(y)))
    return out


def summarize(pu, users):
    us = [u for u in users if u in pu]
    ll = [pu[u][0] for u in us]
    sz = [pu[u][2] for u in us]
    aucs = [pu[u][1] for u in us if pu[u][1] is not None]
    return dict(n_users=len(us), LogLoss_u=round(float(np.mean(ll)), 5),
                LogLoss_w=round(C.weighted_mean(ll, sz), 5),
                AUC_u=round(float(np.mean(aucs)), 4) if aucs else None)


def compare_to_refs(pu, users, result_dir=C.DEFAULT_RESULT_DIR, ci=0.99, n_boot=1000, seed=0):
    """Paired size-weighted bootstrap of our LogLoss minus each reference, on users present with the same size."""
    result_dir = Path(result_dir)
    refs = {"RWKV-Instant": "RWKV-P", "RWKV-Curve": "RWKV", "FSRS-6": "FSRS-6-short"}
    out = {}
    for name, f in refs.items():
        path = result_dir / f"{f}.jsonl"
        if not path.exists():
            continue
        rows = C.load_rows(path)
        us = [u for u in users if u in pu and u in rows and rows[u]["size"] == pu[u][2]]
        if not us:
            continue
        diff = [pu[u][0] - rows[u]["metrics"]["LogLoss"] for u in us]
        sizes = [pu[u][2] for u in us]
        out[name] = C.paired_bootstrap(diff, sizes, n_boot=n_boot, ci=ci, seed=seed)
        out[name]["n_users"] = len(us)
    return out


def compare_pair(pu, other, users, ci=0.99, n_boot=1000, seed=0):
    """Paired bootstrap of our LogLoss minus another per-user table (e.g. the best single expert)."""
    us = [u for u in users if u in pu and u in other]
    diff = [pu[u][0] - other[u][0] for u in us]
    sizes = [pu[u][2] for u in us]
    r = C.paired_bootstrap(diff, sizes, n_boot=n_boot, ci=ci, seed=seed)
    r["n_users"] = len(us)
    return r


def report(df, gate_col, experts, result_dir=C.DEFAULT_RESULT_DIR):
    """Per half and combined: the gate, every expert, the best single expert, and paired CIs."""
    pu_gate = per_user(df.assign(_g=df[gate_col]), "_g")
    singles = {e: per_user(df, e) for e in experts}
    best = min(singles, key=lambda e: summarize(singles[e], list(pu_gate))["LogLoss_w"])
    out = {"best_single": best, "halves": {}}
    for label, users in (("8001-8050", HALF_A), ("8051-8100", HALF_B), ("8001-8100", HALF_A + HALF_B)):
        out["halves"][label] = dict(
            gate=summarize(pu_gate, users),
            **{e: summarize(singles[e], users) for e in experts},
            vs_best_single=compare_pair(pu_gate, singles[best], users),
            vs_refs=compare_to_refs(pu_gate, users, result_dir=result_dir),
        )
    return out


def run(gbdt_online, experts_paths, out, kind="logreg", result_dir=C.DEFAULT_RESULT_DIR):
    """gbdt_online: the parquet written by online_adapt --dump_preds (base rows + p_gbdt, p_online, context).
    experts_paths: {column: parquet path} for optional experts (p_gru, p_tam; (uid, review_th, p) files)."""
    base = pd.read_parquet(gbdt_online)
    experts = {}
    for col, path in experts_paths.items():
        experts[col] = load_preds(path, col)
    df = assemble(base, experts)
    names = ["p_gbdt", "p_online", "p_fsrs"] + list(experts)
    df["p_fsrs"] = np.clip(df["f_R"].to_numpy(np.float64), P_CLIP, 1 - P_CLIP)   # FSRS-6 default R
    p, info = crossfit(df, names, kind=kind)
    df["p_gate"] = p
    rep = report(df, "p_gate", names, result_dir=result_dir)
    rep["gate"] = kind
    rep["folds"] = info
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(str(out) + ".report.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    df[["uid", "review_th", "y", "p_gate"] + names].to_parquet(str(out) + ".parquet", index=False)
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description="MoE gate over the validation experts (6.7).")
    ap.add_argument("--base", required=True, help="parquet from online_adapt --dump_preds")
    ap.add_argument("--gru", default=None, help="parquet (uid, review_th, p) of GBDT+residual GRU")
    ap.add_argument("--tam", default=None, help="parquet (uid, review_th, p) of TAM (a file or merged dir)")
    ap.add_argument("--gate", default="logreg", choices=["logreg", "lgb"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    paths = {}
    if args.gru:
        paths["p_gru"] = args.gru
    if args.tam:
        paths["p_tam"] = args.tam
    rep = run(args.base, paths, args.out, kind=args.gate)
    print(json.dumps({k: v for k, v in rep.items() if k != "folds"}, indent=1, default=str))
    return rep


if __name__ == "__main__":
    main(sys.argv[1:])
