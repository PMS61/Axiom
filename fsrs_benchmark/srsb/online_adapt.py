"""Per-user online logistic layer on top of a saved GBDT (Phase 6, task 6.5a; paper/docs/SRSB_BEAT.md).

For one user, rows are visited in review_th order (the whole stream, same-day reviews included). For each
labelled row t (every non-first review):
  p_t = sigmoid(w . x_t) is predicted BEFORE row t's label is used, then w takes one SGD step on the log loss of
  row t, with a pull toward the initial w0 (L2 strength `lam`).
x_t is the GBDT raw margin m_t with a bias (variant A), or those plus four strong features (variant B), each
scaled by a fixed constant chosen before looking at data. w0 = [1, 0, ...], so with lr = 0 the output is the GBDT.

Leak rule: p_t depends only on rows before t and on row t's features (known before the answer: all of them are
past-only). Labels of rows >= t never enter p_t (tests/test_online_adapt.py).

Selection honesty: variant and lr are chosen on users 8001-8050 only, then reported on 8051-8100 and all 100.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from fsrs_benchmark.models import gbdt as G
from fsrs_benchmark.srsb import compare as C
from fsrs_benchmark.srsb import run_gbdt as R
from fsrs_benchmark.srsb import val_plus as V
from fsrs_benchmark.srsb.features_plus import plus_features

P_CLIP = 1e-6
LAMBDA = 1e-3                                   # L2 pull toward w0 per step (fixed, not tuned)
LRS = (0.003, 0.01, 0.03)
SEL_USERS = range(8001, 8051)                   # selection
REP_USERS = range(8051, 8101)                   # report
ALL_USERS = range(8001, 8101)
EXTRA = ("log_el_frac", "ewm50", "f_logitR", "rate_today")
CTX = ("log_el_frac", "card_n", "u_n", "state", "rate_today", "ewm50", "u_last5", "f_R")   # gate context (6.7)
VARIANTS = {"A": (), "B": EXTRA}


@njit(cache=True)
def online_kernel(X, y, labelled, w0, lr, lam):
    """Rows in order. X: (n, d) with X[:, 0] = margin and X[:, 1] = 1. Returns p (NaN on unlabelled rows)."""
    n, d = X.shape
    w = w0.copy()
    p = np.full(n, np.nan)
    for i in range(n):
        if not labelled[i]:
            continue
        z = 0.0
        for j in range(d):
            z += w[j] * X[i, j]
        pi = 1.0 / (1.0 + np.exp(-z))
        p[i] = pi
        g = pi - y[i]
        for j in range(d):
            w[j] -= lr * (g * X[i, j] + lam * (w[j] - w0[j]))
    return p


def design(margin, extra_cols, variant):
    """Design matrix for a variant: [margin, 1, scaled extras...]. Extras use fixed constants (no data stats)."""
    cols = [margin, np.ones_like(margin)]
    if variant == "B":
        le = np.nan_to_num(extra_cols["log_el_frac"]) / 3.0
        e50 = (np.nan_to_num(extra_cols["ewm50"], nan=0.5) - 0.5) * 4.0
        fl = np.nan_to_num(extra_cols["f_logitR"]) / 3.0
        rt = (np.nan_to_num(extra_cols["rate_today"], nan=0.5) - 0.5) * 4.0
        cols += [le, e50, fl, rt]
    return np.column_stack(cols).astype(np.float64)


def w0_for(variant):
    return np.array([1.0, 0.0] + [0.0] * (len(EXTRA) if variant == "B" else 0))


def user_arrays(uid, booster, names, best_iter, loader, cfg, data_dir: Path):
    """Margins and features for ALL stream rows of one user, plus labels, labelled mask and the eval positions."""
    ds = loader.load_user_data(uid)
    raw = pd.read_parquet(data_dir / "revlogs" / f"user_id={uid}")
    st = R.review_stream(raw, cfg.max_seq_len * 2)
    cards = V._read_opt(data_dir / "cards" / f"user_id={uid}")
    decks = V._read_opt(data_dir / "decks" / f"user_id={uid}")
    base = G.base_features(st, span=20, user_group="row")[R.FEATURE_COLS]
    feats = pd.concat([base, plus_features(st, cards, decks)], axis=1)
    X = feats[list(names)].to_numpy(np.float32)
    margin = booster.predict(X, raw_score=True, num_iteration=best_iter).astype(np.float64)
    first = ~st["card_id"].duplicated().to_numpy()
    keys = ds["review_th"].to_numpy()
    test = R.test_row_mask(len(ds), cfg.n_splits)
    pos = st.index.get_indexer(keys)
    if (pos < 0).any():
        raise RuntimeError(f"user {uid}: scored rows not in the stream")
    extra = {c: feats[c].to_numpy(np.float64) for c in EXTRA}
    ctx = {c: feats[c].to_numpy(np.float64) for c in CTX}
    return dict(uid=uid, margin=margin, extra=extra, ctx=ctx, y=(st["rating"].to_numpy() > 1).astype(np.float64),
                labelled=~first, pos=pos, test=test, y_eval=ds["y"].to_numpy(np.float64), keys=keys)


def eval_user(arr, variant, lr):
    """LogLoss of the layer on the user's test rows, and the test-row count."""
    from sklearn.metrics import log_loss
    Xd = design(arr["margin"], arr["extra"], variant)
    p = online_kernel(Xd, arr["y"], arr["labelled"], w0_for(variant), lr, LAMBDA)
    pe = np.clip(p[arr["pos"]][arr["test"]], P_CLIP, 1 - P_CLIP)
    ye = arr["y_eval"][arr["test"]]
    return float(log_loss(ye, pe, labels=[0, 1])), int(arr["test"].sum()), pe


def gbdt_check(arr):
    """GBDT probabilities on the test rows (sigmoid of the margin), the same as the saved GBDT output."""
    p = 1.0 / (1.0 + np.exp(-arr["margin"][arr["pos"]][arr["test"]]))
    return np.clip(p, P_CLIP, 1 - P_CLIP)


def run(model_path, run_info_path, data_dir, gbdt_jsonl, out, result_dir=C.DEFAULT_RESULT_DIR, users=ALL_USERS,
        dump_preds=None):
    import lightgbm as lgb
    from sklearn.metrics import log_loss

    data_dir = Path(data_dir)
    info = json.loads(Path(run_info_path).read_text(encoding="utf-8"))
    names, best_iter = info["features"], info["best_iter"]
    booster = lgb.Booster(model_file=str(model_path))
    V._init(str(data_dir), names)
    loader, cfg = V._W["loader"], V._W["cfg"]
    gb = C.load_rows(gbdt_jsonl)
    arrays = {}
    t0 = time.time()
    users = list(users)
    for i, u in enumerate(users, 1):
        arrays[u] = user_arrays(u, booster, names, best_iter, loader, cfg, data_dir)
        print(f"featurized user {u} ({i}/{len(users)}), elapsed {time.time() - t0:.0f}s", flush=True)
    # sanity: the GBDT sigmoid reproduces the saved GBDT log loss on every user
    for u in users:
        g = gbdt_check(arrays[u])
        ref = log_loss(arrays[u]["y_eval"][arrays[u]["test"]], g, labels=[0, 1])
        if abs(ref - gb[u]["metrics"]["LogLoss"]) > 1e-4:
            raise RuntimeError(f"user {u}: GBDT check failed ({ref} vs {gb[u]['metrics']['LogLoss']})")

    grid = {}
    pe_all = {}
    for variant in VARIANTS:
        for lr in LRS:
            name = f"{variant}_lr{lr}"
            res = {u: eval_user(arrays[u], variant, lr) for u in users}
            grid[name] = {u: res[u][:2] for u in users}
            pe_all[name] = {u: res[u][2] for u in users}
    rows = {"GBDT": {u: (gb[u]["metrics"]["LogLoss"], gb[u]["size"]) for u in users}}
    rows.update(grid)
    sel = {k: _wmean(v, SEL_USERS) for k, v in grid.items()}
    chosen = min(sel, key=sel.get)
    report = dict(selection_users="8001-8050", chosen=chosen, selection_wLL=sel,
                  lambda_=LAMBDA, lrs=list(LRS), extra_features=list(EXTRA))

    pub = C.load_rows(Path(result_dir) / "RWKV-P.jsonl")
    for label, us in (("8051-8100", REP_USERS), ("8001-8100", ALL_USERS)):
        us = [u for u in us if u in users]
        report[label] = {}
        for ref_name, ref in (("GBDT", rows["GBDT"]), ("online", grid[chosen])):
            report[label][ref_name] = dict(unweighted=round(float(np.mean([ref[u][0] for u in us])), 5),
                                           weighted=round(C.weighted_mean([ref[u][0] for u in us],
                                                                          [ref[u][1] for u in us]), 5))
        common = [u for u in us if u in pub and pub[u]["size"] == grid[chosen][u][1]]
        sizes = [grid[chosen][u][1] for u in common]
        for ref_name, diff_base in (("GBDT", rows["GBDT"]), ("RWKV-Instant", None)):
            if diff_base is None:
                diff = [grid[chosen][u][0] - pub[u]["metrics"]["LogLoss"] for u in common]
            else:
                diff = [grid[chosen][u][0] - diff_base[u][0] for u in common]
            report[label][f"diff_vs_{ref_name}"] = C.paired_bootstrap(diff, sizes, n_boot=1000, ci=0.99)
        report[label]["n_users"] = len(us)
    if dump_preds:
        dump_expert_preds(dump_preds, arrays, pe_all[chosen], users)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for u in users:
            f.write(json.dumps(dict(user=int(u), size=int(grid[chosen][u][1]),
                                    metrics={"LogLoss": round(grid[chosen][u][0], 6)},
                                    chosen=chosen)) + "\n")
    Path(str(out) + ".report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    return report, rows


def dump_expert_preds(path, arrays, pe_chosen, users):
    """Per-row TEST predictions of the GBDT, the GBDT+online layer and FSRS-6 (f_R), with the gate context.
    One row per scored test row: uid, review_th, y, p_gbdt, p_online, f_R and the context columns."""
    frames = []
    for u in users:
        a = arrays[u]
        t = a["test"]
        keys = a["keys"][t]
        pg = 1.0 / (1.0 + np.exp(-a["margin"][a["pos"]][t]))
        df = pd.DataFrame({"uid": np.full(len(keys), int(u), np.int64), "review_th": keys.astype(np.int64),
                           "y": a["y_eval"][t], "p_gbdt": np.clip(pg, P_CLIP, 1 - P_CLIP),
                           "p_online": pe_chosen[u]})
        for c in CTX:
            df[c] = a["ctx"][c][a["pos"]][t]
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out


def _wmean(grid_users, users):
    us = [u for u in users if u in grid_users]
    return C.weighted_mean([grid_users[u][0] for u in us], [grid_users[u][1] for u in us])


def main(argv=None):
    ap = argparse.ArgumentParser(description="Per-user online logistic layer on the saved GBDT (6.5a).")
    ap.add_argument("--model", default="results/srsb_beat/sweep_a_l63.jsonl.model.txt")
    ap.add_argument("--run_info", default="results/srsb_beat/sweep_a_l63.jsonl.run_info.json")
    ap.add_argument("--gbdt_jsonl", default="results/srsb_beat/sweep_a_l63.jsonl")
    ap.add_argument("--data", default="data/anki10k")
    ap.add_argument("--out", default="results/srsb_beat/online_adapt.jsonl")
    ap.add_argument("--dump_preds", default=None, help="parquet of per-row test predictions (uid, review_th, y, ...)")
    args = ap.parse_args(argv)
    report, _ = run(args.model, args.run_info, args.data, args.gbdt_jsonl, args.out, dump_preds=args.dump_preds)
    print(json.dumps(report, indent=1, default=str))
    return report


if __name__ == "__main__":
    main(sys.argv[1:])
