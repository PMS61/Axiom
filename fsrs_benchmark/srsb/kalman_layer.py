"""Kalman-filter user-drift layer on the saved global GBDT (Phase 6, task 6.11; paper/docs/SRSB_BEAT.md).

Per user, a hidden bias b_t follows a random walk. Before row t the prior variance grows by
q0 + q_day * dt_t (dt_t = days since the user's previous review; same-day reviews have dt = 0, so only q0).
The prediction is p_t = sigmoid((m_t + b_t) / sqrt(1 + pi * P_t / 8)) (probit-style variance correction), with
m_t the global GBDT margin. After the label of a labelled (non-first) row, the Laplace / EKF update for a
Bernoulli-logit observation runs with the uncorrected p = sigmoid(m_t + b_t):
    P_new = 1 / (1 / P + p (1 - p)),   b_new = b + P_new * (y - p).

Variant K1: one state for the user. Variant K2: one state per gap bucket (<1d, 1-7d, 7-30d, >30d; bucket
of the row's own elapsed_days, known before the answer); all buckets share the time growth.

Leak rule: p_t uses the state after rows < t, the margin m_t, and dt_t / elapsed_days of row t (all known
before the answer). Row t's label enters only after p_t is taken (tests/test_kalman_layer.py).

Selection honesty: (variant, q0, q_day, P0) chosen on users 8001-8050 only, then reported on 8051-8100 and all
100 users, against the plain GBDT, the online layer (online_adapt_side.jsonl), RWKV-Instant, RWKV-Curve, FSRS-6.

Usage (repo root):
  .venv/bin/python -m fsrs_benchmark.srsb.kalman_layer --workers 4 --out results/srsb_beat/kalman_layer.jsonl
"""
import argparse
import json
import multiprocessing as mp
import resource
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as pads
from numba import njit

from fsrs_benchmark.models import gbdt as G
from fsrs_benchmark.srsb import compare as C
from fsrs_benchmark.srsb import run_gbdt as R
from fsrs_benchmark.srsb import val_plus as V
from fsrs_benchmark.srsb.features_plus import plus_features

P_CLIP = 1e-6
GRID_Q0 = (1e-4, 1e-3, 1e-2)
GRID_QDAY = (1e-3, 1e-2)
GRID_P0 = (0.1, 1.0)
VARIANTS = ("K1", "K2")
SEL_USERS = range(8001, 8051)
REP_USERS = range(8051, 8101)
ALL_USERS = range(8001, 8101)
BIG_ROWS = 150_000                              # users above this many raw rows run one at a time (RAM)
DEFAULT_MODEL = "results/srsb_beat/sweep_a_l63_side.jsonl.model.txt"
DEFAULT_RUN_INFO = "results/srsb_beat/sweep_a_l63_side.jsonl.run_info.json"
DEFAULT_GBDT = "results/srsb_beat/sweep_a_l63_side.jsonl"
DEFAULT_ONLINE = "results/srsb_beat/online_adapt_side.jsonl"


def bucket_of(elapsed_days):
    """Gap bucket 0: <1 day, 1: 1-7 days, 2: 7-30 days, 3: >30 days (NaN counts as the shortest gap)."""
    e = np.nan_to_num(np.asarray(elapsed_days, dtype=np.float64), nan=0.0)
    return np.where(e < 1, 0, np.where(e < 7, 1, np.where(e < 30, 2, 3))).astype(np.int64)


@njit(cache=True)
def kalman_kernel(m, y, labelled, dt, bucket, q0, qday, P0, nb):
    """Per-row predictions p (probit-corrected) over the whole stream. Row t: grow the variances, predict,
    then update the row's bucket with its label (if labelled). Labels enter only after p_t is taken."""
    n = m.shape[0]
    b = np.zeros(nb)
    P = np.full(nb, P0)
    p = np.empty(n)
    for t in range(n):
        if t > 0:
            inc = q0 + qday * dt[t]
            for k in range(nb):
                P[k] += inc
        k = bucket[t]
        s = m[t] + b[k]
        p[t] = 1.0 / (1.0 + np.exp(-s / np.sqrt(1.0 + np.pi * P[k] / 8.0)))
        if labelled[t]:
            pu = 1.0 / (1.0 + np.exp(-s))
            Pn = 1.0 / (1.0 / P[k] + pu * (1.0 - pu))
            b[k] = b[k] + Pn * (y[t] - pu)
            P[k] = Pn
    return p


def configs():
    """{name: (variant, q0, q_day, P0)} for the selection grid (2 variants x 3 x 2 x 2 = 24 settings)."""
    out = {}
    for v in VARIANTS:
        for q0 in GRID_Q0:
            for qd in GRID_QDAY:
                for p0 in GRID_P0:
                    out[f"{v}_q{q0:g}_d{qd:g}_P{p0:g}"] = (v, q0, qd, p0)
    return out


def predict_config(arr, cfg):
    v, q0, qd, p0 = cfg
    bucket = arr["bucket"] if v == "K2" else np.zeros(len(arr["m"]), np.int64)
    nb = 4 if v == "K2" else 1
    return kalman_kernel(arr["m"], arr["y"], arr["labelled"], arr["dt"], bucket, float(q0), float(qd),
                         float(p0), nb)


def user_arrays(uid, booster, names, best_iter, loader, cfg, data_dir, raw=None):
    """Margins, labels, dt and gap buckets for ALL stream rows of one user, plus the scored-row mapping.
    Same feature machinery as online_adapt.user_arrays; `raw` overrides the revlog (tests)."""
    ds = loader.load_user_data(uid)
    if raw is None:
        raw = pd.read_parquet(Path(data_dir) / "revlogs" / f"user_id={uid}")
    st = R.review_stream(raw, cfg.max_seq_len * 2)
    cards = V._read_opt(Path(data_dir) / "cards" / f"user_id={uid}")
    decks = V._read_opt(Path(data_dir) / "decks" / f"user_id={uid}")
    base = G.base_features(st, span=20, user_group="row")[R.FEATURE_COLS]
    feats = pd.concat([base, plus_features(st, cards, decks)], axis=1)
    m = booster.predict(feats[list(names)].to_numpy(np.float32), raw_score=True,
                        num_iteration=best_iter).astype(np.float64)
    keys = ds["review_th"].to_numpy()
    test = R.test_row_mask(len(ds), cfg.n_splits)
    pos = st.index.get_indexer(keys)
    if (pos < 0).any():
        raise RuntimeError(f"user {uid}: scored rows not in the stream")
    return dict(uid=int(uid), m=m, y=(st["rating"].to_numpy() > 1).astype(np.float64),
                labelled=~st["card_id"].duplicated().to_numpy(),
                dt=np.nan_to_num(st["dt_days"].to_numpy(np.float64)),
                bucket=bucket_of(st["elapsed_days"].to_numpy(np.float64)),
                pos=pos, test=test, y_eval=ds["y"].to_numpy(np.float64), keys=keys)


def _ll(y, p):
    from sklearn.metrics import log_loss
    return float(log_loss(y, p, labels=[0, 1]))


def _auc(y, p):
    from sklearn.metrics import roc_auc_score
    try:
        a = float(roc_auc_score(y, p))
    except ValueError:
        return None
    return a if np.isfinite(a) else None


def summarise_user(arr, cfgs):
    """Test-row LogLoss and AUC per config, and the GBDT LogLoss, for one user."""
    t = arr["test"]
    pe = arr["pos"][t]
    y = arr["y_eval"][t]
    p_gb = np.clip(1.0 / (1.0 + np.exp(-arr["m"][pe])), P_CLIP, 1 - P_CLIP)
    metrics = {}
    for name, cfg in cfgs.items():
        p = np.clip(predict_config(arr, cfg)[pe], P_CLIP, 1 - P_CLIP)
        metrics[name] = (_ll(y, p), _auc(y, p))
    return dict(uid=arr["uid"], size=int(t.sum()), gbdt_ll=_ll(y, p_gb), gbdt_auc=_auc(y, p_gb),
                metrics=metrics)


_BW: dict = {}


def _init_worker(model_path, names, best_iter, data_dir, cfgs):
    import lightgbm as lgb
    V._init(str(data_dir), names)
    _BW.update(booster=lgb.Booster(model_file=str(model_path)), names=list(names), best_iter=int(best_iter),
               data_dir=Path(data_dir), cfgs=cfgs)


def _task(uid):
    try:
        b = _BW
        t0 = time.time()
        arr = user_arrays(uid, b["booster"], b["names"], b["best_iter"], V._W["loader"], V._W["cfg"],
                          b["data_dir"])
        res = summarise_user(arr, b["cfgs"])
        res["stream_rows"] = int(len(arr["y"]))
        res["seconds"] = round(time.time() - t0, 1)
        return res
    except Exception:
        raise RuntimeError(f"user {uid} failed:\n{traceback.format_exc()}")


def raw_rows(data_dir, uid):
    return int(pads.dataset(str(Path(data_dir) / "revlogs" / f"user_id={uid}"), format="parquet").count_rows())


def _progress(r, i, n, t0, big=False):
    tag = "big " if big else ""
    where = "" if i is None else f" ({i}/{n})"
    print(f"{tag}user {r['uid']}{where} rows {r['stream_rows']} seconds {r['seconds']} "
          f"total_elapsed {time.time() - t0:.0f}s", flush=True)


def _wmean(vals, sizes):
    return C.weighted_mean(vals, sizes) if len(vals) else float("nan")


def _online_rows(path):
    return C.load_rows(path) if Path(path).exists() else None


def summarise(results, names, pub, online, users, chosen=None):
    users = [int(u) for u in users]
    sel = [u for u in users if u in set(SEL_USERS)]
    if not sel:
        raise ValueError("no selection users (8001-8050) among the evaluated users")
    sel_ll = {n: C.weighted_mean([results[u]["metrics"][n][0] for u in sel], [results[u]["size"] for u in sel])
              for n in names}
    if chosen is None:
        chosen = min(sel_ll, key=sel_ll.get)
    report = dict(selection_users="8001-8050", selection_wLL={k: round(v, 5) for k, v in sel_ll.items()},
                  chosen=chosen, grid=list(names), configs={k: list(v) for k, v in configs().items()},
                  mean_seconds=None)
    for label, us in (("8051-8100", [u for u in users if u in set(REP_USERS)]),
                      ("8001-8100", [u for u in users if u in set(ALL_USERS)])):
        if not us:
            continue
        report[label] = dict(n_users=len(us), summary=_block(results, us, chosen),
                             paired=_paired(results, us, chosen, pub, online))
    report["all_settings_8001-8100"] = {n: _block(results, [u for u in users if u in set(ALL_USERS)], n)
                                        for n in names}
    return report


def _block(results, us, name):
    sizes = [results[u]["size"] for u in us]
    ll = [results[u]["metrics"][name][0] for u in us]
    aucs = [(results[u]["metrics"][name][1], results[u]["size"]) for u in us]
    aucs = [(a, s) for a, s in aucs if a is not None]
    gb_ll = [results[u]["gbdt_ll"] for u in us]
    out = dict(n_users=len(us), unweighted=round(float(np.mean(ll)), 5), weighted=round(C.weighted_mean(ll, sizes), 5),
               auc_weighted=round(_wmean([a for a, _ in aucs], [s for _, s in aucs]), 5),
               gbdt_weighted=round(C.weighted_mean(gb_ll, sizes), 5))
    return out


def _paired(results, us, chosen, pub, online):
    ours = {u: results[u]["metrics"][chosen][0] for u in us}
    out = {}
    out["vs_GBDT"] = C.paired_bootstrap([ours[u] - results[u]["gbdt_ll"] for u in us],
                                        [results[u]["size"] for u in us], n_boot=1000, ci=0.99)
    refs = dict(pub)
    if online is not None:
        refs["online_layer"] = {u: online[u] for u in online}
    for name, rows in refs.items():
        common = [u for u in us if u in rows and rows[u]["size"] == results[u]["size"]]
        if not common:
            continue
        diff = [ours[u] - rows[u]["metrics"]["LogLoss"] for u in common]
        out[f"vs_{name}"] = C.paired_bootstrap(diff, [results[u]["size"] for u in common], n_boot=1000, ci=0.99)
    return out


def run(model_path, run_info_path, data_dir, gbdt_jsonl, out, online_jsonl=DEFAULT_ONLINE,
        result_dir=C.DEFAULT_RESULT_DIR, users=ALL_USERS, workers=4, big_rows=BIG_ROWS):
    data_dir = Path(data_dir)
    info = json.loads(Path(run_info_path).read_text(encoding="utf-8"))
    names, best_iter = info["features"], info["best_iter"]
    cfgs = configs()
    gb = C.load_rows(gbdt_jsonl)
    pub = {k: C.load_rows(Path(result_dir) / f"{f}.jsonl") for k, f in V.PUBLISHED.items()}
    online = _online_rows(online_jsonl)
    users = [int(u) for u in users]
    t0 = time.time()
    args = (str(model_path), names, best_iter, str(data_dir), cfgs)
    results = {}
    big = [u for u in users if raw_rows(data_dir, u) > big_rows] if workers > 1 else []
    small = [u for u in users if u not in set(big)]
    if workers <= 1 or len(small) < 2:
        _init_worker(*args)
        for i, u in enumerate(small, 1):
            results[u] = _task(u)
            _progress(results[u], i, len(small), t0)
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(workers, initializer=_init_worker, initargs=args) as pool:
            for i, r in enumerate(pool.imap(_task, small, chunksize=1), 1):
                results[r["uid"]] = r
                _progress(r, i, len(small), t0)
    if big:
        _init_worker(*args)
        for u in big:
            results[u] = _task(u)
            _progress(results[u], None, None, t0, big=True)
    for u in users:
        ref = gb[u]["metrics"]["LogLoss"]
        if abs(results[u]["gbdt_ll"] - ref) > 1e-4:
            raise RuntimeError(f"user {u}: GBDT check failed ({results[u]['gbdt_ll']} vs {ref})")
    report = summarise(results, list(cfgs), pub, online, users)
    chosen = report["chosen"]
    report["elapsed_min"] = round((time.time() - t0) / 60, 2)
    report["peak_rss_gb_parent"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for u in users:
            r = results[u]
            f.write(json.dumps(dict(user=u, size=r["size"], chosen=chosen,
                                    metrics={"LogLoss": round(r["metrics"][chosen][0], 6),
                                             "AUC": r["metrics"][chosen][1]})) + "\n")
    Path(str(out) + ".report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description="Kalman user-drift layer on the saved GBDT (task 6.11).")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--run_info", default=DEFAULT_RUN_INFO)
    ap.add_argument("--gbdt_jsonl", default=DEFAULT_GBDT)
    ap.add_argument("--online_jsonl", default=DEFAULT_ONLINE)
    ap.add_argument("--data", default="data/anki10k")
    ap.add_argument("--out", default="results/srsb_beat/kalman_layer.jsonl")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--users", default=None,
                    help="ids (comma separated) or a file (one id per line or JSON list); default 8001-8100")
    ap.add_argument("--big_rows", type=int, default=BIG_ROWS)
    ap.add_argument("--result_dir", default=str(C.DEFAULT_RESULT_DIR))
    args = ap.parse_args(argv)
    users = ALL_USERS if args.users is None else R.load_ids(args.users)
    report = run(args.model, args.run_info, args.data, args.gbdt_jsonl, args.out, online_jsonl=args.online_jsonl,
                 result_dir=args.result_dir, users=users, workers=args.workers, big_rows=args.big_rows)
    print(json.dumps(report, indent=1, default=str))
    return report


if __name__ == "__main__":
    main(sys.argv[1:])
