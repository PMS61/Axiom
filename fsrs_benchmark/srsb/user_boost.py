"""Per-user boosting on top of the saved global GBDT (Phase 6, task 6.8; paper/docs/SRSB_BEAT.md).

For one user, rows are visited in review_th order (the whole stream, same-day reviews included). The global
GBDT gives the raw margin m_t for every stream row. The walk is cut into blocks by refit points:
  - a refit point is the start of each new day_offset, with at most one refit per `refit_rows` rows, and only
    once the user has at least `min_rows` labelled past rows (labelled = non-first review);
  - at a refit point t0, a tiny LightGBM is trained on the user's labelled rows < t0 only, with init_score = m
    (the global margin) and the features [SUBSET..., m]; it predicts the delta for rows t0 .. next refit
    (exclusive), so p_t = sigmoid(m_t + delta_t). Before the first refit, p_t = sigmoid(m_t).
Each refit retrains from scratch on all past labelled rows. Fixed hyper-parameters, no early stopping.

Leak rule: the block starting at t0 is predicted by a model trained on rows < t0 only, and the refit points
depend only on day_offset and the first-review flags (not on labels). So p_t never depends on labels of rows
>= t (tests/test_user_boost.py). Features are past-only, as in the rest of the pipeline.

Selection honesty: rounds are chosen on users 8001-8050 only (at most 3 settings), then reported on
8051-8100 and on all 100 users.

Usage (repo root):
  .venv/bin/python -m fsrs_benchmark.srsb.user_boost --workers 4 --dump_preds <parquet>
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

from fsrs_benchmark.models import gbdt as G
from fsrs_benchmark.srsb import compare as C
from fsrs_benchmark.srsb import run_gbdt as R
from fsrs_benchmark.srsb import val_plus as V
from fsrs_benchmark.srsb.features_plus import plus_features

P_CLIP = 1e-6
MIN_ROWS = 100                                  # labelled past rows before the first refit
REFIT_ROWS = 200                                # at most one refit per this many rows
SUBSET = ("log_el_frac", "f_logitR", "card_n", "state", "rate_today", "ewm50", "g1", "card_rate",
          "dur_card_last_rel")                  # plus the margin m itself (appended last)
ROUNDS_GRID = (10, 20, 40)                      # selection grid: at most 3 settings
LEAVES = 4
MAX_SETTINGS = 3
SEL_USERS = range(8001, 8051)                   # selection
REP_USERS = range(8051, 8101)                   # report
ALL_USERS = range(8001, 8101)
USER_PARAMS = dict(objective="binary", learning_rate=0.1, min_data_in_leaf=30, feature_fraction=0.8,
                   lambda_l2=10.0, num_threads=1, seed=0, deterministic=True, force_row_wise=True,
                   verbose=-1)
BIG_ROWS = 150_000                              # users with more raw revlog rows run one at a time (RAM)
DEFAULT_MODEL = "results/srsb_beat/sweep_a_l63_side.jsonl.model.txt"
DEFAULT_RUN_INFO = "results/srsb_beat/sweep_a_l63_side.jsonl.run_info.json"
DEFAULT_GBDT = "results/srsb_beat/sweep_a_l63_side.jsonl"


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def refit_points(day, labelled, refit_rows=REFIT_ROWS, min_rows=MIN_ROWS):
    """Sorted row indices where a refit happens. Uses only day_offset and the labelled (non-first) flags."""
    day = np.asarray(day)
    labelled = np.asarray(labelled, dtype=bool)
    n_lab_before = np.concatenate([[0], np.cumsum(labelled)])   # n_lab_before[t] = labelled rows < t
    pts, last = [], None
    for t in range(1, len(day)):
        if day[t] == day[t - 1]:
            continue
        if n_lab_before[t] < min_rows:
            continue
        if last is not None and t - last < refit_rows:
            continue
        pts.append(t)
        last = t
    return pts


def _fit(X, init, y, rounds, leaves):
    import lightgbm as lgb
    params = dict(USER_PARAMS, num_leaves=int(leaves))
    ds = lgb.Dataset(X, label=y, init_score=init, params={"verbose": -1})
    return lgb.train(params, ds, num_boost_round=int(rounds))


def boost_predict(margin, sub, y, labelled, day, rounds, leaves, refit_rows=REFIT_ROWS, min_rows=MIN_ROWS):
    """Per-row logit delta over the whole stream (n,) and the number of refits.
    margin: global GBDT raw margin m (n,); sub: SUBSET features (n, k); y: labels (n,) (used on labelled
    rows < refit point only); labelled: bool (n,); day: day_offset (n,).
    delta[t] comes from a model trained on rows < block start, so p = sigmoid(margin + delta)."""
    margin = np.asarray(margin, dtype=np.float64)
    n = len(margin)
    delta = np.zeros(n)
    pts = refit_points(day, labelled, refit_rows, min_rows)
    X = np.column_stack([np.asarray(sub, dtype=np.float64), margin])     # margin m is the last feature
    labelled = np.asarray(labelled, dtype=bool)
    bounds = pts + [n]
    for a, b in zip(pts, bounds[1:]):
        train = np.flatnonzero(labelled[:a])
        booster = _fit(X[train], margin[train], np.asarray(y, dtype=np.float64)[train], rounds, leaves)
        delta[a:b] = booster.predict(X[a:b], raw_score=True)
    return delta, len(pts)


def settings_grid(rounds_list=ROUNDS_GRID, leaves=LEAVES):
    """{name: (rounds, leaves)} for the selection grid. At most MAX_SETTINGS settings."""
    grid = {f"r{r}_l{leaves}": (int(r), int(leaves)) for r in rounds_list}
    if len(grid) > MAX_SETTINGS:
        raise ValueError(f"at most {MAX_SETTINGS} settings may be tried (got {len(grid)})")
    return grid


def predict_stream(arr, settings, refit_rows=REFIT_ROWS):
    """{name: (p over the whole stream, n_refits)} for one user's arrays (from user_arrays)."""
    out = {}
    for name, (rounds, leaves) in settings.items():
        delta, nref = boost_predict(arr["margin"], arr["sub"], arr["y"], arr["labelled"], arr["day"],
                                    rounds, leaves, refit_rows=refit_rows)
        out[name] = (_sigmoid(arr["margin"] + delta), nref)
    return out


def user_arrays(uid, booster, names, best_iter, loader, cfg, data_dir, raw=None):
    """Global margins, SUBSET features, labels and day offsets for ALL stream rows of one user, plus the
    scored-row mapping (positions, test mask, labels). `raw` overrides the revlog (used by the tests)."""
    ds = loader.load_user_data(uid)
    if raw is None:
        raw = pd.read_parquet(Path(data_dir) / "revlogs" / f"user_id={uid}")
    st = R.review_stream(raw, cfg.max_seq_len * 2)
    cards = V._read_opt(Path(data_dir) / "cards" / f"user_id={uid}")
    decks = V._read_opt(Path(data_dir) / "decks" / f"user_id={uid}")
    base = G.base_features(st, span=20, user_group="row")[R.FEATURE_COLS]
    feats = pd.concat([base, plus_features(st, cards, decks)], axis=1)
    margin = booster.predict(feats[list(names)].to_numpy(np.float32), raw_score=True,
                             num_iteration=best_iter).astype(np.float64)
    keys = ds["review_th"].to_numpy()
    test = R.test_row_mask(len(ds), cfg.n_splits)
    pos = st.index.get_indexer(keys)
    if (pos < 0).any():
        raise RuntimeError(f"user {uid}: scored rows not in the stream")
    return dict(uid=int(uid), margin=margin, sub=feats[list(SUBSET)].to_numpy(np.float64),
                y=(st["rating"].to_numpy() > 1).astype(np.float64),
                labelled=~st["card_id"].duplicated().to_numpy(),
                day=st["day_offset"].to_numpy(np.int64), rt=st.index.to_numpy(np.int64),
                pos=pos, test=test, y_eval=ds["y"].to_numpy(np.float64), keys=keys)


def _ll(y, p):
    from sklearn.metrics import log_loss
    return float(log_loss(y, p, labels=[0, 1]))


def _auc(y, p):
    from sklearn.metrics import roc_auc_score
    try:
        a = float(roc_auc_score(y, p))
    except ValueError:                       # one class only
        return None
    return a if np.isfinite(a) else None


def summarise_user(arr, settings, refit_rows=REFIT_ROWS):
    """Test-row results for one user: size, labels, review_th, GBDT and each setting's probabilities."""
    t = arr["test"]
    pe = arr["pos"][t]
    p_gb_full = _sigmoid(arr["margin"])
    p_gb = np.clip(p_gb_full[pe], P_CLIP, 1 - P_CLIP)
    y = arr["y_eval"][t]
    gb_ll = _ll(y, p_gb)
    preds, refits = {}, {}
    for name, (p_full, nref) in predict_stream(arr, settings, refit_rows).items():
        preds[name] = np.clip(p_full[pe], P_CLIP, 1 - P_CLIP)
        refits[name] = int(nref)
    return dict(uid=arr["uid"], size=int(t.sum()), y=y, review_th=arr["keys"][t].astype(np.int64),
                p_gbdt=p_gb, gbdt_ll=gb_ll, p=preds, refits=refits)


_BW: dict = {}


def _init_worker(model_path, names, best_iter, data_dir, settings, refit_rows):
    import lightgbm as lgb
    V._init(str(data_dir), names)
    _BW.update(booster=lgb.Booster(model_file=str(model_path)), names=list(names), best_iter=int(best_iter),
               data_dir=Path(data_dir), settings=settings, refit_rows=refit_rows)


def _task(uid):
    try:
        b = _BW
        arr = user_arrays(uid, b["booster"], b["names"], b["best_iter"], V._W["loader"], V._W["cfg"],
                          b["data_dir"])
        return summarise_user(arr, b["settings"], b["refit_rows"])
    except Exception:
        raise RuntimeError(f"user {uid} failed:\n{traceback.format_exc()}")


def raw_rows(data_dir, uid):
    """Revlog rows of a user from the parquet metadata (cheap; an upper bound of the stream length)."""
    return int(pads.dataset(str(Path(data_dir) / "revlogs" / f"user_id={uid}"), format="parquet").count_rows())


def _check_gbdt(res, gb):
    ref = gb[res["uid"]]["metrics"]["LogLoss"]
    if abs(res["gbdt_ll"] - ref) > 1e-4:
        raise RuntimeError(f"user {res['uid']}: GBDT check failed ({res['gbdt_ll']} vs {ref})")


def _wmean_over(values, sizes):
    return C.weighted_mean(values, sizes) if len(values) else float("nan")


def _block_summary(results, users, names):
    """Size-weighted and unweighted LogLoss and size-weighted AUC for GBDT and each setting over `users`."""
    us = [u for u in users if u in results]
    if not us:
        return None
    sizes = [results[u]["size"] for u in us]
    out = {}
    for name in ["GBDT"] + list(names):
        if name == "GBDT":
            ll = [results[u]["gbdt_ll"] for u in us]
            auc_pairs = [(results[u]["y"], results[u]["p_gbdt"]) for u in us]
        else:
            ll = [_ll(results[u]["y"], results[u]["p"][name]) for u in us]
            auc_pairs = [(results[u]["y"], results[u]["p"][name]) for u in us]
        aucs = [(_auc(y, p), s) for (y, p), s in zip(auc_pairs, sizes)]
        aucs = [(a, s) for a, s in aucs if a is not None]
        out[name] = dict(n_users=len(us), unweighted=round(float(np.mean(ll)), 5),
                         weighted=round(C.weighted_mean(ll, sizes), 5),
                         auc_weighted=round(_wmean_over([a for a, _ in aucs], [s for _, s in aucs]), 5))
    return out


def _paired(results, users, chosen, pub):
    """Paired size-weighted LogLoss differences (chosen minus reference), 99% user-bootstrap CI."""
    us = [u for u in users if u in results]
    ours = {u: _ll(results[u]["y"], results[u]["p"][chosen]) for u in us}
    out = {}
    diff = [ours[u] - results[u]["gbdt_ll"] for u in us]
    out["vs_GBDT"] = C.paired_bootstrap(diff, [results[u]["size"] for u in us], n_boot=1000, ci=0.99)
    for ref_name, rows in pub.items():
        common = [u for u in us if u in rows and rows[u]["size"] == results[u]["size"]]
        if not common:
            continue
        diff = [ours[u] - rows[u]["metrics"]["LogLoss"] for u in common]
        out[f"vs_{ref_name}"] = C.paired_bootstrap(diff, [results[u]["size"] for u in common],
                                                   n_boot=1000, ci=0.99)
    return out


def summarise(results, gb, names, pub, users, chosen=None):
    """Full report: selection on 8001-8050, then the chosen setting on 8051-8100 and 8001-8100."""
    users = [int(u) for u in users]
    sel = [u for u in users if u in set(SEL_USERS)]
    if not sel:
        raise ValueError("no selection users (8001-8050) among the evaluated users")
    sel_ll = {}
    for name in names:
        sel_ll[name] = C.weighted_mean([_ll(results[u]["y"], results[u]["p"][name]) for u in sel],
                                       [results[u]["size"] for u in sel])
    if chosen is None:
        chosen = min(sel_ll, key=sel_ll.get)
    report = dict(selection_users="8001-8050", selection_wLL={k: round(v, 5) for k, v in sel_ll.items()},
                  chosen=chosen, settings=list(names), refit_rows=REFIT_ROWS, min_rows=MIN_ROWS,
                  subset=list(SUBSET), leaves=LEAVES, lgb_params=USER_PARAMS,
                  mean_refits={n: round(float(np.mean([results[u]["refits"][n] for u in users])), 2)
                               for n in names})
    for label, us in (("8051-8100", [u for u in users if u in set(REP_USERS)]),
                      ("8001-8100", [u for u in users if u in set(ALL_USERS)])):
        if not us:
            continue
        block = _block_summary(results, us, [chosen])
        report[label] = dict(n_users=len(us), summary=block,
                             paired=_paired(results, us, chosen, pub),
                             pub_summary=_pub_summary(results, us, pub))
    report["all_settings_8001-8100"] = _block_summary(results, [u for u in users if u in set(ALL_USERS)], names)
    return report


def _pub_summary(results, users, pub):
    """AUC and LogLoss of the published models on the same users (same size), size-weighted."""
    common = [u for u in users if all(u in rows and rows[u]["size"] == results[u]["size"]
                                      for rows in pub.values())]
    if not common:
        return None
    sizes = [results[u]["size"] for u in common]
    out = {}
    for name, rows in pub.items():
        ll = [rows[u]["metrics"]["LogLoss"] for u in common]
        aucs = [(rows[u]["metrics"].get("AUC"), results[u]["size"]) for u in common]
        aucs = [(a, s) for a, s in aucs if a is not None]
        out[name] = dict(n_users=len(common), weighted=round(C.weighted_mean(ll, sizes), 5),
                         auc_weighted=round(_wmean_over([a for a, _ in aucs], [s for _, s in aucs]), 5))
    return out


def run(model_path, run_info_path, data_dir, gbdt_jsonl, out, result_dir=C.DEFAULT_RESULT_DIR, users=ALL_USERS,
        rounds_list=ROUNDS_GRID, leaves=LEAVES, refit_rows=REFIT_ROWS, workers=4, dump_preds=None,
        big_rows=BIG_ROWS):
    data_dir = Path(data_dir)
    info = json.loads(Path(run_info_path).read_text(encoding="utf-8"))
    names, best_iter = info["features"], info["best_iter"]
    settings = settings_grid(rounds_list, leaves)
    gb = C.load_rows(gbdt_jsonl)
    pub = {k: C.load_rows(Path(result_dir) / f"{f}.jsonl") for k, f in V.PUBLISHED.items()}
    users = [int(u) for u in users]
    t0 = time.time()
    args = (str(model_path), names, best_iter, str(data_dir), settings, refit_rows)
    results = {}
    # Featurizing a user takes RAM in proportion to its stream length (about 10 KB per row). Big users run one
    # at a time in this process after the pool has finished; the small ones share the pool.
    big = [u for u in users if raw_rows(data_dir, u) > big_rows] if workers > 1 else []
    small = [u for u in users if u not in set(big)]
    if workers <= 1 or len(small) < 2:
        _init_worker(*args)
        for i, u in enumerate(small, 1):
            results[u] = _task(u)
            print(f"user {u} ({i}/{len(small)}) refits {results[u]['refits']}, elapsed {time.time() - t0:.0f}s",
                  flush=True)
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(workers, initializer=_init_worker, initargs=args) as pool:
            for i, r in enumerate(pool.imap(_task, small, chunksize=1), 1):
                results[r["uid"]] = r
                print(f"user {r['uid']} ({i}/{len(small)}), elapsed {time.time() - t0:.0f}s", flush=True)
    if big:
        _init_worker(*args)
        for u in big:
            results[u] = _task(u)
            print(f"big user {u} (sequential), elapsed {time.time() - t0:.0f}s", flush=True)
    for u in users:
        _check_gbdt(dict(results[u], uid=u), gb)
    names_s = list(settings)
    report = summarise(results, gb, names_s, pub, users)
    chosen = report["chosen"]
    report["elapsed_min"] = round((time.time() - t0) / 60, 2)
    report["peak_rss_gb_parent"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)
    report["peak_rss_gb_largest_child"] = round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6, 2)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for u in users:
            r = results[u]
            f.write(json.dumps(dict(user=u, size=r["size"], chosen=chosen,
                                    metrics={"LogLoss": round(_ll(r["y"], r["p"][chosen]), 6),
                                             "AUC": _auc(r["y"], r["p"][chosen])})) + "\n")
    Path(str(out) + ".report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    if dump_preds:
        frames = []
        for u in users:
            r = results[u]
            frames.append(pd.DataFrame({"uid": np.full(r["size"], u, np.int64), "review_th": r["review_th"],
                                        "y": r["y"], "p_boost": r["p"][chosen], "p_gbdt": r["p_gbdt"]}))
        Path(dump_preds).parent.mkdir(parents=True, exist_ok=True)
        pd.concat(frames, ignore_index=True).to_parquet(dump_preds, index=False)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description="Per-user boosting on the saved GBDT (task 6.8).")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--run_info", default=DEFAULT_RUN_INFO)
    ap.add_argument("--gbdt_jsonl", default=DEFAULT_GBDT)
    ap.add_argument("--data", default="data/anki10k")
    ap.add_argument("--out", default="results/srsb_beat/user_boost.jsonl")
    ap.add_argument("--dump_preds", default=None, help="parquet of per-row test predictions")
    ap.add_argument("--refit_rows", type=int, default=REFIT_ROWS, help="at most one refit per N rows")
    ap.add_argument("--rounds", default=",".join(str(r) for r in ROUNDS_GRID),
                    help="comma list of n_rounds to try (at most 3)")
    ap.add_argument("--leaves", type=int, default=LEAVES)
    ap.add_argument("--workers", type=int, default=4, help="parallel users")
    ap.add_argument("--users", default=None, help="ids (comma separated) or a file; default 8001-8100")
    ap.add_argument("--result_dir", default=str(C.DEFAULT_RESULT_DIR))
    ap.add_argument("--big_rows", type=int, default=BIG_ROWS,
                    help="users with more raw revlog rows run sequentially (RAM guard)")
    args = ap.parse_args(argv)
    rounds = [int(x) for x in args.rounds.split(",") if x.strip()]
    users = ALL_USERS if args.users is None else R.load_ids(args.users)
    report = run(args.model, args.run_info, args.data, args.gbdt_jsonl, args.out, result_dir=args.result_dir,
                 users=users, rounds_list=rounds, leaves=args.leaves, refit_rows=args.refit_rows,
                 workers=args.workers, dump_preds=args.dump_preds, big_rows=args.big_rows)
    print(json.dumps(report, indent=1, default=str))
    return report


if __name__ == "__main__":
    main(sys.argv[1:])
