"""SRS-Beat validation harness (Phase 6, task 6.2). See paper/docs/SRSB_BEAT.md.

Pooled GBDT on FEATURE_COLS (run_gbdt: gap, card, fsrs, user) plus features_plus.PLUS_COLS. It is the same
protocol as run_gbdt (train on the train users' scored rows, early stopping on a fixed 10% holdout of them,
score the eval users' TimeSeriesSplit test rows as srs-benchmark does for FSRS-6-short), with a feature-group
choice and cards/decks side tables.

Honesty guard: users 1-1000 are the test set. Any train or eval user in 1-1000 raises unless --final is given
(task 6.5 only). Train and eval must not overlap.

Output (resumable, like run_gbdt):
  <out>                  srs-benchmark style jsonl, one line per eval user {"metrics", "user", "size"}
  <out>.model.txt/.json  the LightGBM model and its best iteration
  <out>.run_info.json    groups, features, params, best_iter, users, fit rows, skipped users, comparison

Usage (repo root):
  .venv/bin/python -m fsrs_benchmark.srsb.val_plus --data data/anki10k --train_users T.txt \
      --eval_users V.txt --out result/val_plus/all.jsonl [--groups gap,card,sib] [--download] [--workers 4]
"""
import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fsrs_benchmark import data as D
from fsrs_benchmark.models import gbdt as G
from fsrs_benchmark.srsb import compare as C
from fsrs_benchmark.srsb import run_gbdt as R
from fsrs_benchmark.srsb.features_plus import PLUS_GROUPS, plus_features

TEST_LO, TEST_HI = 1, 1000             # the test set: never used here unless final=True
OLD_GROUPS = ("gap", "card", "fsrs", "user")
ALL_GROUPS = OLD_GROUPS + tuple(PLUS_GROUPS)
DEFAULT_MAX_ROUNDS = 10000
DEFAULT_LEAVES = 15
DEFAULT_LR = 0.02
PUBLISHED = {"RWKV-Instant": "RWKV-P", "RWKV-Curve": "RWKV", "FSRS-6": "FSRS-6-short"}

_W: dict = {}


def group_columns(groups=None):
    """Feature columns of the chosen groups, in a fixed order. groups=None means all groups."""
    chosen = ALL_GROUPS if groups is None else tuple(groups)
    bad = [g for g in chosen if g not in ALL_GROUPS]
    if bad:
        raise ValueError(f"unknown feature group(s) {bad}; choose from {ALL_GROUPS}")
    cols = []
    for g in ALL_GROUPS:
        if g in chosen:
            cols += G.GROUP_FEATURES[g] if g in OLD_GROUPS else PLUS_GROUPS[g]
    if not cols:
        raise ValueError("no feature groups selected")
    return cols


def check_users(train_uids, eval_uids, final=False):
    """Raise on an empty side, a train/eval overlap, or a user in 1-1000 (test set) unless final=True."""
    train, ev = {int(u) for u in train_uids}, {int(u) for u in eval_uids}
    if not train:
        raise ValueError("no train users")
    if not ev:
        raise ValueError("no eval users")
    overlap = train & ev
    if overlap:
        raise ValueError(f"train and eval users overlap: {sorted(overlap)[:10]}")
    if not final:
        bad = sorted(u for u in train | ev if TEST_LO <= u <= TEST_HI)
        if bad:
            raise ValueError(f"users {TEST_LO}-{TEST_HI} are the test set and may not be used here "
                             f"(only task 6.5 passes final=True): {bad[:10]}")


def _read_opt(path: Path):
    return pd.read_parquet(path) if path.exists() else None


def user_table(uid, loader, cfg, data_dir: Path, cols):
    """Features (cols, float32) and labels of one user's scored rows, plus the test mask and meta."""
    ds = loader.load_user_data(uid)
    raw = pd.read_parquet(data_dir / "revlogs" / f"user_id={uid}")
    st = R.review_stream(raw, cfg.max_seq_len * 2)
    cards = _read_opt(data_dir / "cards" / f"user_id={uid}")
    decks = _read_opt(data_dir / "decks" / f"user_id={uid}")
    base = G.base_features(st, span=20, user_group="row")[R.FEATURE_COLS]
    feats = pd.concat([base, plus_features(st, cards, decks)], axis=1)
    keys = ds["review_th"].to_numpy()
    if np.setdiff1d(keys, feats.index.to_numpy()).size:
        raise RuntimeError(f"user {uid}: scored rows missing from the feature stream")
    X = feats.reindex(keys)[list(cols)].to_numpy(np.float32)
    return dict(
        X=X,
        y=ds["y"].to_numpy(np.int64),
        test=R.test_row_mask(len(ds), cfg.n_splits),
        side=cards is not None and decks is not None,
        meta=pd.DataFrame({"y": ds["y"].to_numpy(np.float64),
                           "elapsed_days": ds["elapsed_days"].to_numpy(np.float64),
                           "i": ds["i"].to_numpy(np.float64),
                           "rmse_bins_lapse": ds["rmse_bins_lapse"].to_numpy(np.float64)}),
    )


def _init(data_dir: str, cols) -> None:
    S = R._srs_modules()
    cfg = S["Config"](S["create_parser"]().parse_args(["--algo", "FSRS-6", "--short", "--data", data_dir]))
    if cfg.get_evaluation_file_name() != "FSRS-6-short":
        raise RuntimeError(f"unexpected srs-benchmark setting {cfg.get_evaluation_file_name()}")
    _W.clear()
    _W.update(cfg=cfg, loader=S["UserDataLoader"](cfg), data_dir=Path(data_dir), cols=list(cols))


def _task_train(uid):
    try:
        t = user_table(uid, _W["loader"], _W["cfg"], _W["data_dir"], _W["cols"])
        return dict(uid=uid, X=t["X"], y=t["y"], side=t["side"])
    except Exception as e:                  # srs-benchmark skips these users too
        return dict(uid=uid, error=f"{type(e).__name__}: {str(e)[:100]}")


def _task_eval(uid):
    try:
        t = user_table(uid, _W["loader"], _W["cfg"], _W["data_dir"], _W["cols"])
        m = t["test"]
        if not m.any():
            return dict(uid=uid, error="no test rows")
        return dict(uid=uid, X=t["X"][m], y=t["y"][m], meta=t["meta"][m].reset_index(drop=True),
                    side=t["side"])
    except Exception as e:
        return dict(uid=uid, error=f"{type(e).__name__}: {str(e)[:100]}")


def _map(task, uids, data_dir: Path, cols, workers: int):
    """Yield task(uid) in order; workers > 1 uses a fork pool (featurization in parallel)."""
    if workers <= 1 or len(uids) < 2:
        _init(str(data_dir), cols)
        for uid in uids:
            yield task(uid)
        return
    ctx = mp.get_context("fork")
    with ctx.Pool(workers, initializer=_init, initargs=(str(data_dir), list(cols))) as pool:
        yield from pool.imap(task, uids, chunksize=1)


def compare_published(ours_path, result_dir=C.DEFAULT_RESULT_DIR, ci: float = 0.99, n_boot: int = 1000,
                      seed: int = 0):
    """Our jsonl against the published RWKV-Instant (RWKV-P), RWKV-Curve (RWKV) and FSRS-6 (FSRS-6-short)
    files, on the users present in all of them with the same size. Returns None when a file or a common
    user is missing."""
    result_dir = Path(result_dir)
    if not all((result_dir / f"{f}.jsonl").exists() for f in PUBLISHED.values()):
        return None
    ours = C.load_rows(ours_path)
    refs = {name: C.load_rows(result_dir / f"{f}.jsonl") for name, f in PUBLISHED.items()}
    users = sorted(u for u in ours if all(u in r and r[u]["size"] == ours[u]["size"] for r in refs.values()))
    if not users:
        return None
    sizes = [ours[u]["size"] for u in users]
    ll = {"GBDT (ours)": [ours[u]["metrics"]["LogLoss"] for u in users]}
    for name, rows in refs.items():
        ll[name] = [rows[u]["metrics"]["LogLoss"] for u in users]
    table = {name: dict(unweighted=round(float(np.mean(v)), 5),
                        weighted=round(C.weighted_mean(v, sizes), 5)) for name, v in ll.items()}
    boots = {}
    for name in refs:
        diff = np.asarray(ll["GBDT (ours)"]) - np.asarray(ll[name])
        boots[f"GBDT - {name}"] = C.paired_bootstrap(diff, sizes, n_boot=n_boot, ci=ci, seed=seed)
    return dict(n_users=len(users), n_rows=int(sum(sizes)), table=table, bootstrap=boots, ci=ci)


def _print_comparison(cmp):
    if cmp is None:
        print("comparison: no eval users with a published result of the same size")
        return
    print(f"comparison on {cmp['n_users']} users ({cmp['n_rows']} test rows), LogLoss, lower is better")
    for name, t in cmp["table"].items():
        print(f"  {name:14s} unweighted {t['unweighted']:.5f}  size-weighted {t['weighted']:.5f}")
    for name, b in cmp["bootstrap"].items():
        print(f"  {name:24s} size-weighted diff {b['mean']:+.5f}  {100 * cmp['ci']:.0f}% CI "
              f"[{b['lo']:+.5f}, {b['hi']:+.5f}]")


def run(train_uids, eval_uids, data_dir, out, groups=None, num_leaves=DEFAULT_LEAVES,
        learning_rate=DEFAULT_LR, max_rounds=DEFAULT_MAX_ROUNDS, patience=R.PATIENCE, max_train_rows=None,
        workers=1, final=False, stop_after=None, result_dir=C.DEFAULT_RESULT_DIR):
    """Train on train_uids (or load <out>.model.txt), score eval_uids (resumable), write run_info and the
    comparison. Returns the run_info dict."""
    check_users(train_uids, eval_uids, final)
    import lightgbm as lgb

    cols = group_columns(groups)
    params = dict(R.LGB_PARAMS, num_leaves=num_leaves, learning_rate=learning_rate)
    data_dir, out = Path(data_dir), Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model_txt = Path(str(out) + ".model.txt")
    model_json = Path(str(out) + ".model.json")
    info_path = Path(str(out) + ".run_info.json")
    skipped = []
    train_no_side, eval_no_side = [], []

    if model_txt.exists() and model_json.exists():
        booster = lgb.Booster(model_file=str(model_txt))
        saved = json.loads(model_json.read_text(encoding="utf-8"))
        if saved["features"] != cols:
            raise RuntimeError("saved model was trained on other features; use a new --out")
        best_iter, info = saved["best_iter"], saved["info"]
        trained_now = False
    else:
        pool = R.TrainPool(hold=R.holdout_users(train_uids), max_rows=max_train_rows, n_feat=len(cols))
        for r in _map(_task_train, train_uids, data_dir, cols, workers):
            if "error" in r:
                skipped.append((r["uid"], r["error"]))
                continue
            if not r["side"]:
                train_no_side.append(r["uid"])
            pool.add(r["uid"], r["X"], r["y"])
        booster, best_iter, info = R.fit_from_pool(pool, max_rounds, patience, max_train_rows,
                                                   names=cols, params=params)
        booster.save_model(str(model_txt))
        model_json.write_text(json.dumps(dict(best_iter=int(best_iter), info=info, features=cols,
                                              params=params)), encoding="utf-8")
        trained_now = True

    done = R._load_done(out)
    pending = list(dict.fromkeys(u for u in eval_uids if u not in done))
    written, stopped = 0, False
    gen = _map(_task_eval, pending, data_dir, cols, workers)
    try:
        with open(out, "a", encoding="utf-8") as f:
            for r in gen:
                if "error" in r:
                    skipped.append((r["uid"], r["error"]))
                    continue
                if not r["side"]:
                    eval_no_side.append(r["uid"])
                p = np.clip(booster.predict(r["X"], num_iteration=best_iter), G.P_CLIP, 1 - G.P_CLIP)
                y = r["y"].astype(np.float64)
                line = {"metrics": R.user_metrics(y, p, r["meta"]), "user": int(r["uid"]),
                        "size": int(len(y))}
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
                f.flush()
                written += 1
                if stop_after is not None and written >= stop_after:
                    stopped = True
                    break
    finally:
        gen.close()
    if not stopped:
        R._sort_file(out)
    n_lines = sum(1 for x in out.read_text(encoding="utf-8").splitlines() if x.strip())
    cmp = compare_published(out, result_dir) if not stopped else None
    run_info = dict(
        groups=list(groups) if groups is not None else list(ALL_GROUPS), features=cols, params=params,
        best_iter=int(best_iter), info=info, trained_now=trained_now, final=final,
        train_users=sorted(int(u) for u in train_uids), eval_users=sorted(int(u) for u in eval_uids),
        eval_lines=n_lines, new_lines=written, stopped=stopped, skipped=skipped,
        train_users_without_side_tables=train_no_side, eval_users_without_side_tables=eval_no_side,
        max_rounds=max_rounds, patience=patience, max_train_rows=max_train_rows, out=str(out),
        comparison=cmp,
    )
    info_path.write_text(json.dumps(run_info, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in run_info.items() if k not in ("train_users", "eval_users", "comparison")},
                     indent=1, default=str))
    _print_comparison(cmp)
    return run_info


def main(argv=None):
    ap = argparse.ArgumentParser(description="SRS-Beat GBDT validation harness (task 6.2).")
    ap.add_argument("--data", default="data/anki10k", help="dir with revlogs/, cards/, decks/")
    ap.add_argument("--train_users", required=True, help="ids (comma separated) or a file of ids")
    ap.add_argument("--eval_users", required=True, help="ids (comma separated) or a file of ids")
    ap.add_argument("--out", required=True, help="output jsonl (resumable)")
    ap.add_argument("--groups", default=None, help=f"comma list of feature groups (default all): {ALL_GROUPS}")
    ap.add_argument("--max_rounds", type=int, default=DEFAULT_MAX_ROUNDS)
    ap.add_argument("--num_leaves", type=int, default=DEFAULT_LEAVES)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR, help="learning rate")
    ap.add_argument("--max_train_rows", type=int, default=None, help="cap training rows (seeded subsample)")
    ap.add_argument("--workers", type=int, default=1, help="parallel featurization processes")
    ap.add_argument("--download", action="store_true",
                    help="download revlogs, cards and decks of the requested users from HF first")
    ap.add_argument("--final", action="store_true", help="allow users 1-1000 (task 6.5 only)")
    args = ap.parse_args(argv)

    train_uids = R.load_ids(args.train_users)
    eval_uids = R.load_ids(args.eval_users)
    check_users(train_uids, eval_uids, args.final)          # before any download
    groups = None if args.groups is None else [g.strip() for g in args.groups.split(",") if g.strip()]
    group_columns(groups)                                    # validates the names early
    if args.download:
        want = sorted(set(train_uids) | set(eval_uids))
        _, missing_rev = D.download_revlogs(want, args.data)
        side = D.download_side_tables(want, args.data)
        print(f"download: revlogs missing {missing_rev}; side-table files missing {len(side['missing'])}")
    return run(train_uids, eval_uids, args.data, args.out, groups=groups, num_leaves=args.num_leaves,
               learning_rate=args.lr, max_rounds=args.max_rounds, max_train_rows=args.max_train_rows,
               workers=args.workers, final=args.final)


if __name__ == "__main__":
    main(sys.argv[1:])
