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

# Give direct file execution the same imports as invocation from the repository.
if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fsrs_benchmark import data as D
from fsrs_benchmark.models import gbdt as G
from fsrs_benchmark.srsb import compare as C
from fsrs_benchmark.srsb import run_gbdt as R
from fsrs_benchmark.srsb.features_plus import PLUS_COLS, PLUS_GROUPS, plus_features

TEST_LO, TEST_HI = 1, 1000             # the test set: never used here unless final=True
VAL_LO, VAL_HI = 8001, 8300            # validation users: never train users (SRS-Beat), unless final=True
SUBSAMPLE_SEED = 0
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
    """Raise on an empty side or a train/eval overlap (always); on a user in 1-1000 (test set) on either side,
    or a train user in 8001-8300 (validation), unless final=True."""
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
        bad_val = sorted(u for u in train if VAL_LO <= u <= VAL_HI)
        if bad_val:
            raise ValueError(f"train users in {VAL_LO}-{VAL_HI} are validation users and may not be used "
                             f"for training (final=True only): {bad_val[:10]}")


def _read_opt(path: Path):
    """The side table of one user, or None when it is absent or empty (no cards on the Hub). `path` is the
    user's directory (cards/user_id=N, which holds data.parquet) or the parquet file itself."""
    p = Path(path)
    if p.is_dir():
        p = p / "data.parquet"
    if not p.is_file():
        return None
    df = pd.read_parquet(p)
    return df if len(df) else None


ALL_COLS = R.FEATURE_COLS + PLUS_COLS
CODE_FILES = ("srsb/features_plus.py", "models/gbdt.py", "srsb/run_gbdt.py")   # relative to fsrs_benchmark/
_CODE_HASH = None
CACHE_VERSION = 2        # bump when the cached tables change meaning (v1 tables had no cards/decks: NaN sib/deck)


def code_hash():
    """sha256 of the feature code (features_plus, gbdt, run_gbdt): a cache entry is valid only for this code."""
    global _CODE_HASH
    if _CODE_HASH is None:
        import hashlib
        root = Path(__file__).resolve().parents[3] / "fsrs_benchmark"
        h = hashlib.sha256()
        h.update(json.dumps([ALL_COLS, CACHE_VERSION]).encode())
        for f in CODE_FILES:
            code = Path(R.__file__) if f == "srsb/run_gbdt.py" else root / f
            h.update(code.read_bytes())
        _CODE_HASH = h.hexdigest()[:16]
    return _CODE_HASH


def cache_dir_for(data_dir) -> Path:
    return Path(data_dir) / "feat_cache" / code_hash()


def _full_table(uid, loader, cfg, data_dir: Path, cache_dir=None):
    """All ALL_COLS features of one user's scored rows (float32), labels, test mask, side flag and meta.
    With cache_dir, a saved table for this user and feature code is loaded instead of rebuilt."""
    path = None if cache_dir is None else Path(cache_dir) / f"u{uid}.npz"
    if path is not None and path.is_file():
        with np.load(path) as z:
            return dict(X=z["X"], y=z["y"], test=z["test"], side=bool(z["side"]),
                        meta=pd.DataFrame({k: z["meta_" + k] for k in META_COLS}))
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
    out = dict(
        X=feats.reindex(keys)[ALL_COLS].to_numpy(np.float32),
        y=ds["y"].to_numpy(np.int64),
        test=R.test_row_mask(len(ds), cfg.n_splits),
        side=cards is not None and decks is not None,
        meta=pd.DataFrame({"y": ds["y"].to_numpy(np.float64),
                           "elapsed_days": ds["elapsed_days"].to_numpy(np.float64),
                           "i": ds["i"].to_numpy(np.float64),
                           "rmse_bins_lapse": ds["rmse_bins_lapse"].to_numpy(np.float64)}),
    )
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as f:
            np.savez(f, X=out["X"], y=out["y"], test=out["test"], side=np.array(out["side"]),
                     **{"meta_" + k: out["meta"][k].to_numpy() for k in META_COLS})
        os.replace(tmp, path)
    return out


META_COLS = ("y", "elapsed_days", "i", "rmse_bins_lapse")


def user_table(uid, loader, cfg, data_dir: Path, cols, cache_dir=None):
    """Features (cols, float32) and labels of one user's scored rows, plus the test mask and meta."""
    full = _full_table(uid, loader, cfg, data_dir, cache_dir)
    idx = [ALL_COLS.index(c) for c in cols]
    return dict(X=full["X"][:, idx], y=full["y"], test=full["test"], side=full["side"], meta=full["meta"])


def _init(data_dir: str, cols, cache_dir=None) -> None:
    S = R._srs_modules()
    cfg = S["Config"](S["create_parser"]().parse_args(["--algo", "FSRS-6", "--short", "--data", data_dir]))
    if cfg.get_evaluation_file_name() != "FSRS-6-short":
        raise RuntimeError(f"unexpected srs-benchmark setting {cfg.get_evaluation_file_name()}")
    _W.clear()
    _W.update(cfg=cfg, loader=S["UserDataLoader"](cfg), data_dir=Path(data_dir), cols=list(cols),
              cache_dir=None if cache_dir is None else str(cache_dir))


def _task_count(uid):
    """Number of scored rows of a user (only the labels are read; no features)."""
    try:
        return dict(uid=uid, n=int(len(_W["loader"].load_user_data(uid))))
    except Exception as e:
        return dict(uid=uid, error=f"{type(e).__name__}: {str(e)[:100]}")


def _task_train(item):
    """item = (uid, quota): the user's features and labels, subsampled to `quota` rows (seeded per user)."""
    uid, quota = item
    try:
        t = user_table(uid, _W["loader"], _W["cfg"], _W["data_dir"], _W["cols"], _W["cache_dir"])
        X, y = t["X"], t["y"]
        if quota < len(y):
            rng = np.random.default_rng([SUBSAMPLE_SEED, int(uid)])
            keep = np.sort(rng.choice(len(y), size=int(quota), replace=False))
            X, y = X[keep], y[keep]
        return dict(uid=uid, X=X, y=y, side=t["side"])
    except Exception as e:                  # srs-benchmark skips these users too
        return dict(uid=uid, error=f"{type(e).__name__}: {str(e)[:100]}")


def _task_eval(uid):
    try:
        t = user_table(uid, _W["loader"], _W["cfg"], _W["data_dir"], _W["cols"], _W["cache_dir"])
        m = t["test"]
        if not m.any():
            return dict(uid=uid, error="no test rows")
        return dict(uid=uid, X=t["X"][m], y=t["y"][m], meta=t["meta"][m].reset_index(drop=True),
                    side=t["side"])
    except Exception as e:
        return dict(uid=uid, error=f"{type(e).__name__}: {str(e)[:100]}")


def _map(task, items, data_dir: Path, cols, workers: int, cache_dir=None):
    """Yield task(item) in order; workers > 1 uses a fork pool (featurization in parallel)."""
    items = list(items)
    if workers <= 1 or len(items) < 2:
        _init(str(data_dir), cols, cache_dir)
        for it in items:
            yield task(it)
        return
    ctx = mp.get_context("fork")
    with ctx.Pool(workers, initializer=_init, initargs=(str(data_dir), list(cols), cache_dir)) as pool:
        yield from pool.imap(task, items, chunksize=1)


def plan_quotas(counts, max_train_rows=None, rows_per_user=None):
    """Rows kept per train user {uid: q}. q = min(scored rows, rows_per_user); if the total is above
    max_train_rows, each user gets floor(max_train_rows * q / total), so the sum never exceeds the cap."""
    q = {}
    for u, n in counts.items():
        m = min(int(n), int(rows_per_user)) if rows_per_user else int(n)
        if m > 0:
            q[int(u)] = m
    if max_train_rows is not None:
        total = sum(q.values())
        if total > max_train_rows:
            q = {u: min(m, int(np.floor(max_train_rows * m / total))) for u, m in q.items()}
            q = {u: m for u, m in q.items() if m > 0}
    return dict(sorted(q.items()))


def peak_ram_estimate(n_rows, n_feat):
    """Bytes for the train arrays plus LightGBM's binned copy: float32 features, int64 labels, ~1 byte per
    feature for the bins. Excludes the Python process and the featurization workers."""
    return int(n_rows) * (4 * int(n_feat) + 8 + int(n_feat))


class PresizedPool:
    """Train rows written straight into arrays sized from the plan, so the peak is the arrays themselves
    (no list of parts). Same interface as run_gbdt.TrainPool for fit_from_pool."""

    def __init__(self, hold, cap, n_feat):
        self.hold = set(hold)
        self.n_feat = int(n_feat)
        self.cap = {k: int(cap.get(k, 0)) for k in ("fit", "val")}
        self.X = {k: np.empty((self.cap[k], self.n_feat), np.float32) for k in self.cap}
        self.Y = {k: np.empty(self.cap[k], np.int64) for k in self.cap}
        self.used = {k: 0 for k in self.cap}
        self.uids = {"fit": [], "val": []}

    def add(self, uid, X, y):
        n = len(y)
        if n == 0:
            return
        kind = "val" if uid in self.hold else "fit"
        s = self.used[kind]
        if s + n > self.cap[kind]:
            raise RuntimeError(f"{kind} rows exceed the plan ({s + n} > {self.cap[kind]})")
        self.X[kind][s:s + n] = X
        self.Y[kind][s:s + n] = y
        self.used[kind] = s + n
        self.uids[kind].append(uid)

    def n_rows(self, kind):
        return self.used[kind]

    def finalize(self, kind, max_rows=None):
        X, y = self.X[kind][:self.used[kind]], self.Y[kind][:self.used[kind]]
        if max_rows is not None and len(y) > max_rows:
            rng = np.random.default_rng(R.SUBSAMPLE_SEED)
            keep = np.sort(rng.choice(len(y), size=max_rows, replace=False))
            X, y = X[keep], y[keep]
        return X, y


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


def _fingerprint(cols, params, groups, train_uids, max_rounds, patience, max_train_rows, rows_per_user):
    """Everything a saved model depends on. A run may reuse a saved model only when all of it matches."""
    fp = dict(features=list(cols), groups=list(groups), params=params, max_rounds=max_rounds,
              patience=patience, max_train_rows=max_train_rows, rows_per_user=rows_per_user,
              train_users=sorted(int(u) for u in train_uids))
    return json.loads(json.dumps(fp))


def run(train_uids, eval_uids, data_dir, out, groups=None, num_leaves=DEFAULT_LEAVES,
        learning_rate=DEFAULT_LR, max_rounds=DEFAULT_MAX_ROUNDS, patience=R.PATIENCE, max_train_rows=None,
        rows_per_user=None, workers=1, final=False, stop_after=None, result_dir=C.DEFAULT_RESULT_DIR,
        min_data_in_leaf=R.LGB_PARAMS["min_data_in_leaf"], feature_fraction=R.LGB_PARAMS["feature_fraction"],
        lambda_l2=R.LGB_PARAMS["lambda_l2"], cache=True):
    """Train on train_uids (or reuse <out>.model.txt when the fingerprint matches), score eval_uids
    (resumable), write run_info and the comparison. Returns the run_info dict.
    cache=True keeps per-user feature tables under <data>/feat_cache/<code hash>/ (skips featurization)."""
    check_users(train_uids, eval_uids, final)
    import lightgbm as lgb

    t0 = time.time()
    cache_dir = cache_dir_for(data_dir) if cache else None
    cols = group_columns(groups)
    group_names = [g for g in ALL_GROUPS if g in (ALL_GROUPS if groups is None else tuple(groups))]
    params = dict(R.LGB_PARAMS, num_leaves=num_leaves, learning_rate=learning_rate,
                  min_data_in_leaf=min_data_in_leaf, feature_fraction=feature_fraction, lambda_l2=lambda_l2)
    data_dir, out = Path(data_dir), Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model_txt = Path(str(out) + ".model.txt")
    model_json = Path(str(out) + ".model.json")
    info_path = Path(str(out) + ".run_info.json")
    fp = _fingerprint(cols, params, group_names, train_uids, max_rounds, patience, max_train_rows, rows_per_user)
    skipped, no_side = [], {"train": [], "eval": []}
    plan = {}

    if model_txt.exists() and model_json.exists():
        saved = json.loads(model_json.read_text(encoding="utf-8"))
        if saved.get("fingerprint") != fp:
            diff = sorted(k for k in set(fp) | set(saved.get("fingerprint", {}))
                          if fp.get(k) != saved.get("fingerprint", {}).get(k))
            raise RuntimeError(f"saved model in {model_txt} was trained with other settings ({diff}); "
                               "use a new --out")
        booster = lgb.Booster(model_file=str(model_txt))
        best_iter, info = saved["best_iter"], saved["info"]
        plan = saved.get("plan", {})
        trained_now = False
    else:
        counts = {}
        for r in _map(_task_count, train_uids, data_dir, cols, workers, cache_dir):
            if "error" in r:
                skipped.append((r["uid"], r["error"]))
            else:
                counts[r["uid"]] = r["n"]
        quotas = plan_quotas(counts, max_train_rows, rows_per_user)
        planned = list(quotas)
        hold = set(R.holdout_users(planned))
        cap = {"val": sum(quotas[u] for u in planned if u in hold),
               "fit": sum(quotas[u] for u in planned if u not in hold)}
        n_rows_plan = cap["fit"] + cap["val"]
        plan = dict(scored_rows_total=int(sum(counts.values())), train_users_counted=len(counts),
                    train_users_planned=len(planned), rows_planned=int(n_rows_plan),
                    n_feat=len(cols), peak_ram_estimate_bytes=peak_ram_estimate(n_rows_plan, len(cols)))
        pool = PresizedPool(hold, cap, len(cols))
        for r in _map(_task_train, [(u, quotas[u]) for u in planned], data_dir, cols, workers, cache_dir):
            if "error" in r:
                skipped.append((r["uid"], r["error"]))
                continue
            if not r["side"]:
                no_side["train"].append(r["uid"])
            pool.add(r["uid"], r["X"], r["y"])
        booster, best_iter, info = R.fit_from_pool(pool, max_rounds, patience, max_train_rows,
                                                   names=cols, params=params)
        booster.save_model(str(model_txt))
        model_json.write_text(json.dumps(dict(best_iter=int(best_iter), info=info, plan=plan,
                                              fingerprint=fp)), encoding="utf-8")
        trained_now = True

    done = R._load_done(out)
    pending = list(dict.fromkeys(u for u in eval_uids if u not in done))
    written, stopped = 0, False
    gen = _map(_task_eval, pending, data_dir, cols, workers, cache_dir)
    try:
        with open(out, "a", encoding="utf-8") as f:
            for r in gen:
                if "error" in r:
                    skipped.append((r["uid"], r["error"]))
                    continue
                if not r["side"]:
                    no_side["eval"].append(r["uid"])
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
        fingerprint=fp, plan=plan, info=info, best_iter=int(best_iter), trained_now=trained_now,
        features=cols, groups=group_names, params=params, max_rounds=max_rounds, patience=patience,
        max_train_rows=max_train_rows, rows_per_user=rows_per_user, cache=cache,
        elapsed_min=round((time.time() - t0) / 60, 2),
        final=final, train_users=sorted(int(u) for u in train_uids), eval_users=sorted(int(u) for u in eval_uids),
        eval_lines=n_lines, new_lines=written, stopped=stopped, skipped=skipped,
        train_users_without_side_tables=no_side["train"], eval_users_without_side_tables=no_side["eval"],
        out=str(out), comparison=cmp,
    )
    info_path.write_text(json.dumps(run_info, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in run_info.items()
                      if k not in ("train_users", "eval_users", "comparison", "fingerprint")},
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
    ap.add_argument("--min_data_in_leaf", type=int, default=R.LGB_PARAMS["min_data_in_leaf"])
    ap.add_argument("--feature_fraction", type=float, default=R.LGB_PARAMS["feature_fraction"])
    ap.add_argument("--lambda_l2", type=float, default=R.LGB_PARAMS["lambda_l2"])
    ap.add_argument("--no_cache", action="store_true", help="do not read or write the feature cache")
    ap.add_argument("--max_train_rows", type=int, default=None,
                    help="cap total training rows; each user keeps a share proportional to its rows")
    ap.add_argument("--rows_per_user", type=int, default=None, help="keep at most N rows per train user (seeded)")
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
               rows_per_user=args.rows_per_user, workers=args.workers, final=args.final,
               min_data_in_leaf=args.min_data_in_leaf, feature_fraction=args.feature_fraction,
               lambda_l2=args.lambda_l2, cache=not args.no_cache)


if __name__ == "__main__":
    main(sys.argv[1:])
