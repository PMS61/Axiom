"""Pooled GBDT driver for srs-benchmark (Phase 2.2), RWKV-style protocol.

One LightGBM model is trained on the scored rows of the train users. It is then evaluated on the
eval users' TimeSeriesSplit(n_splits=5) test rows, which are the rows srs-benchmark itself scores for
`--algo FSRS-6 --short` (the "without same-day reviews" setting).

Per user (exactly as srs-benchmark's script.process()):
  ds   = UserDataLoader(cfg).load_user_data(uid)       scored rows, with review_th as their key
  test = union of the test folds of TimeSeriesSplit(n_splits) over ds (in ds row order)

Features are past-only (gbdt.base_features, user_group="row"). They are computed on the user's whole
revlog stream, including same-day reviews, which srs-benchmark's --short keeps as history. Each row sees
only earlier reviews of its card and user. Features: gap, card, fsrs (FSRS-6 default-state replay) and
user (running recall rate and count). The `dt` and `lexeme` groups are not used.

Output: one jsonl line per eval user, {"metrics": {...}, "user": id, "size": n}, in srs-benchmark's format.
The metric functions are srs-benchmark's (utils.rmse_matrix, utils.mean_bias_error, sklearn log_loss,
roc_auc_score, root_mean_squared_error and statsmodels lowess for ICI, as in utils.evaluate).
Not computed: smECE, precision@90 and recall@90 (utils.evaluate needs the relplot package for smECE).

Usage (from the repo root):
  .venv/bin/python -m fsrs_benchmark.srsb.run_gbdt --data data/anki10k --train_users A.txt \
      --eval_users B.txt --out result/GBDT-A-B.jsonl [--max_train_rows N]
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fsrs_benchmark.models import gbdt as G
from fsrs_benchmark.srsb import ensure_srsb_on_path

# Frozen GBDT settings from fsrs_benchmark/models/gbdt.py (FROZEN_KWARGS: 15 leaves, lr 0.02).
FEATURE_GROUPS = ("gap", "card", "fsrs", "user")
FEATURE_COLS = [c for g in FEATURE_GROUPS for c in G.GROUP_FEATURES[g]]
LGB_PARAMS = dict(objective="binary", metric="binary_logloss", num_leaves=15, learning_rate=0.02,
                  min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  lambda_l2=1.0, num_threads=4, seed=0, deterministic=True, force_row_wise=True, verbose=-1)
MAX_ROUNDS = 2000
PATIENCE = 50
HOLDOUT_FRAC = 0.10          # share of the train users held out for early stopping
HOLDOUT_SEED = 0
SUBSAMPLE_SEED = 0
FALLBACK_ROUNDS = 200        # used only when there are too few train users for a holdout

_srs = None


def _srs_modules():
    """Import srs-benchmark's Config, UserDataLoader, TimeSeriesSplit and metric helpers (once)."""
    global _srs
    if _srs is None:
        ensure_srsb_on_path()
        from config import Config, create_parser
        from data_loader import UserDataLoader
        from sklearn.model_selection import TimeSeriesSplit
        import utils
        _srs = dict(Config=Config, create_parser=create_parser, UserDataLoader=UserDataLoader,
                    TimeSeriesSplit=TimeSeriesSplit, utils=utils)
    return _srs


def load_ids(spec: str) -> list[int]:
    """User ids from a comma/space separated string, a text file (one id per line) or a JSON list file."""
    p = Path(spec)
    if p.exists():
        text = p.read_text(encoding="utf-8")
        if p.suffix == ".json":
            return [int(x) for x in json.loads(text)]
        return [int(x) for x in text.replace(",", " ").split() if x.strip()]
    return [int(x) for x in spec.replace(",", " ").split() if x.strip()]


def review_stream(raw: pd.DataFrame, max_card_reviews: int) -> pd.DataFrame:
    """The user's review stream in chronological order, indexed by review_th (= raw file position + 1).

    Mirrors srs-benchmark's preprocessing that happens before history is built: invalid ratings are
    dropped, and reviews after the first max_card_reviews of a card (max_seq_len * 2) are dropped.
    Same-day reviews stay in the stream, because --short keeps them as history.
    """
    raw = raw.reset_index(drop=True)
    ev = pd.DataFrame({
        "event_id": np.arange(1, len(raw) + 1, dtype=np.int64),
        "card_id": raw["card_id"].to_numpy(np.int64),
        "rating": raw["rating"].to_numpy(np.int64),
        "elapsed_days": raw["elapsed_days"].to_numpy(np.int64),
        "day_offset": raw["day_offset"].to_numpy(np.int64),
    })
    ev = ev[ev["rating"].isin([1, 2, 3, 4])]
    ev = ev.sort_values(["card_id", "event_id"], kind="stable")
    ev["i"] = ev.groupby("card_id").cumcount() + 1
    ev = ev[ev["i"] <= max_card_reviews].drop(columns="i").sort_values("event_id")
    ev.index = pd.Index(ev["event_id"].to_numpy(), name=None)   # index = review_th; the column stays
    ev["user_id"] = 0                       # one user per call; the kernels need only a constant
    ev["y"] = (ev["rating"] >= 2).astype(np.int64)
    ev["is_first"] = ~ev["card_id"].duplicated()
    ev["dt_days"] = ev["day_offset"].diff().fillna(0).clip(lower=0).astype(float)
    return ev


def past_only_features(raw: pd.DataFrame, max_card_reviews: int) -> pd.DataFrame:
    """Past-only features for every review of the stream, indexed by review_th."""
    ev = review_stream(raw, max_card_reviews)
    return G.base_features(ev, span=20, user_group="row")[FEATURE_COLS]


def test_row_mask(n_rows: int, n_splits: int) -> np.ndarray:
    """Boolean mask over the dataset rows that belong to some TimeSeriesSplit test fold (srs-benchmark)."""
    TimeSeriesSplit = _srs_modules()["TimeSeriesSplit"]
    mask = np.zeros(n_rows, dtype=bool)
    for _, test_index in TimeSeriesSplit(n_splits=n_splits).split(np.arange(n_rows)):
        mask[test_index] = True
    return mask


def user_table(uid: int, loader, cfg, data_dir: Path) -> dict:
    """Features, labels and the test mask for one user. Raises if srs-benchmark cannot process the user."""
    ds = loader.load_user_data(uid)
    raw = pd.read_parquet(data_dir / "revlogs" / f"user_id={uid}")
    feats = past_only_features(raw, cfg.max_seq_len * 2)
    keys = ds["review_th"].to_numpy()
    if np.setdiff1d(keys, feats.index.to_numpy()).size:
        raise RuntimeError(f"user {uid}: scored rows missing from the feature stream")
    X = feats.reindex(keys)[FEATURE_COLS].to_numpy(np.float32)
    return dict(
        uid=uid,
        X=X,
        y=ds["y"].to_numpy(np.int64),
        test=test_row_mask(len(ds), cfg.n_splits),
        meta=pd.DataFrame({"y": ds["y"].to_numpy(np.float64),
                           "elapsed_days": ds["elapsed_days"].to_numpy(np.float64),
                           "i": ds["i"].to_numpy(np.float64),
                           "rmse_bins_lapse": ds["rmse_bins_lapse"].to_numpy(np.float64)}),
    )


def holdout_users(train_uids: list[int]) -> list[int]:
    """A fixed 10% of the train users (at least one) for early stopping. Empty if fewer than two users."""
    if len(train_uids) < 2:
        return []
    rng = np.random.default_rng(HOLDOUT_SEED)
    order = sorted(train_uids)
    n_hold = max(1, int(round(HOLDOUT_FRAC * len(order))))
    picked = rng.permutation(len(order))[:n_hold]
    return sorted(order[k] for k in picked)


class TrainPool:
    """Collects train rows user by user (float32, one chunk per user) and routes users to the fit or the
    early-stopping holdout. finalize() gathers the rows into one preallocated (m, F) float32 array. If
    max_rows is set and there are more rows, m = max_rows rows are drawn uniformly at random (seeded)."""

    def __init__(self, hold=(), max_rows=None):
        self.hold = set(hold)
        self.max_rows = max_rows
        self.parts = {"fit": [], "val": []}
        self.uids = {"fit": [], "val": []}

    def add(self, uid, X, y):
        kind = "val" if uid in self.hold else "fit"
        self.parts[kind].append((np.asarray(X, np.float32), np.asarray(y, np.int64)))
        self.uids[kind].append(uid)

    def n_rows(self, kind):
        return sum(len(y) for _, y in self.parts[kind])

    def finalize(self, kind, max_rows=None):
        parts = self.parts[kind]
        self.parts[kind] = []
        n = sum(len(y) for _, y in parts)
        F = len(FEATURE_COLS)
        keep = None
        m = n
        if max_rows is not None and n > max_rows:
            rng = np.random.default_rng(SUBSAMPLE_SEED)
            keep = np.sort(rng.choice(n, size=max_rows, replace=False))
            m = max_rows
        X_out = np.empty((m, F), np.float32)
        y_out = np.empty(m, np.int64)
        pos = 0
        lo = 0
        for X, y in parts:
            hi = lo + len(y)
            if keep is None:
                X_out[pos:pos + len(y)] = X
                y_out[pos:pos + len(y)] = y
                pos += len(y)
            else:
                a, b = np.searchsorted(keep, [lo, hi])
                sel = keep[a:b] - lo
                X_out[pos:pos + len(sel)] = X[sel]
                y_out[pos:pos + len(sel)] = y[sel]
                pos += len(sel)
            lo = hi
            del X, y
        del parts
        return X_out, y_out


def fit_from_pool(pool: TrainPool, max_rounds: int = MAX_ROUNDS, patience: int = PATIENCE, max_train_rows=None):
    """Train one LightGBM on the pool's fit rows, with early stopping on the holdout rows."""
    import lightgbm as lgb

    val_cap = None if max_train_rows is None else max(1, max_train_rows // 5)
    n_fit_users = len(pool.uids["fit"])
    n_val_users = len(pool.uids["val"])
    X_fit, y_fit = pool.finalize("fit", max_train_rows)
    dtr = lgb.Dataset(X_fit, y_fit, feature_name=FEATURE_COLS, free_raw_data=False)
    if n_val_users:
        X_val, y_val = pool.finalize("val", val_cap)
        dva = lgb.Dataset(X_val, y_val, reference=dtr)
        booster = lgb.train(LGB_PARAMS, dtr, max_rounds, valid_sets=[dva],
                            callbacks=[lgb.early_stopping(patience, verbose=False)])
        best_iter = booster.best_iteration or booster.current_iteration()
        del X_val, y_val, dva
    else:
        booster = lgb.train(LGB_PARAMS, dtr, min(max_rounds, FALLBACK_ROUNDS))
        best_iter = booster.current_iteration()
    info = dict(train_users=n_fit_users, holdout_users=n_val_users, fit_rows=int(len(y_fit)),
                best_iter=int(best_iter))
    return booster, best_iter, info


def train_model(tables: list[dict], max_train_rows: int | None = None, max_rounds: int = MAX_ROUNDS,
                patience: int = PATIENCE):
    """Train on a list of per-user tables ({uid, X, y}); holdout users come from holdout_users()."""
    pool = TrainPool(hold=holdout_users([t["uid"] for t in tables]), max_rows=max_train_rows)
    for t in tables:
        pool.add(t["uid"], t["X"], t["y"])
    return fit_from_pool(pool, max_rounds, patience, max_train_rows)


def user_metrics(y: np.ndarray, p: np.ndarray, meta: pd.DataFrame) -> dict:
    """srs-benchmark metric set (see the module docstring), rounded as in utils.evaluate."""
    from sklearn.metrics import log_loss, roc_auc_score, root_mean_squared_error
    from statsmodels.nonparametric.smoothers_lowess import lowess

    utils = _srs_modules()["utils"]
    df = meta.copy()
    df["p"] = p
    p_cal = lowess(y, p, it=0, delta=0.01 * (max(p) - min(p)), return_sorted=False)
    try:
        auc = round(roc_auc_score(y_true=y, y_score=p), 6)
        if not np.isfinite(auc):
            auc = None
    except ValueError:                       # one class only: srs-benchmark writes null
        auc = None
    return {
        "RMSE": round(float(root_mean_squared_error(y_true=y, y_pred=p)), 6),
        "LogLoss": round(float(log_loss(y, p, labels=[0, 1])), 6),
        "RMSE(bins)": round(float(utils.rmse_matrix(df)), 6),
        "ICI": round(float(np.mean(np.abs(p_cal - p))), 6),
        "AUC": auc,
        "MBE": round(float(utils.mean_bias_error(y, p)), 6),
    }


# ---- per-user work, run in the main process or in a worker pool ----------------------------------

_W: dict = {}


def _init_worker(data_dir: str) -> None:
    """Build srs-benchmark's Config and loader once per process."""
    S = _srs_modules()
    cfg = S["Config"](S["create_parser"]().parse_args(["--algo", "FSRS-6", "--short", "--data", data_dir]))
    if cfg.get_evaluation_file_name() != "FSRS-6-short":
        raise RuntimeError(f"unexpected srs-benchmark setting {cfg.get_evaluation_file_name()}")
    _W.clear()
    _W.update(cfg=cfg, loader=S["UserDataLoader"](cfg), data_dir=Path(data_dir))


def _task_train(uid: int) -> dict:
    try:
        t = user_table(uid, _W["loader"], _W["cfg"], _W["data_dir"])
        return dict(uid=uid, X=t["X"], y=t["y"])
    except Exception as e:                   # srs-benchmark skips these users too
        return dict(uid=uid, error=f"{type(e).__name__}: {str(e)[:100]}")


def _task_eval(uid: int) -> dict:
    try:
        t = user_table(uid, _W["loader"], _W["cfg"], _W["data_dir"])
        m = t["test"]
        if not m.any():
            return dict(uid=uid, error="no test rows")
        return dict(uid=uid, X=t["X"][m], y=t["y"][m], meta=t["meta"][m].reset_index(drop=True))
    except Exception as e:
        return dict(uid=uid, error=f"{type(e).__name__}: {str(e)[:100]}")


def _map_tasks(task, uids, data_dir: Path, workers: int):
    """Yield task(uid) for each uid, in order. workers > 1 uses a fork pool (featurization in parallel)."""
    if workers <= 1 or len(uids) < 2:
        _init_worker(str(data_dir))
        for uid in uids:
            yield task(uid)
        return
    ctx = mp.get_context("fork")
    with ctx.Pool(workers, initializer=_init_worker, initargs=(str(data_dir),)) as pool:
        yield from pool.imap(task, uids, chunksize=1)


def _load_done(out: Path) -> set[int]:
    """Users already written to `out`. A truncated last line (killed mid-write) is removed, and a duplicate
    user keeps its first line, so the file is always valid jsonl with one line per user."""
    if not out.exists():
        return set()
    good, seen, dropped = [], set(), False
    for raw in out.read_text(encoding="utf-8").splitlines(keepends=True):
        try:
            obj = json.loads(raw)
            uid = int(obj["user"])
        except (ValueError, KeyError, TypeError):
            dropped = True
            continue
        if uid in seen:
            dropped = True
            continue
        seen.add(uid)
        good.append(raw if raw.endswith("\n") else raw + "\n")
    if dropped or any(not r.endswith("\n") for r in good):
        out.write_text("".join(good), encoding="utf-8")
    return seen


def _sort_file(out: Path) -> None:
    lines = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x.strip()]
    lines.sort(key=lambda x: x["user"])
    tmp = out.with_suffix(out.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for x in lines:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    os.replace(tmp, out)


def run(train_uids: list[int], eval_uids: list[int], data_dir: Path, out: Path,
        max_train_rows: int | None = None, max_rounds: int = MAX_ROUNDS, patience: int = PATIENCE,
        workers: int = 1, stop_after: int | None = None) -> dict:
    """Train on train_uids (or load <out>.model.txt), then score eval_uids and append one line per user.

    Resumable: users already in `out` are skipped. Each line is flushed as soon as its user is scored.
    stop_after=N stops cleanly after N new lines (used to test resuming). The file is sorted by user at the end.
    """
    overlap = set(train_uids) & set(eval_uids)
    if overlap:
        raise ValueError(f"train and eval users overlap: {sorted(overlap)[:10]}")
    import lightgbm as lgb

    data_dir, out = Path(data_dir), Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model_txt = Path(str(out) + ".model.txt")
    model_json = Path(str(out) + ".model.json")
    skipped = []

    if model_txt.exists() and model_json.exists():
        booster = lgb.Booster(model_file=str(model_txt))
        saved = json.loads(model_json.read_text(encoding="utf-8"))
        best_iter, info = saved["best_iter"], saved["info"]
        trained_now = False
    else:
        pool = TrainPool(hold=holdout_users(train_uids), max_rows=max_train_rows)
        for r in _map_tasks(_task_train, train_uids, data_dir, workers):
            if "error" in r:
                skipped.append((r["uid"], r["error"]))
            else:
                pool.add(r["uid"], r["X"], r["y"])
        booster, best_iter, info = fit_from_pool(pool, max_rounds, patience, max_train_rows)
        booster.save_model(str(model_txt))
        model_json.write_text(json.dumps(dict(best_iter=int(best_iter), info=info)), encoding="utf-8")
        trained_now = True

    done = _load_done(out)
    pending = list(dict.fromkeys(u for u in eval_uids if u not in done))
    written = 0
    stopped = False
    gen = _map_tasks(_task_eval, pending, data_dir, workers)
    try:
        with open(out, "a", encoding="utf-8") as f:
            for r in gen:
                if "error" in r:
                    skipped.append((r["uid"], r["error"]))
                    continue
                p = booster.predict(r["X"], num_iteration=best_iter)
                p = np.clip(p, G.P_CLIP, 1 - G.P_CLIP)
                y = r["y"].astype(np.float64)
                line = {"metrics": user_metrics(y, p, r["meta"]), "user": int(r["uid"]), "size": int(len(y))}
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
                written += 1
                if stop_after is not None and written >= stop_after:
                    stopped = True
                    break
    finally:
        gen.close()
    if not stopped:
        _sort_file(out)
    n_lines = sum(1 for x in out.read_text(encoding="utf-8").splitlines() if x.strip())
    return dict(info, trained_now=trained_now, eval_users=n_lines, new_lines=written,
                skipped=skipped, stopped=stopped, out=str(out))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Pooled GBDT for srs-benchmark (FSRS-6-short protocol).")
    ap.add_argument("--data", required=True, help="srs-benchmark data dir containing revlogs/user_id=<u>/")
    ap.add_argument("--train_users", required=True, help="ids (comma separated) or a file of ids")
    ap.add_argument("--eval_users", required=True, help="ids (comma separated) or a file of ids")
    ap.add_argument("--out", required=True, help="output jsonl path (resumable; model at <out>.model.txt)")
    ap.add_argument("--max_train_rows", type=int, default=None, help="cap training rows at N (seeded subsample)")
    ap.add_argument("--workers", type=int, default=1, help="parallel featurization processes (Kaggle: 4)")
    ap.add_argument("--stop_after", type=int, default=None, help="stop cleanly after N new eval lines (testing)")
    args = ap.parse_args(argv)

    train_uids = load_ids(args.train_users)
    eval_uids = load_ids(args.eval_users)
    if set(train_uids) & set(eval_uids):
        raise SystemExit("train and eval users overlap")
    summary = run(train_uids, eval_uids, Path(args.data), Path(args.out), args.max_train_rows,
                  workers=args.workers, stop_after=args.stop_after)
    print(json.dumps(summary, indent=1, default=str))
    return summary


if __name__ == "__main__":
    main(sys.argv[1:])
