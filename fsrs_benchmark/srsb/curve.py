"""GBDT-Curve (Phase 8). Spec: paper/docs/CURVE_SPEC.md.

A Curve model fixes the card's state at its previous review j and outputs recall as a function of the elapsed
time t only. For a scored row i (review_th), j is the previous review of the same card (stream index).

Every ALL_COLS feature (run_gbdt FEATURE_COLS + features_plus PLUS_COLS, 78 columns) is in exactly one class:
  TIME    functions of t and of the card state after j, taken from row i. Monotone in t.
  CARD_I  depend only on the card's own reviews up to and including j, taken from row i (the card kernels read
          the card state before row i, which is the state after j).
  FROM_J  can change because of other cards' reviews (user, day, deck, sibling, bucket statistics), so they are
          taken from row j (the value just before review j, which is past-only).
  DROP    mix t with statistics after j, or are constant on scored rows.
One derived column, since_first_j = since_first(i) - el(i), is the card age at j (CARD_I). It replaces since_first.

Curve GBDT: LightGBM with the frozen settings of sweep_a_l63_side and monotone constraints on TIME.
Curve online layer: per-user logistic layer on [z, 1, ewm50_j, rate_today_j] with no time terms, w0 = [1,0,0,0],
w[0] >= 0 after every update. The curve emitted at review j uses the weights after the last update at a scored
row with stream index <= j.

Several training variants can run in one invocation with --variants (comma list of base, adv, nomono,
adv_lr02); the train rows are built once and each variant writes <out stem>_<variant>.jsonl and
<out stem>_<variant>_online.jsonl. Default --variants base.

Usage (repo root):
  PYTHONPATH=. .venv/bin/python -m fsrs_benchmark.srsb.curve --data data/anki10k \
      --train_users 2001-2300 --eval_users 8001-8100 --out result/curve/curve.jsonl --workers 4
Writes <out> (curve) and <out stem>_online.jsonl (curve+online), srs-benchmark style. Users 1-1000 are refused
unless --final. --quick uses the first 3 train and first 2 eval users.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
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
from fsrs_benchmark.srsb.features_plus import DECAY, FACTOR, plus_features

# ---- column classes (every ALL_COLS column in exactly one list) --------------------------------------------

TIME_COLS = [
    "el",            # gap in days = t
    "log_el",        # log1p(t)
    "el_frac",       # t in days from elapsed_seconds (features_plus)
    "log_el_frac",   # log1p(el_frac)
    "f_R",           # FSRS-6 default recall at t from the card state after j (fsrs group)
    "f_logitR",      # logit of f_R
    "f_el_over_S",   # t / S, S = card state after j
    "f_R_secs",      # FSRS-6 default recall at the fractional t (features_plus)
]
MONO_NEG = ("el", "log_el", "el_frac", "log_el_frac", "f_el_over_S")   # recall falls as t grows
MONO_POS = ("f_R", "f_logitR", "f_R_secs")                              # recall rises with these

CARD_I_COLS = [
    "card_n", "card_lapses", "card_rate",              # counts and rates of the card's reviews before i (up to j)
    "g1", "g2", "g3", "y1", "y2", "y3",                # ratings and labels of the card's last three reviews (g1, y1 = j)
    "gap1", "gap2", "gap3",                            # gaps before j (the card's own history)
    "f_S", "f_D",                                      # FSRS-6 card state after j (card replay in _card_kernel)
    "card_max_gap",                                    # longest gap of the card up to j (card cummax, shift 1)
    "card_ewm3",                                       # card's EWM of labels up to j (features_plus, grouped by card)
    "card_mean_loggap",                                # mean log gap of the card's reviews up to j (grouped by card)
    "card_r4", "card_r5", "card_gap4", "card_gap5",    # card's own rating / gap 4 and 5 reviews back (grouped by card)
    "dur_card_last",                                   # log duration of the card's previous review (j)
    "dur_card_mean",                                   # mean log duration of the card's reviews up to j (grouped by card)
    "rc1", "rc2", "rc3", "rc4",                        # counts of each rating over the card's reviews (grouped by card)
]
SINCE_FIRST_J = "since_first_j"    # = since_first(i) - el(i) = day_j - first_day (checked equal on user 2001)
CARD_I_DERIVED = [SINCE_FIRST_J]

FROM_J_COLS = [
    # user (running statistics over all cards)
    "u_n", "u_rate", "u_rate_inf", "u_recent", "u_day_n", "u_since_last",
    # same-day statistics (day_offset of i vs j, all cards)
    "n_today", "n_first_today", "rate_today", "lapses_today", "u_last5",
    # durations that use other cards or the user
    "dur_user_mean", "dur_card_last_rel", "dur_user_prev",
    # state: row j's review type. Doubtful: the card's type at i is not provably a function of reviews up to j,
    # so the conservative choice is row j's value (known before review j).
    "state",
    # sibling cards of the same note
    "sib_n", "sib_last_y", "sib_days_since", "sib_rate",
    # deck, preset and parent deck
    "deck_n", "deck_rate", "preset_n", "preset_rate", "parent_n", "parent_rate",
    # user memory statistics (EWMs, residuals, buckets keyed by gap / count / previous rating / state)
    "ewm10", "ewm50", "ewm200", "ewm1000", "resid_mean", "resid_ewm100", "resid_gap_mean",
    "cnt_bkt_rate", "cnt_bkt_n", "g1_rate", "g1_n", "state_rate", "state_n",
]

DROP_COLS = {
    "same_day": "always 0 on scored rows (el < 1 is a function of t)",
    "since_first": "replaced by since_first_j (card age at j)",
    "rows_since_card": "counts stream rows after j",
    "gap_bkt_rate": "keyed by t's gap bucket; user statistics include rows after j",
    "gap_bkt_n": "keyed by t's gap bucket; user statistics include rows after j",
}

CURVE_COLS = TIME_COLS + CARD_I_COLS + CARD_I_DERIVED + FROM_J_COLS      # 74 model features
MONOTONE = [-1 if c in MONO_NEG else (1 if c in MONO_POS else 0) for c in CURVE_COLS]
EWM50_IX = CURVE_COLS.index("ewm50")
RATE_IX = CURVE_COLS.index("rate_today")

CACHE_VERSION = 1
CODE_FILES = ("srsb/curve.py",)           # relative to fsrs_benchmark/; the feature code hash is added too
_CODE_HASH = None

# frozen GBDT settings (= results/srsb_beat/sweep_a_l63_side, run_info "params")
CURVE_PARAMS = dict(R.LGB_PARAMS, num_leaves=63, learning_rate=0.05, min_data_in_leaf=200,
                    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                    monotone_constraints=MONOTONE)
DEFAULT_MAX_ROUNDS = 10000
DEFAULT_MAX_TRAIN_ROWS = 12_000_000

# online layer (fixed, not tuned)
ONLINE_LR = 0.003
ONLINE_LAM = 1e-3
W0 = np.array([1.0, 0.0, 0.0, 0.0])

_W: dict = {}


# ---- classes and helpers -----------------------------------------------------------------------------------

def column_classes():
    """{class: [columns]} for the 78 ALL_COLS (the derived since_first_j is listed under CARD_I)."""
    return dict(TIME=list(TIME_COLS), CARD_I=list(CARD_I_COLS) + CARD_I_DERIVED,
                FROM_J=list(FROM_J_COLS), DROP=list(DROP_COLS))


def time_features(el, el_frac, f_S):
    """The 8 TIME columns for a gap `el` (days), fractional gap `el_frac` (days) and card state S after j.
    Same formulas as gbdt._card_kernel / base_features and features_plus (used by the monotone test)."""
    el = np.asarray(el, dtype=float)
    ef = np.asarray(el_frac, dtype=float)
    S = np.asarray(f_S, dtype=float)
    t = np.maximum(el, 0.0)
    rec = (1.0 + FACTOR * t / S) ** (-DECAY)
    rc = np.clip(rec, 1e-6, 1 - 1e-6)
    return dict(el=el, log_el=np.log1p(np.clip(el, 0, None)), el_frac=ef, log_el_frac=np.log1p(ef),
                f_R=rec, f_logitR=np.log(rc / (1 - rc)), f_el_over_S=t / S,
                f_R_secs=(1.0 + FACTOR * np.maximum(ef, 0.0) / S) ** (-DECAY))


def previous_rows(st):
    """Series over the stream index: stream index of each row's previous review of the same card (NaN if first)."""
    pos = pd.Series(st.index.to_numpy(np.int64), index=st.index)
    return pos.groupby(st["card_id"].to_numpy()).shift(1)


def curve_matrix(feats, st, keys):
    """Curve features of the scored rows `keys` (stream indices, increasing).

    feats: full feature frame (ALL_COLS) indexed by stream index; st: review_stream frame.
    Returns X (float32, len(keys) x len(CURVE_COLS)), prev (int64, stream index of j) and th (= keys, int64).
    TIME and CARD_I come from row i, FROM_J from row j.
    """
    keys = np.asarray(keys, dtype=np.int64)
    prev = previous_rows(st).reindex(keys).to_numpy(float)
    if np.isnan(prev).any():
        raise ValueError("a scored row has no previous review of its card (first reviews are never scored)")
    prev = prev.astype(np.int64)
    at_i = feats.reindex(keys)
    at_j = feats.reindex(prev)
    cols = {}
    for c in TIME_COLS + CARD_I_COLS:
        cols[c] = at_i[c].to_numpy(float)
    cols[SINCE_FIRST_J] = at_i["since_first"].to_numpy(float) - at_i["el"].to_numpy(float)
    for c in FROM_J_COLS:
        cols[c] = at_j[c].to_numpy(float)
    X = pd.DataFrame(cols)[CURVE_COLS].to_numpy(np.float32)
    return X, prev, keys


def code_hash():
    """Cache key: curve.py, the feature code (V.code_hash) and the column list."""
    global _CODE_HASH
    if _CODE_HASH is None:
        root = Path(__file__).resolve().parents[1]
        h = hashlib.sha256()
        h.update(json.dumps([CURVE_COLS, CACHE_VERSION, V.code_hash()]).encode())
        for f in CODE_FILES:
            h.update((root / f).read_bytes())
        _CODE_HASH = h.hexdigest()[:16]
    return _CODE_HASH


def load_users(spec):
    """User ids from a file (as run_gbdt.load_ids) or from a comma/space list with ranges such as 2001-2300."""
    if Path(spec).exists():
        return R.load_ids(spec)
    out = []
    for tok in spec.replace(",", " ").split():
        if "-" in tok:
            a, b = tok.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(tok))
    return out


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


# ---- online layer ------------------------------------------------------------------------------------------

def online_design(z, ewm50_j, rate_j):
    """x = [z, 1, (ewm50_j - 0.5) * 4, (rate_today_j - 0.5) * 4]; NaN -> 0.5 (so the term is 0)."""
    e = (np.nan_to_num(np.asarray(ewm50_j, float), nan=0.5) - 0.5) * 4.0
    r = (np.nan_to_num(np.asarray(rate_j, float), nan=0.5) - 0.5) * 4.0
    z = np.asarray(z, float)
    return np.column_stack([z, np.ones_like(z), e, r])


@njit(cache=True)
def _online_kernel(D, y, w0, lr, lam):
    """Scored rows in order: one SGD step per row on the log loss, pull toward w0, w[0] clipped at 0.
    Returns the weight vector after each row's update."""
    n, d = D.shape
    w = w0.copy()
    out = np.empty((n, d))
    for i in range(n):
        z = 0.0
        for j in range(d):
            z += w[j] * D[i, j]
        pi = 1.0 / (1.0 + np.exp(-z))
        g = pi - y[i]
        for j in range(d):
            w[j] -= lr * (g * D[i, j] + lam * (w[j] - w0[j]))
        if w[0] < 0.0:
            w[0] = 0.0
        for j in range(d):
            out[i, j] = w[j]
    return out


def online_weights(D, y, th, prev_th):
    """Weights used to predict each scored row i: the weights after the last update at a scored row with
    stream index <= prev_th[i] (w0 if there is none). th must be increasing."""
    after = _online_kernel(np.ascontiguousarray(D, dtype=np.float64), np.asarray(y, dtype=np.float64),
                           W0.copy(), ONLINE_LR, ONLINE_LAM)
    pos = np.searchsorted(np.asarray(th), np.asarray(prev_th), side="right") - 1
    W = after[np.clip(pos, 0, None)].copy()
    W[pos < 0] = W0
    return W


def online_p(W, D):
    """Curve+online probability of each row: sigmoid(W_i . x_i)."""
    return _sigmoid((W * D).sum(axis=1))


def curve_online(z, ewm50_j, rate_j, y, th, prev_th):
    """Curve+online probabilities of the scored rows (inputs in stream order)."""
    D = online_design(z, ewm50_j, rate_j)
    return online_p(online_weights(D, y, th, prev_th), D)


# ---- per-user tables (cached) ----------------------------------------------------------------------------------

def _table(uid, loader, cfg, data_dir: Path, cache_dir=None):
    """All scored rows of one user: X (curve features), y, test mask, th, prev_th, side flag and meta."""
    path = None if cache_dir is None else Path(cache_dir) / f"u{uid}.npz"
    if path is not None and path.is_file():
        with np.load(path) as z:
            return dict(X=z["X"], y=z["y"], test=z["test"], th=z["th"], prev_th=z["prev_th"],
                        side=bool(z["side"]), meta=pd.DataFrame({k: z["meta_" + k] for k in V.META_COLS}))
    ds = loader.load_user_data(uid)
    raw = pd.read_parquet(data_dir / "revlogs" / f"user_id={uid}")
    st = R.review_stream(raw, cfg.max_seq_len * 2)
    cards = V._read_opt(data_dir / "cards" / f"user_id={uid}")
    decks = V._read_opt(data_dir / "decks" / f"user_id={uid}")
    base = G.base_features(st, span=20, user_group="row")[R.FEATURE_COLS]
    feats = pd.concat([base, plus_features(st, cards, decks)], axis=1)
    keys = ds["review_th"].to_numpy().astype(np.int64)
    if np.setdiff1d(keys, st.index.to_numpy()).size:
        raise RuntimeError(f"user {uid}: scored rows missing from the feature stream")
    if (np.diff(keys) <= 0).any():
        raise RuntimeError(f"user {uid}: scored rows are not in stream order")
    X, prev, th = curve_matrix(feats, st, keys)
    out = dict(X=X, y=ds["y"].to_numpy(np.int64), test=R.test_row_mask(len(ds), cfg.n_splits),
               th=th, prev_th=prev, side=cards is not None and decks is not None,
               meta=pd.DataFrame({"y": ds["y"].to_numpy(np.float64),
                                  "elapsed_days": ds["elapsed_days"].to_numpy(np.float64),
                                  "i": ds["i"].to_numpy(np.float64),
                                  "rmse_bins_lapse": ds["rmse_bins_lapse"].to_numpy(np.float64)}))
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as f:
            np.savez(f, X=X, y=out["y"], test=out["test"], th=th, prev_th=prev, side=np.array(out["side"]),
                     **{"meta_" + k: out["meta"][k].to_numpy() for k in V.META_COLS})
        os.replace(tmp, path)
    return out


def _init(data_dir: str, cache_dir=None) -> None:
    S = R._srs_modules()
    cfg = S["Config"](S["create_parser"]().parse_args(["--algo", "FSRS-6", "--short", "--data", data_dir]))
    if cfg.get_evaluation_file_name() != "FSRS-6-short":
        raise RuntimeError(f"unexpected srs-benchmark setting {cfg.get_evaluation_file_name()}")
    _W.clear()
    _W.update(cfg=cfg, loader=S["UserDataLoader"](cfg), data_dir=Path(data_dir), cache_dir=cache_dir)


def _task_count(uid):
    try:
        return dict(uid=uid, n=int(len(_W["loader"].load_user_data(uid))))
    except Exception as e:
        return dict(uid=uid, error=f"{type(e).__name__}: {str(e)[:100]}")


def _task_train(item):
    """item = (uid, quota): the user's curve rows, subsampled to `quota` (seeded per user, as val_plus)."""
    uid, quota = item
    try:
        t = _table(uid, _W["loader"], _W["cfg"], _W["data_dir"], _W["cache_dir"])
        X, y = t["X"], t["y"]
        if quota < len(y):
            rng = np.random.default_rng([V.SUBSAMPLE_SEED, int(uid)])
            keep = np.sort(rng.choice(len(y), size=int(quota), replace=False))
            X, y = X[keep], y[keep]
        return dict(uid=uid, X=X, y=y, side=t["side"])
    except Exception as e:
        return dict(uid=uid, error=f"{type(e).__name__}: {str(e)[:100]}")


def _task_eval(uid):
    try:
        t = _table(uid, _W["loader"], _W["cfg"], _W["data_dir"], _W["cache_dir"])
        if not t["test"].any():
            return dict(uid=uid, error="no test rows")
        return dict(uid=uid, **t)
    except Exception as e:
        return dict(uid=uid, error=f"{type(e).__name__}: {str(e)[:100]}")


def _map(task, items, data_dir: Path, workers: int, cache_dir=None):
    """Yield task(item) in order; workers > 1 uses a fork pool."""
    items = list(items)
    if workers <= 1 or len(items) < 2:
        _init(str(data_dir), cache_dir)
        for it in items:
            yield task(it)
        return
    ctx = mp.get_context("fork")
    with ctx.Pool(workers, initializer=_init, initargs=(str(data_dir), cache_dir)) as pool:
        yield from pool.imap(task, items, chunksize=1)


# ---- variants ----------------------------------------------------------------------------------------------

VARIANTS = ("base", "adv", "nomono", "adv_lr02")
VARIANT_MAX_ROUNDS = {"adv_lr02": 10000}


def parse_variants(spec):
    """Comma list of variant names -> list (order kept, duplicates dropped). Empty -> ["base"]."""
    names = []
    for tok in (spec or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok not in VARIANTS:
            raise ValueError(f"unknown variant {tok!r}; choose from {', '.join(VARIANTS)}")
        if tok not in names:
            names.append(tok)
    return names or ["base"]


def variant_params(name):
    """LightGBM params of a variant. base is CURVE_PARAMS unchanged."""
    if name == "base":
        return dict(CURVE_PARAMS)
    if name == "adv":
        return dict(CURVE_PARAMS, monotone_constraints_method="advanced")
    if name == "nomono":
        return {k: v for k, v in CURVE_PARAMS.items() if k != "monotone_constraints"}
    if name == "adv_lr02":
        return dict(CURVE_PARAMS, monotone_constraints_method="advanced", learning_rate=0.02)
    raise ValueError(f"unknown variant {name!r}")


def variant_max_rounds(name, max_rounds):
    return VARIANT_MAX_ROUNDS.get(name, max_rounds)


def variant_out(out, name) -> Path:
    """<stem>_<variant><suffix>; the curve+online file of it is online_path(variant_out(out, name))."""
    out = Path(out)
    return out.with_name(f"{out.stem}_{name}{out.suffix}")


# ---- run ---------------------------------------------------------------------------------------------------

def online_path(out) -> Path:
    out = Path(out)
    return out.with_name(out.stem + "_online" + out.suffix)


def _build_train_pool(train_uids, data_dir: Path, workers, cache_dir, max_train_rows, rows_per_user):
    """Train rows of all variants, built once: counted, subsampled to the plan, written into presized arrays.
    Returns (pool, plan, skipped, users without side tables)."""
    counts, skipped, no_side = {}, [], []
    for r in _map(_task_count, train_uids, data_dir, workers, cache_dir):
        if "error" in r:
            skipped.append((r["uid"], r["error"]))
        else:
            counts[r["uid"]] = r["n"]
    quotas = V.plan_quotas(counts, max_train_rows, rows_per_user)
    planned = list(quotas)
    hold = set(R.holdout_users(planned))
    cap = {"val": sum(quotas[u] for u in planned if u in hold),
           "fit": sum(quotas[u] for u in planned if u not in hold)}
    plan = dict(scored_rows_total=int(sum(counts.values())), train_users_counted=len(counts),
                train_users_planned=len(planned), rows_planned=int(cap["fit"] + cap["val"]),
                n_feat=len(CURVE_COLS))
    pool = V.PresizedPool(hold, cap, len(CURVE_COLS))
    for r in _map(_task_train, [(u, quotas[u]) for u in planned], data_dir, workers, cache_dir):
        if "error" in r:
            skipped.append((r["uid"], r["error"]))
            continue
        if not r["side"]:
            no_side.append(r["uid"])
        pool.add(r["uid"], r["X"], r["y"])
    return pool, plan, skipped, no_side


def run(train_uids, eval_uids, data_dir, out, final=False, workers=1, cache=True, cache_root=None,
        max_rounds=DEFAULT_MAX_ROUNDS, max_train_rows=DEFAULT_MAX_TRAIN_ROWS, rows_per_user=None,
        result_dir=C.DEFAULT_RESULT_DIR, variants=("base",)):
    """For each variant: train the curve GBDT on train_uids (or reuse <stem>_<variant>.model.txt when its
    fingerprint matches), then write <stem>_<variant>.jsonl (curve) and <stem>_<variant>_online.jsonl
    (curve+online) for eval_uids (resumable) and print its comparison. The train rows are built once and shared;
    each booster is freed before the next variant. Returns {variant: run_info}."""
    import gc
    import lightgbm as lgb

    variants = parse_variants(",".join(variants))
    V.check_users(train_uids, eval_uids, final)
    t0 = time.time()
    data_dir, out = Path(data_dir), Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = None
    if cache:
        root = Path(cache_root) if cache_root else data_dir / "feat_cache_curve"
        cache_dir = str(root / code_hash())
    pool, shared = None, dict(plan={}, skipped=[], no_side_train=[])
    run_infos = {}

    for name in variants:
        t_v = time.time()
        params = variant_params(name)
        mr = variant_max_rounds(name, max_rounds)
        curve_path = variant_out(out, name)
        out_on = online_path(curve_path)
        model_txt = Path(str(curve_path) + ".model.txt")
        model_json = Path(str(curve_path) + ".model.json")
        info_path = Path(str(curve_path) + ".run_info.json")
        fp = V._fingerprint(CURVE_COLS, params, ["curve"], train_uids, mr, R.PATIENCE,
                            max_train_rows, rows_per_user)
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
            if pool is None:
                pool, plan, skipped, no_side_train = _build_train_pool(
                    train_uids, data_dir, workers, cache_dir, max_train_rows, rows_per_user)
                shared = dict(plan=plan, skipped=skipped, no_side_train=no_side_train)
            plan = shared["plan"]
            booster, best_iter, info = R.fit_from_pool(pool, mr, R.PATIENCE, max_train_rows,
                                                       names=CURVE_COLS, params=params)
            booster.save_model(str(model_txt))
            model_json.write_text(json.dumps(dict(best_iter=int(best_iter), info=info, plan=plan,
                                                  fingerprint=fp)), encoding="utf-8")
            trained_now = True

        done = R._load_done(curve_path) & R._load_done(out_on)
        pending = list(dict.fromkeys(u for u in eval_uids if u not in done))
        written, skipped_eval, no_side_eval = 0, [], []
        with open(curve_path, "a", encoding="utf-8") as fc, open(out_on, "a", encoding="utf-8") as fo:
            for r in _map(_task_eval, pending, data_dir, workers, cache_dir):
                if "error" in r:
                    skipped_eval.append((r["uid"], r["error"]))
                    continue
                if not r["side"]:
                    no_side_eval.append(r["uid"])
                X, y, th, prev = r["X"], r["y"], r["th"], r["prev_th"]
                z = booster.predict(X, raw_score=True, num_iteration=best_iter)
                p_curve = _sigmoid(z)
                p_on = curve_online(z, X[:, EWM50_IX], X[:, RATE_IX], y, th, prev)
                m = r["test"]
                yt = y[m].astype(np.float64)
                meta = r["meta"][m].reset_index(drop=True)
                for f, p in ((fc, p_curve), (fo, p_on)):
                    pt = np.clip(p[m], G.P_CLIP, 1 - G.P_CLIP)
                    line = {"metrics": R.user_metrics(yt, pt, meta), "user": int(r["uid"]), "size": int(len(yt))}
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")
                    f.flush()
                written += 1
        R._sort_file(curve_path)
        R._sort_file(out_on)
        cmp = compare_curve(curve_path, out_on, result_dir)
        run_info = dict(
            variant=name, fingerprint=fp, plan=plan, info=info, best_iter=int(best_iter), trained_now=trained_now,
            features=CURVE_COLS, monotone=MONOTONE if "monotone_constraints" in params else None, params=params,
            max_rounds=mr, patience=R.PATIENCE, max_train_rows=max_train_rows, rows_per_user=rows_per_user,
            cache=cache, elapsed_min=round((time.time() - t_v) / 60, 2), final=final,
            train_users=sorted(int(u) for u in train_uids), eval_users=sorted(int(u) for u in eval_uids),
            new_eval_users=written, skipped=shared["skipped"] + skipped_eval,
            train_users_without_side_tables=shared["no_side_train"],
            eval_users_without_side_tables=no_side_eval, out=str(curve_path), out_online=str(out_on),
            comparison=cmp,
        )
        info_path.write_text(json.dumps(run_info, indent=1, default=str), encoding="utf-8")
        print(f"===== variant {name} (best_iter {int(best_iter)}, trained now: {trained_now}) =====")
        print(json.dumps({k: v for k, v in run_info.items()
                          if k not in ("train_users", "eval_users", "comparison", "fingerprint", "features",
                                       "monotone", "params")}, indent=1, default=str))
        print_comparison(cmp, label=name)
        run_infos[name] = run_info
        del booster
        gc.collect()
    del pool
    gc.collect()
    print(f"done {len(variants)} variant(s) in {(time.time() - t0) / 60:.1f} min")
    return run_infos


# ---- comparison --------------------------------------------------------------------------------------------

def _stats(rows, users):
    ll = [rows[u]["metrics"]["LogLoss"] for u in users]
    sizes = [rows[u]["size"] for u in users]
    au = [(rows[u]["metrics"].get("AUC"), rows[u]["size"]) for u in users]
    au = [(a, s) for a, s in au if a is not None]
    return dict(LL_unweighted=round(float(np.mean(ll)), 5), LL_weighted=round(C.weighted_mean(ll, sizes), 5),
                AUC_unweighted=round(float(np.mean([a for a, _ in au])), 5) if au else None,
                AUC_weighted=round(C.weighted_mean([a for a, _ in au], [s for _, s in au]), 5) if au else None)


def compare_curve(curve_path, online_path, result_dir=C.DEFAULT_RESULT_DIR, ci=0.99, n_boot=1000, seed=0):
    """Curve and curve+online against the published RWKV-Curve (RWKV) and RWKV-Instant (RWKV-P), on the users
    present in all with the same size. Paired size-weighted bootstrap CIs vs RWKV-Curve (99%, 1000, seed 0)."""
    result_dir = Path(result_dir)
    rows = {"Curve (ours)": C.load_rows(curve_path), "Curve+online (ours)": C.load_rows(online_path)}
    for name in ("RWKV-Curve", "RWKV-Instant"):
        f = result_dir / f"{V.PUBLISHED[name]}.jsonl"
        if f.exists():
            rows[name] = C.load_rows(f)
    ours = rows["Curve (ours)"]
    users = sorted(u for u in ours if all(u in r and r[u]["size"] == ours[u]["size"] for r in rows.values()))
    if not users:
        return dict(n_users=0)
    sizes = [ours[u]["size"] for u in users]
    table = {name: _stats(rows[name], users) for name in rows}
    ll = {name: np.array([rows[name][u]["metrics"]["LogLoss"] for u in users]) for name in rows}
    boots = {}
    if "RWKV-Curve" in rows:
        for name in ("Curve+online (ours)", "Curve (ours)"):
            boots[f"{name} - RWKV-Curve"] = C.paired_bootstrap(ll[name] - ll["RWKV-Curve"], sizes,
                                                              n_boot=n_boot, ci=ci, seed=seed)
    return dict(n_users=len(users), users=users, n_rows=int(sum(sizes)), table=table, bootstrap=boots, ci=ci)


def print_comparison(cmp, label=None):
    tag = f"[{label}] " if label else ""
    if not cmp or not cmp.get("n_users"):
        print(f"{tag}comparison: no eval users with a published result of the same size")
        return
    print(f"{tag}comparison on {cmp['n_users']} users ({cmp['n_rows']} test rows)")
    for name, t in cmp["table"].items():
        print(f"  {name:22s} LogLoss unweighted {t['LL_unweighted']:.5f} weighted {t['LL_weighted']:.5f}"
              f"  AUC unweighted {t['AUC_unweighted']} weighted {t['AUC_weighted']}")
    for name, b in cmp["bootstrap"].items():
        print(f"  {name:32s} size-weighted diff {b['mean']:+.5f}  {100 * cmp['ci']:.0f}% CI "
              f"[{b['lo']:+.5f}, {b['hi']:+.5f}]")


def main(argv=None):
    ap = argparse.ArgumentParser(description="GBDT-Curve with the curve online layer (Phase 8).")
    ap.add_argument("--data", default="data/anki10k", help="dir with revlogs/, cards/, decks/")
    ap.add_argument("--train_users", required=True, help="ids (comma list, ranges like 2001-2300) or a file")
    ap.add_argument("--eval_users", required=True, help="ids (comma list, ranges like 8001-8100) or a file")
    ap.add_argument("--out", required=True, help="curve jsonl; the curve+online file is <stem>_online.jsonl")
    ap.add_argument("--workers", type=int, default=1, help="parallel featurization processes")
    ap.add_argument("--cache_dir", default=None, help="cache root (default <data>/feat_cache_curve)")
    ap.add_argument("--no_cache", action="store_true")
    ap.add_argument("--max_rounds", type=int, default=DEFAULT_MAX_ROUNDS)
    ap.add_argument("--max_train_rows", type=int, default=DEFAULT_MAX_TRAIN_ROWS)
    ap.add_argument("--rows_per_user", type=int, default=None)
    ap.add_argument("--quick", action="store_true", help="first 3 train users and first 2 eval users")
    ap.add_argument("--final", action="store_true", help="allow users 1-1000 (task 6.5 only)")
    ap.add_argument("--result_dir", default=str(C.DEFAULT_RESULT_DIR))
    ap.add_argument("--variants", default="base",
                    help=f"comma list of training variants, one run each ({', '.join(VARIANTS)}); default base")
    args = ap.parse_args(argv)
    variants = parse_variants(args.variants)

    train_uids = load_users(args.train_users)
    eval_uids = load_users(args.eval_users)
    if args.quick:
        train_uids, eval_uids = train_uids[:3], eval_uids[:2]
    V.check_users(train_uids, eval_uids, args.final)        # before any work
    return run(train_uids, eval_uids, args.data, args.out, final=args.final, workers=args.workers,
               cache=not args.no_cache, cache_root=args.cache_dir, max_rounds=args.max_rounds,
               max_train_rows=args.max_train_rows, rows_per_user=args.rows_per_user,
               result_dir=Path(args.result_dir), variants=variants)


if __name__ == "__main__":
    main(sys.argv[1:])
