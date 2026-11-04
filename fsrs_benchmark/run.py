"""Runner: download -> load -> split -> fit and predict every model -> write results/<run>/.

Why one runner: every model must be fitted and scored on exactly the same events, built
the same way. This CLI does the whole pipeline in one place and asserts that every model
predicted exactly the scored event set before anything is merged.

Outputs (under <out_dir>/<run_name>/):
  events.parquet          the full event table from make_split
  pred_<model>.parquet    event_id, p for one model (written as soon as that model finishes)
  predictions.parquet     event_id, user_id, card_id, y, elapsed_days, day_offset, one column per model
  run_info.json           args, git commit, package versions, counts, per-model timings and details

Usage: .venv/bin/python -m fsrs_benchmark.run --run_name smoke --n_users 2 --iters 200 --burn 100
"""
import argparse
import hashlib
import importlib.metadata as importlib_metadata
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from fsrs_benchmark import data, duolingo, split
from fsrs_benchmark.models.fsrs_model import FSRSModel
from fsrs_benchmark.models.gbdt import FROZEN_KWARGS as FROZEN_KWARGS_GBDT, GBDT
from fsrs_benchmark.models.hlr import HLRModel
from fsrs_benchmark.models.laglogit import LagLogitModel
from fsrs_benchmark.models.ours import OursModel
from fsrs_benchmark.models.ours_v2 import FROZEN_KWARGS, OursV2
from fsrs_benchmark.models.ours_v3 import FROZEN_KWARGS as FROZEN_KWARGS_V3, OursV3

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS = ("anki", "duolingo")
DEFAULT_DUOLINGO_PATH = "data/duolingo/settles.acl16.learning_traces.13m.csv.gz"
# Models run by default. The opt-in keys below are allowed in --models but not run unless
# named, so existing runs do not change.
ALL_MODELS = ["fsrs_default", "fsrs_opt", "hlr", "ours", "ours_kappa"]
OPT_IN_MODELS = ["ours_gpu", "ours_kappa_gpu", "laglogit", "ours_v2", "ours_v3", "gbdt"]
# key -> (use_kappa, backend) for the ours variants
OURS_VARIANTS = {"ours": (False, "numba"), "ours_kappa": (True, "numba"),
                 "ours_gpu": (False, "gpu"), "ours_kappa_gpu": (True, "gpu")}
PACKAGES = ["fsrs", "numpy", "pandas", "torch", "numba"]
ID_COLUMNS = ["event_id", "user_id", "card_id", "y", "elapsed_days", "day_offset"]
PROGRESS_EVERY = 100  # Gibbs sweeps between progress lines for the ours variants


def log(message):
    """Print a progress line with a wall-clock timestamp."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the FSRS benchmark end to end.")
    parser.add_argument("--n_users", type=int, default=100)
    parser.add_argument("--max_cards", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iters", type=int, default=2000, help="Gibbs sweeps for ours")
    parser.add_argument("--burn", type=int, default=1000, help="burn-in sweeps for ours")
    parser.add_argument("--models", default=",".join(ALL_MODELS),
                        help=f"comma list from: {', '.join(ALL_MODELS)}")
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--data_dir", default="data/anki10k")
    parser.add_argument("--out_dir", default="results")
    parser.add_argument("--workers", type=int, default=1,
                        help="processes for the FSRS optimiser and replays (per user); 1 = serial")
    parser.add_argument("--resume", action="store_true",
                        help="skip models whose pred_<model>.parquet already exists")
    parser.add_argument("--dataset", choices=DATASETS, default="anki",
                        help="anki (revlogs-10k, downloaded) or duolingo (local csv.gz)")
    parser.add_argument("--duolingo_path", default=DEFAULT_DUOLINGO_PATH)
    parser.add_argument("--min_user_rows", type=int, default=100,
                        help="duolingo: only users with at least this many rows are eligible")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="device for the ours_*_gpu keys; auto = cuda if available")
    parser.add_argument("--thin", type=int, default=4, help="keep every thin-th ours draw")
    args = parser.parse_args(argv)

    allowed = ALL_MODELS + OPT_IN_MODELS
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in models if m not in allowed]
    if unknown or not models:
        parser.error(f"unknown or empty --models: {unknown or args.models!r}")
    if len(set(models)) != len(models):
        parser.error("--models lists a model twice")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if not 1 <= args.max_cards:
        parser.error("--max_cards must be at least 1")
    if not 0 <= args.burn < args.iters:
        parser.error("--burn must be >= 0 and less than --iters")
    if args.thin < 1:
        parser.error("--thin must be at least 1")
    if args.min_user_rows < 1:
        parser.error("--min_user_rows must be at least 1")
    if args.dataset == "anki" and args.n_users < 1:
        parser.error("--n_users must be at least 1 for --dataset anki (0 = all users is duolingo only)")
    if args.n_users < 0:
        parser.error("--n_users must be 0 (all eligible users, duolingo) or more")
    args.models = models
    return args


def device_arg(args):
    """--device as the model expects it: None means auto (cuda if available, else cpu)."""
    device = getattr(args, "device", "auto")
    return None if device == "auto" else device


def make_model(key, args):
    """Build the model object for one key in ALL_MODELS or OPT_IN_MODELS."""
    if key == "fsrs_default":
        return FSRSModel(optimize=False, n_workers=args.workers)
    if key == "fsrs_opt":
        return FSRSModel(optimize=True, n_workers=args.workers)
    if key == "hlr":
        return HLRModel(seed=args.seed)
    if key == "laglogit":
        return LagLogitModel(use_dt=True)
    if key == "ours_v2":
        return OursV2(seed=args.seed, **FROZEN_KWARGS)
    if key == "ours_v3":
        return OursV3(seed=args.seed, **FROZEN_KWARGS_V3)
    if key == "gbdt":
        return GBDT(seed=args.seed, **{**FROZEN_KWARGS_GBDT, "user_group": "row" if getattr(args, "dataset", "") == "anki" else "auto"})
    if key in OURS_VARIANTS:
        use_kappa, backend = OURS_VARIANTS[key]
        # getattr defaults: callers may pass a namespace without the Phase 5 flags (anki defaults).
        return OursModel(use_kappa=use_kappa, iters=args.iters, burn=args.burn,
                         thin=getattr(args, "thin", 4), seed=args.seed,
                         progress_every=PROGRESS_EVERY, backend=backend, device=device_arg(args))
    raise KeyError(key)


def git_info():
    """Commit hash and whether the working tree has uncommitted changes (None if no git)."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
        return {"commit": commit, "dirty": bool(status)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def package_versions():
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
    versions["python"] = sys.version.split()[0]
    return versions


def model_config(key, args, scored_ids):
    """What a model's saved predictions depend on. --resume reuses a file only if this matches."""
    config = {"n_users": args.n_users, "max_cards": args.max_cards, "seed": args.seed,
              "scored_event_sha256": hashlib.sha256(np.asarray(scored_ids, np.int64).tobytes()).hexdigest()}
    if key == "ours_v2":
        config.update(hyperparams=OursV2(seed=args.seed, **FROZEN_KWARGS).hyperparams())
    elif key == "ours_v3":
        config.update(hyperparams=OursV3(seed=args.seed, **FROZEN_KWARGS_V3).hyperparams())
    elif key == "gbdt":
        config.update(hyperparams=make_model("gbdt", args).hyperparams())
    elif key.startswith("ours"):
        config.update(iters=args.iters, burn=args.burn)
    if key.endswith("_gpu"):
        config["backend"] = "gpu"  # added only for the GPU keys, so existing sidecars still resume
    # Added only off anki, so existing Anki sidecars still resume. getattr keeps older
    # argument namespaces (which have no dataset field) on the anki path.
    dataset = getattr(args, "dataset", "anki")
    if dataset != "anki":
        config.update(dataset=dataset, min_user_rows=args.min_user_rows)
    return config


def check_resume_config(pred_path, expected):
    """Refuse to reuse a prediction file whose recorded config differs from this run's."""
    sidecar = pred_path.with_suffix(".json")
    if not sidecar.exists():
        detail = "no config record (written before config tracking)"
    else:
        stored = json.loads(sidecar.read_text())
        diffs = [f"{k}: was {stored.get(k)!r}, now {expected[k]!r}"
                 for k in sorted(expected) if stored.get(k) != expected[k]]
        if not diffs:
            return
        detail = "; ".join(diffs)
    raise AssertionError(
        f"{pred_path} was written by a different config ({detail}). "
        "Config changed; delete it or drop --resume to recompute.")


def check_event_ids(key, pred, scored_ids):
    """Every model must predict exactly the scored event set, each event once."""
    ids = pred["event_id"].to_numpy(np.int64)
    if len(np.unique(ids)) != len(ids):
        raise AssertionError(f"{key}: duplicate event_id in predictions")
    expected = set(scored_ids.tolist())
    got = set(ids.tolist())
    if got != expected:
        missing, extra = len(expected - got), len(got - expected)
        raise AssertionError(
            f"{key}: predicted event set differs from scored set "
            f"({missing} missing, {extra} extra)")


def model_details(key, model):
    """Model-specific facts worth keeping in run_info.json."""
    if key == "fsrs_opt":
        return {"fallback_count": len(model.fallback_),
                "fallback_users": sorted(int(u) for u in model.fallback_)}
    if key == "hlr":
        return {"epochs": int(model.epochs_)}
    if key in ("ours_v2", "ours_v3", "gbdt"):
        return {"hyperparams": model.hyperparams()}
    if key.endswith("_gpu"):
        return {"device": str(model.model_.device)}
    return {}


def load_events_source(args):
    """The raw event rows for the chosen dataset, in the data.load_user column shape."""
    if args.dataset == "anki":
        log(f"download: {args.n_users} users, seed {args.seed}, into {args.data_dir}")
        paths = data.download_users(args.n_users, args.seed, args.data_dir)
        log(f"load: {len(paths)} user files")
        return pd.concat([data.load_user(p) for p in paths], ignore_index=True)

    path = Path(args.duolingo_path)
    cache = path.parent / "duolingo_counts.parquet"
    log(f"scan: {path} (counts cache {cache})")
    counts, t0 = duolingo.scan(path, cache)
    # --n_users 0 means every eligible user; otherwise a seeded sample of that size.
    n_users = None if args.n_users == 0 else args.n_users
    users = duolingo.sample_users(counts, n_users, args.seed, args.min_user_rows)
    log(f"load: {len(users)} users with at least {args.min_user_rows} rows "
        f"({'all' if n_users is None else 'sampled'}), seed {args.seed}")
    return duolingo.load_users(path, users, counts, t0, with_lexeme=bool({"ours_v3", "gbdt"} & set(getattr(args, "models", []))),
                               with_ts="gbdt" in getattr(args, "models", []))


def run(args):
    started = time.perf_counter()
    run_dir = Path(args.out_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    info_path = run_dir / "run_info.json"
    previous = {}
    if args.resume and info_path.exists():
        previous = json.loads(info_path.read_text()).get("models", {})

    df = load_events_source(args)

    log(f"split: max_cards={args.max_cards}, seed={args.seed}")
    events = split.make_split(df, max_cards=args.max_cards, seed=args.seed)
    split.save_events(events, run_dir / "events.parquet")
    scored = events[events["scored"]]
    scored_ids = np.sort(scored["event_id"].to_numpy(np.int64))
    n_train = int(events["is_train"].sum())
    log(f"events: {len(events)} total, {n_train} train, {len(events) - n_train} test, "
        f"{len(scored)} scored")

    users_all = set(events["user_id"].unique().tolist())
    users_with_train = set(events.loc[events["is_train"], "user_id"].unique().tolist())
    zero_train_users = sorted(int(u) for u in users_all - users_with_train)
    if zero_train_users:
        log(f"WARNING: {len(zero_train_users)} users have zero train rows (kept): {zero_train_users}")

    models_info, preds = {}, {}
    for key in args.models:
        pred_path = run_dir / f"pred_{key}.parquet"
        config = model_config(key, args, scored_ids)
        if args.resume and pred_path.exists():
            check_resume_config(pred_path, config)
            log(f"{key}: --resume, skipping (found {pred_path.name})")
            pred = pd.read_parquet(pred_path)
            # The sidecar's details are the run info; older sidecars have none, so fall back
            # to whatever the previous run_info.json recorded.
            stored = json.loads(pred_path.with_suffix(".json").read_text())
            details = stored.get("details")
            if not isinstance(details, dict):  # missing, null or hand-edited: fall back
                details = previous.get(key, {})
            models_info[key] = dict(details, resumed=True)
        else:
            model = make_model(key, args)
            log(f"{key}: fit")
            t0 = time.perf_counter()
            model.fit(events)
            fit_seconds = time.perf_counter() - t0
            log(f"{key}: predict")
            t0 = time.perf_counter()
            p = model.predict(events)
            predict_seconds = time.perf_counter() - t0
            pred = pd.DataFrame({"event_id": p.index.to_numpy(np.int64), "p": p.to_numpy(float)})
            check_event_ids(key, pred, scored_ids)
            pred.to_parquet(pred_path, index=False)
            details = dict(fit_seconds=round(fit_seconds, 3),
                           predict_seconds=round(predict_seconds, 3), **model_details(key, model))
            # check_resume_config compares only the config keys, so details is ignored there.
            pred_path.with_suffix(".json").write_text(json.dumps(dict(config, details=details), indent=2))
            if key in OURS_VARIANTS:
                (run_dir / f"diagnostics_{key}.json").write_text(json.dumps(model.diagnostics_, indent=2))
            models_info[key] = dict(details, resumed=False)
            log(f"{key}: done, fit {fit_seconds:.1f}s, predict {predict_seconds:.1f}s, "
                f"saved {pred_path.name}")
        check_event_ids(key, pred, scored_ids)
        preds[key] = pred.rename(columns={"p": key})

    merged = scored[ID_COLUMNS].sort_values("event_id").reset_index(drop=True)
    for key in args.models:
        merged = merged.merge(preds[key], on="event_id", how="left")
        if merged[key].isna().any():
            raise AssertionError(f"{key}: missing predictions after merge")
    merged.to_parquet(run_dir / "predictions.parquet", index=False)
    log(f"wrote {run_dir / 'predictions.parquet'} ({len(merged)} rows)")

    total_seconds = time.perf_counter() - started
    info = {
        "run_name": args.run_name,
        "dataset": args.dataset,
        "args": {k: v for k, v in vars(args).items()},
        "git": git_info(),
        "packages": package_versions(),
        "counts": {
            "users": len(users_all),
            "cards": int(events[["user_id", "card_id"]].drop_duplicates().shape[0]),
            "events": len(events),
            "train": n_train,
            "test": len(events) - n_train,
            "scored": len(scored),
        },
        "users_zero_train": zero_train_users,
        "models": models_info,
        "total_seconds": round(total_seconds, 3),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    info_path.write_text(json.dumps(info, indent=2, default=str))
    log(f"wrote {info_path} (total {total_seconds:.1f}s)")
    return info


def main(argv=None):
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
