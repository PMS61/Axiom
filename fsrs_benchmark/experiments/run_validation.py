"""
run_validation.py -- Phase 6 model selection on VALIDATION data only (no test events).

  python -m fsrs_benchmark.experiments.run_validation --dataset anki --models fsrs_default fsrs_opt v1
Prints a table (log loss, Brier, AUC, ECE) and saves results/phase6_validation/<dataset>.csv
(rows are merged by model name across invocations). Every ours_v2 run is also appended to
docs/reports/phase-6-experiments.md so no variant is silently dropped.
"""
import argparse
import hashlib
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from fsrs_benchmark import metrics
from fsrs_benchmark.experiments.prepare_data import CACHE, build
from fsrs_benchmark.experiments.validation import make_validation_split
from fsrs_benchmark.models.fsrs_model import FSRSModel
from fsrs_benchmark.models.ours_v2 import OursV2

OUT = Path("results/phase6_validation")
LOG = Path("docs/reports/phase-6-experiments.md")

# name -> (description, kwargs for OursV2). Add new variants here; never delete one.
VARIANTS = {
    "v1": ("candidate 1: hierarchical FSRS-style, 5 user effects, MAP, 6 epochs", dict()),
    "v2": ("v1 with 20 epochs", dict(epochs=20)),
    "v4": ("v1 with 10 epochs", dict(epochs=10)),
    "v5": ("v4 + process noise on log-stability (5-pt Gauss-Hermite marginal, candidate 2)", dict(epochs=10, process_noise=True)),
    "v6": ("v4 + per-card difficulty random effect (learned tau, prior N(0,1))", dict(epochs=10, card_effect=True)),
    "v7": ("v4 + user-level initial-difficulty effect (6 effects)", dict(epochs=10, effects=("s0", "growth", "decay", "fail", "bias", "d0"))),
    "v8": ("v4 with smaller chunks (3000 cards, ~10x more optimiser steps), 3 epochs", dict(epochs=3, chunk_cards=3000)),
    "v9": ("v8 + per-card difficulty effect + user d0 effect, z lr 0.1", dict(epochs=3, chunk_cards=3000, card_effect=True, z_lr=0.1, effects=("s0", "growth", "decay", "fail", "bias", "d0"))),
    "v3": ("v2 with fit_steps=200 (longer card prefix in the fit)", dict(epochs=20, fit_steps=200)),
}


def make_model(name, workers):
    if name == "fsrs_default":
        return FSRSModel(optimize=False)
    if name == "fsrs_opt":
        return FSRSModel(optimize=True, n_workers=workers)
    return OursV2(**VARIANTS[name][1])


def ci_vs(events, pred, ref, n_boot=1000):
    """User-cluster bootstrap 95% CI of logloss(pred) - logloss(ref) (negative = pred better)."""
    sc = events[events["scored"]].set_index("event_id").loc[pred.index]
    return metrics.paired_bootstrap(sc["user_id"].to_numpy(), sc["y"].to_numpy(), pred.to_numpy(),
                                    ref.loc[pred.index].to_numpy(), "log_loss", n_boot=n_boot, seed=0)


def evaluate(events, pred):
    sc = events[events["scored"]].set_index("event_id").loc[pred.index]
    return metrics.summarize(sc["y"].to_numpy(), pred.to_numpy())


def _ci_text(res):
    if "d_vs_opt" not in res:
        return "n/a"
    lo, hi = res["ci_lo"], res["ci_hi"]
    verdict = "WIN" if hi < 0 else ("LOSS" if lo > 0 else "TIE")
    return f"{res['d_vs_opt']:+.4f} [{lo:+.4f}, {hi:+.4f}] {verdict}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["anki", "duolingo"], required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--note", default="")
    args = ap.parse_args(argv)

    full = build(args.dataset)
    events = make_validation_split(full)
    print(f"{args.dataset}: {len(events)} rows (test dropped), fit {int(events.is_train.sum())}, "
          f"validation scored {int(events.scored.sum())}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in args.models:
        sid = hashlib.sha256(np.sort(events.loc[events["scored"], "event_id"].to_numpy(np.int64)).tobytes()).hexdigest()[:12]
        cache = CACHE / f"{args.dataset}_val_{name}_{sid}.parquet"
        started = time.perf_counter()
        if name.startswith("fsrs") and cache.exists():
            pred = pd.read_parquet(cache)["p"]
            secs = float("nan")
        else:
            model = make_model(name, args.workers).fit(events)
            pred = model.predict(events)
            secs = time.perf_counter() - started
            if name.startswith("fsrs"):
                pred.to_frame("p").to_parquet(cache)
        res = {"model": name, **evaluate(events, pred), "seconds": round(secs, 1)}
        ref_path = CACHE / f"{args.dataset}_val_fsrs_opt_{sid}.parquet"
        if name != "fsrs_opt" and ref_path.exists():
            ci = ci_vs(events, pred, pd.read_parquet(ref_path)["p"])
            res.update(d_vs_opt=ci["diff"], ci_lo=ci["lo"], ci_hi=ci["hi"])
        rows.append(res)
        print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in res.items()}, flush=True)
        if name in VARIANTS:
            LOG.parent.mkdir(parents=True, exist_ok=True)
            if not LOG.exists():
                LOG.write_text("# Phase 6 experiments log (validation only)\n\nEvery ours_v2 variant run, in order. "
                               "Metrics on validation scored events; lower log loss / Brier / ECE better, higher AUC better.\n\n"
                               "| date | dataset | variant | description | log_loss | brier | auc | ece | fit+predict s | dLL vs FSRS-opt [95% CI] | note |\n"
                               "|---|---|---|---|---|---|---|---|---|---|---|\n")
            with LOG.open("a") as f:
                f.write(f"| {date.today()} | {args.dataset} | {name} | {VARIANTS[name][0]} | {res['log_loss']:.5f} | "
                        f"{res['brier']:.5f} | {res['auc']:.4f} | {res['ece']:.4f} | {res['seconds']} | {_ci_text(res)} | {args.note} |\n")
    table = pd.DataFrame(rows)
    path = OUT / f"{args.dataset}.csv"
    if path.exists():
        old = pd.read_csv(path)
        table = pd.concat([old[~old["model"].isin(table["model"])], table], ignore_index=True)
    table.to_csv(path, index=False)
    print(table[[c for c in ["model", "log_loss", "brier", "auc", "ece", "d_vs_opt", "ci_lo", "ci_hi", "seconds"] if c in table]].to_string(index=False))


if __name__ == "__main__":
    main()
