"""
run_validation_v3.py -- Phase 7 model selection for ours_v3 on VALIDATION data only (no test events).

  python -m fsrs_benchmark.experiments.run_validation_v3 --dataset anki --models b0 a1
Reads the Phase 6 event cache for Anki and builds its own Duolingo cache WITH the `lexeme` column
(results/phase7_validation/cache). FSRS-opt validation predictions are read from the Phase 6 cache
(read only). Rows are written to results/phase7_validation/<dataset>.csv and appended to
docs/reports/phase-7-experiments.md. Test rows are dropped by make_validation_split.
"""
import argparse
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from fsrs_benchmark import duolingo, split
from fsrs_benchmark.experiments.prepare_data import CACHE as P6_CACHE
from fsrs_benchmark.experiments.run_validation import _ci_text, ci_vs, evaluate
from fsrs_benchmark.experiments.validation import make_validation_split
from fsrs_benchmark.models.ours_v3 import OursV3

OUT = Path("results/phase7_validation")
CACHE = OUT / "cache"
LOG = Path("docs/reports/phase-7-experiments.md")
LEX3 = ("s0", "d0", "bias")

# name -> (description, kwargs for OursV3). Add new variants here; never delete one.
VARIANTS = {
    "b0": ("baseline = ours_v2 FROZEN (epochs 20)", dict(epochs=20)),
    "a1": ("b0 + lexeme effect on initial stability (lex_lr 0.1)", dict(epochs=20, lex_effects=("s0",), lex_lr=0.1)),
    "a2": ("b0 + lexeme effects on s0, d0, bias (lex_lr 0.1)", dict(epochs=20, lex_effects=LEX3, lex_lr=0.1)),
    "a3": ("b0 + lexeme effects on s0, d0, growth, bias (lex_lr 0.1)", dict(epochs=20, lex_effects=("s0", "d0", "growth", "bias"), lex_lr=0.1)),
    "a4": ("a2 with chunk_cards=3000 (more optimiser steps), 8 epochs", dict(epochs=8, chunk_cards=3000, lex_effects=LEX3, lex_lr=0.1)),
    "l1": ("b0 + lagged-outcome term (global beta)", dict(epochs=20, lag=True)),
    "c1": ("b0 + per-card logit random effect", dict(epochs=20, card_bias=True)),
    "l2": ("b0 + lag + per-card logit effect (v1's strengths)", dict(epochs=20, lag=True, card_bias=True)),
    "t1": ("b0 + forgetting-curve time from fractional dt_days", dict(epochs=20, use_dt=True)),
    "m1": ("a2 + lag", dict(epochs=20, lex_effects=LEX3, lex_lr=0.1, lag=True)),
    "m2": ("a2 + lag + dt", dict(epochs=20, lex_effects=LEX3, lex_lr=0.1, lag=True, use_dt=True)),
    "m3": ("a2 + lag + card bias + dt (everything)", dict(epochs=20, lex_effects=LEX3, lex_lr=0.1, lag=True, card_bias=True, use_dt=True)),
    # second round, after round 1 showed free-scale lexeme/card effects overfit: regularise instead
    "e1": ("b0 with 10 epochs (early stopping as regulariser)", dict(epochs=10)),
    "z1": ("b0 with user z lr 0.01 (slower user effects)", dict(epochs=20, z_lr=0.01)),
    "p1": ("b0 + L2 penalty 100 on user-effect scales tau (shrinks overfit tau)", dict(epochs=20, tau_pen=100.0)),
    "p2": ("b0 + tau penalty 300", dict(epochs=20, tau_pen=300.0)),
    "p3": ("b0 + tau penalty 1000", dict(epochs=20, tau_pen=1000.0)),
    "p4": ("b0 + tau penalty 3000", dict(epochs=20, tau_pen=3000.0)),
    "k1": ("b0 + per-card logit effect with FIXED scale 0.2", dict(epochs=20, card_bias=True, card_tau=0.2)),
    "q1": ("p2 (tau penalty 300) + lag", dict(epochs=20, tau_pen=300.0, lag=True)),
    "q2": ("p2 + lag + fixed-scale per-card logit effect 0.2", dict(epochs=20, tau_pen=300.0, lag=True, card_bias=True, card_tau=0.2)),
    "q3": ("p1 (tau penalty 100) + lag + fixed-scale per-card logit effect 0.2", dict(epochs=20, tau_pen=100.0, lag=True, card_bias=True, card_tau=0.2)),
    "x1": ("b0 + lexeme s0 effect with FIXED scale 0.1, lr 0.03", dict(epochs=20, lex_effects=("s0",), lex_tau=0.1, lex_lr=0.03)),
}


Q2 = dict(epochs=20, tau_pen=300.0, lag=True, card_bias=True, card_tau=0.2)
# full-Duolingo lexeme check (dataset "duolingo_full"): q2 +- lexeme s0+bias with a penalised learned scale
VARIANTS.update({
    "f0": ("FULL: v2 (epochs 20)", dict(epochs=20)),
    "f1": ("FULL: q2 (frozen)", Q2),
    "f2": ("FULL: q2 + lexeme s0,bias, learned scale, lex_pen 10, lex_lr 0.03", dict(Q2, lex_effects=("s0", "bias"), lex_lr=0.03, lex_pen=10.0)),
    "f3": ("FULL: q2 + lexeme s0,bias, learned scale, lex_pen 100, lex_lr 0.03", dict(Q2, lex_effects=("s0", "bias"), lex_lr=0.03, lex_pen=100.0)),
    "f4": ("FULL: q2 + lexeme s0,bias, learned scale, lex_pen 1000, lex_lr 0.03", dict(Q2, lex_effects=("s0", "bias"), lex_lr=0.03, lex_pen=1000.0)),
})


def build(dataset):
    if dataset == "duolingo_full":
        CACHE.mkdir(parents=True, exist_ok=True)
        out = CACHE / "duolingo_full_events_lex.parquet"
        if out.exists():
            return pd.read_parquet(out)
        p = Path("data/duolingo/settles.acl16.learning_traces.13m.csv.gz")
        counts, t0 = duolingo.scan(p, p.parent / "duolingo_counts.parquet")
        users = duolingo.sample_users(counts, None, 0, min_rows=1)
        ev = split.make_split(duolingo.load_users(p, users, counts, t0, with_lexeme=True), max_cards=10**9, seed=0)
        ev.to_parquet(out, index=False)
        return ev
    if dataset == "anki":
        return pd.read_parquet(P6_CACHE / "anki_events.parquet")
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / "duolingo_events_lex.parquet"
    if out.exists():
        return pd.read_parquet(out)
    p = Path("data/duolingo/settles.acl16.learning_traces.13m.csv.gz")
    counts, t0 = duolingo.scan(p, p.parent / "duolingo_counts.parquet")
    users = duolingo.sample_users(counts, 5000, seed=0, min_rows=1)
    ev = split.make_split(duolingo.load_users(p, users, counts, t0, with_lexeme=True), max_cards=10**9, seed=0)
    ev.to_parquet(out, index=False)
    return ev


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["anki", "duolingo", "duolingo_full"], required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--note", default="")
    args = ap.parse_args(argv)
    events = make_validation_split(build(args.dataset))
    print(f"{args.dataset}: {len(events)} rows (test dropped), fit {int(events.is_train.sum())}, "
          f"validation scored {int(events.scored.sum())}", flush=True)
    if args.dataset == "duolingo_full":
        ref_path = CACHE / "duolingo_full_val_fsrs_ref.parquet"   # made by fsrs_ref_full.py
    else:
        ref_path = next(P6_CACHE.glob(f"{args.dataset}_val_fsrs_opt_*.parquet"))
    ref = pd.read_parquet(ref_path)["p"] if ref_path.exists() else None   # full Duolingo: may be pending
    assert ref is None or set(ref.index) == set(events.loc[events["scored"], "event_id"]), "scored set differs from Phase 6"
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in args.models:
        started = time.perf_counter()
        model = OursV3(**VARIANTS[name][1]).fit(events)
        pred = model.predict(events)
        secs = time.perf_counter() - started
        res = {"model": name, **evaluate(events, pred), "seconds": round(secs, 1)}
        pred.to_frame("p").to_parquet(CACHE / f"{args.dataset}_pred_{name}.parquet")
        if ref is None:
            print(f"{name}: no FSRS reference yet; predictions saved, add CIs later", flush=True)
            continue
        ci = ci_vs(events, pred, ref)
        res.update(d_vs_opt=ci["diff"], ci_lo=ci["lo"], ci_hi=ci["hi"])
        rows.append(res)
        print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in res.items()}, flush=True)
        LOG.parent.mkdir(parents=True, exist_ok=True)
        if not LOG.exists():
            LOG.write_text("# Phase 7 experiments log (validation only)\n\nEvery ours_v3 variant run, in order. "
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
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
