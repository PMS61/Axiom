"""
run_validation_gbdt.py -- Phase 7 task 7.3: GBDT variants on VALIDATION data only (no test events).

  python -m fsrs_benchmark.experiments.run_validation_gbdt --dataset anki --models all no_card
  python -m fsrs_benchmark.experiments.run_validation_gbdt --select      # frozen choice by the fixed rule
Same data/caches as run_validation_v3 (test rows dropped by make_validation_split); FSRS-opt
validation predictions are read from the Phase 6 cache. Writes results/phase7_validation/gbdt_<dataset>.csv,
per-variant predictions in the cache dir, and appends rows to docs/reports/phase-7-experiments.md.
"""
import argparse
import time
from datetime import date
from pathlib import Path

import pandas as pd

from fsrs_benchmark.experiments.prepare_data import CACHE as P6_CACHE
from fsrs_benchmark.experiments.run_validation import _ci_text, ci_vs, evaluate
from fsrs_benchmark import duolingo, split
from fsrs_benchmark.experiments.run_validation_v3 import CACHE, LOG, OUT, build as build_v3
from fsrs_benchmark.experiments.validation import make_validation_split
from fsrs_benchmark.models.gbdt import ALL_GROUPS, GBDT, base_features

ALL = tuple(ALL_GROUPS)


def build(dataset):
    """Anki: the Phase 6 cache (no timestamp; the user kernel then groups by day). Duolingo: a cache WITH
    lexeme and the raw timestamp `ts` (a session shares one ts)."""
    if dataset == "anki":
        return build_v3(dataset)
    out = CACHE / "duolingo_events_lex_ts.parquet"
    if out.exists():
        return pd.read_parquet(out)
    p = Path("data/duolingo/settles.acl16.learning_traces.13m.csv.gz")
    counts, t0 = duolingo.scan(p, p.parent / "duolingo_counts.parquet")
    users = duolingo.sample_users(counts, 5000, seed=0, min_rows=1)
    ev = split.make_split(duolingo.load_users(p, users, counts, t0, with_lexeme=True, with_ts=True),
                          max_cards=10**9, seed=0)
    ev.to_parquet(out, index=False)
    return ev


def _drop(g):
    return tuple(x for x in ALL if x != g)


# name -> (description, kwargs for GBDT). Add new variants here; never delete one.
VARIANTS = {
    "all": ("all groups, leaves 31, lr 0.05, min_leaf 100", dict()),
    **{f"no_{g}": (f"all minus group {g}", dict(groups=_drop(g))) for g in ALL},
    "o_gap": ("only gap", dict(groups=("gap",))),
    "o_gap_fsrs": ("gap + fsrs state", dict(groups=("gap", "fsrs"))),
    "o_gap_card": ("gap + card history", dict(groups=("gap", "card"))),
    "o_gap_card_fsrs": ("gap + card + fsrs", dict(groups=("gap", "card", "fsrs"))),
    "o_gap_card_fsrs_user": ("gap + card + fsrs + user running", dict(groups=("gap", "card", "fsrs", "user"))),
    "l7": ("all, num_leaves 7", dict(num_leaves=7)),
    "l15": ("all, num_leaves 15", dict(num_leaves=15)),
    "l63": ("all, num_leaves 63", dict(num_leaves=63)),
    "lr02": ("all, learning rate 0.02", dict(learning_rate=0.02)),
    "md30": ("all, min_data_in_leaf 30", dict(min_data_in_leaf=30)),
    "md300": ("all, min_data_in_leaf 300", dict(min_data_in_leaf=300)),
    "nonfirst": ("all, train on all non-first rows (incl. same-day)", dict(train_rows="nonfirst")),
    "sm30": ("all, user/lexeme prior smoothing m=30", dict(m_user=30.0, m_lex=30.0)),
    "sm3": ("all, user/lexeme prior smoothing m=3", dict(m_user=3.0, m_lex=3.0)),
    "l15_md300": ("all, leaves 15, min_leaf 300", dict(num_leaves=15, min_data_in_leaf=300)),
    "l15_lr02": ("all, leaves 15, lr 0.02", dict(num_leaves=15, learning_rate=0.02)),
    "l15_md300_lr02": ("all, leaves 15, min_leaf 300, lr 0.02", dict(num_leaves=15, min_data_in_leaf=300, learning_rate=0.02)),
    # round 2: combine what helped in round 1 (nonfirst training rows, smaller trees, slower lr, low smoothing)
    "nf_sm3": ("nonfirst + smoothing m=3", dict(train_rows="nonfirst", m_user=3.0, m_lex=3.0)),
    "nf_l15": ("nonfirst + leaves 15", dict(train_rows="nonfirst", num_leaves=15)),
    "nf_l15_lr02": ("nonfirst + leaves 15 + lr 0.02", dict(train_rows="nonfirst", num_leaves=15, learning_rate=0.02)),
    "nf_l15_md300_lr02": ("nonfirst + leaves 15 + min_leaf 300 + lr 0.02", dict(train_rows="nonfirst", num_leaves=15, min_data_in_leaf=300, learning_rate=0.02)),
    "nf_l15_lr02_sm3": ("nonfirst + leaves 15 + lr 0.02 + smoothing 3", dict(train_rows="nonfirst", num_leaves=15, learning_rate=0.02, m_user=3.0, m_lex=3.0)),
    "nf_l63_lr02": ("nonfirst + leaves 63 + lr 0.02", dict(train_rows="nonfirst", num_leaves=63, learning_rate=0.02)),
    "nf_nouserprior": ("nonfirst, without the userprior group", dict(train_rows="nonfirst", groups=_drop("userprior"))),
    "nf_l15_lr02_sm3_nup": ("nonfirst + leaves 15 + lr 0.02 + smoothing 3, no userprior", dict(train_rows="nonfirst", num_leaves=15, learning_rate=0.02, m_user=3.0, m_lex=3.0, groups=_drop("userprior"))),
}
HEADER = ("\n## Task 7.3: GBDT\n\nLightGBM (`fsrs_benchmark/models/gbdt.py`), validation only; same split, scored set, FSRS-opt reference "
          "and bootstrap as Task 7.2. Every variant run, in order. Early stopping on an inner 15% user holdout of the fit data.\n\n"
          "| date | dataset | variant | description | log_loss | brier | auc | ece | fit+predict s | dLL vs FSRS-opt [95% CI] | rounds |\n"
          "|---|---|---|---|---|---|---|---|---|---|---|\n")


def verdict(r):
    return "WIN" if r["ci_hi"] < 0 else ("LOSS" if r["ci_lo"] > 0 else "TIE")


def select():
    tabs = {d: pd.read_csv(OUT / f"gbdt_{d}.csv").set_index("model") for d in ("anki", "duolingo")}
    both = sorted(set(tabs["anki"].index) & set(tabs["duolingo"].index))
    rows = []
    for m in both:
        ok = all(verdict(tabs[d].loc[m]) != "LOSS" for d in tabs)
        rows.append((m, ok, (tabs["anki"].loc[m, "log_loss"] + tabs["duolingo"].loc[m, "log_loss"]) / 2))
    rows.sort(key=lambda r: r[2])
    for m, ok, mean in rows:
        print(f"{m:22s} mean {mean:.5f} {'ok' if ok else 'LOSES'}")
    print("FROZEN:", next(m for m, ok, _ in rows if ok))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["anki", "duolingo"])
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--save-pred", default="", help="variant whose predictions go to pred_gbdt_<dataset>.parquet")
    args = ap.parse_args(argv)
    if args.select:
        return select()
    events = make_validation_split(build(args.dataset))
    print(f"{args.dataset}: {len(events)} rows (test dropped), fit {int(events.is_train.sum())}, "
          f"validation scored {int(events.scored.sum())}", flush=True)
    ref = pd.read_parquet(next(P6_CACHE.glob(f"{args.dataset}_val_fsrs_opt_*.parquet")))["p"]
    assert set(ref.index) == set(events.loc[events["scored"], "event_id"]), "scored set differs from Phase 6"
    OUT.mkdir(parents=True, exist_ok=True)
    UG = "row" if args.dataset == "anki" else "auto"   # Anki: individually timed reviews
    base = base_features(events, user_group=UG)
    rows = []
    for name in args.models:
        started = time.perf_counter()
        model = GBDT(seed=0, user_group=UG, **VARIANTS[name][1]).fit(events, base)
        pred = model.predict(events, base)
        secs = time.perf_counter() - started
        res = {"model": name, **evaluate(events, pred), "seconds": round(secs, 1), "rounds": int(model.best_iter_)}
        pred.to_frame("p").to_parquet(CACHE / f"{args.dataset}_pred_gbdt_{name}.parquet")
        if name == args.save_pred:
            pred.to_frame("p").to_parquet(OUT / f"pred_gbdt_{args.dataset}.parquet")
        ci = ci_vs(events, pred, ref)
        res.update(d_vs_opt=ci["diff"], ci_lo=ci["lo"], ci_hi=ci["hi"])
        rows.append(res)
        print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in res.items()}, flush=True)
        text = LOG.read_text() if LOG.exists() else ""
        with LOG.open("a") as f:
            if "## Task 7.3: GBDT" not in text:
                f.write(HEADER)
            f.write(f"| {date.today()} | {args.dataset} | {name} | {VARIANTS[name][0]} | {res['log_loss']:.5f} | "
                    f"{res['brier']:.5f} | {res['auc']:.4f} | {res['ece']:.4f} | {res['seconds']} | {_ci_text(res)} | {res['rounds']} |\n")
    table = pd.DataFrame(rows)
    path = OUT / f"gbdt_{args.dataset}.csv"
    if path.exists():
        old = pd.read_csv(path)
        table = pd.concat([old[~old["model"].isin(table["model"])], table], ignore_index=True)
    table.to_csv(path, index=False)


if __name__ == "__main__":
    main()
