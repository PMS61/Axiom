"""
Task E driver: v1 + v2 ensemble diagnostic, VALIDATION ONLY.

  python -m fsrs_benchmark.experiments.run_ensemble --dataset anki      # per-dataset curve + v2 preds
  python -m fsrs_benchmark.experiments.run_ensemble --finalize          # shared w, CI table, doc section
Needs pred_v1_<dataset>.parquet (from v1_validation) and the cached FSRS-opt validation preds.
Saves under results/phase6_validation/: pred_v2_<ds>.parquet, ensemble_<ds>_curve.csv,
pred_ens_best_<ds>.parquet, pred_ens_shared_<ds>.parquet, ensemble_table.csv.
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from fsrs_benchmark import metrics
from fsrs_benchmark.experiments import ensemble as E
from fsrs_benchmark.experiments.prepare_data import CACHE, build
from fsrs_benchmark.experiments.run_validation import ci_vs
from fsrs_benchmark.experiments.validation import make_validation_split

OUT = Path("results/phase6_validation")
DOC = Path("docs/reports/phase-6-experiments.md")
DATASETS = ("anki", "duolingo")


def _val(dataset):
    ev = make_validation_split(build(dataset))
    sc = ev[ev["scored"]].set_index("event_id").sort_index()
    sid = hashlib.sha256(np.sort(sc.index.to_numpy(np.int64)).tobytes()).hexdigest()[:12]
    return ev, sc, sid


def per_dataset(dataset):
    from fsrs_benchmark.models.ours_v2 import FROZEN_KWARGS, OursV2
    ev, sc, sid = _val(dataset)
    v2_path = OUT / f"pred_v2_{dataset}.parquet"
    if v2_path.exists():
        p2 = pd.read_parquet(v2_path)["p"]
    else:
        p2 = OursV2(**FROZEN_KWARGS).fit(ev).predict(ev).rename("p")
        p2.to_frame().to_parquet(v2_path)
    p1 = pd.read_parquet(OUT / f"pred_v1_{dataset}.parquet")["p"]
    idx = sc.index
    p1, p2 = p1.loc[idx], p2.loc[idx]
    curve = E.loss_curve(sc["y"].to_numpy(), p1, p2)
    pd.DataFrame({"w_v1": E.GRID, "log_loss": curve}).to_csv(OUT / f"ensemble_{dataset}_curve.csv", index=False)
    w = E.best_w(curve)
    E.blend(p1, p2, w).rename("p").to_frame().to_parquet(OUT / f"pred_ens_best_{dataset}.parquet")
    print(dataset, "best w (on v1) =", w, "min log loss", curve.min(), "v1", curve[-1], "v2", curve[0])


def finalize():
    curves = {d: pd.read_csv(OUT / f"ensemble_{d}_curve.csv")["log_loss"].to_numpy() for d in DATASETS}
    w_shared = E.shared_w(list(curves.values()))
    rows, lines = [], []
    for d in DATASETS:
        ev, sc, sid = _val(d)
        idx = sc.index
        p1 = pd.read_parquet(OUT / f"pred_v1_{d}.parquet")["p"].loc[idx]
        p2 = pd.read_parquet(OUT / f"pred_v2_{d}.parquet")["p"].loc[idx]
        opt = pd.read_parquet(CACHE / f"{d}_val_fsrs_opt_{sid}.parquet")["p"].loc[idx]
        w_best = E.best_w(curves[d])
        pe_best = E.blend(p1, p2, w_best)
        pe_sh = E.blend(p1, p2, w_shared)
        pe_sh.rename("p").to_frame().to_parquet(OUT / f"pred_ens_shared_{d}.parquet")
        for name, p in [("fsrs_opt", opt), ("v1", p1), ("v2", p2),
                        (f"ens_best(w_v1={w_best:.2f})", pe_best), (f"ens_shared(w_v1={w_shared:.2f})", pe_sh)]:
            r = dict(dataset=d, model=name, log_loss=metrics.log_loss(sc["y"], p.to_numpy()))
            if name != "fsrs_opt":
                ci = ci_vs(ev, p, opt)
                r.update(d_vs_opt=ci["diff"], ci_lo=ci["lo"], ci_hi=ci["hi"])
            rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "ensemble_table.csv", index=False)
    print(t.to_string(index=False))
    md = ["", "## Task E: ensemble (v1 + v2, validation only)", "",
          "p = w*p_v1 + (1-w)*p_v2, w on a 0.05 grid. v1 = OursModel (numba, 2000 sweeps, burn 1000, thin 4, seed 0), "
          "v2 = frozen ours_v2. Fit on the fit part, scored on validation events; no test rows. "
          "dLL vs FSRS-opt with user-cluster bootstrap 95% CI (1000 reps, seed 0).", "",
          f"Shared w (v1 weight) minimising the mean of the two datasets' val log loss: **{w_shared:.2f}**. "
          "Per-dataset best w is in the model names below.", "",
          "| dataset | model | log loss | dLL vs FSRS-opt [95% CI] |", "|---|---|---|---|"]
    for r in rows:
        if "d_vs_opt" in r:
            v = "WIN" if r["ci_hi"] < 0 else ("LOSS" if r["ci_lo"] > 0 else "TIE")
            ci = f"{r['d_vs_opt']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] {v}"
        else:
            ci = "reference"
        md.append(f"| {r['dataset']} | {r['model']} | {r['log_loss']:.5f} | {ci} |")
    with DOC.open("a") as f:
        f.write("\n".join(md) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--finalize", action="store_true")
    a = ap.parse_args()
    finalize() if a.finalize else per_dataset(a.dataset)
