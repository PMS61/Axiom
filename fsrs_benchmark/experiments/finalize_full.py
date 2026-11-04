"""Compute metrics + CIs for saved full-Duolingo variant predictions vs a reference and log them.
  python -m fsrs_benchmark.experiments.finalize_full REF_PARQUET REF_NAME f0 f1 ..."""
import sys
from datetime import date
import pandas as pd
from fsrs_benchmark.experiments.run_validation import _ci_text, ci_vs, evaluate
from fsrs_benchmark.experiments.run_validation_v3 import CACHE, LOG, OUT, VARIANTS, build
from fsrs_benchmark.experiments.validation import make_validation_split

ref_path, ref_name, names = sys.argv[1], sys.argv[2], sys.argv[3:]
events = make_validation_split(build("duolingo_full"))
ref = pd.read_parquet(ref_path)["p"]
rows = []
for n in names:
    pred = pd.read_parquet(CACHE / f"duolingo_full_pred_{n}.parquet")["p"]
    res = {"model": n, "ref": ref_name, **evaluate(events, pred)}
    ci = ci_vs(events, pred, ref)
    res.update(d_vs_ref=ci["diff"], ci_lo=ci["lo"], ci_hi=ci["hi"])
    res["d_vs_opt"] = res["d_vs_ref"]
    rows.append(res)
    with LOG.open("a") as f:
        f.write(f"| {date.today()} | duolingo_full | {n} | {VARIANTS[n][0]} | {res['log_loss']:.5f} | {res['brier']:.5f} | {res['auc']:.4f} | "
                f"{res['ece']:.4f} | - | vs {ref_name}: {_ci_text(res)} | full-Duolingo validation (all users) |\n")
    print(res, flush=True)
pd.DataFrame(rows).to_csv(OUT / f"duolingo_full_vs_{ref_name}.csv", index=False)
