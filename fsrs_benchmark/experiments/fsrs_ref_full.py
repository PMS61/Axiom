"""FSRS reference predictions on FULL Duolingo validation (fit/val split, no test rows).
  python -m fsrs_benchmark.experiments.fsrs_ref_full opt|default [workers]"""
import sys
import pandas as pd
from fsrs_benchmark.experiments.run_validation_v3 import CACHE, build
from fsrs_benchmark.experiments.validation import make_validation_split
from fsrs_benchmark.models.fsrs_model import FSRSModel

kind = sys.argv[1]
workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
ev = make_validation_split(build("duolingo_full"))
pred = FSRSModel(optimize=(kind == "opt"), n_workers=workers).fit(ev).predict(ev)
pred.to_frame("p").to_parquet(CACHE / ("duolingo_full_val_fsrs_ref.parquet" if kind == "opt" else "duolingo_full_val_fsrs_default.parquet"))
print("done", kind, len(pred))
