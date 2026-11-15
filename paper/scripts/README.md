# Paper scripts

Everything in `paper/tables/` and `paper/figures/` is generated here. Do not edit those files by hand.
Run from the repo root with the project venv.

Run order:

1. `.venv/bin/python -m fsrs_benchmark.paper_export ...` writes `paper/data/*.csv` and `paper/FACTS.md`
   (run once per dataset, see its docstring).
2. `.venv/bin/python paper/scripts/make_tables.py` reads `paper/data/*.csv` and writes
   `paper/tables/{overall,paired,seeds}.tex`.
3. `.venv/bin/python paper/scripts/make_figures.py` reads `paper/data/*_gaps.csv` and
   `results/paper_{anki,duolingo}/{predictions,events}.parquet`, and writes
   `paper/figures/*.pdf` and `paper/data/*_reliability.csv`.
4. `paper/build.sh` builds `paper/build/main.pdf`.

Tests: `.venv/bin/python -m pytest -q fsrs_benchmark/tests/test_paper_tables.py`.
