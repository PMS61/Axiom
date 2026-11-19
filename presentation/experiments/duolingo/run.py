"""Run a small chronological FSRS versus GBDT benchmark on Duolingo."""

import argparse
from pathlib import Path
import sys

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from presentation.experiments.support import ROOT, score_predictions, scores, show_saved


def main():
    parser = argparse.ArgumentParser(description="Run the Axiom duolingo benchmark.")
    parser.add_argument("--saved", action="store_true", help="Display the historical run instead of fitting.")
    parser.add_argument("--users", type=int, default=4, help="Users in the live run (default: 4).")
    args = parser.parse_args()
    title = "Axiom Phase 7 · Duolingo held-out benchmark"
    if args.saved:
        show_saved(title, ROOT / "results/duolingo_final/metrics.csv", overall=True)
        return
    if args.users < 2:
        parser.error("--users must be at least 2")

    from fsrs_benchmark import duolingo, split
    from fsrs_benchmark.models.fsrs_model import FSRSModel
    from fsrs_benchmark.models.gbdt import FROZEN_KWARGS, GBDT

    print(f"{title}\nLive sample · {args.users} users · FSRS default and GBDT\n", flush=True)
    path = ROOT / "data/duolingo/settles.acl16.learning_traces.13m.csv.gz"
    if not path.is_file():
        parser.exit(1, f"Missing Duolingo data: {path}\n")
    counts, t0 = duolingo.scan(path, path.parent / "duolingo_counts.parquet")
    users = duolingo.sample_users(counts, args.users, 0, min_rows=100)
    raw = duolingo.load_users(path, users, counts, t0, with_lexeme=True, with_ts=True)
    events = split.make_split(raw, max_cards=300, seed=0)
    models = [
        ("FSRS default", FSRSModel(optimize=False)),
        ("GBDT", GBDT(seed=0, **{**FROZEN_KWARGS, "user_group": 'auto'})),
    ]
    rows = []
    for name, model in models:
        print(f"Fitting and scoring {name}…", flush=True)
        predictions = model.fit(events).predict(events)
        rows.append({"model": name, **score_predictions(events, predictions)})
    print()
    scores(rows)


if __name__ == "__main__":
    main()
