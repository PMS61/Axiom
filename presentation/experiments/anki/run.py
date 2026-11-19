"""Run a small chronological FSRS versus GBDT benchmark on Anki."""

import argparse
from pathlib import Path
import sys

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from presentation.experiments.support import ROOT, score_predictions, scores, show_saved


def main():
    parser = argparse.ArgumentParser(description="Run the Axiom anki benchmark.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--saved", action="store_true", help="Display the historical run instead of fitting.")
    mode.add_argument("--demo", action="store_true", help="Generate local synthetic revlogs and run the same pipeline.")
    parser.add_argument("--users", type=int, default=4, help="Users in the live run (default: 4).")
    args = parser.parse_args()
    title = "Axiom Phase 7 · Anki held-out benchmark"
    if args.saved:
        show_saved(title, ROOT / "results/full_u100_c1000_final/metrics.csv", overall=True)
        return
    if args.users < 2:
        parser.error("--users must be at least 2")

    import pandas as pd

    from fsrs_benchmark import data, split
    from fsrs_benchmark.models.fsrs_model import FSRSModel
    from fsrs_benchmark.models.gbdt import FROZEN_KWARGS, GBDT

    if args.demo:
        from tempfile import TemporaryDirectory

        from presentation.experiments.srs_first.run import synthetic_revlog

        print(f"{title}\nLive synthetic sample · {args.users} users\n", flush=True)
        with TemporaryDirectory(prefix="axiom-anki-") as temp:
            reviews = []
            for user in range(args.users):
                folder = Path(temp) / f"user_id={user}"
                folder.mkdir()
                path = folder / "data.parquet"
                synthetic_revlog(user).to_parquet(path, index=False)
                reviews.append(data.load_user(path))
            raw = pd.concat(reviews, ignore_index=True)
    else:
        print(f"{title}\nLive Anki sample · {args.users} users\n", flush=True)
        try:
            paths = data.download_users(args.users, 0, ROOT / "data/anki10k")
        except RuntimeError as error:
            parser.exit(1, f"Anki data unavailable: {error}\n")
        raw = pd.concat([data.load_user(path) for path in paths], ignore_index=True)
    events = split.make_split(raw, max_cards=300, seed=0)
    models = [
        ("FSRS default", FSRSModel(optimize=False)),
        ("GBDT", GBDT(seed=0, **{**FROZEN_KWARGS, "user_group": 'row'})),
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
