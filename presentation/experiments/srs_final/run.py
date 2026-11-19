"""Run the frozen GBDT and Elo-style update on a synthetic stream or the actual test cohort."""

import argparse
import json
import math
from pathlib import Path
import sys

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from presentation.experiments.support import ROOT, scores, table

ARTIFACTS = ROOT / "results/srsb_beat"


def demo():
    import lightgbm as lgb
    import numpy as np
    import pandas as pd

    from fsrs_benchmark.metrics import summarize
    from fsrs_benchmark.models.gbdt import base_features
    from fsrs_benchmark.srsb.features_plus import plus_features
    from presentation.experiments.srs_first.run import FEATURE_COLS, review_stream, synthetic_revlog
    from presentation.experiments.srs_online_adaptation.run import (
        EXTRA, FINAL_LR, FINAL_VARIANT, LAMBDA, design, online_kernel, w0_for,
    )

    model = lgb.Booster(model_file=str(ARTIFACTS / "sweep_a_l63_side.jsonl.model.txt"))
    info = json.loads((ARTIFACTS / "sweep_a_l63_side.jsonl.run_info.json").read_text())
    stream = review_stream(synthetic_revlog(42), 128)
    base = base_features(stream, span=20, user_group="row")[FEATURE_COLS]
    features = pd.concat([base, plus_features(stream)], axis=1)

    # Score the same reviews under the two models. Feed every labelled review
    # into the online update, including warm-up rows that are outside scoring.
    margins = model.predict(features[info["features"]], raw_score=True, num_iteration=info["best_iter"])
    outcomes = stream["y"].to_numpy(dtype=float)
    labelled = (~stream["is_first"]).to_numpy()
    scored = labelled & (np.arange(len(stream)) >= len(stream) // 2)
    inputs = design(margins, {name: features[name].to_numpy() for name in EXTRA}, FINAL_VARIANT)
    online = online_kernel(inputs, outcomes, labelled, w0_for(FINAL_VARIANT), FINAL_LR, LAMBDA)
    baseline = 1 / (1 + np.exp(-margins))

    print("Frozen GBDT + online update · live inference on synthetic reviews")
    print(f"{labelled.sum():,} labelled reviews · {scored.sum():,} scored in the second half\n")
    scores([
        {"model": "Frozen GBDT", **summarize(outcomes[scored], baseline[scored])},
        {"model": "Frozen GBDT + online", **summarize(outcomes[scored], online[scored])},
    ])


def show_saved():
    path = ARTIFACTS / "final/online_test1000.jsonl.comparison.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    print_comparison(result, "saved test result")


def print_comparison(result, mode):
    print(f"Public srs-benchmark · {mode} · {result['n_users']:,} matched users\n")
    if not result["n_users"]:
        return
    print(f"{result['n_rows']:,} reviews\n")

    def metric(value):
        return f"{value:.5f}" if value is not None and math.isfinite(value) else "—"

    table(
        ("Model", "Weighted log loss ↓", "Weighted AUC ↑"),
        ((name, metric(row["weighted"]), metric(row["auc_weighted"]))
         for name, row in sorted(result["table"].items(), key=lambda item: item[1]["weighted"])),
    )
    print()
    table(
        ("Final model minus reference", "Δ log loss", f"{result['ci']:.0%} paired interval"),
        ((name.removeprefix("online - "), f"{row['mean']:+.5f}", f"[{row['lo']:+.5f}, {row['hi']:+.5f}]")
         for name, row in result["bootstrap"].items()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--saved", action="store_true", help="Show the historical held-out scores.")
    mode.add_argument("--live", action="store_true", help="Recompute the frozen test on the actual cohort.")
    parser.add_argument("--data", type=Path, default=ROOT / "data/anki10k")
    parser.add_argument("--out", type=Path, help="New JSONL path for the real test output.")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.saved:
        show_saved()
    elif args.live:
        if not args.out or args.out.exists():
            parser.error("--live requires --out pointing to a new file")
        if not (args.data / "revlogs").is_dir():
            parser.error("--live requires the Anki revlogs and side tables under --data")
        from fsrs_benchmark.srsb.compare import DEFAULT_RESULT_DIR
        from presentation.experiments.srs_feature_sweep.run import PUBLISHED
        from presentation.experiments.srs_online_adaptation.run import run_final

        missing = [name for name in PUBLISHED.values() if not (DEFAULT_RESULT_DIR / f"{name}.jsonl").is_file()]
        if missing:
            parser.error(f"Missing published results under {DEFAULT_RESULT_DIR}: {', '.join(missing)}")

        users = [int(user) for user in (ARTIFACTS / "final/test_ids.txt").read_text().split()]
        result = run_final(
            ARTIFACTS / "sweep_a_l63_side.jsonl.model.txt",
            ARTIFACTS / "sweep_a_l63_side.jsonl.run_info.json",
            args.data, args.out, users, workers=args.workers,
            gbdt_jsonl=ARTIFACTS / "final/gbdt_test1000.jsonl", final=True,
        )
        print_comparison(result, "live held-out evaluation")
        print(f"Per-user results written to {args.out}")
    else:
        demo()


if __name__ == "__main__":
    main()
