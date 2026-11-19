"""Run the Elo-band ablation with FSRS features masked in all or only stronger bands."""

import argparse
from pathlib import Path
import sys

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("none", "mixed"), default="none")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from presentation.experiments.srs_elo_experts.run import main as train_experts

    # The same trainer, quantiles, and scoring cohort are used in both variants.
    # Changing the mode masks FSRS-derived inputs in both fit and evaluation.
    train_experts(fsrs_mode=args.mode, out=args.out)


if __name__ == "__main__":
    main()
