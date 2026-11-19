"""Compatibility entry point for the experiment in `presentation/experiments/gbdt_ablation`."""

import sys

from presentation.experiments.gbdt_ablation import run as experiment

if __name__ == "__main__":
    experiment.main()
else:
    sys.modules[__name__] = experiment
