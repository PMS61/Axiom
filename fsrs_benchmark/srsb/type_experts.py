"""Compatibility entry point for the experiment in `presentation/experiments/srs_user_experts`."""

import sys

from presentation.experiments.srs_user_experts import run as experiment

if __name__ == "__main__":
    experiment.main()
else:
    sys.modules[__name__] = experiment
