"""Compatibility entry point for the experiment in `presentation/experiments/srs_expert_gate`."""

import sys

from presentation.experiments.srs_expert_gate import run as experiment

if __name__ == "__main__":
    experiment.main()
else:
    sys.modules[__name__] = experiment
