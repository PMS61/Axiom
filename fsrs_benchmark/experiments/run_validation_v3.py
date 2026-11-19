"""Compatibility entry point for the experiment in `presentation/experiments/bayesian_variants`."""

import sys

from presentation.experiments.bayesian_variants import run as experiment

if __name__ == "__main__":
    experiment.main()
else:
    sys.modules[__name__] = experiment
