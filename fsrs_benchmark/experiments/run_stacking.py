"""Compatibility entry point for the experiment in `presentation/experiments/model_stacking`."""

import sys

from presentation.experiments.model_stacking import run as experiment

if __name__ == "__main__":
    experiment.main()
else:
    sys.modules[__name__] = experiment
