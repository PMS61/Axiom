"""Compatibility entry point for the experiment in `presentation/experiments/sampler_diagnostics`."""

import sys

from presentation.experiments.sampler_diagnostics import run as experiment

if __name__ == "__main__":
    experiment.main()
else:
    sys.modules[__name__] = experiment
