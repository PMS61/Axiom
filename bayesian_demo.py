"""JSON bridge for the Bayesian recall interface."""

import json
import sys

from presentation.experiments.bayesian_recall.run import fit


if __name__ == "__main__":
    print(json.dumps(fit(json.loads(sys.argv[1]))))
