"""FSRSModel.predict in worker processes must give the same numbers as the serial run."""
import numpy as np

from fsrs_benchmark.models.fsrs_model import FSRSModel
from fsrs_benchmark.tests._synthetic_events import synthetic_events


def test_parallel_predict_matches_serial():
    events = synthetic_events(n_users=4)
    serial = FSRSModel(optimize=False, n_workers=1).fit(events).predict(events)
    parallel = FSRSModel(optimize=False, n_workers=2).fit(events).predict(events)
    assert serial.index.equals(parallel.index)
    assert np.allclose(serial.to_numpy(), parallel.to_numpy(), rtol=0, atol=1e-12)
