"""Tests for OursModel's backend switch: numba stays the default, gpu builds the GPU model."""
import pytest

from fsrs_benchmark.models.dynamic_retention_gpu import DynamicRetentionModelGPU
from fsrs_benchmark.models.ours import DynamicRetentionModelDT, OursModel
from fsrs_benchmark.tests._synthetic_events import synthetic_events


def test_default_backend_is_numba():
    assert OursModel().backend == "numba"


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="backend must be one of"):
        OursModel(backend="cuda")


def test_gpu_backend_fits_and_predicts_scored_events():
    events = synthetic_events()
    model = OursModel(iters=6, burn=2, thin=1, max_draws=5, backend="gpu", device="cpu").fit(events)
    assert isinstance(model.model_, DynamicRetentionModelGPU)
    assert str(model.model_.device) == "cpu"
    p = model.predict(events)
    assert set(p.index) == set(events.loc[events["scored"], "event_id"].astype(int))
    assert ((p >= 1e-6) & (p <= 1 - 1e-6)).all()


def test_numba_backend_uses_the_cpu_model_class():
    model = OursModel(iters=3, burn=1, thin=1, max_draws=3).fit(synthetic_events())
    assert isinstance(model.model_, DynamicRetentionModelDT)
