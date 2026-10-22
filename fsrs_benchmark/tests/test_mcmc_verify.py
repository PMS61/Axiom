"""Verifier checks for the MCMC diagnostics (ours.py _diagnostics, run.py diagnostics files).

Tiny fits only (iters=20, burn=10) so this stays light.
"""
import json

import numpy as np
import pandas as pd
import pytest

from fsrs_benchmark.models import ours as ours_mod
from fsrs_benchmark.models.ours import OursModel, _diagnostics
from fsrs_benchmark.tests._real import real_events


@pytest.fixture(scope="module")
def tiny_events():
    ev = real_events()
    users = sorted(ev["user_id"].unique())[:2]
    return ev[ev["user_id"].isin(users)].copy()


@pytest.mark.parametrize("use_kappa", [False, True])
def test_diagnostics_are_strict_json(tiny_events, use_kappa):
    m = OursModel(use_kappa=use_kappa, iters=20, burn=10).fit(tiny_events)
    text = json.dumps(m.diagnostics_, allow_nan=False)  # NaN would make this raise
    assert "NaN" not in text and "Infinity" not in text


@pytest.mark.parametrize("use_kappa", [False, True])
def test_kappa_only_in_kappa_variant(tiny_events, use_kappa):
    m = OursModel(use_kappa=use_kappa, iters=20, burn=10).fit(tiny_events)
    assert ("kappa" in m.diagnostics_) == use_kappa


def test_summary_fields_match_model_summary(tiny_events):
    m = OursModel(iters=20, burn=10).fit(tiny_events)
    raw = m.model_.summary()["lam"]
    assert m.diagnostics_["lam"]["mean"] == pytest.approx(raw["mean"])
    assert m.diagnostics_["lam"]["lo"] == pytest.approx(raw["lo"])
    assert m.diagnostics_["lam"]["hi"] == pytest.approx(raw["hi"])
    assert m.diagnostics_["n_samples"] == len(m.model_.samples["lam"])


def test_mh_acc_rate_is_the_average_over_sweeps(tiny_events):
    """Documented as the Metropolis-Hastings acceptance rate of the run. The model overwrites
    self.acc_rate on every Gibbs sweep, so the reported value must be the mean over sweeps,
    not just the last sweep. Expected to fail while the model stores only the last sweep."""
    seen = []

    class Recording(ours_mod.DynamicRetentionModelDT):
        def __setattr__(self, name, value):
            if name == "acc_rate" and np.isfinite(value):
                seen.append(float(value))
            object.__setattr__(self, name, value)

    original = ours_mod.DynamicRetentionModelDT
    ours_mod.DynamicRetentionModelDT = Recording
    try:
        m = OursModel(iters=20, burn=10).fit(tiny_events)
    finally:
        ours_mod.DynamicRetentionModelDT = original
    assert len(seen) == 20
    assert m.diagnostics_["mh_acc_rate"] == pytest.approx(np.mean(seen), abs=1e-12)



def _fit_samples(tiny_events, use_kappa, wrap):
    original = ours_mod._record_lam_acceptance
    if not wrap:
        ours_mod._record_lam_acceptance = lambda sampler, **kwargs: []
    try:
        m = OursModel(use_kappa=use_kappa, iters=20, burn=10, seed=3).fit(tiny_events)
    finally:
        ours_mod._record_lam_acceptance = original
    return m.model_.samples


@pytest.mark.parametrize("use_kappa", [False, True])
def test_wrapper_does_not_change_the_samples(tiny_events, use_kappa):
    """Same seed with and without the acceptance wrapper must give identical draws."""
    with_w = _fit_samples(tiny_events, use_kappa, wrap=True)
    without = _fit_samples(tiny_events, use_kappa, wrap=False)
    assert with_w.keys() == without.keys()
    for k in with_w:
        np.testing.assert_array_equal(np.asarray(with_w[k]), np.asarray(without[k]), err_msg=k)


def test_acceptance_record_is_per_instance(tiny_events):
    """Fitting one model must not wrap, record into, or read from another model's sampler."""
    a = OursModel(iters=20, burn=10, seed=1)
    b = OursModel(iters=20, burn=10, seed=2)
    a.fit(tiny_events)
    b.fit(tiny_events)
    assert a.model_ is not b.model_
    assert "_update_lam_sigma2" in vars(a.model_) and "_update_lam_sigma2" in vars(b.model_)
    assert vars(a.model_)["_update_lam_sigma2"] is not vars(b.model_)["_update_lam_sigma2"]
    assert "_update_lam_sigma2" not in vars(ours_mod.DynamicRetentionModelDT.__new__(ours_mod.DynamicRetentionModelDT))
    assert a.diagnostics_["mh_acc_rate"] is not None and b.diagnostics_["mh_acc_rate"] is not None
