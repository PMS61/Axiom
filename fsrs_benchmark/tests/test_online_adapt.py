"""Tests for fsrs_benchmark/srsb/online_adapt.py (Phase 6, task 6.5a): the leak rule of the online layer,
lr=0 reproduces the GBDT, and first reviews are never predicted or updated."""
import numpy as np
import pytest

from fsrs_benchmark.srsb import online_adapt as O


def _toy(n=60, seed=0):
    rng = np.random.default_rng(seed)
    margin = rng.normal(0, 1.5, n)
    extra = {c: rng.normal(0, 1, n) for c in O.EXTRA}
    y = (rng.random(n) < 0.6).astype(np.float64)
    labelled = np.ones(n, bool)
    labelled[::7] = False                       # some first reviews
    return margin, extra, y, labelled


@pytest.mark.parametrize("variant", ["A", "B"])
def test_labels_at_and_after_t_do_not_change_p_up_to_t(variant):
    margin, extra, y, labelled = _toy()
    X = O.design(margin, extra, variant)
    w0 = O.w0_for(variant)
    base = O.online_kernel(X, y, labelled, w0, 0.03, O.LAMBDA)
    for k in (1, 15, 40, 58):
        y2 = y.copy()
        y2[k:] = 1.0 - y2[k:]                   # flip every label from row k on
        other = O.online_kernel(X, y2, labelled, w0, 0.03, O.LAMBDA)
        assert np.array_equal(base[:k + 1], other[:k + 1], equal_nan=True), f"row <= {k} changed"
        assert not np.array_equal(base[k + 1:], other[k + 1:], equal_nan=True)   # the change is visible later


def test_lr_zero_reproduces_the_gbdt_probability():
    margin, extra, y, labelled = _toy()
    for variant in ("A", "B"):
        p = O.online_kernel(O.design(margin, extra, variant), y, labelled, O.w0_for(variant), 0.0, O.LAMBDA)
        expect = 1.0 / (1.0 + np.exp(-margin))
        assert np.allclose(p[labelled], expect[labelled], rtol=1e-12)
        assert np.isnan(p[~labelled]).all()


def test_first_reviews_are_not_predicted_or_used_for_updates():
    margin, extra, y, labelled = _toy()
    base = O.online_kernel(O.design(margin, extra, "A"), y, labelled, O.w0_for("A"), 0.01, O.LAMBDA)
    y2 = y.copy()
    y2[~labelled] = 1.0 - y2[~labelled]         # labels of first reviews are not used
    other = O.online_kernel(O.design(margin, extra, "A"), y2, labelled, O.w0_for("A"), 0.01, O.LAMBDA)
    assert np.array_equal(base, other, equal_nan=True)


def test_update_moves_w_toward_the_labels():
    # one user who always answers 1 while the GBDT says 0.1: the layer must raise its predictions
    n = 200
    margin = np.full(n, np.log(0.1 / 0.9))
    y = np.ones(n)
    labelled = np.ones(n, bool)
    p = O.online_kernel(O.design(margin, {}, "A"), y, labelled, O.w0_for("A"), 0.03, O.LAMBDA)
    assert p[-1] > p[0] > 0.1
