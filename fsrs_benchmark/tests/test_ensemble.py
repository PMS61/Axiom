import numpy as np
import pandas as pd
import pytest

from fsrs_benchmark.experiments import ensemble as E


def _pair():
    idx = pd.Index(np.arange(200), name="event_id")
    rng = np.random.default_rng(0)
    y = (rng.random(200) < 0.7).astype(int)
    good = pd.Series(np.where(y == 1, 0.8, 0.4), index=idx)
    bad = pd.Series(np.full(200, 0.5), index=idx)
    return y, good, bad


def test_blend_endpoints_and_mix():
    _, a, b = _pair()
    assert E.blend(a, b, 1.0).equals(a) and E.blend(a, b, 0.0).equals(b)
    assert E.blend(a, b, 0.25).iloc[0] == pytest.approx(0.25 * a.iloc[0] + 0.75 * b.iloc[0])


def test_best_w_prefers_better_model_and_shared_w():
    y, good, bad = _pair()
    c = E.loss_curve(y, good, bad)
    assert len(c) == 21 and E.best_w(c) == 1.0
    assert E.shared_w([c, c[::-1]]) in E.GRID


def test_mismatched_index_rejected():
    _, a, b = _pair()
    with pytest.raises(ValueError):
        E.blend(a, b.iloc[::-1], 0.5)
