"""Task E helper: linear ensemble p = w*p_a + (1-w)*p_b and the w grid search (validation only)."""
import numpy as np

from fsrs_benchmark import metrics

GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)


def blend(p_a, p_b, w):
    """Series blend; both must be indexed by the same event_ids."""
    if not p_a.index.equals(p_b.index):
        raise ValueError("predictions must cover the same events in the same order")
    return w * p_a + (1.0 - w) * p_b


def loss_curve(y, p_a, p_b, grid=GRID):
    """Log loss for each w in grid (array aligned with grid)."""
    return np.array([metrics.log_loss(y, blend(p_a, p_b, w).to_numpy()) for w in grid])


def best_w(curve, grid=GRID):
    return float(grid[int(np.argmin(curve))])


def shared_w(curves, grid=GRID):
    """w minimising the mean of several datasets' log-loss curves."""
    return best_w(np.mean(np.vstack(curves), axis=0), grid)
