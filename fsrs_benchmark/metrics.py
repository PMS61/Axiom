"""
metrics.py -- Scoring rules and calibration metrics, written from scratch in numpy.

Why these metrics: a model that predicts recall probabilities has to be judged on
three things at once. Log loss and Brier score measure overall probability quality.
AUC measures ranking (does it put forgotten cards before remembered ones?). ECE,
mean bias and the reliability curve measure calibration (are the probabilities
numerically right?). A model can rank well and still be badly calibrated, so all
of them are reported.

Every function takes array-likes: y in {0,1}, p in [0,1].
"""
import numpy as np
import pandas as pd


def _as_arrays(y, p):
    y = np.asarray(y, dtype=float).ravel()
    p = np.asarray(p, dtype=float).ravel()
    if y.shape != p.shape:
        raise ValueError(f"y and p must have the same length, got {y.shape} and {p.shape}")
    if y.size == 0:
        raise ValueError("empty input")
    # Reject bad values up front. Otherwise NaN silently falls into bin 0 or a
    # rank tie, and the metric comes out as a plausible-looking wrong number.
    if not np.all(np.isfinite(p)):
        raise ValueError("predictions contain NaN or inf")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("predictions must lie in [0, 1]")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("labels must be 0 or 1")
    return y, p


def log_loss(y, p, eps=1e-6):
    """Mean binary cross-entropy. p is clipped to [eps, 1-eps] so log(0) cannot occur."""
    y, p = _as_arrays(y, p)
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p):
    """Mean squared error between the probability and the 0/1 outcome."""
    y, p = _as_arrays(y, p)
    return float(np.mean((p - y) ** 2))


def auc(y, p):
    """Area under the ROC curve, via the Mann-Whitney U statistic.

    Why rank-based: it equals the probability that a random positive gets a higher
    score than a random negative, and it needs only one sort. Tied scores get their
    average rank, so a constant predictor scores exactly 0.5.
    Returns NaN when only one class is present, since AUC is then undefined.
    """
    y, p = _as_arrays(y, p)
    pos = y == 1
    n_pos = int(pos.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(p).rank(method="average").to_numpy()
    u = ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def _bin_index(p, n_bins):
    """Equal-width bin index in 0..n_bins-1. p == 1.0 goes into the last bin."""
    idx = np.floor(p * n_bins).astype(int)
    return np.clip(idx, 0, n_bins - 1)


def reliability_curve(y, p, n_bins=10):
    """Per-bin counts and means: the data behind a reliability diagram.

    Columns: bin_lo, bin_hi, n, mean_p, mean_y. Empty bins have n = 0 and NaN means.
    """
    y, p = _as_arrays(y, p)
    idx = _bin_index(p, n_bins)
    counts = np.bincount(idx, minlength=n_bins).astype(int)
    sum_p = np.bincount(idx, weights=p, minlength=n_bins)
    sum_y = np.bincount(idx, weights=y, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_p = np.where(counts > 0, sum_p / counts, np.nan)
        mean_y = np.where(counts > 0, sum_y / counts, np.nan)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    return pd.DataFrame({
        "bin_lo": edges[:-1], "bin_hi": edges[1:],
        "n": counts, "mean_p": mean_p, "mean_y": mean_y,
    })


def ece(y, p, n_bins=10):
    """Expected calibration error: bin-size-weighted mean of |mean_p - mean_y|."""
    rc = reliability_curve(y, p, n_bins)
    n_total = rc["n"].sum()
    gap = np.abs(rc["mean_p"] - rc["mean_y"]).fillna(0.0)
    return float(np.sum(rc["n"] * gap) / n_total)


def rmse_bins(y, p, n_bins=20):
    """RMSE over bins: sqrt of the bin-size-weighted mean of (mean_p - mean_y)^2.

    Optional metric from the srs-benchmark convention. Like ECE, but squared, so
    large calibration gaps in one bin count more.
    """
    rc = reliability_curve(y, p, n_bins)
    n_total = rc["n"].sum()
    sq = ((rc["mean_p"] - rc["mean_y"]) ** 2).fillna(0.0)
    return float(np.sqrt(np.sum(rc["n"] * sq) / n_total))


def mean_bias(y, p):
    """Mean predicted probability minus the observed recall rate.

    Positive means the model is too optimistic overall. It is zero when the average
    prediction matches the average outcome.
    """
    y, p = _as_arrays(y, p)
    return float(np.mean(p) - np.mean(y))


def summarize(y, p, n_bins=10):
    """All scalar metrics for one set of predictions, as a dict."""
    y, p = _as_arrays(y, p)
    return {
        "n": int(y.size),
        "log_loss": log_loss(y, p),
        "brier": brier(y, p),
        "auc": auc(y, p),
        "ece": ece(y, p, n_bins),
        "rmse_bins": rmse_bins(y, p),
        "mean_bias": mean_bias(y, p),
    }


METRICS = {
    "log_loss": log_loss,
    "brier": brier,
    "auc": auc,
    "ece": ece,
    "rmse_bins": rmse_bins,
    "mean_bias": mean_bias,
}


def _paired_bootstrap_reference(user_ids, y, p_a, p_b, metric, n_boot=1000, seed=0):
    """Slow reference: rebuilds each resample and recomputes the metric. Kept to test the fast path."""
    """Cluster bootstrap CI for metric(model A) - metric(model B).

    Why resample users, not events: reviews from one user are correlated (the same
    person, the same habits), so resampling single events would make the interval
    too narrow. Users are drawn with replacement; all of a drawn user's events come
    along. Model A and B are resampled with the same users, so the interval is for
    the paired difference.

    metric is a name from METRICS or a callable(y, p). Returns dict(diff, lo, hi),
    with a 95% percentile interval. Bootstrap samples where the metric is NaN
    (for example AUC with one class) are dropped.
    """
    fn = METRICS[metric] if isinstance(metric, str) else metric
    user_ids = np.asarray(user_ids).ravel()
    y, p_a = _as_arrays(y, p_a)
    _, p_b = _as_arrays(y, p_b)
    if user_ids.shape != y.shape:
        raise ValueError("user_ids must have one entry per event")

    # Index array per user, built once; each bootstrap just concatenates these.
    order = np.argsort(user_ids, kind="stable")
    sorted_ids = user_ids[order]
    uniq, starts = np.unique(sorted_ids, return_index=True)
    ends = np.append(starts[1:], len(sorted_ids))
    per_user = [order[s:e] for s, e in zip(starts, ends)]

    diff = fn(y, p_a) - fn(y, p_b)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(per_user), size=len(per_user))
        idx = np.concatenate([per_user[i] for i in pick])
        d = fn(y[idx], p_a[idx]) - fn(y[idx], p_b[idx])
        if np.isfinite(d):
            draws.append(d)
    if not draws:
        return {"diff": float(diff), "lo": float("nan"), "hi": float("nan")}
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"diff": float(diff), "lo": float(lo), "hi": float(hi)}


# ---------------------------------------------------------------------------
# Fast paired bootstrap. A resample only changes how many times each user is drawn,
# so each metric is computed as a weighted version of the full data: user weights
# w_e = count of e's user in the draw. Same RNG draws as the reference, so the results agree.
# ---------------------------------------------------------------------------

FAST_METRICS = ("log_loss", "brier", "auc", "ece", "rmse_bins", "mean_bias")


class _WeightedMetric:
    """One model's metric on the full data, evaluated for any per-event weight vector w.

    The sort (for AUC) and the bin indices (for ECE / RMSE) are computed once here,
    outside the bootstrap loop.
    """

    def __init__(self, name, y, p):
        self.name = name
        self.y = y
        self.p = p
        self.pos = y == 1
        if name == "log_loss":
            pc = np.clip(p, 1e-6, 1 - 1e-6)
            self.loss = -(y * np.log(pc) + (1 - y) * np.log(1 - pc))
        elif name == "brier":
            self.sq = (p - y) ** 2
        elif name == "ece":
            self.idx = _bin_index(p, 10)
            self.nb = 10
        elif name == "rmse_bins":
            self.idx = _bin_index(p, 20)
            self.nb = 20
        elif name == "auc":
            # One sort: groups of tied p values, ascending.
            _, self.grp = np.unique(p, return_inverse=True)
            self.n_groups = int(self.grp.max()) + 1
        elif name != "mean_bias":
            raise ValueError(f"no fast path for metric {name!r}")

    def value(self, w):
        """Metric on the resample described by weights w (w.sum() > 0). NaN if undefined."""
        total = w.sum()
        if self.name == "log_loss":
            return float(np.sum(w * self.loss) / total)
        if self.name == "brier":
            return float(np.sum(w * self.sq) / total)
        if self.name == "mean_bias":
            return float(np.sum(w * self.p) / total - np.sum(w * self.y) / total)
        if self.name in ("ece", "rmse_bins"):
            bw = np.bincount(self.idx, weights=w, minlength=self.nb)
            sp = np.bincount(self.idx, weights=w * self.p, minlength=self.nb)
            sy = np.bincount(self.idx, weights=w * self.y, minlength=self.nb)
            if self.name == "ece":
                return float(np.sum(np.abs(sp - sy)) / total)
            occupied = bw > 0
            term = np.zeros(self.nb)
            term[occupied] = (sp[occupied] - sy[occupied]) ** 2 / bw[occupied]
            return float(np.sqrt(np.sum(term) / total))
        # AUC: weighted Mann-Whitney. Ties count 0.5, as in auc().
        w_pos = np.bincount(self.grp, weights=w * self.pos, minlength=self.n_groups)
        w_neg = np.bincount(self.grp, weights=w * (~self.pos), minlength=self.n_groups)
        n_pos, n_neg = w_pos.sum(), w_neg.sum()
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        below = np.cumsum(w_neg) - w_neg  # negatives strictly below each tie group
        u = np.sum(w_pos * (below + 0.5 * w_neg))
        return float(u / (n_pos * n_neg))


def paired_bootstrap(user_ids, y, p_a, p_b, metric, n_boot=1000, seed=0):
    """Cluster bootstrap CI for metric(model A) - metric(model B). Same contract as the reference.

    Users are resampled with replacement (same draws as _paired_bootstrap_reference), and
    each draw is turned into a count per user, so metrics are computed with weights rather
    than by building the resampled rows. Named metrics in FAST_METRICS take this path;
    a callable metric falls back to the reference.
    """
    if not (isinstance(metric, str) and metric in FAST_METRICS):
        return _paired_bootstrap_reference(user_ids, y, p_a, p_b, metric, n_boot=n_boot, seed=seed)
    user_ids = np.asarray(user_ids).ravel()
    y, p_a = _as_arrays(y, p_a)
    _, p_b = _as_arrays(y, p_b)
    if user_ids.shape != y.shape:
        raise ValueError("user_ids must have one entry per event")

    fn = METRICS[metric]
    diff = fn(y, p_a) - fn(y, p_b)
    uniq, codes = np.unique(user_ids, return_inverse=True)
    codes = codes.ravel()
    n_users = len(uniq)
    fast_a = _WeightedMetric(metric, y, p_a)
    fast_b = _WeightedMetric(metric, y, p_b)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, n_users, size=n_users)  # identical draw to the reference
        counts = np.bincount(pick, minlength=n_users)
        w = counts[codes].astype(float)
        d = fast_a.value(w) - fast_b.value(w)
        if np.isfinite(d):
            draws.append(d)
    if not draws:
        return {"diff": float(diff), "lo": float("nan"), "hi": float("nan")}
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"diff": float(diff), "lo": float(lo), "hi": float(hi)}
