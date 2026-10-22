"""
hlr.py -- Half-Life Regression baseline (Settles & Meeder, ACL 2016), trained as in the
official implementation.

Why this model: HLR is the simple, well-known baseline for spaced repetition. It
assumes a forgetting curve p = 2^(-t/h), where t is the days since the last review and
h is the half-life, with h = 2^(w . x). The benchmark compares FSRS and our model
against it, so it must use the same events and the same no-leakage rules.

Training replicates duolingo/halflife-regression, experiment.py (MIT), its
HalfLifeRegression.train_update: SGD with the paper's loss (squared error on p plus
hlwt times squared error on h, with L2 on the weights). The official code:
  - starts from zero weights;
  - clips h to [MIN_HALF_LIFE, MAX_HALF_LIFE] = [15 min, 274 days] (hclip) and p to
    [1e-4, 1 - 1e-4] (pclip), and uses h_obs = hclip(-t / log2(pclip(y)));
  - uses no gradient mask at the clip, so the clip is applied as a value clamp only;
  - uses per-feature learning rates lrate / sqrt(1 + fcount_k) and halves the rate on
    recall (factor 1 / (1 + pclip(y)));
  - defaults: lrate = 0.001, hlwt = 0.01, l2wt = 0.1, sigma = 1.

Deviations from the official code (documented):
  - Epochs. The official code makes one shuffled pass. Our TRAIN set is small, so we
    repeat shuffled passes (seeded) until the largest weight change over one pass is
    below `tol` (1e-4), with at most `max_epochs` (1000). On the two-user run it
    converges after about 600 passes (about 8 s). The mean train loss is not
    used as the stop rule: with the h term on binary labels it oscillates from pass to
    pass (about 83 to 88), so it never settles below a tolerance. `epochs_` says how
    many passes ran; if it equals max_epochs the fit did not converge.
  - Features. The official code also has per-lexeme indicator features (not available
    here). We use only fv = [sqrt(1 + n_right), sqrt(1 + n_wrong), 1 (bias)].

Features only use past history: n_right and n_wrong count the card's PRIOR kept reviews
(same user and card, strictly earlier in review_idx). Same-day and first reviews
count as history.
"""
import numba as nb
import numpy as np
import pandas as pd

P_MIN = 1e-4
P_MAX = 1.0 - P_MIN
MIN_HALF_LIFE = 15.0 / 1440.0  # days (15 minutes), as in the official code
MAX_HALF_LIFE = 274.0          # days, as in the official code
LN2 = np.log(2.0)


def hclip(h):
    return np.clip(h, MIN_HALF_LIFE, MAX_HALF_LIFE)


def pclip(p):
    return np.clip(p, P_MIN, P_MAX)


def observed_half_life(dt, y):
    """h_obs = hclip(-dt / log2(pclip(y))), as in the official code."""
    return hclip(-dt / np.log2(pclip(y)))


def features(events):
    """Feature matrix X (one row per event, indexed by event_id) from prior history only.

    Each card is identified by (user_id, card_id), because card ids are per user.
    Columns: sqrt(1 + n_right), sqrt(1 + n_wrong), 1 (bias).
    """
    ordered = events.sort_values(["user_id", "card_id", "review_idx"])
    keys = [ordered["user_id"], ordered["card_id"]]
    y = ordered["y"].astype(float)
    wrong = 1.0 - y
    # Cumulative counts include the current row; subtract it so only the past is counted.
    n_right = y.groupby(keys).cumsum() - y
    n_wrong = wrong.groupby(keys).cumsum() - wrong
    return pd.DataFrame({
        "x0": np.sqrt(1.0 + n_right).to_numpy(),
        "x1": np.sqrt(1.0 + n_wrong).to_numpy(),
        "x2": 1.0,
    }, index=ordered["event_id"].to_numpy())


def sgd_update(w, fcounts, x, t, y, h_obs, lrate=0.001, hlwt=0.01, l2wt=0.1, sigma=1.0):
    """One official train_update step, in place on w (list) and fcounts (list).

    Mirrors experiment.py: h = hclip(2^(w.x)), p = pclip(2^(-t/h)), then per feature k:
      rate = (1 / (1 + pclip(y))) * lrate / sqrt(1 + fcounts[k])
      w_k -= rate * (dlp_dw + hlwt * dlh_dw) * x_k + rate * l2wt * w_k / sigma^2
    with dlp_dw = 2 (p - y) ln2^2 p (t / h) and dlh_dw = 2 (h - h_obs) ln2 h.
    """
    dot = sum(wk * xk for wk, xk in zip(w, x))
    h = min(max(2.0 ** dot, MIN_HALF_LIFE), MAX_HALF_LIFE)
    p = min(max(2.0 ** (-t / h), P_MIN), P_MAX)
    p_obs = pclip(y)  # the official code uses the clipped observed p in loss and rate
    dlp_dw = 2.0 * (p - p_obs) * LN2 ** 2 * p * (t / h)
    dlh_dw = 2.0 * (h - h_obs) * LN2 * h
    for k in range(len(w)):
        rate = (1.0 / (1.0 + p_obs)) * lrate / np.sqrt(1.0 + fcounts[k])
        w[k] -= rate * dlp_dw * x[k]
        w[k] -= rate * hlwt * dlh_dw * x[k]
        w[k] -= rate * l2wt * w[k] / sigma ** 2
        fcounts[k] += 1
    return w, fcounts


@nb.njit(cache=True)
def _sgd_epoch(w, fcounts, X, t, y, h_obs, order, lrate, hlwt, l2wt, sigma):
    """One shuffled pass of sgd_update over the rows in `order`, compiled with numba.

    Same arithmetic and the same sequential order as sgd_update (the verifier tests check
    this against the official arithmetic). Writes into w and fcounts in place.
    """
    ln2 = np.log(2.0)
    for idx in range(order.shape[0]):
        i = order[idx]
        dot = 0.0
        for k in range(w.shape[0]):
            dot += w[k] * X[i, k]
        h = min(max(2.0 ** dot, MIN_HALF_LIFE), MAX_HALF_LIFE)
        p_obs = min(max(y[i], P_MIN), P_MAX)
        p = min(max(2.0 ** (-t[i] / h), P_MIN), P_MAX)
        dlp_dw = 2.0 * (p - p_obs) * ln2 ** 2 * p * (t[i] / h)
        dlh_dw = 2.0 * (h - h_obs[i]) * ln2 * h
        for k in range(w.shape[0]):
            rate = (1.0 / (1.0 + p_obs)) * lrate / np.sqrt(1.0 + fcounts[k])
            w[k] -= rate * dlp_dw * X[i, k]
            w[k] -= rate * hlwt * dlh_dw * X[i, k]
            w[k] -= rate * l2wt * w[k] / sigma ** 2
            fcounts[k] += 1


class HLRModel:
    """Half-life regression with the common interface: fit(events) and predict(events).

    Attributes after fit():
      theta_    the learned weights (length 3)
      epochs_   number of passes over TRAIN rows
      loss_history_  mean train loss after each pass (monitor only, not the stop rule)
    """

    def __init__(self, lrate=0.001, hlwt=0.01, l2wt=0.1, sigma=1.0,
                 max_epochs=1000, tol=1e-4, seed=0):
        self.lrate = lrate
        self.hlwt = hlwt
        self.l2wt = l2wt
        self.sigma = sigma
        self.max_epochs = max_epochs
        self.tol = tol
        self.seed = seed
        self.theta_ = None
        self.epochs_ = 0
        self.loss_history_ = []

    def _loss(self, w, X, t, y, h_obs):
        """Mean train loss: squared error on p plus hlwt times squared error on h."""
        h = hclip(np.exp2(np.clip(X @ w, -60.0, 60.0)))
        p = pclip(np.exp2(-t / h))
        return float(np.mean((p - y) ** 2) + self.hlwt * np.mean((h - h_obs) ** 2))

    def fit(self, events):
        """Train on TRAIN, non-first, inter-day rows (see module docstring)."""
        X_all = features(events)
        mask = (events["is_train"].astype(bool) & ~events["is_first"].astype(bool)
                & (events["elapsed_days"] >= 1)).to_numpy()
        rows = events.loc[mask]
        X = X_all.loc[rows["event_id"].to_numpy()].to_numpy()
        t = rows["dt_days"].to_numpy(dtype=float)
        y = rows["y"].to_numpy(dtype=float)
        h_obs = observed_half_life(t, y)

        if len(y) == 0:  # no train rows: keep the zero weights, no passes, no loss to average
            self.theta_ = np.zeros(X_all.shape[1])
            self.epochs_ = 0
            self.loss_history_ = []
            return self

        rng = np.random.default_rng(self.seed)
        # The SGD steps are sequential, so each pass runs as a numba loop (same arithmetic
        # as sgd_update, which the verifier tests check against the official formulas).
        Xf = np.ascontiguousarray(X, dtype=float)
        wf = np.zeros(X.shape[1]); cf = np.zeros(X.shape[1])
        prev_w = wf.copy()
        self.loss_history_ = []
        epochs = 0
        for epoch in range(self.max_epochs):
            order = rng.permutation(len(y)).astype(np.int64)
            _sgd_epoch(wf, cf, Xf, t, y, h_obs, order, self.lrate, self.hlwt, self.l2wt, self.sigma)
            epochs = epoch + 1
            cur = wf.copy()
            self.loss_history_.append(self._loss(cur, X, t, y, h_obs))
            if np.abs(cur - prev_w).max() < self.tol:
                break
            prev_w = cur
        self.theta_ = wf.copy()
        self.epochs_ = epochs
        return self

    def predict(self, events):
        """Recall probabilities for scored rows, indexed by event_id."""
        if self.theta_ is None:
            raise RuntimeError("call fit() before predict()")
        scored = events.loc[events["scored"].astype(bool)]
        X = features(events).loc[scored["event_id"].to_numpy()].to_numpy()
        t = scored["dt_days"].to_numpy(dtype=float)
        h = hclip(np.exp2(np.clip(X @ self.theta_, -60.0, 60.0)))
        p = pclip(np.exp2(-t / h))
        out = pd.Series(p, index=pd.Index(scored["event_id"].to_numpy(), name="event_id"), name="p")
        return out.sort_index()
