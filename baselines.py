"""
baselines.py

Sanity-check baselines that every result table should include.

  ConstantBaseline      predict the training recall rate everywhere
  AlwaysOneBaseline     predict ~1 everywhere (what a model that "never expects
                        forgetting" scores; HLR's MAE must beat THIS to mean anything)
  GroupMeanBaseline     shrunken per-item or per-user mean recall
  LaggedLogistic        logistic regression on [lagged outcome, sqrt right/wrong
                        counts, (log dt)] + item/user fixed effects. This is the
                        fair control for "is the latent STATE doing the work, or
                        just the lagged-outcome feature?"

Interface for every model in this project:
    model.fit(train_seqs)            -> self
    model.predict(test_seqs)         -> list of 1-D arrays aligned with test_seqs

IMPORTANT: test sequences are the held-out SUFFIX of a (user,item) pair whose
PREFIX is in train_seqs. `build_history` captures the prefix's counts and last
outcome so that test-time features CONTINUE from the training history instead
of restarting at zero.
"""
import numpy as np
from scipy import sparse
from scipy.optimize import minimize


# ------------------------------------------------------------------ shared helpers
def build_history(train_seqs):
    hist = {}
    for s in train_seqs:
        r = np.asarray(s['r']).astype(int)
        hist[(s['user'], s['item'])] = dict(
            n=len(r), right=int(r.sum()), wrong=int(len(r) - r.sum()), last_r=int(r[-1]))
    return hist


def count_features(seq, hist=None):
    """(n_prior, right_prior, wrong_prior) before each event, continuing from
    `hist` (the training prefix of the same pair) when given."""
    r = np.asarray(seq['r']).astype(float)
    n0, r0, w0 = (hist['n'], hist['right'], hist['wrong']) if hist else (0, 0, 0)
    cum_r = np.concatenate([[0.0], np.cumsum(r)[:-1]])
    cum_w = np.concatenate([[0.0], np.cumsum(1.0 - r)[:-1]])
    return n0 + np.arange(len(r), dtype=float), r0 + cum_r, w0 + cum_w


def lagged_outcome(seq, hist=None):
    r = np.asarray(seq['r']).astype(float)
    first = float(hist['last_r']) if hist else float(np.asarray(seq['x'])[0])
    return np.concatenate([[first], r[:-1]])


# ------------------------------------------------------------------ trivial baselines
class ConstantBaseline:
    def fit(self, train_seqs):
        self.p = float(np.mean(np.concatenate([np.asarray(s['r']) for s in train_seqs])))
        return self

    def predict(self, test_seqs):
        return [np.full(len(s['r']), self.p) for s in test_seqs]


class AlwaysOneBaseline:
    def __init__(self, p=0.999):
        self.p = p

    def fit(self, train_seqs):
        return self

    def predict(self, test_seqs):
        return [np.full(len(s['r']), self.p) for s in test_seqs]


class GroupMeanBaseline:
    """Shrunken mean recall per 'item' or 'user': (sum r + k*global)/(n + k)."""
    def __init__(self, by="item", k=5.0):
        self.by, self.k = by, k

    def fit(self, train_seqs):
        allr = np.concatenate([np.asarray(s['r']) for s in train_seqs])
        self.g = float(allr.mean())
        tot, cnt = {}, {}
        for s in train_seqs:
            key = s[self.by]
            tot[key] = tot.get(key, 0.0) + float(np.sum(s['r']))
            cnt[key] = cnt.get(key, 0) + len(s['r'])
        self.mean = {kk: (tot[kk] + self.k * self.g) / (cnt[kk] + self.k) for kk in tot}
        return self

    def predict(self, test_seqs):
        return [np.full(len(s['r']), self.mean.get(s[self.by], self.g)) for s in test_seqs]


# ------------------------------------------------------------------ lagged logistic
class LaggedLogistic:
    """
    logit P(recall) = b0 + a*lag + b*sqrt(right) + c*sqrt(wrong) [+ d*log1p(dt)]
                      + item effect + user effect
    L2 penalties are prior precisions on the SUM-of-NLL scale (so 1.0 ~ N(0,1) prior).
    use_dt=False  -> the control for the no-dt dynamic model
    use_dt=True   -> the control for the dt-aware dynamic model
    """
    def __init__(self, use_dt=False, use_item=True, use_user=True,
                 prec_base=0.1, prec_fixed=1.0):
        self.use_dt, self.use_item, self.use_user = use_dt, use_item, use_user
        self.prec_base, self.prec_fixed = prec_base, prec_fixed

    def _design(self, seqs, hist_map):
        base, ii, uu = [], [], []
        for s in seqs:
            h = hist_map.get((s['user'], s['item'])) if hist_map is not None else None
            _, cr, cw = count_features(s, h)
            lag = (np.asarray(s['x']).astype(float) if hist_map is None
                   else lagged_outcome(s, h))
            cols = [np.ones(len(cr)), lag, np.sqrt(cr), np.sqrt(cw)]
            if self.use_dt:
                cols.append(np.log1p(np.asarray(s['dt'], float)))
            base.append(np.column_stack(cols))
            ii.append(np.full(len(cr), s['item']))
            uu.append(np.full(len(cr), s['user']))
        base = np.vstack(base); ii = np.concatenate(ii); uu = np.concatenate(uu)
        blocks = [sparse.csr_matrix(base)]
        if self.use_item:
            ok = ii < self.n_items
            blocks.append(sparse.csr_matrix(
                (np.ones(ok.sum()), (np.flatnonzero(ok), ii[ok])), shape=(len(ii), self.n_items)))
        if self.use_user:
            ok = uu < self.n_users
            blocks.append(sparse.csr_matrix(
                (np.ones(ok.sum()), (np.flatnonzero(ok), uu[ok])), shape=(len(uu), self.n_users)))
        return sparse.hstack(blocks).tocsr(), base.shape[1]

    def fit(self, train_seqs):
        self.n_items = max(s['item'] for s in train_seqs) + 1
        self.n_users = max(s['user'] for s in train_seqs) + 1
        self.hist = build_history(train_seqs)
        X, self.n_base = self._design(train_seqs, None)
        y = np.concatenate([np.asarray(s['r'], float) for s in train_seqs])
        prec = np.full(X.shape[1], self.prec_fixed)
        prec[:self.n_base] = self.prec_base
        prec[0] = 0.0                                   # intercept unpenalised

        def fg(w):
            z = X @ w
            f = np.sum(np.logaddexp(0, z) - y * z) + 0.5 * np.sum(prec * w * w)
            g = X.T @ (1 / (1 + np.exp(-z)) - y) + prec * w
            return f, g

        res = minimize(fg, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B",
                       options=dict(maxiter=500))
        self.w = res.x
        return self

    def predict(self, test_seqs):
        X, _ = self._design(test_seqs, self.hist)
        p = 1 / (1 + np.exp(-(X @ self.w)))
        out, k = [], 0
        for s in test_seqs:
            T = len(s['r'])
            out.append(p[k:k + T]); k += T
        return out