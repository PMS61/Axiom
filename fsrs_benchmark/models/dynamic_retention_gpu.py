"""
dynamic_retention_gpu.py -- PyTorch (CPU or CUDA) port of DynamicRetentionModelDT.

Same model, priors, defaults, update order and predict() filter as
AnkiFirstRun/dynamic_retention_dt.py and pg_sampler.py (those files are not modified).
Only the array library changes. Randomness is equal in distribution, not bitwise: a
torch.Generator (seeded from the numpy rng) draws the vector noise; the numpy rng is used
for the few scalar steps (MH proposals/accepts, inverse-gamma, b0/g0), as in the original.

Memory layout (needed for millions of sequences): per-EVENT data (outcome r, lagged outcome
x, gap dt, count feature C, user u, item i, latent psi) is stored FLAT, sequences back to
back in input order (the event order of `R_padded[mask]` in the original). The FFBS runs
per LENGTH BUCKET (lengths 1, 2, 3-4, 5-8, ... powers of two): each bucket is gathered into
its own padded (N_b x T_b) tensors, filtered and sampled, and scattered back. This is
mathematically identical to one big padded matrix, because sequences are independent given
the parameters and every other update sums over observed events only.

Differences in layout versus the original (the math is the same):
  - `_dyn()`, `_mu()`, `_ffbs()`, `psi`, `R`, `X`, `DT`, `C` are flat (E,) tensors, not (N, Tm).
  - `U`, `I`, `lens` are numpy arrays per sequence; `fu`, `fi`, `fx` are per-event tensors.
  - `key2idx` is built lazily (a dict of millions of tuples is only made if you ask for it).
"""
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
for _folder in (_REPO, _REPO / "AnkiFirstRun"):
    if str(_folder) not in sys.path:
        sys.path.append(str(_folder))

from dynamic_retention_dt import ess, split_rhat  # noqa: E402,F401

_SAMPLE_KEYS = ['b0', 'g0', 'lam', 'kappa', 'sigma2', 'tau2_alpha', 'tau2_beta', 'tau2_gamma',
                'alpha', 'beta', 'gamma', 'psi_last']


def sample_pg1_torch(c, gen, n_terms=200, chunk_elems=20_000_000):
    """PG(1, c_i) draws, truncated series as pg_sampler.sample_pg1, Gamma(1,1) = Exponential(1).

    c is a 1-D tensor; processed in chunks of about chunk_elems / n_terms entries.
    """
    c = torch.atleast_1d(c)
    n = c.shape[0]
    out = torch.empty_like(c)
    k2 = ((torch.arange(1, n_terms + 1, device=c.device, dtype=c.dtype) - 0.5) ** 2)[None, :]
    step = max(1, int(chunk_elems // n_terms))
    inv = 1.0 / (4 * np.pi ** 2)
    for s in range(0, n, step):
        cc = c[s:s + step]
        denom = (cc * cc * inv)[:, None] + k2
        g = torch.empty_like(denom)
        g.exponential_(generator=gen)
        g.div_(denom)
        out[s:s + step] = g.sum(dim=1) / (2 * np.pi ** 2)
    return out


def _bucket_ids(lens, bucket=True):
    """Bucket id per sequence: length 1 -> 0, 2 -> 1, 3-4 -> 2, 5-8 -> 3, ..."""
    if not bucket:
        return np.zeros(len(lens), dtype=np.int64)
    out = np.zeros(len(lens), dtype=np.int64)
    big = lens >= 2
    out[big] = np.floor(np.log2(lens[big] - 1)).astype(np.int64) + 1
    return out


def _ranges(starts, lens):
    """Concatenation of arange(start, start+len) for each pair (vectorised)."""
    total = int(lens.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    first = np.cumsum(lens) - lens
    return np.arange(total, dtype=np.int64) - np.repeat(first, lens) + np.repeat(starts, lens)


class DynamicRetentionModelGPU:
    def __init__(self, sequences, rng=None, init_state_var=1.0,
                 sigma2_prior=(2.0, 0.5), tau2_prior=(2.0, 1.0),
                 loglam_prior=(np.log(0.05), 1.5), lam_bounds=(1e-4, 20.0),
                 dt_bounds=(1e-2, 365.0), n_mh=5, fixed_lam=None, use_kappa=False,
                 kappa_prior_sd=1.0, device=None, dtype=torch.float64,
                 chunk_elems=20_000_000, bucket=True, store_dtype=np.float64):
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.device = torch.device(device if device is not None
                                   else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = dtype
        self.chunk_elems = int(chunk_elems)
        self.store_dtype = store_dtype
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(int(self.rng.integers(2 ** 62)))
        self.init_var = init_state_var
        self.sigma2_prior, self.tau2_prior = sigma2_prior, tau2_prior
        self.loglam_prior, self.lam_bounds = loglam_prior, lam_bounds
        self.dt_bounds, self.n_mh = dt_bounds, n_mh
        self.fixed_lam, self.use_kappa, self.kappa_prior_sd = fixed_lam, use_kappa, kappa_prior_sd

        N = len(sequences)
        self.N = N
        self.lens = np.fromiter((len(s['r']) for s in sequences), dtype=np.int64, count=N)
        self.Tm = int(self.lens.max())
        self.U = np.fromiter((s['user'] for s in sequences), dtype=np.int64, count=N)
        self.I = np.fromiter((s['item'] for s in sequences), dtype=np.int64, count=N)
        self.n_users = int(self.U.max()) + 1
        self.n_items = int(self.I.max()) + 1
        E = int(self.lens.sum())
        self.E = E
        self.offsets = np.cumsum(self.lens) - self.lens
        r = np.concatenate([np.asarray(s['r'], float) for s in sequences])
        x = np.concatenate([np.asarray(s['x'], float) for s in sequences])
        dt = np.clip(np.concatenate([np.asarray(s['dt'], float) for s in sequences]), *dt_bounds)
        seq_of = np.repeat(np.arange(N), self.lens)
        first = np.zeros(E, bool)
        first[self.offsets] = True
        dt[first] = 1.0                          # dt of the first event is unused
        cs = np.cumsum(r) - r                    # events before e (all sequences)
        Cnp = np.log1p(cs - cs[self.offsets][seq_of])
        self.sum_r = np.add.reduceat(r, self.offsets)       # per-sequence sum / last outcome
        self.last_r = r[self.offsets + self.lens - 1]
        self._keys = None
        self._key2idx = None

        dev, dty = self.device, self.dtype
        T = lambda a, d=dty: torch.as_tensor(np.ascontiguousarray(a), dtype=d, device=dev)  # noqa: E731
        self.R, self.X, self.DT, self.C = T(r), T(x), T(dt), T(Cnp)
        self.first = torch.as_tensor(first, device=dev)
        self.fu = torch.as_tensor(self.U[seq_of], device=dev)
        self.fi = torch.as_tensor(self.I[seq_of], device=dev)
        self.fx = self.X
        self.tr_idx = torch.nonzero(~self.first).flatten()          # transition events (t >= 1)
        self.tdt, self.tC = self.DT[self.tr_idx], self.C[self.tr_idx]
        self.item_seen_np = np.bincount(self.I[seq_of], minlength=self.n_items) > 0
        self.item_seen = torch.as_tensor(self.item_seen_np, device=dev)
        self.n_seen = int(self.item_seen_np.sum())
        self.last_idx = torch.as_tensor(self.offsets + self.lens - 1, device=dev)

        # length buckets
        bid = _bucket_ids(self.lens, bucket)
        self.buckets = []
        for b in np.unique(bid):
            idx = np.nonzero(bid == b)[0]
            lb = self.lens[idx]
            Tb = int(lb.max())
            Nb = len(idx)
            gv = _ranges(self.offsets[idx], lb)
            pv = np.repeat(np.arange(Nb, dtype=np.int64) * Tb, lb) + _ranges(np.zeros(Nb, np.int64), lb)
            self.buckets.append(dict(
                idx=idx, N=Nb, T=Tb, lens=lb,
                last=torch.as_tensor(lb - 1, device=dev),
                gv=torch.as_tensor(gv, device=dev), pv=torch.as_tensor(pv, device=dev)))

        p0 = float(np.clip(r.mean(), 0.05, 0.95))
        l0 = float(np.log(p0 / (1 - p0)))
        self.b0, self.g0 = l0, 0.0
        z = lambda n: torch.zeros(n, dtype=dty, device=dev)  # noqa: E731
        self.alpha, self.beta, self.gamma = z(self.n_users), z(self.n_items), z(self.n_users)
        self.tau2 = dict(alpha=1.0, beta=1.0, gamma=0.5)
        self.lam = float(fixed_lam) if fixed_lam is not None else 0.05
        self.sigma2 = 0.5
        self.kappa, self.step_k = 0.0, 0.15
        self.psi = torch.full((E,), l0, dtype=dty, device=dev)
        self.step = 0.2
        self.acc_rate = np.nan
        self.samples = {k: [] for k in _SAMPLE_KEYS}

    # ------------------------------------------------------------ lazy key lookup
    @property
    def key2idx(self):
        if self._key2idx is None:
            self._key2idx = {(int(u), int(i)): k for k, (u, i) in enumerate(zip(self.U, self.I))}
        return self._key2idx

    @key2idx.setter
    def key2idx(self, v):
        self._key2idx = v

    def _lookup(self, users, items):
        """Training sequence index for each (user, item) pair, or -1 (vectorised)."""
        if self._keys is None:
            stride = self.n_items + 1
            keys = self.U * stride + self.I
            order = np.argsort(keys, kind='stable')
            self._keys = (keys[order], order, stride)
        ks, order, stride = self._keys
        users, items = np.asarray(users, np.int64), np.asarray(items, np.int64)
        ok = (users >= 0) & (users < self.n_users) & (items >= 0) & (items < self.n_items)
        q = np.where(ok, users, 0) * stride + np.where(ok, items, 0)
        # side='right' - 1: duplicated training pairs resolve to the LAST sequence (like the dict)
        pos = np.maximum(np.searchsorted(ks, q, side='right') - 1, 0)
        hit = ok & (ks[pos] == q)
        return np.where(hit, order[pos], -1)

    # ------------------------------------------------------------ PIECES (flat, per event)
    def _dyn(self):
        L = self.lam * torch.exp(-self.kappa * self.C)
        phi = torch.exp(-L * self.DT)
        c = -torch.expm1(-2.0 * L * self.DT) / (2.0 * L)
        q = self.sigma2 * c
        phi = torch.where(self.first, torch.zeros_like(phi), phi)    # t=0: prior N(mu_0, init_var)
        q = torch.where(self.first, torch.full_like(q, self.init_var), q)
        return phi, q

    def _mu(self):
        return self.b0 + self.alpha[self.fu] + self.beta[self.fi] + self.gamma[self.fu] * self.fx

    def _to_pad(self, b, vec, fill):
        out = torch.full((b['N'] * b['T'],), fill, dtype=vec.dtype, device=vec.device)
        out[b['pv']] = vec[b['gv']]
        return out.view(b['N'], b['T'])

    def _ffbs_forward_T(self, omega, phi, q, mu, R):
        """Forward filter on (T, N) tensors (time-major). Returns m, P, a, Rp, all (T, N)."""
        Tn, Nn = omega.shape
        ov = 1.0 / omega
        z = (R - 0.5) / omega
        m = torch.empty_like(omega); P = torch.empty_like(omega)
        a = torch.empty_like(omega); Rp = torch.empty_like(omega)
        mp = torch.zeros(Nn, dtype=omega.dtype, device=omega.device)
        Pp = torch.zeros_like(mp)
        for t in range(Tn):
            ph = phi[t]
            a_t = ph * mp + (1.0 - ph) * mu[t]
            R_t = ph * ph * Pp + q[t]
            K = R_t / (R_t + ov[t])
            m_t = a_t + K * (z[t] - a_t)
            P_t = (1.0 - K) * R_t
            a[t] = a_t; Rp[t] = R_t; m[t] = m_t; P[t] = P_t
            mp, Pp = m_t, P_t
        return m, P, a, Rp

    def _ffbs_backward_T(self, m, P, a, Rp, phi, eps, last):
        """Backward sampler on (T, N) tensors; last is (N,) index of each sequence's last event."""
        Tn = m.shape[0]
        psi = torch.zeros_like(m)
        zero = torch.zeros((), dtype=m.dtype, device=m.device)
        for t in range(Tn - 1, -1, -1):
            e = eps[t]
            P_t = P[t]
            var_end = m[t] + torch.sqrt(P_t.clamp(min=1e-12)) * e
            if t == Tn - 1:
                val = var_end
            else:
                pn = phi[t + 1]
                J = P_t * pn / Rp[t + 1]
                h = m[t] + J * (psi[t + 1] - a[t + 1])
                V = P_t - J * pn * P_t
                cond = h + torch.sqrt(V.clamp(min=1e-12)) * e
                val = torch.where(last == t, var_end, cond)
            psi[t] = torch.where(last >= t, val, zero)
        return psi

    # public (N, T) versions, for tests and for single-bucket use
    def _ffbs_forward(self, omega, phi, q, mu, R=None):
        if R is None:
            if len(self.buckets) != 1:
                raise ValueError("pass R explicitly when the model has several buckets")
            R = self._to_pad(self.buckets[0], self.R, 0.0)
        tt = lambda y: y.T.contiguous()  # noqa: E731
        m, P, a, Rp = self._ffbs_forward_T(tt(omega), tt(phi), tt(q), tt(mu), tt(R))
        return m.T, P.T, a.T, Rp.T

    def _ffbs_backward(self, m, P, a, Rp, phi, eps, lens=None):
        if lens is None:
            lens = self.lens
        last = torch.as_tensor(np.asarray(lens) - 1, device=m.device)
        tt = lambda y: y.T.contiguous()  # noqa: E731
        return self._ffbs_backward_T(tt(m), tt(P), tt(a), tt(Rp), tt(phi), tt(eps), last).T

    def _ffbs(self, omega, phi, q, mu, eps_list=None):
        """FFBS over all buckets. omega, phi, q, mu, result: flat (E,). eps_list (tests): one
        (N_b, T_b) standard-normal tensor per bucket instead of fresh draws."""
        out = torch.empty_like(omega)
        for bi, b in enumerate(self.buckets):
            om = self._to_pad(b, omega, 1.0).T.contiguous()
            ph = self._to_pad(b, phi, 0.0).T.contiguous()
            qq = self._to_pad(b, q, 1.0).T.contiguous()
            mm = self._to_pad(b, mu, 0.0).T.contiguous()
            Rb = self._to_pad(b, self.R, 0.0).T.contiguous()
            m, P, a, Rp = self._ffbs_forward_T(om, ph, qq, mm, Rb)
            if eps_list is None:
                eps = torch.randn(m.shape, generator=self.gen, dtype=m.dtype, device=m.device)
            else:
                eps = eps_list[bi].T.contiguous()
            psi = self._ffbs_backward_T(m, P, a, Rp, ph, eps, b['last'])
            out[b['gv']] = psi.T.reshape(-1)[b['pv']]
        return out

    def _gauss(self, ngroups, idx, design, resid, var, prior_mean, prior_var):
        w = 1.0 / var
        if ngroups == 1:
            s = torch.stack([(design * design * w).sum(), (design * resid * w).sum()]).tolist()
            prec = s[0] + 1.0 / prior_var
            pv = 1.0 / prec
            return torch.tensor([self.rng.normal(pv * (s[1] + prior_mean / prior_var), np.sqrt(pv))],
                                dtype=self.dtype, device=self.device)
        z = torch.zeros(ngroups, dtype=design.dtype, device=design.device)
        prec = z.clone().index_add_(0, idx, design * design * w) + 1.0 / prior_var
        num = z.index_add_(0, idx, design * resid * w) + prior_mean / prior_var
        pv = 1.0 / prec
        noise = torch.randn(ngroups, generator=self.gen, dtype=design.dtype, device=design.device)
        return pv * num + torch.sqrt(pv) * noise

    def _inv_gamma_tau2(self, theta, center, count=None):
        a, b = self.tau2_prior
        n = len(theta) if count is None else count
        ss = float(((theta - center) ** 2).sum())
        return float(1.0 / self.rng.gamma(a + n / 2.0, 1.0 / (b + 0.5 * ss)))

    def _update_effects(self, psi, phi, q):
        prev = torch.zeros_like(psi)
        prev[1:] = psi[:-1]
        y = psi - phi * prev
        c = 1.0 - phi
        v = q
        u, i, x = self.fu, self.fi, self.fx
        zero = torch.zeros_like(u)
        res = y - c * (self.alpha[u] + self.beta[i] + self.gamma[u] * x)
        self.b0 = float(self._gauss(1, zero, c, res, v, 0.0, 100.0)[0])
        res = y - c * (self.b0 + self.beta[i] + self.gamma[u] * x)
        self.alpha = self._gauss(self.n_users, u, c, res, v, 0.0, self.tau2['alpha'])
        res = y - c * (self.b0 + self.alpha[u] + self.gamma[u] * x)
        self.beta = self._gauss(self.n_items, i, c, res, v, 0.0, self.tau2['beta'])
        res = y - c * (self.b0 + self.alpha[u] + self.beta[i])
        self.gamma = self._gauss(self.n_users, u, c * x, res, v, self.g0, self.tau2['gamma'])
        prec = self.n_users / self.tau2['gamma'] + 1.0 / 100.0
        pv = 1.0 / prec
        self.g0 = float(self.rng.normal(pv * float(self.gamma.sum()) / self.tau2['gamma'], np.sqrt(pv)))
        self.tau2['alpha'] = self._inv_gamma_tau2(self.alpha, 0.0)
        self.tau2['beta'] = self._inv_gamma_tau2(self.beta * self.item_seen, 0.0, self.n_seen)
        self.tau2['gamma'] = self._inv_gamma_tau2(self.gamma, self.g0)

    def _update_lam_sigma2(self, psi, it, adapt):
        ti = self.tr_idx
        tpsi, tprev = psi[ti], psi[ti - 1]
        tmu = self._mu()[ti]
        tdt, tC = self.tdt, self.tC
        n = int(ti.numel())
        sa, sb = self.sigma2_prior
        m0, s0 = self.loglam_prior
        lo, hi = self.lam_bounds

        def lp(lam, kappa):
            Lt = lam * torch.exp(-kappa * tC)
            ph = torch.exp(-Lt * tdt)
            c = -torch.expm1(-2.0 * Lt * tdt) / (2.0 * Lt)
            e = tpsi - ph * tprev - (1.0 - ph) * tmu
            S = 0.5 * torch.sum(e * e / c)
            val = -0.5 * torch.sum(torch.log(c)) - (sa + n / 2.0) * torch.log(sb + S)
            return float(val), S

        if self.fixed_lam is None:
            cur = lp(self.lam, self.kappa)[0] - 0.5 * ((np.log(self.lam) - m0) / s0) ** 2
            acc = 0
            for _ in range(self.n_mh):
                prop = float(np.exp(np.log(self.lam) + self.step * self.rng.standard_normal()))
                if lo < prop < hi:
                    new = lp(prop, self.kappa)[0] - 0.5 * ((np.log(prop) - m0) / s0) ** 2
                    if np.log(self.rng.random()) < new - cur:
                        self.lam, cur = prop, new
                        acc += 1
            self.acc_rate = acc / self.n_mh
            if adapt:
                self.step = float(np.clip(self.step * np.exp((self.acc_rate - 0.3) / (it + 1) ** 0.6), 0.01, 2.0))
        if self.use_kappa:
            cur = lp(self.lam, self.kappa)[0] - 0.5 * (self.kappa / self.kappa_prior_sd) ** 2
            acc = 0
            for _ in range(self.n_mh):
                prop = self.kappa + self.step_k * self.rng.standard_normal()
                if abs(prop) < 3.0:
                    new = lp(self.lam, prop)[0] - 0.5 * (prop / self.kappa_prior_sd) ** 2
                    if np.log(self.rng.random()) < new - cur:
                        self.kappa, cur = prop, new
                        acc += 1
            if adapt:
                self.step_k = float(np.clip(self.step_k * np.exp((acc / self.n_mh - 0.3) / (it + 1) ** 0.6), 0.01, 1.0))
        S = float(lp(self.lam, self.kappa)[1])
        self.sigma2 = float(1.0 / self.rng.gamma(sa + n / 2.0, 1.0 / (sb + S)))

    # ------------------------------------------------------------ MAIN LOOP
    def run(self, n_iter=1000, burn_in=400, thin=4, verbose=False):
        n_keep = len(range(burn_in, n_iter, thin))
        sd = self.store_dtype
        sm = {k: np.zeros(n_keep) for k in ['b0', 'g0', 'lam', 'kappa', 'sigma2',
                                            'tau2_alpha', 'tau2_beta', 'tau2_gamma']}
        sm['alpha'] = np.zeros((n_keep, self.n_users), sd)
        sm['beta'] = np.zeros((n_keep, self.n_items), sd)
        sm['gamma'] = np.zeros((n_keep, self.n_users), sd)
        sm['psi_last'] = np.zeros((n_keep, self.N), sd)
        kept = 0
        for it in range(n_iter):
            phi, q = self._dyn()
            c = self.psi
            omega = sample_pg1_torch(c, self.gen, 200, self.chunk_elems).clamp_(min=1e-8)
            self.psi = self._ffbs(omega, phi, q, self._mu())
            self._update_effects(self.psi, phi, q)
            self._update_lam_sigma2(self.psi, it, adapt=(it < burn_in))
            if it >= burn_in and (it - burn_in) % thin == 0:
                sm['b0'][kept] = self.b0; sm['g0'][kept] = self.g0
                sm['lam'][kept] = self.lam; sm['kappa'][kept] = self.kappa
                sm['sigma2'][kept] = self.sigma2
                sm['tau2_alpha'][kept] = self.tau2['alpha']
                sm['tau2_beta'][kept] = self.tau2['beta']
                sm['tau2_gamma'][kept] = self.tau2['gamma']
                sm['alpha'][kept] = self.alpha.cpu().numpy()
                sm['beta'][kept] = self.beta.cpu().numpy()
                sm['gamma'][kept] = self.gamma.cpu().numpy()
                sm['psi_last'][kept] = self.psi[self.last_idx].cpu().numpy()
                kept += 1
            if verbose and it % 100 == 0:
                print(f"[dyn-gpu] it {it:5d} lam={self.lam:.4f} sigma2={self.sigma2:.3f} "
                      f"b0={self.b0:.2f} g0={self.g0:.2f} mh_acc={self.acc_rate:.2f}")
        self.samples = sm
        return self.samples

    def fit(self, train_seqs=None, **kw):
        """Uniform interface: sequences were given at construction."""
        self.run(**kw)
        return self

    # ------------------------------------------------------------ REPORTING
    def summary(self):
        out = {}
        keys = ['lam'] + (['kappa'] if self.use_kappa else []) + ['sigma2', 'b0', 'g0', 'tau2_alpha', 'tau2_beta', 'tau2_gamma']
        for k in keys:
            x = self.samples[k]
            out[k] = dict(mean=float(x.mean()),
                          lo=float(np.percentile(x, 2.5)), hi=float(np.percentile(x, 97.5)),
                          ess=ess(x), rhat=split_rhat(x))
        return out

    def print_summary(self):
        print(f"{'param':<11}{'mean':>9}{'2.5%':>9}{'97.5%':>9}{'ESS':>8}{'Rhat':>7}")
        for k, d in self.summary().items():
            print(f"{k:<11}{d['mean']:>9.3f}{d['lo']:>9.3f}{d['hi']:>9.3f}{d['ess']:>8.0f}{d['rhat']:>7.2f}")

    # ------------------------------------------------------------ PREDICTION
    def predict(self, test_seqs, max_draws=100, return_draws=False):
        sm = self.samples
        S_all = len(sm['lam'])
        sel = np.unique(np.linspace(0, S_all - 1, min(max_draws, S_all)).astype(int))
        S = len(sel)
        K = len(test_seqs)
        dev, dty = self.device, self.dtype
        tl = np.fromiter((len(s['r']) for s in test_seqs), dtype=np.int64, count=K)
        Et = int(tl.sum())
        toff = np.cumsum(tl) - tl
        seq_of = np.repeat(np.arange(K), tl)
        r = np.concatenate([np.asarray(s['r'], float) for s in test_seqs])
        dt = np.clip(np.concatenate([np.asarray(s['dt'], float) for s in test_seqs]), *self.dt_bounds)
        tu = np.fromiter((s['user'] for s in test_seqs), dtype=np.int64, count=K)
        ti = np.fromiter((s['item'] for s in test_seqs), dtype=np.int64, count=K)
        ui = np.where(tu < self.n_users, tu, self.n_users)
        ii = np.where(ti < self.n_items, ti, self.n_items)
        j = self._lookup(tu, ti)
        warm = j >= 0
        jj = np.where(warm, j, 0)
        x0 = np.where(warm, self.last_r[jj], 0.0)
        base = np.where(warm, self.sum_r[jj], 0.0)
        cs = np.cumsum(r) - r
        Cn = np.log1p(base[seq_of] + cs - cs[toff][seq_of])
        xn = np.empty(Et)
        xn[1:] = r[:-1]
        xn[toff] = x0
        widx = np.where(warm, j, 0)

        out_flat = np.zeros(Et)
        draws_flat = np.zeros((S, Et), dtype=self.store_dtype) if return_draws else None
        order = np.argsort(tl, kind='stable')
        sl = tl[order]
        budget = max(self.chunk_elems, 1)
        start = 0
        while start < K:
            lo_, hi_ = start + 1, K        # largest end with (end-start)*sl[end-1]*S <= budget
            while lo_ < hi_:
                mid = (lo_ + hi_ + 1) // 2
                if (mid - start) * int(sl[mid - 1]) * S <= budget:
                    lo_ = mid
                else:
                    hi_ = mid - 1
            end = lo_
            ch = order[start:end]
            start = end
            Kc, Tc = len(ch), int(tl[ch].max())
            lc = tl[ch]
            ev = _ranges(toff[ch], lc)
            row = np.repeat(np.arange(Kc), lc)
            col = _ranges(np.zeros(Kc, np.int64), lc)
            Rt = np.zeros((Kc, Tc)); Xt = np.zeros((Kc, Tc)); Ct = np.zeros((Kc, Tc))
            DTt = np.ones((Kc, Tc)); Mt = np.zeros((Kc, Tc), bool)
            Rt[row, col] = r[ev]; Xt[row, col] = xn[ev]; Ct[row, col] = Cn[ev]
            DTt[row, col] = dt[ev]; Mt[row, col] = True
            uc, ic, wc, jc = ui[ch], ii[ch], warm[ch], widx[ch]
            A = np.asarray(sm['alpha'][np.ix_(sel, np.minimum(uc, self.n_users - 1))]) * (uc < self.n_users)
            B = np.asarray(sm['beta'][np.ix_(sel, np.minimum(ic, self.n_items - 1))]) * (ic < self.n_items)
            G = np.asarray(sm['gamma'][np.ix_(sel, np.minimum(uc, self.n_users - 1))]) * (uc < self.n_users)
            PL = np.asarray(sm['psi_last'][np.ix_(sel, jc)]) * wc
            to = lambda a_: torch.as_tensor(np.ascontiguousarray(a_), dtype=dty, device=dev)  # noqa: E731
            alpha, beta, gamma, m = to(A), to(B), to(G), to(PL)
            Rt_, Xt_, Ct_, DTt_ = to(Rt), to(Xt), to(Ct), to(DTt)
            Mt_ = torch.as_tensor(Mt, device=dev)
            b0 = to(sm['b0'][sel])[:, None]
            lam = to(sm['lam'][sel])[:, None]
            kap = to(sm['kappa'][sel])[:, None]
            sig2 = to(sm['sigma2'][sel])[:, None]
            cold = torch.as_tensor(~wc, device=dev)[None, :]
            P = torch.zeros((S, Kc), dtype=dty, device=dev)
            outc = torch.zeros((Kc, Tc), dtype=dty, device=dev)
            drawc = torch.zeros((S, Kc, Tc), dtype=dty, device=dev) if return_draws else None
            for t in range(Tc):
                dt_t = DTt_[:, t][None, :]
                mu = b0 + alpha + beta + gamma * Xt_[:, t][None, :]
                rate = lam * torch.exp(-kap * Ct_[:, t][None, :])
                phi = torch.exp(-rate * dt_t)
                q = sig2 * (-torch.expm1(-2.0 * rate * dt_t) / (2.0 * rate))
                phi = torch.where(cold, torch.zeros_like(phi), phi)
                q = torch.where(cold, torch.full_like(q, self.init_var), q)
                a = phi * m + (1.0 - phi) * mu
                Rr = phi ** 2 * P + q
                p_t = 0.5 * (1.0 + torch.tanh(0.5 * (a / torch.sqrt(1.0 + np.pi * Rr / 8.0))))
                outc[:, t] = p_t.mean(dim=0)
                if return_draws:
                    drawc[:, :, t] = p_t
                w = torch.where(a.abs() < 1e-6, torch.full_like(a, 0.25),
                                torch.tanh(a / 2.0) / (2.0 * torch.where(a == 0, torch.ones_like(a), a)))
                w = w.clamp(min=1e-6)
                zt = (Rt_[:, t][None, :] - 0.5) / w
                Kg = Rr / (Rr + 1.0 / w)
                m_new = a + Kg * (zt - a)
                P_new = (1.0 - Kg) * Rr
                upd = Mt_[:, t][None, :]
                m = torch.where(upd, m_new, m)
                P = torch.where(upd, P_new, P)
            out_flat[ev] = outc.cpu().numpy()[row, col]
            if return_draws:
                draws_flat[:, ev] = drawc.cpu().numpy()[:, row, col]
        preds = [out_flat[toff[k]:toff[k] + tl[k]].copy() for k in range(K)]
        if return_draws:
            return preds, [draws_flat[:, toff[k]:toff[k] + tl[k]] for k in range(K)]
        return preds
