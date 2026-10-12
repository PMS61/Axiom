#Original dynamic_retention_dt : without optimisation

"""
dynamic_retention_dt.py

Time-aware hierarchical dynamic retention model (Polya-Gamma + FFBS Gibbs).

MODEL  (psi = logit-scale latent retention state, Delta_t = days since previous review)

    phi_t   = exp(-lambda * Delta_t)
    mu_t    = b0 + alpha_u + beta_i + gamma_u * x_t          (x_t = previous outcome)
    psi_t   = phi_t * psi_{t-1} + (1 - phi_t) * mu_t + w_t,
    w_t     ~ N(0, sigma2 * (1 - phi_t^2) / (2 * lambda))
    psi_0   ~ N(mu_0, init_state_var)
    r_t     ~ Bernoulli(sigmoid(psi_t))

This is the EXACT discretisation of an Ornstein-Uhlenbeck process that mean-reverts
to the level mu_t, with sigma2 the per-day diffusion variance. Two consequences that
the plain "phi_t * psi + beta + gamma x" version does NOT have:
   * Delta_t -> 0 : state is unchanged (no free drift from two reviews seconds apart)
   * Delta_t -> inf: state decays to mu_t (item/user baseline, higher after a success
                     because gamma_u > 0), with the right amount of added uncertainty.

SAMPLER (all conjugate except lambda)
    PG augmentation -> batched FFBS -> Gaussian updates of b0, alpha_u, beta_i, gamma_u, g0
    -> inverse-gamma updates of the tau^2 hyper-variances
    -> lambda by random-walk Metropolis on log(lambda) with sigma2 INTEGRATED OUT
       (partially-collapsed step), then sigma2 | lambda is exact inverse-gamma.
    So: "exact Gibbs, plus one Metropolis step for lambda."

Effects are hierarchical with LEARNED shrinkage (tau^2 are sampled, not fixed at 1).
alpha_u (user ability) also makes the model usable on Anki, where cards are nested in
users: beta_card ~ N(0, tau_beta^2) around the user's alpha_u.

PREDICTION  predict() is an ONLINE posterior-predictive: for each stored posterior draw
it propagates the state across the real gap, forms E[sigmoid(psi)] (probit approx),
then updates the state with the observed outcome (PG-mean Kalman step) before the next
held-out event. Predictions are averaged over draws, so parameter + state + process
uncertainty are all included.

Requires pg_sampler.sample_pg1(z, rng) from your project (PG(1, z), elementwise).
"""
import numpy as np
from pg_sampler import sample_pg1


def _sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


# ---------------------------------------------------------------------- diagnostics
def split_rhat(x):
    x = np.asarray(x, float)
    h = len(x) // 2
    if h < 4:
        return np.nan
    ch = np.vstack([x[:h], x[h:2 * h]])
    W = ch.var(axis=1, ddof=1).mean()
    if W <= 0:
        return np.nan
    B = h * ch.mean(axis=1).var(ddof=1)
    return float(np.sqrt(((h - 1) / h * W + B / h) / max(W, 1e-300)))


def ess(x):
    x = np.asarray(x, float)
    n = len(x)
    if n < 8 or x.std() == 0:
        return float(n)
    x = x - x.mean()
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    acf /= acf[0]
    s = 0.0
    for k in range(1, n - 1, 2):
        pair = acf[k] + acf[k + 1]
        if pair < 0:
            break
        s += pair
    return float(n / max(1.0 + 2.0 * s, 1.0))


# ---------------------------------------------------------------------- model
class DynamicRetentionModelDT:
    def __init__(self, sequences, rng=None, init_state_var=1.0,
                 sigma2_prior=(2.0, 0.5), tau2_prior=(2.0, 1.0),
                 loglam_prior=(np.log(0.05), 1.5), lam_bounds=(1e-4, 20.0),
                 dt_bounds=(1e-2, 365.0), n_mh=5, fixed_lam=None, use_kappa=False,
                 kappa_prior_sd=1.0):
        self.seqs = sequences
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.init_var = init_state_var
        self.sigma2_prior, self.tau2_prior = sigma2_prior, tau2_prior
        self.loglam_prior, self.lam_bounds = loglam_prior, lam_bounds
        self.dt_bounds, self.n_mh = dt_bounds, n_mh
        self.fixed_lam, self.use_kappa, self.kappa_prior_sd = fixed_lam, use_kappa, kappa_prior_sd

        N = len(sequences)
        self.N = N
        self.lens = np.array([len(s['r']) for s in sequences])
        Tm = int(self.lens.max())
        self.Tm = Tm
        self.n_users = max(s['user'] for s in sequences) + 1
        self.n_items = max(s['item'] for s in sequences) + 1
        self.U = np.array([s['user'] for s in sequences])
        self.I = np.array([s['item'] for s in sequences])
        self.R = np.zeros((N, Tm)); self.X = np.zeros((N, Tm))
        self.DT = np.ones((N, Tm)); self.M = np.zeros((N, Tm), bool)
        for n, s in enumerate(sequences):
            T = self.lens[n]
            self.R[n, :T] = s['r']
            self.X[n, :T] = s['x']
            self.DT[n, :T] = np.clip(np.asarray(s['dt'], float), *dt_bounds)
            self.M[n, :T] = True
        self.DT[:, 0] = 1.0                      # dt of the first event is unused
        self.TR = self.M.copy(); self.TR[:, 0] = False   # transition entries (t >= 1)
        self.fu = np.broadcast_to(self.U[:, None], (N, Tm))[self.M]
        self.fi = np.broadcast_to(self.I[:, None], (N, Tm))[self.M]
        self.fx = self.X[self.M]
        self.item_seen = np.bincount(self.fi, minlength=self.n_items) > 0
        self.key2idx = {(s['user'], s['item']): k for k, s in enumerate(sequences)}

        p0 = float(np.clip(self.R[self.M].mean(), 0.05, 0.95))
        l0 = np.log(p0 / (1 - p0))
        self.b0, self.g0 = l0, 0.0
        self.alpha = np.zeros(self.n_users)
        self.beta = np.zeros(self.n_items)
        self.gamma = np.zeros(self.n_users)
        self.tau2 = dict(alpha=1.0, beta=1.0, gamma=0.5)
        self.lam = fixed_lam if fixed_lam is not None else 0.05
        self.sigma2 = 0.5
        self.kappa, self.step_k = 0.0, 0.15
        # log(1 + successes strictly before event t): drives 'decay slows with practice'
        cs = np.cumsum(self.R * self.M, axis=1) - self.R * self.M
        self.C = np.log1p(cs) * self.M
        self.psi = np.full((N, Tm), l0)
        self.step = 0.2          # MH step on log(lambda), adapted during burn-in
        self.acc_rate = np.nan
        self.samples = {k: [] for k in
                        ['b0', 'g0', 'lam', 'kappa', 'sigma2', 'tau2_alpha', 'tau2_beta', 'tau2_gamma',
                         'alpha', 'beta', 'gamma', 'psi_last']}

    # ------------------------------------------------------------ pieces
    def _dyn(self):
        L = self.lam * np.exp(-self.kappa * self.C)      # per-event decay rate
        phi = np.exp(-L * self.DT)
        c = -np.expm1(-2.0 * L * self.DT) / (2.0 * L)
        q = self.sigma2 * c
        phi[:, 0] = 0.0                         # t=0: no history, prior N(mu_0, init_var)
        q[:, 0] = self.init_var
        return phi, q

    def _mu(self):
        return (self.b0 + self.alpha[self.U][:, None] + self.beta[self.I][:, None]
                + self.gamma[self.U][:, None] * self.X)

    def _ffbs(self, omega, phi, q, mu):
        rng = self.rng
        N, Tm = self.N, self.Tm
        z = (self.R - 0.5) / omega
        ov = 1.0 / omega
        m = np.zeros((N, Tm)); P = np.zeros((N, Tm))
        a = np.zeros((N, Tm)); Rp = np.zeros((N, Tm))
        mp = np.zeros(N); Pp = np.zeros(N)
        for t in range(Tm):
            a_t = phi[:, t] * mp + (1 - phi[:, t]) * mu[:, t]
            R_t = phi[:, t] ** 2 * Pp + q[:, t]
            K = R_t / (R_t + ov[:, t])
            m_t = a_t + K * (z[:, t] - a_t)
            P_t = (1 - K) * R_t
            a[:, t], Rp[:, t], m[:, t], P[:, t] = a_t, R_t, m_t, P_t
            mp, Pp = m_t, P_t
        eps = rng.standard_normal((N, Tm))
        last = self.lens - 1
        psi = np.zeros((N, Tm))
        for t in range(Tm - 1, -1, -1):
            end = m[:, t] + np.sqrt(np.maximum(P[:, t], 1e-12)) * eps[:, t]
            if t == Tm - 1:
                val = end
            else:
                J = P[:, t] * phi[:, t + 1] / Rp[:, t + 1]
                h = m[:, t] + J * (psi[:, t + 1] - a[:, t + 1])
                V = P[:, t] - J * phi[:, t + 1] * P[:, t]
                cond = h + np.sqrt(np.maximum(V, 1e-12)) * eps[:, t]
                val = np.where(last == t, end, cond)
            psi[:, t] = np.where(t <= last, val, 0.0)
        return psi

    def _gauss(self, ngroups, idx, design, resid, var, prior_mean, prior_var):
        prec = np.bincount(idx, weights=design ** 2 / var, minlength=ngroups) + 1.0 / prior_var
        num = np.bincount(idx, weights=design * resid / var, minlength=ngroups) + prior_mean / prior_var
        pv = 1.0 / prec
        return self.rng.normal(pv * num, np.sqrt(pv))

    def _inv_gamma_tau2(self, theta, center):
        a, b = self.tau2_prior
        return float(1.0 / self.rng.gamma(a + len(theta) / 2.0,
                                          1.0 / (b + 0.5 * np.sum((theta - center) ** 2))))

    def _update_effects(self, psi, phi, q):
        prev = np.zeros_like(psi); prev[:, 1:] = psi[:, :-1]
        y = (psi - phi * prev)[self.M]
        c = (1.0 - phi)[self.M]
        v = q[self.M]
        u, i, x = self.fu, self.fi, self.fx
        zero = np.zeros_like(u)
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
        self.g0 = float(self.rng.normal(pv * self.gamma.sum() / self.tau2['gamma'], np.sqrt(pv)))
        self.tau2['alpha'] = self._inv_gamma_tau2(self.alpha, 0.0)
        self.tau2['beta'] = self._inv_gamma_tau2(self.beta[self.item_seen], 0.0)
        self.tau2['gamma'] = self._inv_gamma_tau2(self.gamma, self.g0)

    def _update_lam_sigma2(self, psi, it, adapt):
        prev = np.zeros_like(psi); prev[:, 1:] = psi[:, :-1]
        mu = self._mu()
        tpsi, tprev, tdt, tmu = psi[self.TR], prev[self.TR], self.DT[self.TR], mu[self.TR]
        tC = self.C[self.TR]
        n = len(tpsi)
        sa, sb = self.sigma2_prior
        m0, s0 = self.loglam_prior
        lo, hi = self.lam_bounds

        def lp(lam, kappa):          # log-lik with sigma2 integrated out
            Lt = lam * np.exp(-kappa * tC)
            ph = np.exp(-Lt * tdt)
            c = -np.expm1(-2.0 * Lt * tdt) / (2.0 * Lt)
            e = tpsi - ph * tprev - (1.0 - ph) * tmu
            S = 0.5 * np.sum(e * e / c)
            return -0.5 * np.sum(np.log(c)) - (sa + n / 2.0) * np.log(sb + S), S

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
        _, S = lp(self.lam, self.kappa)
        self.sigma2 = float(1.0 / self.rng.gamma(sa + n / 2.0, 1.0 / (sb + S)))

    # ------------------------------------------------------------ main loop
    def run(self, n_iter=1000, burn_in=400, thin=4, verbose=False):
        rows = np.arange(self.N)
        for it in range(n_iter):
            phi, q = self._dyn()
            omega = np.ones((self.N, self.Tm))
            omega[self.M] = np.clip(sample_pg1(self.psi[self.M], self.rng), 1e-8, None)
            self.psi = self._ffbs(omega, phi, q, self._mu())
            self._update_effects(self.psi, phi, q)
            self._update_lam_sigma2(self.psi, it, adapt=(it < burn_in))
            if it >= burn_in and (it - burn_in) % thin == 0:
                sm = self.samples
                sm['b0'].append(self.b0); sm['g0'].append(self.g0)
                sm['lam'].append(self.lam); sm['kappa'].append(self.kappa)
                sm['sigma2'].append(self.sigma2)
                sm['tau2_alpha'].append(self.tau2['alpha'])
                sm['tau2_beta'].append(self.tau2['beta'])
                sm['tau2_gamma'].append(self.tau2['gamma'])
                sm['alpha'].append(self.alpha.copy()); sm['beta'].append(self.beta.copy())
                sm['gamma'].append(self.gamma.copy())
                sm['psi_last'].append(self.psi[rows, self.lens - 1].copy())
            if verbose and it % 100 == 0:
                print(f"[dyn-dt] it {it:5d} lam={self.lam:.4f} sigma2={self.sigma2:.3f} "
                      f"b0={self.b0:.2f} g0={self.g0:.2f} mh_acc={self.acc_rate:.2f}")
        for k in self.samples:
            self.samples[k] = np.array(self.samples[k])
        return self.samples

    def fit(self, train_seqs=None, **kw):
        """Uniform interface: sequences were given at construction."""
        self.run(**kw)
        return self

    # ------------------------------------------------------------ reporting
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

    # ------------------------------------------------------------ prediction
    def predict(self, test_seqs, max_draws=100, return_draws=False):
        sm = self.samples
        S_all = len(sm['lam'])
        sel = np.unique(np.linspace(0, S_all - 1, min(max_draws, S_all)).astype(int))
        S = len(sel)
        K = len(test_seqs)
        Tt = max(len(s['r']) for s in test_seqs)
        Rt = np.zeros((K, Tt)); Xt = np.zeros((K, Tt))
        DTt = np.ones((K, Tt)); Mt = np.zeros((K, Tt), bool)
        ui = np.zeros(K, int); ii = np.zeros(K, int)
        Ct = np.zeros((K, Tt))
        warm = np.zeros(K, bool); widx = np.zeros(K, int)
        for k, s in enumerate(test_seqs):
            r = np.asarray(s['r'], float); T = len(r)
            Rt[k, :T] = r
            DTt[k, :T] = np.clip(np.asarray(s['dt'], float), *self.dt_bounds)
            Mt[k, :T] = True
            ui[k] = s['user'] if s['user'] < self.n_users else self.n_users
            ii[k] = s['item'] if s['item'] < self.n_items else self.n_items
            j = self.key2idx.get((s['user'], s['item']))
            x0 = 0.0
            if j is not None:
                warm[k], widx[k] = True, j
                x0 = float(self.R[j, self.lens[j] - 1])
            Xt[k, 0] = x0
            base = float(self.R[j, :self.lens[j]].sum()) if j is not None else 0.0
            Ct[k, :T] = np.log1p(base + np.concatenate([[0.0], np.cumsum(r)[:-1]]))
            if T > 1:
                Xt[k, 1:T] = r[:T - 1]
        pad = np.zeros((S, 1))
        alpha = np.hstack([sm['alpha'][sel], pad])[:, ui]
        beta = np.hstack([sm['beta'][sel], np.zeros((S, 1))])[:, ii]
        gamma = np.hstack([sm['gamma'][sel], pad])[:, ui]
        b0 = sm['b0'][sel][:, None]
        lam = sm['lam'][sel][:, None]
        kap = sm['kappa'][sel][:, None]
        sig2 = sm['sigma2'][sel][:, None]
        m = np.where(warm[None, :], sm['psi_last'][sel][:, widx], 0.0)
        P = np.zeros((S, K))
        cold = ~warm[None, :]
        out = np.zeros((K, Tt))
        draws = np.zeros((S, K, Tt)) if return_draws else None
        for t in range(Tt):
            dt_t = DTt[:, t][None, :]
            mu = b0 + alpha + beta + gamma * Xt[:, t][None, :]
            rate = lam * np.exp(-kap * Ct[:, t][None, :])
            phi = np.exp(-rate * dt_t)
            q = sig2 * (-np.expm1(-2.0 * rate * dt_t) / (2.0 * rate))
            phi = np.where(cold, 0.0, phi)
            q = np.where(cold, self.init_var, q)
            a = phi * m + (1.0 - phi) * mu
            Rr = phi ** 2 * P + q
            p_t = _sigmoid(a / np.sqrt(1.0 + np.pi * Rr / 8.0))
            out[:, t] = p_t.mean(axis=0)
            if return_draws:
                draws[:, :, t] = p_t
            # online update with the observed outcome (PG-mean pseudo-observation)
            w = np.where(np.abs(a) < 1e-6, 0.25, np.tanh(a / 2.0) / (2.0 * np.where(a == 0, 1, a)))
            w = np.clip(w, 1e-6, None)
            zt = (Rt[:, t][None, :] - 0.5) / w
            Kg = Rr / (Rr + 1.0 / w)
            m_new = a + Kg * (zt - a)
            P_new = (1.0 - Kg) * Rr
            upd = Mt[:, t][None, :]
            m = np.where(upd, m_new, m)
            P = np.where(upd, P_new, P)
        preds = [out[k, :len(s['r'])].copy() for k, s in enumerate(test_seqs)]
        if return_draws:
            return preds, [draws[:, k, :len(s['r'])] for k, s in enumerate(test_seqs)]
        return preds