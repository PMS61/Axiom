"""
dynamic_retention_model.py

Hierarchical DYNAMIC (state-space) retention model -- Section 3.3/3.4/4.1
of the report. The latent, logit-scale memory state follows an AR(1)
process driven by item difficulty and user learning-rate effects:

    psi_{u,i,t} = phi * psi_{u,i,t-1} + beta_i + gamma_u * x_t + w_t,
    w_t ~ N(0, sigma_w2)
    r_{u,i,t} ~ Bernoulli(sigmoid(psi_{u,i,t}))

Fit via a Polya-Gamma-augmented Gibbs sampler with Forward-Filtering
Backward-Sampling (FFBS) for each (user, item) trajectory (Carter & Kohn,
1994; Fruhwirth-Schnatter, 1994; Polson, Scott & Windle, 2013). This is
Algorithm 2 of the report.

NOTE ON THE LOGIT REPARAMETERISATION: this model works on logit(p), not
directly on the exponential-decay half-life link p = 2^{-dt/h}. As
discussed in the report's Discussion/Limitations section, this is a
deliberate trade-off: it is what makes exact Polya-Gamma augmentation (and
therefore exact FFBS) apply, at the cost of not being literally the
half-life-regression forgetting curve. The 'dt' field is kept in the
sequence schema for parity with data_pipeline.py and as a hook for a
future dt-aware extension (e.g. feeding dt as an additional regressor, or
switching to a Laplace/EKF approximation of the exponential link
directly), but is NOT used by this implementation.

Expected sequence schema (list of dicts), one entry per (user, item) pair
with an observed review sequence -- matches data_pipeline.py and
synthetic_validation.py:
    {
        'user': int,                    # 0-indexed user id
        'item': int,                    # 0-indexed item id
        'r':    np.ndarray[int], (T,)   # binary recall/completion outcomes
        'x':    np.ndarray[int], (T,)   # previous review's outcome (drift covariate)
        'dt':   np.ndarray[float], (T,) # days since last review (not used here)
    }
"""
import numpy as np
from pg_sampler import sample_pg1


class DynamicRetentionModel:
    def __init__(self, sequences, rng=None,
                 tau_beta2=1.0, tau_gamma2=1.0,
                 phi_prior_mean=0.5, phi_prior_var=0.25,
                 phi_bounds=(0.01, 0.99),
                 sigma_w2_prior_a=2.0, sigma_w2_prior_b=1.0,
                 init_state_var=1.0):
        self.seqs = sequences
        self.n_users = max(s['user'] for s in sequences) + 1
        self.n_items = max(s['item'] for s in sequences) + 1
        self.rng = rng if rng is not None else np.random.default_rng(0)

        self.tau_beta2 = tau_beta2
        self.tau_gamma2 = tau_gamma2
        self.phi_prior_mean = phi_prior_mean
        self.phi_prior_var = phi_prior_var
        self.phi_bounds = phi_bounds
        self.sigma_w2_prior_a = sigma_w2_prior_a
        self.sigma_w2_prior_b = sigma_w2_prior_b
        self.init_state_var = init_state_var

        self.beta = np.zeros(self.n_items)
        self.gamma = np.zeros(self.n_users)
        self.phi = phi_prior_mean
        self.sigma_w2 = sigma_w2_prior_b / max(sigma_w2_prior_a - 1, 1e-3)

        # Current state trajectories (one array per sequence); starts at 0
        # and is refined by FFBS sweeps.
        self.psi = [np.zeros(len(s['r'])) for s in self.seqs]

        self.samples = {'beta': [], 'gamma': [], 'phi': [], 'sigma_w2': []}
        self._psi_sum = [np.zeros(len(s['r'])) for s in self.seqs]
        self._psi_n = 0

    # ---------------------------------------------------------------
    # Step 1: Polya-Gamma augmentation + FFBS (Algorithm 2, lines 3-6)
    # ---------------------------------------------------------------
    def _ffbs_one_sequence(self, s, psi_current):
        T = len(s['r'])
        i, u = s['item'], s['user']
        mu = self.beta[i] + self.gamma[u] * s['x']  # drift at each t, shape (T,)

        omega = sample_pg1(psi_current, self.rng)
        omega = np.clip(omega, 1e-8, None)
        z = (s['r'] - 0.5) / omega        # conditionally-Gaussian pseudo-observation
        obs_var = 1.0 / omega

        # --- forward filter (Kalman recursion) ---
        m = np.zeros(T)    # filtered means
        P = np.zeros(T)    # filtered variances
        a = np.zeros(T)    # one-step-ahead predicted means (needed for backward pass)
        Rp = np.zeros(T)   # one-step-ahead predicted variances

        for t in range(T):
            if t == 0:
                a_t, Rp_t = mu[0], self.init_state_var
            else:
                a_t = self.phi * m[t - 1] + mu[t]
                Rp_t = self.phi ** 2 * P[t - 1] + self.sigma_w2
            a[t], Rp[t] = a_t, Rp_t

            K_t = Rp_t / (Rp_t + obs_var[t])
            m[t] = a_t + K_t * (z[t] - a_t)
            P[t] = (1 - K_t) * Rp_t

        # --- backward sample (Carter & Kohn 1994 / Fruhwirth-Schnatter 1994) ---
        psi_new = np.zeros(T)
        psi_new[T - 1] = self.rng.normal(m[T - 1], np.sqrt(max(P[T - 1], 1e-12)))
        for t in range(T - 2, -1, -1):
            J_t = P[t] * self.phi / Rp[t + 1]
            h_t = m[t] + J_t * (psi_new[t + 1] - a[t + 1])
            V_t = P[t] - J_t * self.phi * P[t]
            psi_new[t] = self.rng.normal(h_t, np.sqrt(max(V_t, 1e-12)))

        return psi_new

    # ---------------------------------------------------------------
    # Step 2: conjugate updates for beta, gamma, phi, sigma_w2
    # (Algorithm 2, lines 7-11)
    # ---------------------------------------------------------------
    def _increments(self, s, psi):
        """Returns (delta_t, var_t) where delta_t is the 'observed'
        increment implied by the current state trajectory at each t
        (delta_0 = psi_0, delta_t = psi_t - phi*psi_{t-1} for t>=1), and
        var_t is its known conditional variance (init_state_var at t=0,
        sigma_w2 thereafter). These are the quantities beta/gamma/phi are
        regressed against."""
        T = len(psi)
        delta = np.empty(T)
        var_t = np.empty(T)
        delta[0] = psi[0]
        var_t[0] = self.init_state_var
        if T > 1:
            delta[1:] = psi[1:] - self.phi * psi[:-1]
            var_t[1:] = self.sigma_w2
        return delta, var_t

    def _update_beta(self):
        resid_sum = np.zeros(self.n_items)
        prec_sum = np.zeros(self.n_items)
        for s, psi in zip(self.seqs, self.psi):
            i, u = s['item'], s['user']
            delta, var_t = self._increments(s, psi)
            resid = delta - self.gamma[u] * s['x']
            resid_sum[i] += np.sum(resid / var_t)
            prec_sum[i] += np.sum(1.0 / var_t)
        post_var = 1.0 / (prec_sum + 1.0 / self.tau_beta2)
        post_mean = post_var * resid_sum
        self.beta = self.rng.normal(post_mean, np.sqrt(post_var))

    def _update_gamma(self):
        resid_sum = np.zeros(self.n_users)
        prec_sum = np.zeros(self.n_users)
        for s, psi in zip(self.seqs, self.psi):
            i, u = s['item'], s['user']
            delta, var_t = self._increments(s, psi)
            resid = delta - self.beta[i]
            x = s['x'].astype(float)
            resid_sum[u] += np.sum(resid * x / var_t)
            prec_sum[u] += np.sum((x ** 2) / var_t)
        post_var = 1.0 / (prec_sum + 1.0 / self.tau_gamma2)
        post_mean = post_var * resid_sum
        self.gamma = self.rng.normal(post_mean, np.sqrt(post_var))

    def _update_phi(self):
        """psi_t - mu_t = phi * psi_{t-1} + w_t for t=1..T-1 (t=0 carries
        no information about phi)."""
        num = self.phi_prior_mean / self.phi_prior_var
        denom = 1.0 / self.phi_prior_var
        for s, psi in zip(self.seqs, self.psi):
            if len(psi) < 2:
                continue
            i, u = s['item'], s['user']
            mu = self.beta[i] + self.gamma[u] * s['x']
            y = psi[1:] - mu[1:]
            xprev = psi[:-1]
            num += np.sum(xprev * y) / self.sigma_w2
            denom += np.sum(xprev ** 2) / self.sigma_w2
        post_var = 1.0 / denom
        post_mean = post_var * num
        draw = self.rng.normal(post_mean, np.sqrt(post_var))
        # Pragmatic simplification (see report Discussion): clip to a
        # plausible persistence range rather than truncating the sampling
        # distribution itself.
        lo, hi = self.phi_bounds
        self.phi = float(np.clip(draw, lo, hi))

    def _update_sigma_w2(self):
        a_post = self.sigma_w2_prior_a
        b_post = self.sigma_w2_prior_b
        for s, psi in zip(self.seqs, self.psi):
            if len(psi) < 2:
                continue
            i, u = s['item'], s['user']
            mu = self.beta[i] + self.gamma[u] * s['x']
            resid = psi[1:] - self.phi * psi[:-1] - mu[1:]
            a_post += len(resid) / 2.0
            b_post += np.sum(resid ** 2) / 2.0
        self.sigma_w2 = float(1.0 / self.rng.gamma(shape=a_post, scale=1.0 / b_post))

    # ---------------------------------------------------------------
    # Main Gibbs loop
    # ---------------------------------------------------------------
    def run(self, n_iter=1000, burn_in=400, thin=2, verbose=False):
        for it in range(n_iter):
            self.psi = [self._ffbs_one_sequence(s, psi_cur)
                        for s, psi_cur in zip(self.seqs, self.psi)]

            self._update_beta()
            self._update_gamma()
            self._update_phi()
            self._update_sigma_w2()

            if it >= burn_in and (it - burn_in) % thin == 0:
                self.samples['beta'].append(self.beta.copy())
                self.samples['gamma'].append(self.gamma.copy())
                self.samples['phi'].append(self.phi)
                self.samples['sigma_w2'].append(self.sigma_w2)
                for idx, psi in enumerate(self.psi):
                    self._psi_sum[idx] += psi
                self._psi_n += 1

            if verbose and it % 100 == 0:
                print(f"[dynamic] iter {it:4d}  phi={self.phi:.3f}  "
                      f"sigma_w2={self.sigma_w2:.3f}")

        for k in self.samples:
            self.samples[k] = np.array(self.samples[k])

        # Overwrite self.psi with the post-burn-in posterior MEAN
        # trajectory per sequence -- this is what evaluation.py reads via
        # model.psi[idx][-1] as the "last fitted state" to forecast from.
        if self._psi_n > 0:
            self.psi = [ssum / self._psi_n for ssum in self._psi_sum]

        return self.samples
