"""Controlled variants of the branch's pure NumPy time-aware model.

Retains its priors, FFBS and collapsed MH transition. Uses the reference
polyagamma package in this harness instead of the truncated-series sampler.
The original implementations and their single-chain diagnostics stay untouched.
"""
from pathlib import Path
import numpy as np
from polyagamma import random_polyagamma
from scipy.linalg import solve_triangular
from AnkiFirstRun.dynamic_retention_model_optimised import DynamicRetentionModelDT, _sigmoid

SAMPLERS = ('reference', 'joint', 'blocked', 'joint_blocked')

VARIANTS = ('full', 'no_practice', 'no_hierarchy', 'no_dynamics')


class DiagnosticModel(DynamicRetentionModelDT):
    def __init__(self, sequences, variant='full', seed=0, chain=0, sampler='reference', mh_steps=5, dt_bounds=(.01, 365.)):
        if variant not in VARIANTS:
            raise ValueError(f'Unknown variant: {variant}')
        super().__init__(sequences, rng=np.random.default_rng(seed),
                         use_kappa=variant not in ('no_practice', 'no_dynamics'), dt_bounds=dt_bounds)
        if sampler not in SAMPLERS:
            raise ValueError(f'Unknown sampler: {sampler}')
        self.sampler = sampler
        self.n_mh = mh_steps
        self.joint_cov = np.diag([.2**2, .15**2])
        self.joint_scale = 1.0
        self.joint_count = 0
        self.joint_mean = np.zeros(2)
        self.joint_m2 = np.zeros((2, 2))
        self.variant = variant
        self.dynamic = variant != 'no_dynamics'
        self.hierarchy = variant != 'no_hierarchy'
        self.b0 += [-1.0, 0.5, 1.5, -0.5][chain % 4]
        self.lam = [0.005, 0.05, 0.5, 2.0][chain % 4]
        self.sigma2 = [0.05, 0.2, 1.0, 3.0][chain % 4]
        self.kappa = [-1., 0., 0.5, 1.5][chain % 4] if self.use_kappa else 0.
        if self.hierarchy:
            self.alpha = self.rng.normal(0, 0.5, self.n_users)
            self.beta = self.rng.normal(0, 0.5, self.n_items)
            self.gamma = self.rng.normal(0.5, 0.3, self.n_users)
        else:
            self.g0 = 0.5
            self.gamma[:] = self.g0
        self.psi[:] = self.b0

    def _update_effects(self, psi, phi, q):
        if self.sampler in ('blocked', 'joint_blocked'):
            return self._blocked_effects(psi, phi, q)
        if self.hierarchy:
            return super()._update_effects(psi, phi, q)
        prev = np.zeros_like(psi)
        prev[:, 1:] = psi[:, :-1]
        y, c, v = (psi - phi * prev)[self.M], (1 - phi)[self.M], q[self.M]
        zeros = np.zeros(len(y), dtype=int)
        self.b0 = float(self._gauss(1, zeros, c, y - c * self.g0 * self.fx, v, 0, 100)[0])
        self.g0 = float(self._gauss(1, zeros, c * self.fx, y - c * self.b0, v, 0, 100)[0])
        self.gamma[:] = self.g0

    def _effect_system(self, psi, phi, q):
        """Precision/information for the complete conditional Gaussian block.

        theta = [b0, alpha, beta, gamma, g0]. The gamma-g0 prior contributes
        off-diagonal precision; no sum-to-zero constraint or prior is changed.
        """
        prev = np.zeros_like(psi)
        prev[:, 1:] = psi[:, :-1]
        y = (psi - phi * prev)[self.M]
        c = (1 - phi)[self.M]
        if not self.hierarchy:
            design = np.column_stack([c, c * self.fx])
            precision = np.eye(2) / 100
        else:
            u, i = self.n_users, self.n_items
            dim = 2 + 2 * u + i
            design = np.zeros((len(y), dim))
            rows = np.arange(len(y))
            design[:, 0] = c
            design[rows, 1 + self.fu] = c
            design[rows, 1 + u + self.fi] = c
            design[rows, 1 + u + i + self.fu] = c * self.fx
            precision = np.zeros((dim, dim))
            precision[0, 0] = precision[-1, -1] = .01
            precision[1:1+u, 1:1+u] = np.eye(u) / self.tau2['alpha']
            precision[1+u:1+u+i, 1+u:1+u+i] = np.eye(i) / self.tau2['beta']
            g = np.arange(1+u+i, dim-1)
            precision[g, g] = 1 / self.tau2['gamma']
            precision[g, -1] = precision[-1, g] = -1 / self.tau2['gamma']
            precision[-1, -1] += u / self.tau2['gamma']
        weight = 1 / q[self.M]
        return precision + design.T @ (weight[:, None] * design), design.T @ (weight * y)

    def _blocked_effects(self, psi, phi, q):
        precision, information = self._effect_system(psi, phi, q)
        chol = np.linalg.cholesky(precision)
        mean = solve_triangular(chol.T, solve_triangular(chol, information, lower=True), lower=False)
        draw = mean + solve_triangular(chol.T, self.rng.standard_normal(len(mean)), lower=False)
        self.b0, self.g0 = float(draw[0]), float(draw[-1])
        if self.hierarchy:
            u, i = self.n_users, self.n_items
            self.alpha = draw[1:1+u]
            self.beta = draw[1+u:1+u+i]
            self.gamma = draw[1+u+i:-1]
            self.tau2['alpha'] = self._inv_gamma_tau2(self.alpha, 0.)
            self.tau2['beta'] = self._inv_gamma_tau2(self.beta[self.item_seen], 0.)
            self.tau2['gamma'] = self._inv_gamma_tau2(self.gamma, self.g0)
        else:
            self.gamma[:] = self.g0

    def _decay_log_density(self, theta, psi):
        """Density with respect to d log(lambda) d kappa; sigma2 integrated out."""
        loglam, kappa = theta
        if not np.log(self.lam_bounds[0]) < loglam < np.log(self.lam_bounds[1]) or abs(kappa) >= 3:
            return -np.inf, np.nan
        rate = np.exp(loglam - kappa * self.C[self.TR])
        gap = self.DT[self.TR]
        phi = np.exp(-rate * gap)
        c = -np.expm1(-2 * rate * gap) / (2 * rate)
        prev = np.zeros_like(psi)
        prev[:, 1:] = psi[:, :-1]
        residual = psi[self.TR] - phi * prev[self.TR] - (1-phi) * self._mu()[self.TR]
        scale = self.sigma2_prior[1] + .5 * np.sum(residual**2 / c)
        shape = self.sigma2_prior[0] + self.TR.sum() / 2
        m, sd = self.loglam_prior
        lp = -.5 * np.log(c).sum() - shape * np.log(scale)
        lp -= .5 * ((loglam-m)/sd)**2 + .5 * (kappa/self.kappa_prior_sd)**2
        return float(lp), float(scale)

    def _update_lam_sigma2(self, psi, it, adapt):
        if self.sampler not in ('joint', 'joint_blocked') or not self.use_kappa:
            return super()._update_lam_sigma2(psi, it, adapt)
        theta = np.array([np.log(self.lam), self.kappa])
        current, scale = self._decay_log_density(theta, psi)
        chol = np.linalg.cholesky(self.joint_cov) * self.joint_scale
        accepted = 0
        for _ in range(self.n_mh):
            proposal = theta + chol @ self.rng.standard_normal(2)
            target, proposed_scale = self._decay_log_density(proposal, psi)
            if np.log(self.rng.random()) < target - current:
                theta, current, scale = proposal, target, proposed_scale
                accepted += 1
        self.lam, self.kappa = float(np.exp(theta[0])), float(theta[1])
        self.acc_rate = accepted / self.n_mh
        # Restore sigma2 immediately after the collapsed transition.
        shape = self.sigma2_prior[0] + self.TR.sum() / 2
        self.sigma2 = float(1 / self.rng.gamma(shape, 1 / scale))
        if adapt:
            self.joint_count += 1
            delta = theta - self.joint_mean
            self.joint_mean += delta / self.joint_count
            self.joint_m2 += np.outer(delta, theta - self.joint_mean)
            if self.joint_count >= 50:
                self.joint_cov = .95 * self.joint_m2 / (self.joint_count-1) + .05 * np.diag([.2**2, .15**2])
            self.joint_scale = float(np.clip(self.joint_scale * np.exp((self.acc_rate-.234)/(it+1)**.6), .01, 10))

    def run(self, n_iter=2000, burn_in=1000, thin=1, verbose=False):
        if not 0 <= burn_in < n_iter or thin < 1:
            raise ValueError('Require 0 <= warmup < iterations and thinning >= 1')
        keys = ['b0', 'g0', 'alpha', 'beta', 'gamma']
        if self.hierarchy:
            keys += ['tau2_alpha', 'tau2_beta', 'tau2_gamma']
        if self.dynamic:
            keys += ['lam', 'sigma2', 'psi_last']
            if self.use_kappa:
                keys += ['kappa']
        self.samples = {k: [] for k in keys}
        self.warmup = {k: [] for k in ['b0'] + (['lam', 'sigma2'] if self.dynamic else [])
                       + (['kappa'] if self.use_kappa else [])}
        acceptance, kappa_moves = [], []
        rows = np.arange(self.N)
        for it in range(n_iter):
            if it % 25 == 0 and getattr(self, 'cancel_file', None) and Path(self.cancel_file).exists():
                raise InterruptedError('Cancelled by user')
            omega = np.ones((self.N, self.Tm))
            omega[self.M] = np.maximum(random_polyagamma(1., self.psi[self.M], random_state=self.rng), 1e-10)
            if self.dynamic:
                phi, q = self._dyn()
                self.psi = self._ffbs(omega, phi, q, self._mu())
                self._update_effects(self.psi, phi, q)
                before_kappa = self.kappa
                self._update_lam_sigma2(self.psi, it, adapt=it < burn_in)
            else:
                # PG logistic regression with exactly the same mean effects and priors.
                z = (self.R - .5) / omega
                self._update_effects(z, np.zeros_like(omega), 1. / omega)
                self.psi = self._mu()
            if it < burn_in:
                for k in self.warmup:
                    self.warmup[k].append(float(getattr(self, k)))
            else:
                if self.dynamic:
                    acceptance.append(self.acc_rate)
                    if self.use_kappa:
                        kappa_moves.append(float(self.kappa != before_kappa))
                if (it - burn_in) % thin == 0:
                    for k in keys:
                        if k.startswith('tau2_'):
                            value = self.tau2[k[5:]]
                        elif k == 'psi_last':
                            value = self.psi[rows, self.lens - 1]
                        else:
                            value = getattr(self, k)
                        self.samples[k].append(np.array(value).copy())
        self.samples = {k: np.array(v) for k, v in self.samples.items()}
        # Keep prediction interface compatible; constants are not monitored parameters.
        n = len(self.samples['b0'])
        if not self.use_kappa:
            self.samples['kappa'] = np.zeros(n)
        self.sampling_info = dict(lambda_acceptance=float(np.mean(acceptance)) if acceptance else None,
                                  kappa_sweep_move_fraction=float(np.mean(kappa_moves)) if kappa_moves else None,
                                  lambda_step=self.step, kappa_step=self.step_k, sampler=self.sampler,
                                  mh_steps=self.n_mh, joint_covariance=self.joint_cov.tolist(),
                                  joint_scale=self.joint_scale, adaptation_draws=self.joint_count)
        return self.samples

    def predict(self, test_seqs, max_draws=200, return_draws=False):
        if self.dynamic:
            return super().predict(test_seqs, max_draws=max_draws, return_draws=return_draws)
        sm = self.samples
        sel = np.unique(np.linspace(0, len(sm['b0']) - 1, min(max_draws, len(sm['b0']))).astype(int))
        draws = []
        for s in test_seqs:
            u, i = s['user'], s['item']
            p = _sigmoid(sm['b0'][sel, None] + sm['alpha'][sel, u, None]
                         + sm['beta'][sel, i, None] + sm['gamma'][sel, u, None] * s['x'][None, :])
            draws.append(p)
        preds = [p.mean(axis=0) for p in draws]
        return (preds, draws) if return_draws else preds
