"""Controlled variants of the branch's pure NumPy time-aware model.

Retains its priors, FFBS and collapsed MH transition. Uses the reference
polyagamma package in this harness instead of the truncated-series sampler.
The original implementations and their single-chain diagnostics stay untouched.
"""
from pathlib import Path
import numpy as np
from polyagamma import random_polyagamma
from AnkiFirstRun.dynamic_retention_model_optimised import DynamicRetentionModelDT, _sigmoid

VARIANTS = ('full', 'no_practice', 'no_hierarchy', 'no_dynamics')


class DiagnosticModel(DynamicRetentionModelDT):
    def __init__(self, sequences, variant='full', seed=0, chain=0):
        if variant not in VARIANTS:
            raise ValueError(f'Unknown variant: {variant}')
        super().__init__(sequences, rng=np.random.default_rng(seed),
                         use_kappa=variant not in ('no_practice', 'no_dynamics'))
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
        if self.hierarchy:
            return super()._update_effects(psi, phi, q)
        prev = np.zeros_like(psi)
        prev[:, 1:] = psi[:, :-1]
        y, c, v = (psi - phi * prev)[self.M], (1 - phi)[self.M], q[self.M]
        zeros = np.zeros(len(y), dtype=int)
        self.b0 = float(self._gauss(1, zeros, c, y - c * self.g0 * self.fx, v, 0, 100)[0])
        self.g0 = float(self._gauss(1, zeros, c * self.fx, y - c * self.b0, v, 0, 100)[0])
        self.gamma[:] = self.g0

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
                                  lambda_step=self.step, kappa_step=self.step_k)
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
