"""
Static hierarchical Bayesian retention model -- the phi=0, sigma_w2->0
special case of the dynamic model (Section 3.3 of the report), implemented
as its own standalone (non-state-space) Gibbs sampler.

This is REQUIRED as a baseline for Experiment 1 (Table 1): it isolates
whether allowing half-life/log-odds to evolve over time actually buys
predictive improvement over a single fixed per-(user,item) effect.

Model (mirrors the original static-report Eq 4-6, on the logit scale to
match the dynamic model's operationalisation and keep the comparison fair
-- both models are evaluated on the same observation scale):

    psi_{u,i} = beta_i + gamma_u          (ONE value per (u,i) pair, no t)
    r_{u,i,t} ~ Bernoulli(sigmoid(psi_{u,i}))   for all t in that pair's sequence

This is fit via the same Polya-Gamma augmentation trick (still needed,
since the likelihood is still Bernoulli), but WITHOUT any FFBS step, since
there is no time-varying state -- psi_{u,i} is a single conjugate-Normal
unknown per pair, given the PG augmentation.
"""
import numpy as np
from pg_sampler import sample_pg1


class StaticRetentionModel:
    def __init__(self, sequences, rng=None):
        self.seqs = sequences
        self.n_users = max(s['user'] for s in sequences) + 1
        self.n_items = max(s['item'] for s in sequences) + 1
        self.rng = rng if rng is not None else np.random.default_rng(0)

        self.beta = np.zeros(self.n_items)
        self.gamma = np.zeros(self.n_users)
        self.tau_beta2 = 0.5
        self.tau_gamma2 = 0.5

        # One static psi per sequence pair, informed by beta_i + gamma_u
        self.psi_pair = np.zeros(len(sequences))

        self.samples = {'beta': [], 'gamma': []}

    def run(self, n_iter=1000, burn_in=300, thin=2, verbose=True):
        for s_iter in range(n_iter):
            # 1. Update per-pair static psi (conjugate given PG augmentation)
            for idx, s in enumerate(self.seqs):
                i, u = s['item'], s['user']
                mu_pair = self.beta[i] + self.gamma[u]
                T = len(s['r'])
                psi_vec = np.full(T, self.psi_pair[idx])
                omega = sample_pg1(psi_vec, rng=self.rng)
                omega = np.clip(omega, 1e-6, None)
                r = s['r']
                z = (r - 0.5) / omega
                # Precision-weighted conjugate normal update:
                # likelihood precision sum(omega), prior N(mu_pair, tau_pair2)
                prior_var_pair = 0.3  # tight-ish: psi_pair should track mu_pair closely
                post_var = 1.0 / (omega.sum() + 1.0 / prior_var_pair)
                post_mean = post_var * (np.sum(omega * z) + mu_pair / prior_var_pair)
                self.psi_pair[idx] = self.rng.normal(post_mean, np.sqrt(post_var))

            # 2. Update beta_i (item effects) given psi_pair - gamma_u residuals
            resid_sum = np.zeros(self.n_items)
            n_obs = np.zeros(self.n_items)
            for idx, s in enumerate(self.seqs):
                i, u = s['item'], s['user']
                resid_sum[i] += self.psi_pair[idx] - self.gamma[u]
                n_obs[i] += 1
            post_var = 1.0 / (n_obs + 1.0 / self.tau_beta2)
            post_mean = post_var * resid_sum
            self.beta = self.rng.normal(post_mean, np.sqrt(post_var))

            # 3. Update gamma_u (user effects) given psi_pair - beta_i residuals
            resid_sum = np.zeros(self.n_users)
            n_obs = np.zeros(self.n_users)
            for idx, s in enumerate(self.seqs):
                i, u = s['item'], s['user']
                resid_sum[u] += self.psi_pair[idx] - self.beta[i]
                n_obs[u] += 1
            post_var = 1.0 / (n_obs + 1.0 / self.tau_gamma2)
            post_mean = post_var * resid_sum
            self.gamma = self.rng.normal(post_mean, np.sqrt(post_var))

            if s_iter >= burn_in and (s_iter - burn_in) % thin == 0:
                self.samples['beta'].append(self.beta.copy())
                self.samples['gamma'].append(self.gamma.copy())

            if verbose and s_iter % 100 == 0:
                print(f"[static] iter {s_iter}")

        for k in self.samples:
            self.samples[k] = np.array(self.samples[k])
        return self.samples

    def predict_proba(self, user, item):
        """Posterior mean recall probability for a given (user, item) pair,
        constant across all future reviews (this is the model's whole point --
        it CANNOT adapt within a sequence)."""
        beta_mean = self.samples['beta'].mean(axis=0)[item]
        gamma_mean = self.samples['gamma'].mean(axis=0)[user]
        psi = beta_mean + gamma_mean
        return 1.0 / (1.0 + np.exp(-psi))
