"""CLI-only exact PG coefficient/path interweaving for the unchanged full OU model."""
import numpy as np
from scipy.linalg import solve_triangular
from .innovation import InnovationModel


class CoefficientInnovationModel(InnovationModel):
    def __init__(self, *args, coefficient_interweave=True, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.hierarchy or not self.dynamic:
            raise ValueError('Coefficient/path pilot requires the full hierarchical dynamics model.')
        self.coefficient_interweave = coefficient_interweave
        self.coefficient_moves = 0

    def coefficients(self):
        return np.r_[self.b0, self.alpha, self.beta, self.gamma, self.g0]

    def set_coefficients(self, theta):
        u, i = self.n_users, self.n_items
        self.b0, self.g0 = float(theta[0]), float(theta[-1])
        self.alpha = theta[1:1+u].copy()
        self.beta = theta[1+u:1+u+i].copy()
        self.gamma = theta[1+u+i:-1].copy()

    def coefficient_prior_precision(self):
        u, i = self.n_users, self.n_items
        dim = 2 + 2*u + i
        precision = np.zeros((dim, dim))
        precision[0, 0] = precision[-1, -1] = .01
        precision[1:1+u, 1:1+u] = np.eye(u)/self.tau2['alpha']
        precision[1+u:1+u+i, 1+u:1+u+i] = np.eye(i)/self.tau2['beta']
        g = np.arange(1+u+i, dim-1)
        precision[g, g] = 1/self.tau2['gamma']
        precision[g, -1] = precision[-1, g] = -1/self.tau2['gamma']
        precision[-1, -1] += u/self.tau2['gamma']
        return precision

    def affine_path(self, z):
        """psi[M] = offset + H theta at fixed z/OU parameters.

        Initial phi=0, q=1 is supplied by _dyn, not stationary initialization.
        Constant means have response one; lag slopes obey the same OU recursion.
        g0 has zero likelihood column and enters through the gamma hierarchy prior.
        """
        phi, q = self._dyn()
        offset = np.zeros_like(z)
        response = np.zeros_like(z)
        previous = np.zeros(self.N)
        lag = np.zeros(self.N)
        for t in range(self.Tm):
            previous = phi[:, t]*previous + np.sqrt(q[:, t])*z[:, t]
            lag = phi[:, t]*lag + (1-phi[:, t])*self.X[:, t]
            offset[:, t] = np.where(self.M[:, t], previous, 0.)
            response[:, t] = lag
        u, i = self.n_users, self.n_items
        design = np.zeros((self.M.sum(), 2+2*u+i))
        rows = np.arange(len(design))
        design[:, 0] = 1
        design[rows, 1+self.fu] = 1
        design[rows, 1+u+self.fi] = 1
        design[rows, 1+u+i+self.fu] = response[self.M]
        return offset[self.M], design

    def coefficient_system(self, z, omega):
        offset, design = self.affine_path(z)
        w = omega[self.M]
        precision = self.coefficient_prior_precision() + design.T @ (w[:, None]*design)
        info = design.T @ (self.R[self.M]-.5-w*offset)
        return precision, info

    def _coefficient_path_move(self):
        z = self.innovations(self.psi)
        precision, info = self.coefficient_system(z, self._omega)
        chol = np.linalg.cholesky(precision)
        mean = solve_triangular(chol.T, solve_triangular(chol, info, lower=True), lower=False)
        theta = mean + solve_triangular(chol.T, self.rng.standard_normal(len(mean)), lower=False)
        self.set_coefficients(theta)
        self.psi = self.reconstruct(z)
        # Given the new coefficients, these are the original exact IG conditionals.
        # At fixed z the state Jacobian depends only on OU parameters, not scales.
        self.tau2['alpha'] = self._inv_gamma_tau2(self.alpha, 0.)
        self.tau2['beta'] = self._inv_gamma_tau2(self.beta[self.item_seen], 0.)
        self.tau2['gamma'] = self._inv_gamma_tau2(self.gamma, self.g0)
        self.coefficient_moves += 1

    def _update_lam_sigma2(self, psi, it, adapt):
        # First restore sigma after collapsed decay, then complete the original
        # innovation parameter move. The extra Gibbs move uses the resulting z
        # and the SAME PG variables. Next sweep refreshes PG from the new path.
        super()._update_lam_sigma2(psi, it, adapt)
        if self.coefficient_interweave:
            self._coefficient_path_move()

    def run(self, *args, **kwargs):
        result = super().run(*args, **kwargs)
        self.sampling_info.update(coefficient_interweave=self.coefficient_interweave,
                                  coefficient_moves=self.coefficient_moves)
        return result
