"""Independent density/conditional checks for coefficient/path interweaving."""
import unittest
import numpy as np
from .coefficient_innovation import CoefficientInnovationModel
from .innovation import InnovationModel
from .data import simulate


class CoefficientInnovationTests(unittest.TestCase):
    def setUp(self):
        seqs, _, _ = simulate(seed=52, users=2, items=2, events=8)
        # Ragged histories, nonzero first lag, extreme valid gaps.
        seqs[0] = {k: v[:5] if isinstance(v, np.ndarray) else v for k, v in seqs[0].items()}
        seqs[0]['x'][0] = 1
        self.seqs = seqs
        self.m = CoefficientInnovationModel(seqs, sampler='joint_blocked', interweave=True, seed=93)
        self.m.DT[1, 1:3] = [1/86400, 2000]

    def test_affine_path_matches_recursion_and_coefficient_derivatives(self):
        m = self.m
        z = m.rng.normal(size=m.psi.shape)
        offset, design = m.affine_path(z)
        theta = m.coefficients()
        np.testing.assert_allclose(offset+design@theta, m.reconstruct(z)[m.M], atol=1e-12)
        self.assertTrue(np.all(design[:, -1] == 0))
        first = np.cumsum(np.r_[0, m.lens[:-1]])
        np.testing.assert_array_equal(design[first, 1+m.n_users+m.n_items+m.U], m.X[:, 0])
        for j in range(len(theta)):
            step = np.eye(len(theta))[j]*1e-5
            m.set_coefficients(theta+step); plus = m.reconstruct(z)[m.M]
            m.set_coefficients(theta-step); minus = m.reconstruct(z)[m.M]
            np.testing.assert_allclose((plus-minus)/2e-5, design[:, j], atol=1e-9)
        m.set_coefficients(theta)

    def test_gaussian_system_matches_independent_centered_transformed_target(self):
        m = self.m
        z = m.rng.normal(size=m.psi.shape)
        omega = m.rng.uniform(.05, .5, size=m.psi.shape)
        precision, info = m.coefficient_system(z, omega)
        u, i = m.n_users, m.n_items
        def density(theta):
            m.set_coefficients(theta)
            psi = m.reconstruct(z)
            b, a, beta, gamma, g = theta[0], theta[1:1+u], theta[1+u:1+u+i], theta[1+u+i:-1], theta[-1]
            prior = -(b*b+g*g)/200 - .5*(a@a/m.tau2['alpha']+beta@beta/m.tau2['beta']+np.sum((gamma-g)**2)/m.tau2['gamma'])
            phi, q = m._dyn()
            previous = np.zeros_like(psi); previous[:, 1:] = psi[:, :-1]
            residual = psi-phi*previous-(1-phi)*m._mu()
            # Centered transition plus triangular path Jacobian, not assumed cancellation.
            transition = -.5*np.sum((np.log(q)+residual**2/q)[m.M])
            jacobian = .5*np.log(q[m.M]).sum()
            pg = np.sum(((m.R-.5)*psi-.5*omega*psi**2)[m.M])
            return prior+transition+jacobian+pg
        theta = np.linspace(-.3, .6, len(info)); eps = 1e-4
        numeric = np.empty_like(precision)
        for j, ej in enumerate(np.eye(len(theta))*eps):
            for k, ek in enumerate(np.eye(len(theta))*eps):
                numeric[j, k] = -(density(theta+ej+ek)-density(theta+ej-ek)-density(theta-ej+ek)+density(theta-ej-ek))/(4*eps**2)
        np.testing.assert_allclose(numeric, precision, atol=2e-5)
        gradient = np.array([(density(theta+e*eps)-density(theta-e*eps))/(2*eps) for e in np.eye(len(theta))])
        np.testing.assert_allclose(gradient, info-precision@theta, atol=1e-7)
        # Independent prior must match the centered implementation, including gamma/g0 coupling.
        centered, _ = m._effect_system(m.psi, np.ones_like(m.psi), np.ones_like(m.psi))
        np.testing.assert_allclose(m.coefficient_prior_precision(), centered)

    def test_draws_match_gaussian_and_preserve_innovations(self):
        m = self.m
        m._omega = np.full_like(m.psi, .2)
        z = m.innovations(m.psi)
        precision, info = m.coefficient_system(z, m._omega)
        covariance = np.linalg.inv(precision); mean = covariance@info
        scales = m.tau2.copy(); draws = []
        for _ in range(6000):
            m.tau2 = scales.copy()
            m._coefficient_path_move()
            draws.append(m.coefficients())
        draws = np.asarray(draws)
        np.testing.assert_array_less(np.abs(draws.mean(0)-mean), 5*np.sqrt(np.diag(covariance)/len(draws)))
        np.testing.assert_allclose(np.cov(draws.T), covariance, atol=.06, rtol=.08)
        np.testing.assert_allclose(m.innovations(m.psi), z, atol=1e-9)

    def test_disabled_move_reproduces_baseline_and_active_metadata(self):
        old = InnovationModel(self.seqs, sampler='joint_blocked', interweave=True, seed=44)
        off = CoefficientInnovationModel(self.seqs, sampler='joint_blocked', interweave=True, seed=44, coefficient_interweave=False)
        a, b = old.run(120, 60), off.run(120, 60)
        for key in a: np.testing.assert_array_equal(a[key], b[key])
        active = self.m; samples = active.run(120, 60)
        self.assertTrue(all(np.isfinite(v).all() for v in samples.values()))
        self.assertEqual(active.sampling_info['coefficient_moves'], 120)
        self.assertEqual(active.innovation_adaptations, 60)
        z = active.innovations(active.psi)
        np.testing.assert_allclose(active.reconstruct(z), active.psi, atol=1e-9)

    def test_order_audit_exhibits_feasible_different_predictors(self):
        from .selected_order_audit import compare_orders
        value = compare_orders([0, 2, 2, 2], np.array([0, 1.8, .04, .03])*86400, [0, 1, 0, 1])
        self.assertTrue(value['both_strictly_feasible'])
        self.assertEqual(value['different_outcome_positions'], 2)
        self.assertGreater(value['different_lag_positions'], 0)
        self.assertGreater(value['different_practice_positions'], 0)
        # Two events on a day have no alternative under this restricted audit.
        same = compare_orders([0, 2, 2], np.array([0, 1.8, .03])*86400, [0, 1, 0])
        self.assertEqual(same['different_row_positions'], 0)

    def test_study_and_report_refuse_existing_outputs(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from .coefficient_study import execute
        from .coefficient_report import summarize
        with TemporaryDirectory() as folder:
            root = Path(folder); source = root/'source'; source.mkdir()
            (source/'status.json').write_text('{"state":"complete"}')
            out = root/'preserve'; out.mkdir(); sentinel = out/'keep'; sentinel.write_text('keep')
            with self.assertRaises(FileExistsError): execute(out, source)
            with self.assertRaises(FileExistsError): summarize(source, out)
            self.assertEqual(sentinel.read_text(), 'keep')
            (source/'status.json').write_text('{"state":"running"}')
            with self.assertRaises(ValueError): summarize(source, root/'new')
            self.assertFalse((root/'new').exists())


if __name__ == '__main__': unittest.main()
