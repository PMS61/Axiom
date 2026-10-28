"""Run with: python -m unittest model_diagnostics.test_diagnostics -v."""
import unittest
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
from .analysis import diagnostics, metrics, scalar_draws
from .data import simulate, split_suffix, validate_sequences
from .model import DiagnosticModel
from .run import validate_config


class DiagnosticsTests(unittest.TestCase):
    def test_rank_diagnostics_detect_chain_disagreement_and_autocorrelation(self):
        rng = np.random.default_rng(7)
        iid = rng.normal(size=(4, 2000))
        shifted = iid + np.arange(4)[:, None] * 2
        ar = np.zeros_like(iid)
        for t in range(1, 2000): ar[:, t] = .98 * ar[:, t-1] + iid[:, t]
        rows = diagnostics(dict(iid=iid, shifted=shifted, ar=ar), 1)
        self.assertTrue(rows[0]['passed'])
        self.assertGreater(rows[1]['rhat'], 1.1)
        self.assertFalse(rows[1]['passed'])
        self.assertLess(rows[2]['ess_bulk'], rows[0]['ess_bulk'] / 10)

    def test_generated_data_are_reproducible_and_lags_are_chronological(self):
        seqs, truth, states = simulate(seed=42, users=2, items=2, events=12)
        other, _, _ = simulate(seed=42, users=2, items=2, events=12)
        train, test = split_suffix(seqs)
        for s, o, tr, te in zip(seqs, other, train, test):
            np.testing.assert_array_equal(s['r'], o['r'])
            np.testing.assert_array_equal(s['x'][1:], s['r'][:-1])
            self.assertEqual(te['x'][0], tr['r'][-1])
            self.assertEqual(len(tr['r']) + len(te['r']), len(s['r']))
        self.assertAlmostEqual(float(truth['alpha'].mean()), 0)
        self.assertEqual(len(states), 4)

    def test_validation_rejects_invalid_data_and_settings(self):
        for setting in [{'warmup': 2000}, {'variants': ['no_practice']}, {'chains': 1}, {'source': 'upload'}, {'seed': True}]:
            with self.assertRaises(ValueError): validate_config(setting)
        raw = [dict(user=100000, item=i, r=[1, 0] * 4, dt=[1] * 8) for i in (7, 9)]
        seqs, mapping = validate_sequences(raw)
        self.assertEqual([s['item'] for s in seqs], [0, 1])
        self.assertEqual(seqs[0]['user'], 0)
        self.assertEqual(mapping['users'][100000], 0)
        saved = validate_config(dict(source='upload', sequences=raw))
        self.assertEqual(validate_config(saved)['scenarios'], ['uploaded'])
        for broken in [[raw[0]] * 2, [{**s, 'dt': [float('nan')] * 8} for s in raw],
                       [{**s, 'r': [True] * 8} for s in raw]]:
            with self.assertRaises(ValueError): validate_sequences(broken)

    def test_existing_run_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            (out / 'config.json').write_text('preserve original settings')
            (out / 'input.json').write_text('{}')
            result = subprocess.run([sys.executable, '-m', 'model_diagnostics.run',
                                     '--config', str(out / 'input.json'), '--out', folder],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertEqual((out / 'config.json').read_text(), 'preserve original settings')
            self.assertFalse((out / 'status.json').exists())

    def test_ablation_constraints_and_prediction_has_no_future_leakage(self):
        seqs, _, _ = simulate(seed=3, users=2, items=2, events=12)
        train, test = split_suffix(seqs)
        for variant in ('full', 'no_practice', 'no_hierarchy', 'no_dynamics'):
            model = DiagnosticModel(train, variant=variant, seed=11)
            samples = model.run(n_iter=100, burn_in=50)
            altered = [{**s, 'r': 1-s['r'], 'x': np.concatenate([[s['x'][0]], 1-s['r'][:-1]])} for s in test]
            before, after = model.predict(test), model.predict(altered)
            for p, q in zip(before, after):
                self.assertAlmostEqual(p[0], q[0])
                self.assertTrue(np.all((p >= 0) & (p <= 1)))
            if variant == 'no_practice': np.testing.assert_array_equal(samples['kappa'], 0)
            if variant == 'no_hierarchy':
                np.testing.assert_array_equal(samples['alpha'], 0)
                np.testing.assert_array_equal(samples['beta'], 0)
                self.assertNotIn('alpha[0]', scalar_draws({k: np.stack([v] * 4) for k, v in samples.items()}, variant))
            if variant == 'no_dynamics': self.assertNotIn('lam', samples)

    def test_blocked_gaussian_matches_analytic_conditional(self):
        seqs, _, _ = simulate(seed=31, users=2, items=2, events=8)
        for variant in ('full', 'no_hierarchy', 'no_dynamics'):
            model = DiagnosticModel(seqs, variant=variant, seed=21, sampler='blocked')
            phi, q = model._dyn()
            precision, info = model._effect_system(model.psi, phi, q)
            covariance = np.linalg.inv(precision)
            expected = covariance @ info
            # Hold hypervariances fixed to test the Gaussian draw itself.
            scales = model.tau2.copy()
            draws = []
            for _ in range(6000):
                model.tau2 = scales.copy()
                model._blocked_effects(model.psi, phi, q)
                draws.append(np.r_[model.b0, model.alpha, model.beta, model.gamma, model.g0]
                             if model.hierarchy else [model.b0, model.g0])
            draws = np.asarray(draws)
            np.testing.assert_array_less(np.abs(draws.mean(0)-expected), 5*np.sqrt(np.diag(covariance)/len(draws)))
            np.testing.assert_allclose(np.cov(draws.T), covariance, atol=.06, rtol=.08)
            # Independently assemble the joint prior/likelihood Hessian by finite differences.
            def log_density(t):
                b, a, beta, gamma, g = t[0], t[1:3], t[3:5], t[5:7], t[-1]
                if not model.hierarchy:
                    b, g = t
                    a, beta, gamma = np.zeros(2), np.zeros(2), np.full(2, g)
                mu = b + a[model.U, None] + beta[model.I, None] + gamma[model.U, None]*model.X
                previous = np.zeros_like(model.psi)
                previous[:, 1:] = model.psi[:, :-1]
                residual = model.psi - phi*previous - (1-phi)*mu
                lp = -.5*np.sum((residual**2/q)[model.M]) - (b*b+g*g)/200
                if model.hierarchy:
                    lp -= .5*(np.sum(a*a)/scales['alpha'] + np.sum(beta*beta)/scales['beta'] + np.sum((gamma-g)**2)/scales['gamma'])
                return lp
            theta = np.linspace(-.3, .4, len(info))
            eps = 1e-4
            numeric = np.empty_like(precision)
            for i in range(len(info)):
                for j in range(len(info)):
                    ei, ej = np.eye(len(info))[i]*eps, np.eye(len(info))[j]*eps
                    numeric[i,j] = -(log_density(theta+ei+ej)-log_density(theta+ei-ej)-log_density(theta-ei+ej)+log_density(theta-ei-ej))/(4*eps**2)
            np.testing.assert_allclose(numeric, precision, atol=2e-5)
            gradient = np.array([(log_density(theta+e*eps)-log_density(theta-e*eps))/(2*eps)
                                 for e in np.eye(len(info))])
            np.testing.assert_allclose(gradient, info-precision@theta, atol=1e-7)

    def test_joint_target_matches_integrated_transition_density_and_freezes(self):
        from scipy.special import gammaln
        seqs, _, _ = simulate(seed=5, users=2, items=2, events=12)
        model = DiagnosticModel(seqs, seed=12, sampler='joint_blocked')
        model.psi = model.rng.normal(size=model.psi.shape)
        values = []
        for lam, kappa in ((.1, -.3), (.5, .8)):
            model.lam, model.kappa = lam, kappa
            phi, q = model._dyn()
            c = (q/model.sigma2)[model.TR]
            previous = np.zeros_like(model.psi)
            previous[:, 1:] = model.psi[:, :-1]
            e = (model.psi - phi*previous - (1-phi)*model._mu())[model.TR]
            a, b = model.sigma2_prior
            n = len(e)
            lp = (gammaln(a+n/2)-gammaln(a)+a*np.log(b)-(a+n/2)*np.log(b+.5*np.sum(e*e/c))
                  -.5*np.log(c).sum()-n/2*np.log(2*np.pi)
                  -.5*((np.log(lam)-model.loglam_prior[0])/model.loglam_prior[1])**2-.5*kappa**2)
            target, _ = model._decay_log_density([np.log(lam), kappa], model.psi)
            values.append(target-lp)
        self.assertAlmostEqual(*values)
        model.run(n_iter=110, burn_in=60)
        self.assertEqual(model.joint_count, 60)
        cov, scale = model.joint_cov.copy(), model.joint_scale
        for it in range(20): model._update_lam_sigma2(model.psi, 110+it, adapt=False)
        np.testing.assert_array_equal(cov, model.joint_cov)
        self.assertEqual(scale, model.joint_scale)
        self.assertTrue(np.isfinite(model.sigma2) and model.sigma2 > 0)
        for bad in ('bad', None):
            with self.assertRaises(ValueError): validate_config({'sampler': bad})

    def test_new_kernels_preserve_ablation_constraints_and_prediction_order(self):
        seqs, _, _ = simulate(seed=3, users=2, items=2, events=12)
        train, test = split_suffix(seqs)
        for sampler in ('joint', 'blocked', 'joint_blocked'):
            for variant in ('full', 'no_practice', 'no_hierarchy', 'no_dynamics'):
                model = DiagnosticModel(train, variant=variant, sampler=sampler, seed=11)
                samples = model.run(n_iter=80, burn_in=50)
                self.assertTrue(all(np.isfinite(v).all() for v in samples.values()))
                changed = [{**s, 'r': 1-s['r']} for s in test]
                for p, q in zip(model.predict(test), model.predict(changed)):
                    self.assertAlmostEqual(p[0], q[0])
                if variant == 'no_practice': np.testing.assert_array_equal(samples['kappa'], 0)
                if variant == 'no_hierarchy':
                    np.testing.assert_array_equal(samples['alpha'], 0)
                    np.testing.assert_array_equal(samples['beta'], 0)
                if variant == 'no_dynamics': self.assertNotIn('lam', samples)

    def test_comparison_requires_matched_data_and_preserves_output(self):
        import json
        from .compare import compare
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = [root/'reference', root/'blocked']
            for path in paths:
                path.mkdir()
                config = validate_config({'sampler': path.name})
                result = dict(scenario='full', replicate=0, variant='full', min_ess_bulk=500,
                              min_ess_tail=600, fit_cpu_seconds=10, log_loss=.6, brier=.2,
                              passed=True, max_rhat=1.001)
                diagnostic = dict(scenario='full', replicate=0, variant='full', parameter='b0',
                                  mean=0, mcse_mean=.01, ess_per_cpu_second=50)
                (path/'summary.json').write_text(json.dumps(dict(config=config, results=[result],
                                                                diagnostics=[diagnostic], recovery=[])))
                (path/'full-0-dataset.json').write_text('same data')
            compare(paths, root/'output')
            self.assertTrue((root/'output'/'parameter-comparison.csv').exists())
            with self.assertRaises(ValueError): compare(paths, root/'output')
            (paths[1]/'full-0-dataset.json').write_text('different data')
            with self.assertRaises(ValueError): compare(paths, root/'mismatched')
            self.assertFalse((root/'mismatched').exists())

    def test_anki_cohorts_preserve_partition_timing_and_nested_ids(self):
        from .anki import select_cohort, ANKI_BOUNDS
        train, test = [], []
        for user in range(6):
            for item in range(3):
                for event in range(8):
                    outcome = event % 2
                    row = dict(user_id=user, item_id=item, timestamp=2*event,
                               elapsed_seconds=2*86400, y=outcome, rating=3 if outcome else 1)
                    (train if event < 6 else test).append(row)
        a, b, full, meta = select_cohort(train, test, 81, 0, 2, 2, 60)
        _, _, _, other = select_cohort(train, test, 81, 1, 2, 2, 60)
        self.assertFalse(set(r['user'] for r in meta['mapping']) & set(r['user'] for r in other['mapping']))
        self.assertEqual(len({s['item'] for s in full}), 4)
        for tr, te, seq in zip(a, b, full):
            self.assertEqual((len(tr['r']), len(te['r'])), (6, 2))
            self.assertEqual(te['x'][0], tr['r'][-1])
            np.testing.assert_array_equal(seq['dt'][1:], 2)
        # Outcome changes do not change cohort/card selection.
        changed = [{**r, 'y': 1-r['y'], 'rating': 1 if r['y'] else 3} for r in test]
        _, _, _, altered = select_cohort(train, changed, 81, 0, 2, 2, 60)
        self.assertEqual(meta['mapping'], altered['mapping'])
        timed = [{**s, 'dt': s['dt'].copy()} for s in a]
        timed[0]['dt'][1:3] = [1/86400, 1000.]
        model = DiagnosticModel(timed, dt_bounds=ANKI_BOUNDS)
        self.assertEqual(model.dt_bounds, ANKI_BOUNDS)
        self.assertEqual(model.DT[0, 1], 1/86400)
        self.assertEqual(model.DT[0, 2], 1000.)
        self.assertEqual(validate_config({'source':'anki'})['scenarios'], ['anki'])

    def test_anki_bad_partitions_and_missing_history_are_excluded(self):
        from .anki import select_cohort
        train, test = [], []
        for user in range(2):
            for item in range(4):
                for event in range(8):
                    row = dict(user_id=user, item_id=item, timestamp=2*event,
                               elapsed_seconds=2*86400, y=1, rating=3)
                    if item == 1 and event == 6: row['timestamp'] = 10  # tied partition boundary
                    if item == 2 and event == 3: row['elapsed_seconds'] = 0
                    if item == 3 and event == 3: row['elapsed_seconds'] = 10*86400
                    (train if event < 6 else test).append(row)
        _, _, _, metadata = select_cohort(train, test, 9, 0, 2, 1, 60)
        self.assertEqual(metadata['exclusions']['partition_overlap'], 2)
        self.assertEqual(metadata['exclusions']['invalid_elapsed'], 2)
        self.assertEqual(metadata['exclusions']['inconsistent_time'], 2)
        with self.assertRaises(ValueError): select_cohort(train, test, 9, 0, 2, 2, 60)

    def test_metrics_ties_single_class_and_perfect_predictions(self):
        score = metrics(np.array([0, 1]), np.array([.5, .5]))
        self.assertEqual(score['auc'], .5)
        self.assertAlmostEqual(score['log_loss'], np.log(2))
        self.assertIsNone(metrics(np.array([1, 1]), np.array([.9, .8]))['auc'])
        self.assertLess(metrics(np.array([0, 1]), np.array([0, 1]))['brier'], 1e-15)


if __name__ == '__main__': unittest.main()
