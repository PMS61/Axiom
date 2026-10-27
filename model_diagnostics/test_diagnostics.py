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

    def test_metrics_ties_single_class_and_perfect_predictions(self):
        score = metrics(np.array([0, 1]), np.array([.5, .5]))
        self.assertEqual(score['auc'], .5)
        self.assertAlmostEqual(score['log_loss'], np.log(2))
        self.assertIsNone(metrics(np.array([1, 1]), np.array([.9, .8]))['auc'])
        self.assertLess(metrics(np.array([0, 1]), np.array([0, 1]))['brier'], 1e-15)


if __name__ == '__main__': unittest.main()
