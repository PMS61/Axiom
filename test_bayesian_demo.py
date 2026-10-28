"""Numerical integration checks for the local demo adapter."""
import unittest
import numpy as np
from bayesian_demo import fit
from dynamic_retention_model import DynamicRetentionModel

HISTORIES = [[1, 1, 1, 0, 0, 0, 0, 0], [0, 0, 1, 1, 1, 1, 1, 1],
             [1, 0, 1, 0, 1, 0, 1, 0], [1, 1, 1, 1, 0, 0, 1, 0],
             [0, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 1, 0, 0, 0, 0]]


class DemoTests(unittest.TestCase):
    def test_reproducible_fit_and_feedback(self):
        original = fit(HISTORIES)
        self.assertEqual(original, fit(HISTORIES))
        self.assertEqual(original['observations'], 48)
        for item in original['items']:
            self.assertTrue(0 <= item['low'] <= item['recall'] <= item['high'] <= 1)
            self.assertTrue(0 <= item['static'] <= 1)
        # Distinguish consistently successful from consistently failed recall.
        self.assertGreater(original['items'][4]['recall'], original['items'][5]['recall'])
        remembered = fit([h + [1] if i == 0 else h for i, h in enumerate(HISTORIES)])
        forgot = fit([h + [0] if i == 0 else h for i, h in enumerate(HISTORIES)])
        self.assertEqual(remembered['observations'], 49)
        self.assertGreater(remembered['items'][0]['recall'], forgot['items'][0]['recall'])

    def test_saved_states_are_paired_with_retained_draws(self):
        model = DynamicRetentionModel([dict(user=0, item=0, r=np.array([1, 0, 1, 1]),
                                           x=np.array([0, 1, 0, 1]), dt=np.ones(4))])
        result = model.run(n_iter=12, burn_in=4, thin=2)
        self.assertEqual(result['last_psi'].shape, (4, 1))
        np.testing.assert_allclose(result['last_psi'].mean(axis=0)[0], model.psi[0][-1])

    def test_invalid_histories(self):
        for histories in [[], [[1, 0, 2, 0]] * 6, [[1, 0]] * 6, [[True] * 4] * 6]:
            with self.assertRaises(ValueError):
                fit(histories)


if __name__ == '__main__':
    unittest.main()
