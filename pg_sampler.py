"""
pg_sampler.py

Polya-Gamma random variate sampler, PG(1, c), used by both
static_retention_model.py and dynamic_retention_model.py to make the
Bernoulli observation model conditionally Gaussian (Polson, Scott &
Windle, 2013).

IMPLEMENTATION NOTE (read this before trusting results on real data):
This uses the truncated sum-of-Gammas representation of the Polya-Gamma
distribution,

    PG(b, c) =_d (1 / (2*pi^2)) * sum_{k=1}^inf  g_k / ((k - 1/2)^2 + c^2/(4*pi^2))

where g_k ~iid Gamma(b, 1). For b=1 this series converges quickly (terms
decay like 1/k^2), so truncating at a few hundred terms gives a close
approximation, verified below against the known closed-form mean of
PG(1,c).

This is NOT the exact Devroye (2009) rejection sampler used in the
Polson-Scott-Windle reference implementation. That algorithm is exact but
involves an alternating-series acceptance test that is easy to get subtly
wrong without extensive unit testing, and a silent bug there would
quietly bias every downstream Gibbs update. The truncated-sum sampler
below is simpler to verify (see the self-test at the bottom of this file)
and is accurate enough for prototyping and for a first-draft thesis
result. If exactness matters for a final submitted result, swap this
function's body for a call into the `polyagamma` package
(`pip install polyagamma`), which wraps Windle's reference C
implementation and has the same (c, rng) -> omega signature shape.
"""
import numpy as np


def _pg1_mean(c):
    """Closed-form E[PG(1,c)] = tanh(c/2) / (2c), with c->0 limit = 1/4.
    Used only for the self-test below."""
    c = np.asarray(c, dtype=float)
    out = np.full_like(c, 0.25)
    nonzero = np.abs(c) > 1e-8
    out[nonzero] = np.tanh(c[nonzero] / 2.0) / (2.0 * c[nonzero])
    return out


def sample_pg1(c, rng, n_terms=200):
    """
    Draw one PG(1, c_i) random variable for each entry of c.

    Parameters
    ----------
    c : array-like, shape (n,)
        Tilting parameter for each draw (in this codebase: the current
        value of the latent logit-scale state psi_t at each review event).
    rng : np.random.Generator
    n_terms : int
        Number of terms kept in the truncated series.

    Returns
    -------
    np.ndarray, shape (n,)
    """
    c = np.atleast_1d(np.asarray(c, dtype=float))
    k = np.arange(1, n_terms + 1)
    denom = (k - 0.5) ** 2 + (c[:, None] ** 2) / (4 * np.pi ** 2)  # (n, n_terms)
    gammas = rng.gamma(shape=1.0, scale=1.0, size=(len(c), n_terms))
    omega = (1.0 / (2 * np.pi ** 2)) * np.sum(gammas / denom, axis=1)
    return omega


if __name__ == "__main__":
    # Self-test: empirical mean of many draws should match the closed-form
    # PG(1,c) mean for a spread of c values, including c=0.
    rng = np.random.default_rng(0)
    print("--- Polya-Gamma sampler self-test ---")
    max_diff = 0.0
    for c_val in [0.0, 0.5, 1.0, 2.0, 5.0, -3.0]:
        draws = sample_pg1(np.full(20000, c_val), rng)
        empirical_mean = draws.mean()
        theoretical_mean = _pg1_mean(np.array([c_val]))[0]
        diff = abs(empirical_mean - theoretical_mean)
        max_diff = max(max_diff, diff)
        print(f"c={c_val:6.2f}  empirical={empirical_mean:.4f}  "
              f"theoretical={theoretical_mean:.4f}  diff={diff:.4f}")
    assert max_diff < 0.01, "PG(1,c) sampler mean does not match theory closely enough!"
    print(f"OK: max |empirical - theoretical| = {max_diff:.4f} (< 0.01)")
