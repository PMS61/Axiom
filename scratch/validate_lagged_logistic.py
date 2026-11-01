"""
Independent validation of baselines.LaggedLogistic.

Tests only. Does not modify project implementation.
"""

import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import baselines
from baselines import (
    LaggedLogistic,
    build_history,
    count_features,
    lagged_outcome,
)

def check(name, condition):
    if not condition:
        raise AssertionError(f"FAILED: {name}")
    print(f"PASS: {name}")


def make_sequence(user, item, r, dt):
    r = np.asarray(r, dtype=int)
    dt = np.asarray(dt, dtype=float)
    x = np.zeros(len(r), dtype=float)
    if len(r) > 1:
        x[1:] = r[:-1]
    return {
        "user": user,
        "item": item,
        "r": r,
        "dt": dt,
        "x": x,
    }

def test_optimizer_convergence():
    train = make_training_data()

    original_minimize = baselines.minimize
    captured = {}

    def capture_minimize(*args, **kwargs):
        result = original_minimize(*args, **kwargs)
        captured["result"] = result
        return result

    baselines.minimize = capture_minimize

    try:
        model = LaggedLogistic(use_dt=True)
        model.fit(train)
    finally:
        baselines.minimize = original_minimize

    result = captured["result"]

    print("\nOptimizer diagnostics:")
    print("Success:", result.success)
    print("Status:", result.status)
    print("Message:", result.message)
    print("Iterations:", result.nit)
    print("Final objective:", result.fun)
    print("Gradient infinity norm:",
          np.linalg.norm(result.jac, ord=np.inf))

    check("L-BFGS-B reports successful convergence", result.success)
    check("optimizer objective is finite", np.isfinite(result.fun))
    check("optimizer gradient is finite",
          np.all(np.isfinite(result.jac)))

def test_history_features():
    train = [
        make_sequence(0, 0, [1, 0, 1], [0, 1, 2]),
        make_sequence(0, 1, [0, 0], [0, 1]),
    ]

    hist_map = build_history(train)

    seq = make_sequence(0, 0, [0, 1], [4, 5])

    # Extract history for this specific user-item pair.
    hist = hist_map[(seq["user"], seq["item"])]

    n, right, wrong = count_features(seq, hist)

    check("training and cumulative review counts",
      np.array_equal(n, [3, 4]))

    check("training and cumulative success counts",
        np.array_equal(right, [2, 2]))

    check("training and cumulative failure counts",
        np.array_equal(wrong, [1, 2]))

    lag = lagged_outcome(seq, hist)

    check("first test lag uses last training outcome", lag[0] == 1)
    check("second test lag uses preceding test outcome", lag[1] == 0)

def test_design_features():
    model = LaggedLogistic(use_dt=True, use_item=False, use_user=False)

    seq = make_sequence(0, 0, [1, 0], [0, 3])
    X, _ = model._design([seq], None)

    expected = np.array([
        [1.0, 0.0, 0.0, 0.0, np.log1p(0.0)],
        [1.0, 1.0, 1.0, 0.0, np.log1p(3.0)],
    ])

    check("design matrix shape", X.shape == expected.shape)
    check("intercept, lag, count and dt features",
          np.allclose(X.toarray(), expected))


def make_training_data():
    rng = np.random.default_rng(42)
    sequences = []

    for user in range(5):
        for item in range(4):
            n = 35
            dt = rng.uniform(0, 30, n)
            r = np.zeros(n, dtype=int)
            x = np.zeros(n, dtype=float)

            for t in range(n):
                if t:
                    x[t] = r[t - 1]

                logit = (
                    0.4
                    + 0.9 * x[t]
                    + 0.15 * np.sqrt(max(0, r[:t].sum()))
                    - 0.12 * np.sqrt(max(0, t - r[:t].sum()))
                    - 0.25 * np.log1p(dt[t])
                    + 0.05 * user
                    - 0.04 * item
                )
                p = 1.0 / (1.0 + np.exp(-logit))
                r[t] = rng.binomial(1, p)

            sequences.append({
                "user": user,
                "item": item,
                "r": r,
                "dt": dt,
                "x": x,
            })

    return sequences


def test_fit_and_predictions():
    train = make_training_data()

    model = LaggedLogistic(use_dt=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(train)

    check("fitted coefficients exist", model.w is not None)
    check("coefficients are finite", np.all(np.isfinite(model.w)))

    predictions = model.predict(train)

    check("one prediction array per sequence",
          len(predictions) == len(train))

    for i, (p, seq) in enumerate(zip(predictions, train)):
        check(f"prediction length for sequence {i}",
              len(p) == len(seq["r"]))
        check(f"finite predictions for sequence {i}",
              np.all(np.isfinite(p)))
        check(f"probability range for sequence {i}",
              np.all((p >= 0) & (p <= 1)))

    print(f"Optimizer-related warnings captured: {len(caught)}")


def test_prediction_no_future_leakage():
    train = make_training_data()
    model = LaggedLogistic(use_dt=True)
    model.fit(train)

    test_a = [
        make_sequence(0, 0, [1, 0, 1, 0], [2, 3, 4, 5])
    ]
    test_b = [
        make_sequence(0, 0, [1, 0, 0, 1], [2, 3, 4, 5])
    ]

    p_a = model.predict(test_a)[0]
    p_b = model.predict(test_b)[0]

    # Prediction at index t may depend on outcomes strictly before t,
    # but not on the outcome at t or any later outcome.
    check("first prediction independent of all test outcomes",
          np.isclose(p_a[0], p_b[0]))

    check("second prediction independent of second/future outcomes",
          np.isclose(p_a[1], p_b[1]))

    check("third prediction independent of third/future outcomes",
          np.isclose(p_a[2], p_b[2]))


def test_reproducibility():
    train = make_training_data()

    m1 = LaggedLogistic(use_dt=True)
    m2 = LaggedLogistic(use_dt=True)

    m1.fit(train)
    m2.fit(train)

    p1 = m1.predict(train)
    p2 = m2.predict(train)

    check("identical fits produce identical coefficients",
          np.allclose(m1.w, m2.w, atol=1e-7))

    check("identical fits produce identical predictions",
          all(np.allclose(a, b, atol=1e-7)
              for a, b in zip(p1, p2)))


if __name__ == "__main__":
    print("\n=== LaggedLogistic independent validation ===\n")

    test_history_features()
    test_design_features()
    test_fit_and_predictions()
    test_prediction_no_future_leakage()
    test_reproducibility()
    test_optimizer_convergence()

    print("\n=== ALL VALIDATION CHECKS PASSED ===")
