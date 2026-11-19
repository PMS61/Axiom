"""Fit the project's static and dynamic Bayesian models to six synthetic card histories."""

from pathlib import Path
import sys

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from dynamic_retention_model import DynamicRetentionModel
from presentation.experiments.support import table
from static_retention_model import StaticRetentionModel


def fit(histories):
    if (len(histories) != 6 or any(not 4 <= len(h) <= 40 for h in histories)
            or any(type(r) is not int or r not in (0, 1) for h in histories for r in h)):
        raise ValueError('Expected six binary review histories of length 4–40.')
    sequences = [dict(user=0, item=i, r=np.array(h),
                      x=np.array([0] + h[:-1]), dt=np.ones(len(h)))
                 for i, h in enumerate(histories)]
    model = DynamicRetentionModel(sequences, rng=np.random.default_rng(42))
    samples = model.run(n_iter=480, burn_in=160, thin=2)
    static = StaticRetentionModel(sequences, rng=np.random.default_rng(42))
    static.run(n_iter=480, burn_in=160, thin=2, verbose=False)
    rng = np.random.default_rng(2026)
    items = []
    for i, h in enumerate(histories):
        # Integrate the next transition over paired state/parameter draws and
        # process noise. These are predictive probability intervals, not CIs
        # for an observed binary answer or a measured scheduling benefit.
        draws = (samples['phi'] * samples['last_psi'][:, i]
                 + samples['beta'][:, i] + samples['gamma'][:, 0] * h[-1])
        forecast = 1 / (1 + np.exp(-np.clip(
            draws[:, None] + rng.normal(size=(len(draws), 30))
            * np.sqrt(samples['sigma_w2'][:, None]), -30, 30)))
        low, high = np.quantile(forecast, [0.05, 0.95])
        items.append(dict(id=i, recall=float(forecast.mean()), low=float(low),
                          high=float(high), static=float(static.predict_proba(0, i))))
    return dict(items=items, draws=len(samples['phi']), seed=42,
                phi=float(samples['phi'].mean()),
                observations=sum(map(len, histories)))


def main():
    histories = [
        [0, 1, 0, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 1, 1],
        [1, 1, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 1],
    ]
    print("Bayesian recall · live fit on six synthetic cards\n", flush=True)
    result = fit(histories)
    print(f"{result['observations']} reviews · {result['draws']} retained posterior draws")
    print("Next-review recall at a one-day gap")
    table(
        ("Card", "Past recalls (0/1)", "Static", "Dynamic", "90% probability interval"),
        (
            (item["id"] + 1, " ".join(map(str, histories[item["id"]])),
             f"{item['static']:.1%}", f"{item['recall']:.1%}",
             f"{item['low']:.1%} to {item['high']:.1%}")
            for item in result["items"]
        ),
    )


if __name__ == "__main__":
    main()
