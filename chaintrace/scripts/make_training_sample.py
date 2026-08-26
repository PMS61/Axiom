"""Generate data/training_sample.csv - the toy training set for the ML flag.

Run:  python scripts/make_training_sample.py

The POC has no real labeled wallet corpus, so this synthesizes one with the
shape a real one would have: licit wallets make a few payments to a few
addresses, illicit ones fan value out across many addresses in round-number
chunks. Deliberately noisy so the model is not trivially perfect.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

ROWS = 600
SEED = 20260826
OUT = Path(__file__).resolve().parent.parent / "data" / "training_sample.csv"


def licit_row(rng: random.Random) -> dict[str, float | str]:
    num_inputs = rng.randint(1, 3)
    num_outputs = rng.randint(1, 3)
    total_amount = rng.lognormvariate(11.5, 1.1)          # ~100k sats, wide spread
    fan_out_ratio = num_outputs / max(num_inputs, 1) * rng.uniform(0.6, 1.4)
    return {
        "num_inputs": num_inputs,
        "num_outputs": num_outputs,
        "total_amount": round(total_amount, 2),
        "fan_out_ratio": round(fan_out_ratio, 3),
        "label": "licit",
    }


def illicit_row(rng: random.Random) -> dict[str, float | str]:
    num_inputs = rng.randint(1, 6)
    num_outputs = rng.randint(4, 20)                       # fan-out / peeling
    total_amount = rng.lognormvariate(13.5, 1.3)           # larger sums
    fan_out_ratio = num_outputs / max(num_inputs, 1) * rng.uniform(0.8, 1.6)
    return {
        "num_inputs": num_inputs,
        "num_outputs": num_outputs,
        "total_amount": round(total_amount, 2),
        "fan_out_ratio": round(fan_out_ratio, 3),
        "label": "illicit",
    }


def main() -> None:
    rng = random.Random(SEED)
    rows = []
    for index in range(ROWS):
        row = illicit_row(rng) if index % 2 else licit_row(rng)
        # 8% label noise so the classes overlap, like real data.
        if rng.random() < 0.08:
            row["label"] = "licit" if row["label"] == "illicit" else "illicit"
        rows.append(row)
    rng.shuffle(rows)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["num_inputs", "num_outputs", "total_amount", "fan_out_ratio", "label"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
