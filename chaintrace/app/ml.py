"""A minimal ML hook: "illicit likelihood" for a wallet.

This is a demonstration of where a model plugs into the pipeline, not a tuned
detector. A RandomForest is trained on the small bundled feature set in
data/training_sample.csv, and the same four features are computed for a traced
wallet so the dashboard can show a score next to the trace.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from .config import TRAINING_SAMPLE_FILE
from .fetchers import fetch
from .models import Tx

FEATURES = ["num_inputs", "num_outputs", "total_amount", "fan_out_ratio"]

# Ethereum amounts are in ETH and Bitcoin amounts in satoshis. The training set
# is expressed in satoshis, so ETH is scaled onto the same axis. Crude, and
# fine for a demo - a real model would use fiat value at transaction time.
ETH_TO_SATS = 1e8


@dataclass
class RiskScore:
    address: str
    likelihood: float                 # 0-1, probability of the "illicit" class
    features: dict[str, float]
    model_accuracy: float
    trained_on: int
    note: str = ""

    @property
    def percent(self) -> float:
        return round(self.likelihood * 100, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "address": self.address,
            "likelihood": self.likelihood,
            "percent": self.percent,
            "features": self.features,
            "model_accuracy": self.model_accuracy,
            "trained_on": self.trained_on,
            "note": self.note,
        }


@lru_cache(maxsize=1)
def get_model() -> tuple[Any, float, int]:
    """Train (once per process) and return (model, holdout accuracy, row count)."""
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    frame = pd.read_csv(TRAINING_SAMPLE_FILE)
    x = frame[FEATURES]
    y = (frame["label"] == "illicit").astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=120, max_depth=6, random_state=42)
    model.fit(x_train, y_train)
    accuracy = float(model.score(x_test, y_test))
    return model, round(accuracy, 3), len(frame)


def wallet_features(address: str, transactions: list[Tx]) -> dict[str, float]:
    """The same four features the model was trained on, for a live wallet."""
    outgoing = [tx for tx in transactions if address in tx.inputs]
    if not outgoing:
        return {name: 0.0 for name in FEATURES}

    scale = ETH_TO_SATS if outgoing[0].chain == "ethereum" else 1.0
    total_amount = sum(tx.total_out for tx in outgoing) * scale
    num_inputs = sum(len(tx.inputs) for tx in outgoing) / len(outgoing)
    num_outputs = sum(len(tx.outputs) for tx in outgoing) / len(outgoing)
    unique_recipients = {out for tx in outgoing for out in tx.outputs if out != address}

    return {
        "num_inputs": round(num_inputs, 3),
        "num_outputs": round(num_outputs, 3),
        "total_amount": round(total_amount, 2),
        "fan_out_ratio": round(len(unique_recipients) / len(outgoing), 3),
    }


def score_wallet(address: str, allow_network: bool = True) -> RiskScore:
    """Illicit likelihood for one wallet, computed from its outgoing activity."""
    import pandas as pd

    result = fetch(address, allow_network=allow_network)
    features = wallet_features(address, result.transactions)
    model, accuracy, rows = get_model()

    if not any(features.values()):
        return RiskScore(
            address=address,
            likelihood=0.0,
            features=features,
            model_accuracy=accuracy,
            trained_on=rows,
            note="No outgoing activity to score.",
        )

    frame = pd.DataFrame([features], columns=FEATURES)
    likelihood = float(model.predict_proba(frame)[0][1])
    return RiskScore(
        address=address,
        likelihood=round(likelihood, 4),
        features=features,
        model_accuracy=accuracy,
        trained_on=rows,
        note="Demonstration model trained on a small synthetic sample.",
    )


def feature_importances() -> dict[str, float]:
    """What the model actually leans on - shown in the UI for transparency."""
    model, _, _ = get_model()
    return {
        name: round(float(weight), 3)
        for name, weight in zip(FEATURES, model.feature_importances_)
    }
