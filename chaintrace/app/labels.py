"""The labeled-address database: address -> {entity, type}.

Backed by data/labeled_addresses.json for the POC. In production this is a
table; the lookup interface below is what the rest of the app depends on, so
the storage can change without touching the trace engine.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from .config import LABELED_ADDRESSES_FILE
from .models import Label

EXCHANGE = "exchange"
MIXER = "mixer"
BRIDGE = "bridge"


class LabelStore:
    """Case-insensitive lookup over the labeled-address JSON file."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path or LABELED_ADDRESSES_FILE)
        self._by_lower: dict[str, Label] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        raw = json.loads(self.path.read_text())
        for address, meta in raw.items():
            if address.startswith("_") or not isinstance(meta, dict):
                continue  # skip the "_comment" key
            label = Label(
                address=address,
                entity=str(meta.get("entity", "Unknown entity")),
                type=str(meta.get("type", "unknown")).lower(),
            )
            self._by_lower[address.lower()] = label

    def lookup(self, address: str) -> Label | None:
        return self._by_lower.get((address or "").strip().lower())

    def type_of(self, address: str) -> str | None:
        label = self.lookup(address)
        return label.type if label else None

    def is_exchange(self, address: str) -> bool:
        return self.type_of(address) == EXCHANGE

    def is_mixer(self, address: str) -> bool:
        return self.type_of(address) == MIXER

    def is_bridge(self, address: str) -> bool:
        return self.type_of(address) == BRIDGE

    def all_labels(self) -> list[Label]:
        return list(self._by_lower.values())

    def __len__(self) -> int:
        return len(self._by_lower)


@lru_cache(maxsize=1)
def get_label_store() -> LabelStore:
    """Shared, lazily-loaded label store."""
    return LabelStore()
