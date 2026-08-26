"""Confidence scoring and typology flags.

The confidence score is deliberately a simple, transparent weighted formula -
an investigator has to be able to explain in court why a number came out the
way it did, so every adjustment is returned alongside the score as a factor
with its own explanation. No black box.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import networkx as nx

from .labels import get_label_store
from .trace import (
    TERMINAL_DEAD_END,
    TERMINAL_DEPTH_LIMIT,
    TERMINAL_EXCHANGE,
    TERMINAL_MIXER,
    TracePath,
    TraceResult,
)

BASE_SCORE = 50.0
MIN_SCORE, MAX_SCORE = 5.0, 99.0

# Severity levels for typology flags.
INFO, WARN, HIGH = "info", "warn", "high"


def _parse_ts(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _factor(name: str, delta: float, detail: str) -> dict[str, Any]:
    return {"factor": name, "delta": round(delta, 1), "detail": detail}


def score_path(path: TracePath) -> tuple[float, list[dict[str, Any]]]:
    """Score one traced path 5-99 and return the factors behind the number."""
    factors = [_factor("Base score", BASE_SCORE, "Starting point for any completed trace")]
    score = BASE_SCORE

    # 1. Did we land on something we can actually name?
    if path.terminal_kind == TERMINAL_EXCHANGE:
        delta = 25.0
        factors.append(
            _factor("Terminal is a labeled exchange", delta, f"Attributed to {path.entity}")
        )
    elif path.terminal_kind == TERMINAL_MIXER:
        delta = -40.0
        factors.append(
            _factor("Terminal is a mixer", delta, f"Trail broken at {path.entity or 'mixer'}")
        )
    elif path.terminal_kind == TERMINAL_DEPTH_LIMIT:
        delta = -15.0
        factors.append(
            _factor("Hop limit reached", delta, "Funds were still moving when the trace stopped")
        )
    else:  # dead end
        delta = -10.0
        factors.append(
            _factor("Dead end", delta, "Funds have not moved on from the final address")
        )
    score += delta

    # 2. Fewer hops = a tighter, more defensible link.
    hop_delta = max(-15.0, 20.0 - 6.0 * path.hops)
    factors.append(
        _factor("Hop distance", hop_delta, f"{path.hops} hop(s) from the suspect wallet")
    )
    score += hop_delta

    # 3. Obfuscation services on the way through.
    intermediate_mixers = [m for m in path.mixers if m != path.terminal]
    if intermediate_mixers:
        mixer_delta = max(-45.0, -30.0 * len(intermediate_mixers))
        factors.append(
            _factor(
                "Passed through a mixer",
                mixer_delta,
                f"{len(intermediate_mixers)} mixing service(s) on this path",
            )
        )
        score += mixer_delta

    if path.bridges:
        bridge_delta = max(-24.0, -12.0 * len(path.bridges))
        factors.append(
            _factor(
                "Crossed a bridge",
                bridge_delta,
                f"{len(path.bridges)} cross-chain bridge hop(s); off-chain correlation needed",
            )
        )
        score += bridge_delta

    # 4. Value continuity - if roughly the same amount arrives at the far end,
    #    it is much more likely to be the same money.
    first, last = path.amount_first_hop, path.amount_final_hop
    if first > 0:
        retained = last / first
        if retained >= 0.9:
            cont_delta = 10.0
        elif retained >= 0.5:
            cont_delta = 5.0
        else:
            cont_delta = 0.0
        factors.append(
            _factor(
                "Value continuity",
                cont_delta,
                f"{retained:.0%} of the first-hop amount arrived at the terminal",
            )
        )
        score += cont_delta

    score = max(MIN_SCORE, min(MAX_SCORE, score))
    return round(score, 1), factors


def flag_path(path: TracePath, graph: nx.DiGraph | None = None) -> list[dict[str, Any]]:
    """Typology flags for one path: mixers, bridges, peel chains, speed."""
    labels = get_label_store()
    flags: list[dict[str, Any]] = []

    for mixer in path.mixers:
        broken = mixer == path.terminal
        flags.append(
            {
                "code": "mixer_hop",
                "severity": HIGH,
                "title": "Trail broken here" if broken else "Mixer in the path",
                "address": mixer,
                "detail": (
                    f"{labels.lookup(mixer).entity if labels.lookup(mixer) else 'Mixer'}"
                    + (" - funds enter a mixing pool and cannot be followed further"
                       if broken else " - output attribution is probabilistic beyond this point")
                ),
            }
        )

    for bridge in path.bridges:
        flags.append(
            {
                "code": "bridge_hop",
                "severity": WARN,
                "title": "Cross-chain bridge",
                "address": bridge,
                "detail": (
                    f"{labels.lookup(bridge).entity if labels.lookup(bridge) else 'Bridge'}"
                    " - funds move to another chain; continuation needs the destination chain"
                ),
            }
        )

    # Rapid movement: hops less than an hour apart look automated, not human.
    for previous, current in zip(path.edges, path.edges[1:]):
        t1, t2 = _parse_ts(previous.timestamp), _parse_ts(current.timestamp)
        if t1 and t2:
            minutes = (t2 - t1).total_seconds() / 60
            if 0 <= minutes < 60:
                flags.append(
                    {
                        "code": "rapid_movement",
                        "severity": WARN,
                        "title": "Rapid hop",
                        "address": current.src,
                        "detail": f"Funds moved on after {minutes:.0f} minutes - likely automated",
                    }
                )

    # Peel chain / fan-out, read off the graph around each node on the path.
    if graph is not None:
        for node in path.nodes:
            if node not in graph:
                continue
            out_edges = list(graph.out_edges(node, data=True))
            if len(out_edges) < 2:
                continue
            total = sum(float(d.get("amount", 0.0)) for _, _, d in out_edges) or 1.0
            smallest = min(float(d.get("amount", 0.0)) for _, _, d in out_edges)
            if smallest / total <= 0.2:
                flags.append(
                    {
                        "code": "peel_chain",
                        "severity": WARN,
                        "title": "Possible peel chain",
                        "address": node,
                        "detail": (
                            f"One output takes {smallest / total:.0%} of the value while the rest"
                            " moves on together - classic peeling pattern"
                        ),
                    }
                )
            if len(out_edges) >= 3:
                flags.append(
                    {
                        "code": "fan_out",
                        "severity": INFO,
                        "title": "Fan-out",
                        "address": node,
                        "detail": f"Value split across {len(out_edges)} addresses in one step",
                    }
                )

    # De-duplicate (code, address) while keeping order.
    seen: set[tuple[str, str]] = set()
    unique: list[dict[str, Any]] = []
    for flag in flags:
        key = (flag["code"], flag.get("address", ""))
        if key not in seen:
            seen.add(key)
            unique.append(flag)
    return unique


def apply_scoring(result: TraceResult) -> TraceResult:
    """Score and flag every path in a trace result, in place."""
    for path in result.paths:
        path.confidence, path.confidence_factors = score_path(path)
        path.flags = flag_path(path, result.graph)

    # Keep the ranking honest: same terminal kind, higher confidence first.
    from .trace import _KIND_RANK  # local import to avoid a circular import

    result.paths.sort(
        key=lambda p: (_KIND_RANK.get(p.terminal_kind, 9), -p.confidence, p.hops)
    )
    return result


def confidence_band(score: float) -> str:
    """Plain-English band for a score, for the summary strip."""
    if score >= 75:
        return "High"
    if score >= 50:
        return "Medium"
    if score >= 30:
        return "Low"
    return "Very low"
