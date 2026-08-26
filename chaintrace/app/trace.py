"""The trace engine: walk the money forward, hop by hop, to a cash-out point.

How it works
------------
1. Start at the suspect wallet. Fetch its outgoing transactions.
2. Every output address becomes a node, every transfer an edge carrying the
   amount and timestamp. That is the NetworkX directed graph.
3. Breadth-first search follows every branch outward until a node is either
   - a labeled exchange  -> found the cash-out point, stop this branch
   - a labeled mixer     -> trail broken, stop this branch
   - out of transactions -> dead end, stop this branch
   - at max depth        -> depth limit reached, stop this branch
   Bridges are flagged but the walk continues through them.
4. Enumerate the paths from the suspect wallet to each terminal node and
   report them, best (an exchange, fewest hops) first.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any

import networkx as nx

from .chains import detect_chain
from .config import DEFAULT_MAX_DEPTH, MAX_NODES, MAX_PATHS
from .fetchers import fetch
from .labels import BRIDGE, EXCHANGE, MIXER, get_label_store

# Terminal kinds - why a branch stopped.
TERMINAL_EXCHANGE = "exchange"
TERMINAL_MIXER = "mixer"
TERMINAL_DEAD_END = "dead_end"
TERMINAL_DEPTH_LIMIT = "depth_limit"

# Overall trace outcomes, best to worst.
STATUS_ATTRIBUTED = "attributed"        # reached an exchange
STATUS_TRAIL_BROKEN = "trail_broken"    # reached a mixer and stopped
STATUS_DEPTH_LIMIT = "depth_limit"      # still moving when we ran out of hops
STATUS_DEAD_END = "dead_end"            # funds have not moved on
STATUS_NO_ACTIVITY = "no_activity"      # nothing left this wallet at all

# Node roles, used for colouring the graph in the UI.
ROLE_SUSPECT = "suspect"
ROLE_INTERMEDIATE = "intermediate"

# Safety rails so a wide graph cannot hang the demo. MAX_NODES and MAX_PATHS
# come from app/config.py (both overridable by environment variable).
MAX_PATHS_PER_TERMINAL = 3

# Enumerating routes to every terminal of a wide graph is the expensive part,
# so stop scanning well before it matters and rank what we have.
PATH_SCAN_LIMIT = max(MAX_PATHS * 4, 200)


@dataclass
class PathEdge:
    """One transfer along a traced path."""

    src: str
    dst: str
    txid: str
    amount: float
    timestamp: str
    chain: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TracePath:
    """One route from the suspect wallet to a terminal node."""

    nodes: list[str]
    edges: list[PathEdge]
    terminal: str
    terminal_kind: str                       # exchange | mixer | dead_end | depth_limit
    entity: str | None = None                # labeled entity at the terminal
    mixers: list[str] = field(default_factory=list)
    bridges: list[str] = field(default_factory=list)
    confidence: float = 0.0                  # filled in by app/scoring.py
    confidence_factors: list[dict[str, Any]] = field(default_factory=list)
    flags: list[dict[str, Any]] = field(default_factory=list)

    @property
    def hops(self) -> int:
        return len(self.edges)

    @property
    def amount_first_hop(self) -> float:
        return self.edges[0].amount if self.edges else 0.0

    @property
    def amount_final_hop(self) -> float:
        return self.edges[-1].amount if self.edges else 0.0

    @property
    def is_success(self) -> bool:
        return self.terminal_kind == TERMINAL_EXCHANGE

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": self.nodes,
            "edges": [e.to_dict() for e in self.edges],
            "terminal": self.terminal,
            "terminal_kind": self.terminal_kind,
            "entity": self.entity,
            "hops": self.hops,
            "mixers": self.mixers,
            "bridges": self.bridges,
            "amount_first_hop": self.amount_first_hop,
            "amount_final_hop": self.amount_final_hop,
            "confidence": self.confidence,
            "confidence_factors": self.confidence_factors,
            "flags": self.flags,
        }


@dataclass
class TraceResult:
    """Everything the UI, the API and the PDF report need."""

    address: str
    chain: str
    status: str
    paths: list[TracePath] = field(default_factory=list)
    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)  # address -> metadata
    max_depth: int = DEFAULT_MAX_DEPTH
    data_sources: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    graph: nx.DiGraph | None = None
    elapsed_ms: float = 0.0

    # --- convenience accessors used all over the UI ---------------------------
    @property
    def best_path(self) -> TracePath | None:
        return self.paths[0] if self.paths else None

    @property
    def attributed_entity(self) -> str | None:
        for path in self.paths:
            if path.is_success:
                return path.entity
        return None

    @property
    def attributed_address(self) -> str | None:
        for path in self.paths:
            if path.is_success:
                return path.terminal
        return None

    @property
    def confidence(self) -> float:
        return self.best_path.confidence if self.best_path else 0.0

    @property
    def hops(self) -> int | None:
        return self.best_path.hops if self.best_path else None

    @property
    def mixers_seen(self) -> list[str]:
        return sorted({m for p in self.paths for m in p.mixers})

    @property
    def bridges_seen(self) -> list[str]:
        return sorted({b for p in self.paths for b in p.bridges})

    def to_dict(self) -> dict[str, Any]:
        return {
            "address": self.address,
            "chain": self.chain,
            "status": self.status,
            "attributed_entity": self.attributed_entity,
            "attributed_address": self.attributed_address,
            "confidence": self.confidence,
            "hops": self.hops,
            "max_depth": self.max_depth,
            "paths": [p.to_dict() for p in self.paths],
            "nodes": self.nodes,
            "mixers_seen": self.mixers_seen,
            "bridges_seen": self.bridges_seen,
            "data_sources": self.data_sources,
            "notes": self.notes,
            "elapsed_ms": self.elapsed_ms,
        }


def _role_for(address: str, is_source: bool, has_outgoing: bool) -> str:
    """Node role, used for graph colouring and for stopping the walk."""
    if is_source:
        return ROLE_SUSPECT
    label_type = get_label_store().type_of(address)
    if label_type in {EXCHANGE, MIXER, BRIDGE}:
        return label_type
    return ROLE_INTERMEDIATE if has_outgoing else TERMINAL_DEAD_END


def build_graph(
    address: str,
    max_depth: int = DEFAULT_MAX_DEPTH,
    allow_network: bool = True,
) -> tuple[nx.DiGraph, list[str], list[str]]:
    """BFS outward from `address`, returning (graph, data_sources, notes).

    Node attributes: role, depth, label, entity, chain.
    Edge attributes: txid, amount, timestamp, chain, transfers (all of them).
    """
    labels = get_label_store()
    graph = nx.DiGraph()
    sources: list[str] = []
    notes: list[str] = []

    source_chain = detect_chain(address)
    graph.add_node(
        address,
        role=ROLE_SUSPECT,
        depth=0,
        label=None,
        entity=None,
        chain=source_chain,
        amount_in=0.0,
        timestamp="",
    )

    queue: deque[tuple[str, int]] = deque([(address, 0)])
    expanded: set[str] = set()

    while queue:
        current, depth = queue.popleft()
        if current in expanded:
            continue
        expanded.add(current)

        # Stop conditions - we do not walk past these.
        if depth >= max_depth:
            graph.nodes[current]["stopped"] = TERMINAL_DEPTH_LIMIT
            continue
        if current != address and labels.type_of(current) in {EXCHANGE, MIXER}:
            # Exchange = cash-out point found. Mixer = trail broken.
            graph.nodes[current]["stopped"] = labels.type_of(current)
            continue
        if len(graph) >= MAX_NODES:
            notes.append(f"stopped expanding at {MAX_NODES} nodes (graph too wide)")
            break

        result = fetch(current, allow_network=allow_network)
        if result.source not in sources and result.source != "none":
            sources.append(result.source)
        interesting = result.truncated or "live fetch failed" in result.note
        if result.note and interesting and result.note not in notes:
            notes.append(result.note)

        # Only transactions where this address is spending - money leaving it.
        outgoing = [tx for tx in result.transactions if current in tx.inputs]
        if not outgoing:
            graph.nodes[current]["stopped"] = TERMINAL_DEAD_END
            continue

        for tx in outgoing:
            for out_address in tx.outputs:
                if out_address in tx.inputs:
                    continue  # change coming back to the sender, not a real hop
                amount = tx.amount_to(out_address)

                if out_address not in graph:
                    graph.add_node(
                        out_address,
                        role=ROLE_INTERMEDIATE,
                        depth=depth + 1,
                        label=None,
                        entity=None,
                        chain=tx.chain,
                        amount_in=amount,
                        timestamp=tx.timestamp,
                    )

                transfer = {
                    "txid": tx.txid,
                    "amount": amount,
                    "timestamp": tx.timestamp,
                    "chain": tx.chain,
                }
                if graph.has_edge(current, out_address):
                    graph[current][out_address]["transfers"].append(transfer)
                    graph[current][out_address]["amount"] += amount
                else:
                    graph.add_edge(current, out_address, transfers=[transfer], **transfer)

                queue.append((out_address, depth + 1))

    # Second pass: now that we know which nodes have outgoing edges, assign the
    # final role and attach any label metadata.
    for node in graph.nodes:
        label = labels.lookup(node)
        has_outgoing = graph.out_degree(node) > 0
        graph.nodes[node]["role"] = _role_for(node, node == address, has_outgoing)
        graph.nodes[node]["label"] = label.type if label else None
        graph.nodes[node]["entity"] = label.entity if label else None

    return graph, sources, notes


def _terminal_kind(graph: nx.DiGraph, node: str, max_depth: int) -> str | None:
    """Why this node ends a branch, or None if it does not."""
    role = graph.nodes[node].get("role")
    if role == EXCHANGE:
        return TERMINAL_EXCHANGE
    if role == MIXER:
        return TERMINAL_MIXER
    if graph.out_degree(node) == 0:
        if graph.nodes[node].get("depth", 0) >= max_depth:
            return TERMINAL_DEPTH_LIMIT
        return TERMINAL_DEAD_END
    return None


# Ranking: an exchange beats a mixer beats a depth limit beats a dead end.
_KIND_RANK = {
    TERMINAL_EXCHANGE: 0,
    TERMINAL_MIXER: 1,
    TERMINAL_DEPTH_LIMIT: 2,
    TERMINAL_DEAD_END: 3,
}


def _build_path(graph: nx.DiGraph, nodes: list[str], max_depth: int) -> TracePath:
    labels = get_label_store()
    edges: list[PathEdge] = []
    for src, dst in zip(nodes, nodes[1:]):
        data = graph[src][dst]
        edges.append(
            PathEdge(
                src=src,
                dst=dst,
                txid=data.get("txid", ""),
                amount=float(data.get("amount", 0.0)),
                timestamp=data.get("timestamp", ""),
                chain=data.get("chain", "unknown"),
            )
        )

    terminal = nodes[-1]
    label = labels.lookup(terminal)
    return TracePath(
        nodes=nodes,
        edges=edges,
        terminal=terminal,
        terminal_kind=_terminal_kind(graph, terminal, max_depth) or TERMINAL_DEAD_END,
        entity=label.entity if label else None,
        mixers=[n for n in nodes if labels.is_mixer(n)],
        bridges=[n for n in nodes if labels.is_bridge(n)],
    )


def trace(
    address: str,
    max_depth: int = DEFAULT_MAX_DEPTH,
    allow_network: bool = True,
) -> TraceResult:
    """Trace a suspect wallet forward and report where the money went."""
    import time

    started = time.perf_counter()
    address = (address or "").strip()

    graph, sources, notes = build_graph(address, max_depth=max_depth, allow_network=allow_network)

    # Collect every node that ends a branch, then enumerate routes to it.
    # Nodes come out in BFS insertion order, so the nearest terminals - the ones
    # an investigator cares about - are scanned first.
    paths: list[TracePath] = []
    terminals_scanned = 0
    for node in graph.nodes:
        if node == address:
            continue
        if _terminal_kind(graph, node, max_depth) is None:
            continue
        if len(paths) >= PATH_SCAN_LIMIT:
            notes.append(
                f"stopped enumerating routes after {len(paths)} paths "
                f"({terminals_scanned} terminal nodes scanned)"
            )
            break
        terminals_scanned += 1
        routes = nx.all_simple_paths(graph, address, node, cutoff=max_depth)
        for count, route in enumerate(routes):
            if count >= MAX_PATHS_PER_TERMINAL:
                break
            paths.append(_build_path(graph, list(route), max_depth))

    # Best first: exchanges before mixers, then fewest hops.
    paths.sort(key=lambda p: (_KIND_RANK.get(p.terminal_kind, 9), p.hops))

    # Keep the ranked head. Hundreds of dead-end branches are noise, and every
    # consumer downstream (UI, JSON, PDF) has to render whatever we return.
    if len(paths) > MAX_PATHS:
        notes.append(f"showing the {MAX_PATHS} best-ranked of {len(paths)} paths found")
        paths = paths[:MAX_PATHS]

    if not graph.out_degree(address):
        status = STATUS_NO_ACTIVITY
    elif any(p.terminal_kind == TERMINAL_EXCHANGE for p in paths):
        status = STATUS_ATTRIBUTED
    elif any(p.terminal_kind == TERMINAL_MIXER for p in paths):
        status = STATUS_TRAIL_BROKEN
    elif any(p.terminal_kind == TERMINAL_DEPTH_LIMIT for p in paths):
        status = STATUS_DEPTH_LIMIT
    else:
        status = STATUS_DEAD_END

    node_meta = {
        node: {
            "role": data.get("role"),
            "depth": data.get("depth"),
            "label": data.get("label"),
            "entity": data.get("entity"),
            "chain": data.get("chain"),
            "amount_in": data.get("amount_in", 0.0),
            "timestamp": data.get("timestamp", ""),
        }
        for node, data in graph.nodes(data=True)
    }

    result = TraceResult(
        address=address,
        chain=detect_chain(address),
        status=status,
        paths=paths,
        nodes=node_meta,
        max_depth=max_depth,
        data_sources=sources,
        notes=notes,
        graph=graph,
    )

    # Phase 4: attach a confidence score and typology flags to every path.
    from .scoring import apply_scoring  # local import keeps the modules decoupled

    apply_scoring(result)
    result.elapsed_ms = (time.perf_counter() - started) * 1000
    return result
