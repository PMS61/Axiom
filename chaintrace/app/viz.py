"""Graph rendering for the dashboard.

Primary renderer is pyvis (interactive: drag, zoom, hover for node details),
with all of its JavaScript inlined so it works with no internet connection.
If pyvis is unavailable for any reason there is a matplotlib fallback, so the
demo always shows a picture.
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any

import networkx as nx

from .trace import (
    ROLE_INTERMEDIATE,
    ROLE_SUSPECT,
    TERMINAL_DEAD_END,
    TraceResult,
)

# Node colours by role - the legend in the UI uses the same map.
COLORS: dict[str, str] = {
    ROLE_SUSPECT: "#2563eb",        # blue   - the wallet under investigation
    ROLE_INTERMEDIATE: "#94a3b8",   # slate  - pass-through address
    "exchange": "#16a34a",          # green  - cash-out point
    "mixer": "#dc2626",             # red    - trail broken
    "bridge": "#9333ea",            # purple - crossed chains
    TERMINAL_DEAD_END: "#6b7280",   # gray   - funds stopped here
}

ROLE_LABELS: dict[str, str] = {
    ROLE_SUSPECT: "Suspect wallet",
    ROLE_INTERMEDIATE: "Intermediate address",
    "exchange": "Exchange (cash-out)",
    "mixer": "Mixer (trail broken)",
    "bridge": "Cross-chain bridge",
    TERMINAL_DEAD_END: "Dead end",
}


def format_amount(amount: float, chain: str) -> str:
    """Bitcoin amounts are satoshis, Ethereum amounts are ETH."""
    if not amount:
        return "-"
    if chain == "ethereum":
        return f"{amount:,.4f} ETH".replace(".0000 ", " ")
    return f"{amount:,.0f} sats"


def short(address: str, head: int = 10, tail: int = 6) -> str:
    """Shorten an address for a node caption."""
    if len(address) <= head + tail + 3:
        return address
    return f"{address[:head]}...{address[-tail:]}"


def _node_caption(address: str, meta: dict[str, Any]) -> str:
    entity = meta.get("entity")
    return f"{entity}\n{short(address)}" if entity else short(address)


def _node_tooltip(address: str, meta: dict[str, Any]) -> str:
    lines = [
        f"Address: {address}",
        f"Role: {ROLE_LABELS.get(meta.get('role'), meta.get('role'))}",
        f"Chain: {meta.get('chain', 'unknown')}",
        f"Hop: {meta.get('depth', 0)}",
    ]
    if meta.get("entity"):
        lines.append(f"Entity: {meta['entity']} ({meta.get('label')})")
    if meta.get("amount_in"):
        lines.append(f"Received: {format_amount(meta['amount_in'], meta.get('chain', ''))}")
    if meta.get("timestamp"):
        lines.append(f"Seen: {meta['timestamp']}")
    return "\n".join(lines)


def _vis_network_bundle() -> str:
    """Read pyvis's bundled vis-network JavaScript straight off disk.

    We do not use pyvis's own `cdn_resources="in_line"` because it reads the
    bundle line by line, which rewrites the exotic control characters inside it
    and leaves the script with a syntax error. Reading the file whole keeps it
    byte-for-byte correct, and inlining it is what makes the graph render with
    no internet connection.
    """
    import pyvis

    package_root = Path(pyvis.__file__).resolve().parent
    candidates = sorted(package_root.glob("templates/lib/vis-*/vis-network.min.js"), reverse=True)
    candidates += sorted(package_root.glob("lib/vis-*/vis-network.min.js"), reverse=True)
    for candidate in candidates:
        return candidate.read_text(encoding="utf-8")
    raise FileNotFoundError("vis-network.min.js not found inside the pyvis package")


def _localize_resources(document: str) -> str:
    """Swap every CDN reference for a local copy (or drop it).

    vis-network is inlined because the graph needs it. Bootstrap is cosmetic,
    and with no internet its request just stalls the page, so it goes.
    """
    document = re.sub(
        r'<script[^>]*src="https://cdnjs\.cloudflare\.com/[^"]*vis-network[^"]*"[^>]*>\s*</script>',
        lambda _: f"<script>{_vis_network_bundle()}</script>",
        document,
        flags=re.S,
    )
    # Remaining remote tags are styling only.
    document = re.sub(r'<link[^>]*href="https://[^"]*"[^>]*>', "", document, flags=re.S)
    document = re.sub(r'<script[^>]*src="https://[^"]*"[^>]*>\s*</script>', "", document, flags=re.S)
    return document


def build_pyvis_html(result: TraceResult, height: str = "620px") -> str:
    """Return a self-contained HTML document with the interactive graph."""
    from pyvis.network import Network  # imported lazily so tests do not need it

    net = Network(
        height=height,
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="#111827",
        cdn_resources="remote",   # the CDN tags are swapped for local copies below
        notebook=False,
    )

    graph = result.graph or nx.DiGraph()
    for address, data in graph.nodes(data=True):
        role = data.get("role", ROLE_INTERMEDIATE)
        meta = result.nodes.get(address, data)
        net.add_node(
            address,
            label=_node_caption(address, meta),
            title=_node_tooltip(address, meta),
            color=COLORS.get(role, COLORS[ROLE_INTERMEDIATE]),
            shape="box" if role == ROLE_SUSPECT else "ellipse",
            level=int(data.get("depth", 0)),
            borderWidth=3 if role == ROLE_SUSPECT else 1,
            font={"color": "#111827", "size": 14},
        )

    for src, dst, data in graph.edges(data=True):
        transfers = data.get("transfers", [])
        tooltip = "\n".join(
            f"{t.get('txid', '')} | {format_amount(t.get('amount', 0), t.get('chain', ''))}"
            f" | {t.get('timestamp', '')}"
            for t in transfers
        )
        net.add_edge(
            src,
            dst,
            label=format_amount(float(data.get("amount", 0.0)), data.get("chain", "")),
            title=tooltip or data.get("txid", ""),
            color="#64748b",
            arrows="to",
            font={"size": 11, "align": "middle", "color": "#475569"},
        )

    # Left-to-right hierarchy by hop count reads like a flow of funds.
    net.set_options(
        """
        {
          "layout": {
            "hierarchical": {
              "enabled": true,
              "direction": "LR",
              "sortMethod": "directed",
              "levelSeparation": 260,
              "nodeSpacing": 130
            }
          },
          "physics": { "enabled": false },
          "interaction": { "hover": true, "tooltipDelay": 80, "navigationButtons": true },
          "edges": { "smooth": { "type": "cubicBezier", "roundness": 0.4 } }
        }
        """
    )
    return _localize_resources(net.generate_html(notebook=False))


def build_matplotlib_png(result: TraceResult) -> bytes:
    """Fallback renderer: a static PNG of the same graph."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    graph = result.graph or nx.DiGraph()
    figure, axis = plt.subplots(figsize=(11, 6))

    # Place nodes by hop depth (x) so the flow reads left to right.
    by_depth: dict[int, list[str]] = {}
    for node, data in graph.nodes(data=True):
        by_depth.setdefault(int(data.get("depth", 0)), []).append(node)
    positions = {
        node: (depth, index - (len(nodes) - 1) / 2)
        for depth, nodes in by_depth.items()
        for index, node in enumerate(nodes)
    }

    colors = [
        COLORS.get(data.get("role", ROLE_INTERMEDIATE), COLORS[ROLE_INTERMEDIATE])
        for _, data in graph.nodes(data=True)
    ]
    nx.draw_networkx_nodes(graph, positions, node_color=colors, node_size=1400, ax=axis)
    nx.draw_networkx_edges(graph, positions, edge_color="#64748b", arrows=True,
                           arrowsize=14, ax=axis)
    nx.draw_networkx_labels(
        graph, positions,
        labels={n: short(n, 8, 4) for n in graph.nodes},
        font_size=7, ax=axis,
    )
    axis.set_axis_off()
    axis.set_title(f"Fund flow from {short(result.address)}")

    buffer = io.BytesIO()
    figure.tight_layout()
    figure.savefig(buffer, format="png", dpi=140)
    plt.close(figure)
    return buffer.getvalue()
