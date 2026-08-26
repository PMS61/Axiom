"""ChainTrace POC - investigator dashboard.

Run from the chaintrace/ directory:

    streamlit run ui/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Make the sibling `app` package importable when Streamlit runs this file.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from app.cache import get_cache
from app.fetchers import get_sample_fetcher
from app.labels import get_label_store
from app.ml import feature_importances, score_wallet
from app.report import build_pdf
from app.storage import case_from_trace, get_case_store
from app.scoring import confidence_band
from app.trace import (
    STATUS_ATTRIBUTED,
    STATUS_DEAD_END,
    STATUS_DEPTH_LIMIT,
    STATUS_NO_ACTIVITY,
    STATUS_TRAIL_BROKEN,
    TraceResult,
    trace,
)
from app.viz import COLORS, ROLE_LABELS, build_matplotlib_png, build_pyvis_html, format_amount, short

st.set_page_config(page_title="ChainTrace POC", page_icon="C", layout="wide")

SEVERITY_ICON = {"high": "HIGH", "warn": "WARN", "info": "INFO"}


# --------------------------------------------------------------------------- #
# Sidebar - controls
# --------------------------------------------------------------------------- #
def sidebar() -> tuple[int, bool]:
    st.sidebar.title("ChainTrace")
    st.sidebar.caption("Wallet tracing POC")

    st.sidebar.subheader("Demo wallets")
    wallets = get_sample_fetcher().demo_wallets()
    for wallet in wallets:
        if st.sidebar.button(short(wallet, 16, 4), key=f"demo-{wallet}", use_container_width=True):
            st.session_state.address = wallet

    st.sidebar.subheader("Trace settings")
    max_depth = st.sidebar.slider("Max hops", min_value=1, max_value=10, value=5)
    offline = st.sidebar.toggle(
        "Offline mode (bundled sample data only)",
        value=True,
        help="Off: try live Esplora / Etherscan first, fall back to sample data on failure.",
    )

    if not offline:
        st.sidebar.caption(
            "Live mode paces calls (~4/s per API) and pages through full address "
            "history, so a deep trace on a busy address takes a while. Start with "
            "2-3 hops."
        )

    if st.sidebar.button("Clear cache"):
        get_cache().clear()
        st.sidebar.success("Cache cleared")

    labels = get_label_store()
    st.sidebar.caption(f"{len(labels)} labeled addresses loaded")
    return max_depth, not offline


# --------------------------------------------------------------------------- #
# Result rendering
# --------------------------------------------------------------------------- #
def status_banner(result: TraceResult) -> None:
    if result.status == STATUS_ATTRIBUTED:
        st.success(
            f"Cash-out point identified: **{result.attributed_entity}** "
            f"({short(result.attributed_address or '', 14, 6)}) after {result.hops} hop(s)."
        )
    elif result.status == STATUS_TRAIL_BROKEN:
        st.error(
            f"**Trail broken.** The funds enter **{result.best_path.entity or 'a mixer'}** "
            "and cannot be followed further on-chain."
        )
    elif result.status == STATUS_DEPTH_LIMIT:
        st.warning(
            f"Funds were still moving at the {result.max_depth}-hop limit. "
            "Raise 'Max hops' in the sidebar to keep going."
        )
    elif result.status == STATUS_DEAD_END:
        st.warning("Dead end: the funds have not moved on from the final address yet.")
    elif result.status == STATUS_NO_ACTIVITY:
        st.info("No outgoing transactions found for this wallet.")


def summary_strip(result: TraceResult) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Attributed exchange", result.attributed_entity or "-")
    col2.metric("Hops", result.hops if result.hops is not None else "-")
    confidence = result.confidence
    # delta_color="off": the band is a label, not a movement, so it should not
    # render as a green upward arrow on a low score.
    col3.metric(
        "Confidence",
        f"{confidence:.0f}%" if confidence else "-",
        confidence_band(confidence) if confidence else None,
        delta_color="off",
    )
    col4.metric("Trace time", f"{result.elapsed_ms:.0f} ms")


def legend_html() -> str:
    items = "".join(
        f'<span style="display:inline-flex;align-items:center;margin-right:18px;">'
        f'<span style="width:12px;height:12px;border-radius:3px;background:{color};'
        f'display:inline-block;margin-right:6px;"></span>'
        f'<span style="font-size:13px;color:#475569;">{ROLE_LABELS.get(role, role)}</span></span>'
        for role, color in COLORS.items()
    )
    return f'<div style="margin:6px 0 10px 0;">{items}</div>'


def graph_tab(result: TraceResult) -> None:
    st.markdown(legend_html(), unsafe_allow_html=True)
    try:
        components.html(build_pyvis_html(result), height=640, scrolling=False)
        st.caption("Drag to pan, scroll to zoom, hover a node for its details.")
    except Exception as exc:  # pyvis missing or template problem - still show a picture
        st.warning(f"Interactive graph unavailable ({exc}); showing a static render.")
        st.image(build_matplotlib_png(result), use_container_width=True)

    with st.expander("Node details"):
        rows = [
            {
                "Address": address,
                "Role": ROLE_LABELS.get(meta.get("role"), meta.get("role")),
                "Entity": meta.get("entity") or "-",
                "Hop": meta.get("depth"),
                "Received": format_amount(meta.get("amount_in", 0.0), meta.get("chain", "")),
                "First seen": meta.get("timestamp") or "-",
            }
            for address, meta in result.nodes.items()
        ]
        frame = pd.DataFrame(rows).sort_values("Hop")
        st.dataframe(frame, use_container_width=True, hide_index=True)


# Streamlit allows at most 500 elements inside one container, and a real
# address fans out into hundreds of branches, so both loops below are capped.
MAX_PATHS_SHOWN = 10


def paths_tab(result: TraceResult) -> None:
    if not result.paths:
        st.info("No paths to show.")
        return

    shown = result.paths[:MAX_PATHS_SHOWN]
    if len(result.paths) > len(shown):
        st.caption(
            f"Showing the {len(shown)} best-ranked of {len(result.paths)} paths. "
            "The full set is in the Evidence tab."
        )

    for index, path in enumerate(shown, start=1):
        headline = path.entity or short(path.terminal, 16, 6)
        title = (
            f"Path {index} - {path.terminal_kind.replace('_', ' ')} at {headline} "
            f"({path.hops} hops, confidence {path.confidence:.0f}%)"
        )
        with st.expander(title, expanded=(index == 1)):
            st.markdown(
                "  ->  ".join(f"`{short(node, 12, 4)}`" for node in path.nodes)
            )
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Hop": position,
                            "From": short(edge.src, 12, 4),
                            "To": short(edge.dst, 12, 4),
                            "Amount": format_amount(edge.amount, edge.chain),
                            "Transaction": edge.txid,
                            "Timestamp": edge.timestamp or "-",
                        }
                        for position, edge in enumerate(path.edges, start=1)
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("**How this confidence score was reached**")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Factor": factor["factor"],
                            "Effect": f"{factor['delta']:+.0f}",
                            "Why": factor["detail"],
                        }
                        for factor in path.confidence_factors
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.progress(
                min(int(path.confidence), 100),
                text=f"{path.confidence:.0f}% - {confidence_band(path.confidence)} confidence",
            )


SEVERITY_ORDER = {"high": 0, "warn": 1, "info": 2}


def flags_tab(result: TraceResult) -> None:
    # The same flag turns up on every path that shares an address, so collapse
    # by (code, address) before rendering.
    unique: dict[tuple[str, str], dict] = {}
    for path in result.paths:
        for flag in path.flags:
            unique.setdefault((flag["code"], flag.get("address", "")), flag)

    flags = sorted(unique.values(), key=lambda f: SEVERITY_ORDER.get(f["severity"], 9))
    if not flags:
        st.success("No mixer, bridge or structuring typologies detected on the traced paths.")
        return

    # The most serious few as alerts, the rest as one table - a table is a
    # single element, so this stays well under Streamlit's 500-per-container cap.
    for flag in [f for f in flags if f["severity"] == "high"][:5]:
        st.error(
            f"**{flag['title']}** ({short(flag.get('address', ''), 14, 6)}): {flag['detail']}"
        )

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Severity": SEVERITY_ICON.get(flag["severity"], flag["severity"]),
                    "Flag": flag["title"],
                    "Address": short(flag.get("address", ""), 14, 6),
                    "Detail": flag["detail"],
                }
                for flag in flags
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )


def risk_tab(result: TraceResult, allow_network: bool) -> None:
    """The ML hook: an illicit-likelihood score for the suspect wallet."""
    risk = score_wallet(result.address, allow_network=allow_network)

    left, right = st.columns([1, 2])
    left.metric("Illicit likelihood", f"{risk.percent:.0f}%")
    left.caption(
        f"RandomForest, {risk.model_accuracy:.0%} holdout accuracy on "
        f"{risk.trained_on} synthetic rows"
    )
    right.markdown("**Wallet features fed to the model**")
    right.dataframe(
        pd.DataFrame(
            [
                {"Feature": name, "Value": value, "Model weight": feature_importances()[name]}
                for name, value in risk.features.items()
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
    if risk.note:
        st.caption(risk.note + " Treat it as a triage hint, not evidence.")


def evidence_tab(result: TraceResult, case_id: str | None = None,
                 metadata: dict | None = None) -> None:
    st.write("**Data sources used:** " + (", ".join(result.data_sources) or "none"))
    for note in result.notes:
        st.caption(note)

    # Built on render rather than behind a "generate" button: clicking a button
    # reruns the script, and a rerun sends Streamlit back to the first tab.
    st.download_button(
        "Download PDF report",
        data=build_pdf(
            result,
            case_id=case_id,
            risk=score_wallet(result.address, allow_network=False),
            metadata=metadata,
        ),
        file_name=f"chaintrace_report_{case_id or short(result.address, 10, 4)}.pdf",
        mime="application/pdf",
        key=f"pdf-{case_id or result.address}",
    )
    payload = json.dumps(result.to_dict(), indent=2, default=str)
    st.download_button(
        "Download trace as JSON",
        data=payload,
        file_name=f"chaintrace_{short(result.address, 10, 4)}.json",
        mime="application/json",
    )
    st.json(result.to_dict(), expanded=False)


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
def render_result(result: TraceResult, allow_network: bool,
                  case_id: str | None = None, metadata: dict | None = None) -> None:
    """Everything below the input box: banner, metrics and the four tabs."""
    st.divider()
    status_banner(result)
    summary_strip(result)

    graph, paths, flags, risk, evidence = st.tabs(
        ["Fund flow graph", "Traced paths", "Typology flags", "Risk (ML)", "Evidence"]
    )
    with graph:
        graph_tab(result)
    with paths:
        paths_tab(result)
    with flags:
        flags_tab(result)
    with risk:
        risk_tab(result, allow_network)
    with evidence:
        evidence_tab(result, case_id=case_id, metadata=metadata)


def complaint_view(max_depth: int, allow_network: bool) -> None:
    """Second entry point: a wallet arrives from the fraud-complaint feed.

    Same engine, no investigator in the loop, and a clock on the result -
    this is the "act while the money is still moving" mode.
    """
    st.title("Complaint feed - real-time mode")
    st.caption(
        "Simulates a wallet arriving from a fraud complaint. The same trace engine "
        "runs automatically and opens a case."
    )

    with st.form("complaint"):
        column_left, column_right = st.columns(2)
        wallet = column_left.text_input("Reported wallet", value="SUSPECT_WALLET_DEMO_1")
        reference = column_right.text_input("Complaint reference", value="NCRP/2026/00123")
        complainant = column_left.text_input("Complainant", value="Victim A")
        amount = column_right.number_input("Amount reported", min_value=0.0, value=46000.0)
        description = st.text_area(
            "Complaint summary", value="Victim transferred funds after an investment scam call."
        )
        submitted = st.form_submit_button("Receive complaint and trace", type="primary")

    if submitted and wallet.strip():
        started = time.perf_counter()
        with st.spinner("Complaint received - tracing..."):
            result = trace(wallet.strip(), max_depth=max_depth, allow_network=allow_network)
            metadata = {
                "complaint_ref": reference,
                "complainant": complainant,
                "amount_reported": amount,
                "description": description,
            }
            case = get_case_store().save(
                case_from_trace(result, source="complaint", metadata=metadata)
            )
        st.session_state["complaint_result"] = (result, case.case_id, metadata)
        st.session_state["complaint_elapsed"] = (time.perf_counter() - started) * 1000

    stored = st.session_state.get("complaint_result")
    if stored:
        result, case_id, metadata = stored
        elapsed = st.session_state.get("complaint_elapsed", 0.0)
        left, right = st.columns(2)
        left.metric("Time to result", f"{elapsed:.0f} ms")
        right.metric("Case opened", case_id)
        render_result(result, allow_network, case_id=case_id, metadata=metadata)

    cases = get_case_store().list(limit=10)
    if cases:
        st.subheader("Recent cases")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Case": case.case_id,
                        "Opened": case.created_at,
                        "Wallet": short(case.address, 14, 4),
                        "Outcome": case.status,
                        "Exchange": case.entity or "-",
                        "Confidence": f"{case.confidence:.0f}%" if case.confidence else "-",
                        "Source": case.source,
                    }
                    for case in cases
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )


def investigator_view(max_depth: int, allow_network: bool) -> None:
    st.title("ChainTrace POC")
    st.caption(
        "Trace a suspect wallet hop by hop to the exchange where the money cashed out."
    )

    st.session_state.setdefault("address", "")
    column_input, column_button = st.columns([5, 1])
    address = column_input.text_input(
        "Suspect wallet address",
        key="address",
        placeholder="Paste a wallet address, or pick a demo wallet in the sidebar",
    )
    column_button.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    traced = column_button.button("Trace", type="primary", use_container_width=True)

    if traced:
        if not address.strip():
            st.warning("Enter a wallet address first.")
        else:
            with st.spinner("Following the money..."):
                st.session_state["result"] = trace(
                    address.strip(), max_depth=max_depth, allow_network=allow_network
                )

    result: TraceResult | None = st.session_state.get("result")
    if result is None:
        st.info("Pick a demo wallet in the sidebar and press Trace to see the engine run.")
        return

    render_result(result, allow_network)


def main() -> None:
    view = st.sidebar.radio(
        "View", ["Investigator trace", "Complaint feed (real-time)"], label_visibility="collapsed"
    )
    max_depth, allow_network = sidebar()
    if view.startswith("Investigator"):
        investigator_view(max_depth, allow_network)
    else:
        complaint_view(max_depth, allow_network)


main()
