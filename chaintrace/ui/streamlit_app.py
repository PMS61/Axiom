"""ChainTrace POC - investigator dashboard (Phase 1 skeleton).

Run with:  streamlit run ui/streamlit_app.py
"""

import streamlit as st

st.set_page_config(page_title="ChainTrace POC", page_icon="chains", layout="wide")

st.title("ChainTrace POC")
st.caption("Crypto wallet tracing for investigators - proof of concept")

address = st.text_input("Suspect wallet address", placeholder="e.g. SUSPECT_WALLET_DEMO_1")
traced = st.button("Trace", type="primary")

if traced:
    # Phase 1: nothing is wired up yet, the engine arrives in Phase 3.
    st.info(f"Trace engine not wired yet. You entered: {address or '(nothing)'}")
