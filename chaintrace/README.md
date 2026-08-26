# ChainTrace POC

Crypto-wallet tracing for law enforcement - **proof of concept**.

Give it a suspect wallet address. It walks the money hop by hop across the
blockchain until it reaches a known exchange (a cash-out point), then reports
which exchange, with a confidence score and a fund-flow graph.

> Hackathon POC. Everything runs offline off bundled sample data - no paid API,
> no database server, no Redis required.

## Run it (two commands)

```bash
pip install -r requirements.txt
streamlit run ui/streamlit_app.py
```

Then open http://localhost:8501.

Prefer an isolated environment:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run ui/streamlit_app.py
```

## Layout

```text
chaintrace/
├── app/          # backend logic (fetching, tracing, scoring)
├── data/         # bundled sample transactions + labeled addresses
├── ui/           # Streamlit dashboard
├── tests/        # pytest suite
└── requirements.txt
```

## Demo walkthrough

1. Launch the dashboard and pick `SUSPECT_WALLET_DEMO_1` in the sidebar, then
   press **Trace**. The money splits: one branch cashes out at *SampleExchange
   Alpha* (green, 87% confidence), the other disappears into a mixer (red).
2. `SUSPECT_WALLET_DEMO_2` - the trail breaks in a mixer, marked "trail broken
   here".
3. `0xSuspectWalletDemo3333...` - an Ethereum trace that crosses a bridge
   (purple) before cashing out.
4. `SUSPECT_WALLET_DEMO_4` - a dead end: the funds never moved on.
5. `SUSPECT_WALLET_DEMO_5` - still moving at the 5-hop limit. Raise **Max hops**
   in the sidebar to 8 and it reaches *SampleExchange Zeta*.

## Status

Phase 6 - the POC plus the ML hook. A RandomForest trained on a small bundled
synthetic feature set scores each traced wallet for "illicit likelihood" next
to the trace result.
