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

## Status

Phase 4 - attribution, confidence and typology flags. A trace that reaches an
exchange reports the attributed VASP with a 5-99 confidence score and the full
list of factors behind it; mixer and bridge hops are flagged, and a trace that
dies in a mixer is marked "trail broken here". The dashboard is still the
Phase 1 skeleton.
