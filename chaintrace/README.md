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

Phase 3 - the trace engine works. BFS over a NetworkX graph follows every
branch from the suspect wallet until it hits a labeled exchange, a mixer, a
dead end or the hop limit, and reports the paths it found. The dashboard is
still the Phase 1 skeleton.
