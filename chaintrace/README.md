# ChainTrace POC

Crypto-wallet tracing for law enforcement - **proof of concept**.

Give it a suspect wallet address. It walks the money hop by hop across the
blockchain until it reaches a known exchange (a cash-out point), then reports
which exchange, with a confidence score, typology flags and a visual fund-flow
graph.

Two entry points, one engine: an investigator types a wallet into the
dashboard, or a wallet arrives automatically from a fraud complaint.

> Hackathon POC. It runs offline off bundled sample data - no paid API, no
> database server, no Redis, no API key required.

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

The API (second entry point) runs separately:

```bash
uvicorn app.api:app --port 8000        # http://localhost:8000/docs
```

Or both at once with Docker:

```bash
docker compose up --build              # dashboard :8501, API :8000
```

## Demo walkthrough

Pick a demo wallet in the sidebar and press **Trace**.

| Demo wallet | What it shows |
|---|---|
| `SUSPECT_WALLET_DEMO_1` | Funds split: one branch cashes out at *SampleExchange Alpha* (87% confidence), the other vanishes into a mixer |
| `SUSPECT_WALLET_DEMO_2` | Trail broken - the money enters a mixing service and stops |
| `0xSuspectWalletDemo3333...` | Ethereum trace that crosses a bridge before cashing out |
| `SUSPECT_WALLET_DEMO_4` | Dead end - the funds have not moved on |
| `SUSPECT_WALLET_DEMO_5` | Still moving at the 5-hop limit; raise **Max hops** to 8 and it reaches *SampleExchange Zeta* |

Then switch the sidebar to **Complaint feed (real-time)**: submit the form and
the same engine runs automatically, opens a case and shows the time to result.

## How it works

```text
address ->  chain detection  ->  fetcher  ->  NetworkX graph  ->  BFS trace
                                     |                                |
                          live API or sample data          exchange / mixer /
                                                           dead end / hop limit
                                                                      |
                                        confidence score + typology flags + ML
                                                                      |
                                         graph view, evidence JSON, PDF report
```

1. **Chain detection** (`app/chains.py`) - Bitcoin (legacy, P2SH, bech32) vs
   Ethereum, from the address format.
2. **Fetching** (`app/fetchers/`) - Blockstream Esplora (with mempool.space as
   a backup) for Bitcoin, Etherscan `txlist`/`txlistinternal` for Ethereum.
   Everything is normalized to one `Tx` dataclass. If the network is
   unreachable, a key is missing, or the address is one of the demo wallets,
   the bundled sample data is used instead - the app never crashes on a
   missing dependency.
3. **Tracing** (`app/trace.py`) - transfers become a NetworkX directed graph;
   BFS follows every branch until a node is a labeled exchange (cash-out
   found), a mixer (trail broken), out of transactions (dead end), or at the
   hop limit (default 5). Bridges are flagged and walked through.
4. **Scoring** (`app/scoring.py`) - a transparent weighted formula, plus the
   typology flags (mixer, bridge, rapid movement, peel chain, fan-out).
5. **Visualization** (`app/viz.py`, `ui/streamlit_app.py`) - an interactive
   pyvis graph coloured by node role, with a matplotlib fallback.
6. **ML flag** (`app/ml.py`) - a RandomForest trained on a small synthetic
   feature set gives an "illicit likelihood" triage hint.
7. **Report and intake** (`app/report.py`, `app/api.py`, `app/storage.py`) -
   a PDF report, a `/complaint` endpoint, and SQLite case storage.

### The confidence score

Starts at 50 and adjusts. Every adjustment is shown in the UI and the PDF, so
the number can always be explained:

| Factor | Effect |
|---|---|
| Terminal is a labeled exchange | +25 |
| Terminal is a mixer | -40 |
| Hop limit reached / dead end | -15 / -10 |
| Hop distance | `20 - 6 x hops`, floor -15 |
| Passed through a mixer | -30 each, floor -45 |
| Crossed a bridge | -12 each, floor -24 |
| Value continuity (>=90% / >=50% arrives) | +10 / +5 |

Clamped to 5-99.

## Layout

```text
chaintrace/
├── app/
│   ├── api.py          # FastAPI: /trace, /complaint, /cases
│   ├── cache.py        # in-memory cache with a Redis-shaped interface
│   ├── chains.py       # address -> chain detection
│   ├── config.py       # paths, env vars, defaults
│   ├── fetchers/       # esplora / etherscan / bundled sample data
│   ├── labels.py       # labeled-address database
│   ├── ml.py           # RandomForest illicit-likelihood flag
│   ├── models.py       # Tx, Label, FetchResult
│   ├── report.py       # PDF report (reportlab)
│   ├── scoring.py      # confidence score + typology flags
│   ├── storage.py      # SQLite case storage (swappable for PostgreSQL)
│   ├── trace.py        # the BFS trace engine
│   └── viz.py          # pyvis / matplotlib graph rendering
├── data/               # sample transactions, labels, ML training sample
├── scripts/            # regenerate the ML training sample
├── ui/streamlit_app.py # the dashboard
└── tests/              # pytest suite
```

## Configuration

Everything has a working default. Set these only if you want live data:

| Variable | Effect |
|---|---|
| `ETHERSCAN_API_KEY` | Enables live Ethereum fetching. Missing = sample data. |
| `CHAINTRACE_OFFLINE` | `1` = never touch the network. |
| `CHAINTRACE_DB` | Where the SQLite case database lives. |
| `CHAINTRACE_HTTP_TIMEOUT` | Seconds before a live API call gives up (default 6). |
| `CHAINTRACE_HTTP_RETRIES` | Retries on throttling or a 5xx (default 3). |
| `CHAINTRACE_ESPLORA_RATE` / `CHAINTRACE_ETHERSCAN_RATE` | Requests per second per API (default 4). |
| `CHAINTRACE_MAX_TXS` | Transactions to pull per address before reporting truncation (default 200). |
| `CHAINTRACE_MAX_NODES` | Stop expanding the graph past this many addresses (default 400). |
| `CHAINTRACE_MAX_PATHS` | Keep this many best-ranked paths (default 50). |
| `CHAINTRACE_SAMPLE_DATA` | Point the offline fetcher at a different sample file. |

Deliberate POC substitutions, each behind an interface so it can be swapped:
SQLite stands in for PostgreSQL (`app/storage.py`), and an in-memory dict
stands in for Redis (`app/cache.py`).

## Tests

```bash
python -m pytest
```

49 tests covering chain detection, the fetch fallback chain, all five trace
scenarios, the scoring rules, the ML features, case storage, the PDF, the
API, and pagination/rate limiting/retries against a stubbed HTTP layer.

## Live data

Bitcoin needs no key. Ethereum needs a free `ETHERSCAN_API_KEY`. Turn off
**Offline mode** in the sidebar and the fetchers page through an address's full
history - Esplora 25 confirmed transactions at a time, Etherscan 100 - rather
than reading the first page and silently returning a partial (and therefore
wrong) picture. If an address is busier than `CHAINTRACE_MAX_TXS`, the trace
says so in its notes instead of pretending it saw everything.

Every call goes through one rate limiter per API (`app/httpclient.py`), which
paces requests and retries throttling and 5xx responses with backoff, honouring
`Retry-After`. That pacing is why a deep live trace is slow: the walk makes at
least one request per address it visits.

Live mode has never been exercised against the real APIs from this repo's
development sandbox (no outbound network), but the whole path - pagination,
normalization, graph, attribution - is covered end to end against a stubbed
Esplora in `tests/test_http_and_pagination.py`.

## Real addresses fan out

A busy mainnet address branches into thousands of addresses within a few hops.
Two caps keep that usable, and both say so in the result notes when they trip:
the walk stops expanding at `CHAINTRACE_MAX_NODES` addresses, and only the
`CHAINTRACE_MAX_PATHS` best-ranked routes are returned. The dashboard renders
at most 10 path blocks and collapses flags into one table, because Streamlit
refuses to draw more than 500 elements inside a single container.

## What this POC is not

Bitcoin and Ethereum only. No Tron, Solana or real cross-chain continuation
past a bridge. No authentication, no SAHYOG/NCRP integration, no real labeled
address feed - `data/labeled_addresses.json` holds 20 illustrative placeholder
labels, and the bundled transactions are synthetic. Attribution and scores are
investigative leads, not evidence.
