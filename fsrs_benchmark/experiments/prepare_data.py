"""Build and cache the Phase 6 development event tables (anki 100 users c1000; duolingo 5000 users)."""
import sys
from pathlib import Path

import pandas as pd

from fsrs_benchmark import data, duolingo, split

CACHE = Path("results/phase6_validation/cache")


def build(dataset):
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / f"{dataset}_events.parquet"
    if out.exists():
        return pd.read_parquet(out)
    if dataset == "anki":
        paths = data.download_users(100, 0, "data/anki10k")
        df = pd.concat([data.load_user(p) for p in paths], ignore_index=True)
        ev = split.make_split(df, max_cards=1000, seed=0)
    else:
        p = Path("data/duolingo/settles.acl16.learning_traces.13m.csv.gz")
        counts, t0 = duolingo.scan(p, p.parent / "duolingo_counts.parquet")
        users = duolingo.sample_users(counts, 5000, seed=0, min_rows=1)
        ev = split.make_split(duolingo.load_users(p, users, counts, t0), max_cards=10**9, seed=0)
    ev.to_parquet(out, index=False)
    return ev


if __name__ == "__main__":
    for d in sys.argv[1:] or ["anki", "duolingo"]:
        e = build(d)
        print(d, len(e), e["user_id"].nunique(), int(e["scored"].sum()))
