"""Build the local learner-demo sample. Requires pyarrow (in .venv-diagnostics).

Run from repository root:
  .venv-diagnostics/bin/python client/scripts/import_karl_demo.py
"""
import argparse
from collections import defaultdict
import csv
from datetime import timedelta
import hashlib
import json
from pathlib import Path
from statistics import median
from urllib.request import urlopen

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
HF_REVISION = "198aa632c0e8dca71c579f0647041d27d479a9e7"
GITHUB_REVISION = "2bdcb801232dab5362610fa8297119dc5e9876f2"
URLS = {
    "reviews.parquet": f"https://huggingface.co/datasets/nbalepur/KARL/resolve/{HF_REVISION}/data/train-00000-of-00001.parquet?download=true",
    "facts.csv": f"https://media.githubusercontent.com/media/Pinafore/fact-repetition/{GITHUB_REVISION}/facts.csv?download=true",
}
HASHES = {
    "reviews.parquet": "6ed0667325f2b2d21aa5c726d3c0942cdb7e77af85692d43e4a8701b14f61dca",
    "facts.csv": "3a7812d9e475828ac79a5a0063263267c16c032c42237cdfee17f4e236dd9ef7",
}


def build(cache):
    cache.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        path = cache / name
        if not path.exists():
            with urlopen(url, timeout=60) as response, path.with_suffix(".partial").open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            path.with_suffix(".partial").rename(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != HASHES[name]:
            raise ValueError(f"Source hash mismatch: {path}. Remove the cached file and retry.")

    rows = pq.read_table(cache / "reviews.parquet").to_pylist()
    with (cache / "facts.csv").open(newline="", encoding="utf-8") as source:
        facts = {row["fact_id"]: row for row in csv.DictReader(source)}
    pairs = defaultdict(list)
    for row in rows:
        pairs[(row["user_id"], row["card_id"])].append(row)

    normalize = lambda value: " ".join(value.split())
    eligible = defaultdict(list)
    for (user, card), events in pairs.items():
        fact = facts.get(card)
        if (not fact or len(events) < 4 or len(fact["text"]) > 130
                or fact["answer"] in ("", "NULL")
                or any(normalize(event["card_text"]) != normalize(fact["text"]) for event in events)):
            continue
        events.sort(key=lambda event: event["utc_datetime"])
        if len({event["utc_datetime"] for event in events}) != len(events):
            continue
        eligible[user].append((card, events))
    # Selection depends on history coverage and prompt length, never correctness.
    user = min(eligible, key=lambda key: (-len(eligible[key]), int(key)))
    selected = sorted(eligible[user], key=lambda pair: (-len(pair[1]), int(pair[0])))[:8]
    assert len(selected) == 8
    anchor = max(events[-1]["utc_datetime"] for _, events in selected) + timedelta(days=1)
    cards = []
    for card, events in selected:
        gaps = [(b["utc_datetime"] - a["utc_datetime"]).total_seconds() / 86400
                for a, b in zip(events, events[1:])]
        cards.append(dict(
            cardId=card,
            question=facts[card]["text"],
            answer=facts[card]["answer"],
            lastReviewDay=(events[-1]["utc_datetime"] - anchor).total_seconds() / 86400,
            # A descriptive seed for the existing illustrative scheduler, not a fitted parameter.
            initialStabilityDays=max(1, median(gaps)),
            events=[dict(at=event["utc_datetime"].isoformat(),
                         recalled=event["response"],
                         correctOnFirstTry=event["correct_on_first_try"])
                    for event in events],
        ))
    snapshot = dict(
        dataset="KAR³L", learnerId=user,
        datasetUrl="https://huggingface.co/datasets/nbalepur/KARL",
        factsUrl="https://github.com/Pinafore/fact-repetition/blob/master/facts.csv",
        paperUrl="https://aclanthology.org/2024.emnlp-main.784/",
        hfRevision=HF_REVISION, githubRevision=GITHUB_REVISION,
        sourceHashes={name: hashlib.sha256((cache / name).read_bytes()).hexdigest() for name in URLS},
        sourceLogCount=len(rows), sourceLearnerCount=len({row["user_id"] for row in rows}),
        selection="Learner with most eligible cards; eight most-reviewed cards with at least four logs, prompts <=130 characters, unique timestamps, and exact normalized prompt matches. No outcome filtering.",
        snapshotAt=anchor.isoformat(),
        terms="Repository includes an MIT license; the Hugging Face dataset card does not declare a separate data license. Included for this local research demonstration; no broader content-rights claim.",
        cards=cards,
    )
    output = ROOT / "client/src/lib/bayesian/karl-sample/karl-demo.json"
    output.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Matched {len(cards)} questions and answers to {sum(len(card['events']) for card in cards)} logs for learner {user}.")
    return snapshot


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/axiom-karl-data"))
    build(parser.parse_args().cache_dir)
