"""Export the Solar System dashboard replay from the pinned KARL study logs.

Fetch the pinned source with client/scripts/import_karl_demo.py. From the repo root:
    .venv-diagnostics/bin/python -m client.scripts.export_dashboard_replay
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from client.scripts.export_review_demo import (
    DAY_SECONDS, DAYS, DEMO_REVIEWS_PER_CARD, continuation, interval, project,
)
from client.scripts.import_karl_demo import HASHES
from fsrs_benchmark import duolingo, split
from fsrs_benchmark.models.gbdt import FROZEN_KWARGS, GBDT, W

# Curated KARL card IDs. Questions are lightly edited from the source prompts.
CARDS = {
    "19885": ("Mariner missions", "Which planet did Mariner 4 fly past and Mariner 9 orbit?", "Mars"),
    "19888": ("Olympus Mons", "Which planet is home to Olympus Mons, the tallest volcano in the Solar System?", "Mars"),
    "21358": ("NASA’s Juno probe", "Which planet did NASA’s Juno probe begin orbiting in 2016?", "Jupiter"),
    "21359": ("A planet’s rings", "Adrastea and Amalthea help maintain the rings of which planet?", "Jupiter"),
    "23920": ("The Apollo landing", "Which celestial body did Neil Armstrong walk on?", "The Moon"),
    "26635": ("The moon Mimas", "Mimas orbits which planet in less than one Earth day?", "Saturn"),
    "27701": ("The largest moon", "What is the largest moon in the Solar System?", "Ganymede"),
    "28096": ("The MESSENGER mission", "Which planet did NASA’s MESSENGER spacecraft orbit?", "Mercury"),
    "28099": ("A long solar day", "On which planet is a solar day longer than a year?", "Mercury"),
    "29702": ("The Akatsuki mission", "Which planet was Japan’s Akatsuki spacecraft sent to study?", "Venus"),
    "29706": ("The Vega probes", "Which planet did the Vega 1 and Vega 2 probes study?", "Venus"),
    "30054": ("Predicting a planet", "Which planet did Urbain Le Verrier predict using Newton’s laws?", "Neptune"),
    "30055": ("Lassell’s discovery", "Which planet’s largest moon did William Lassell discover shortly after the planet itself was found?", "Neptune"),
    "33194": ("The moons of Mars", "Which of Mars’s moons is larger and closer to the planet than Deimos?", "Phobos"),
    "35582": ("Planetary motion", "Whose laws describe how planets move in orbit?", "Johannes Kepler"),
    "37918": ("Earth’s magnetic field", "What measures around 30 microteslas at parts of Earth’s surface?", "Earth’s magnetic field"),
    "38353": ("Saturn’s largest moon", "What is Saturn’s largest moon?", "Titan"),
    "40406": ("A returning comet", "Which famous comet returns roughly every 75 years?", "Halley’s Comet"),
    "156871": ("Predicting a planet", "Which planet did Urbain Le Verrier predict using Newton’s laws?", "Neptune"),
    "157783": ("A discovery in 1781", "Which planet was discovered in 1781?", "Uranus"),
}


def build(source):
    if hashlib.sha256(source.read_bytes()).hexdigest() != HASHES["reviews.parquet"]:
        raise ValueError("The KARL reviews file differs from the pinned source")
    logs = pd.read_parquet(source, columns=["user_id", "card_id", "utc_datetime", "response"])
    lexeme, ids = pd.factorize(logs.card_id, sort=True)
    user = pd.factorize(logs.user_id, sort=True)[0]
    timestamp = logs.utc_datetime.dt.as_unit("s").astype("int64").to_numpy()
    t0 = int(timestamp.min())
    raw = duolingo._build_events(
        user, lexeme, timestamp, logs.response.to_numpy(dtype=np.int64),
        np.zeros(len(logs), dtype=np.int64), np.arange(len(logs)), t0,
        with_lexeme=True, with_ts=True,
    )
    events = split.make_split(raw, max_cards=10**9)
    lookup = {code: CARDS[card] for code, card in enumerate(ids) if card in CARDS}
    reviewed_cards = events[events.lexeme.isin(lookup)].groupby(["user_id", "card_id"]).size()
    candidates = reviewed_cards[reviewed_cards >= 2].groupby("user_id").size()
    candidate_ids = candidates[candidates >= 4].index
    fitted = events[~events.user_id.isin(candidate_ids)]
    return fitted, events[events.user_id.isin(candidate_ids)], lookup, t0


def select_case(model, held_out, lookup, t0):
    best = None
    for _, events in held_out.groupby("user_id"):
        selected = events[events.lexeme.isin(lookup)]
        labels = {row.card_id: lookup[row.lexeme] for row in selected.itertuples()}
        for anchor in sorted(selected.ts.unique())[-32:]:
            history = events[events.ts <= anchor]
            seen = history[history.card_id.isin(labels)]
            recent = seen.groupby("card_id").ts.agg(["size", "max"])
            candidates = recent.index[(recent["size"] >= 2) & (recent["max"] >= anchor - 45 * DAY_SECONDS)]
            if len(candidates) < 3:
                continue
            forecasts = project(model, history, candidates, t0)
            active_days = ((anchor - history.ts) // DAY_SECONDS).nunique()
            for target in (0.8, 0.85, 0.75, 0.9, 0.7):
                dates = {card: interval(forecasts[card], target) for card in candidates}
                for early in (card for card in candidates if dates[card] == 1):
                    missed_last = not seen[seen.card_id == early].iloc[-1].y
                    next_review = continuation(model, history, early, forecasts[early], t0, 1, target)
                    gap = next_review["remembered"]["interval"] - next_review["missed"]["interval"]
                    if gap < 2:
                        continue
                    rest = [card for card in candidates if labels[card][2] != labels[early][2]]
                    for middle in [card for card in rest if 2 <= dates[card] <= 10]:
                        for late in [card for card in rest if dates[card] > dates[middle] and labels[card][2] != labels[middle][2]]:
                            # Prefer a returning learner with a longer visible study history.
                            quality = (min(active_days, 30) + min(gap, 10) + 2 * missed_last + (dates[late] < DAYS) + (target == .8)
                                       + max(0, 1 - abs(dates[late] - 10) / 10))
                            if best is None or quality > best[0]:
                                best = (quality, history, labels, forecasts, [early, middle, late], target)
    if best is None:
        raise RuntimeError("No held-out Solar System replay matched the selection criteria")
    _, history, labels, forecasts, selected, target = best
    anchor = int(history.ts.max())
    cards = []
    for card in selected:
        title, question, answer = labels[card]
        trace = history[history.card_id == card]
        cards.append(dict(
            word=title, question=question, meaning=answer,
            practice=[dict(daysAgo=(anchor - int(row.ts)) // DAY_SECONDS, recalled=bool(row.y))
                      for row in trace.itertuples()],
            forecast=np.round(forecasts[card], 6).tolist(),
            plan=continuation(model, history, card, forecasts[card], t0, DEMO_REVIEWS_PER_CARD, target),
        ))
    daily = history.assign(days_ago=(anchor - history.ts) // DAY_SECONDS).groupby("days_ago").y.agg(["size", "sum"])
    activity = [
        dict(daysAgo=int(days_ago), remembered=int(row["sum"]), missed=int(row["size"] - row["sum"]))
        for days_ago, row in daily.iterrows()
    ]
    return cards, target, activity


def export_booster(model, path):
    """Serialize the numeric LightGBM trees for browser-side review scheduling."""
    def flatten(node, result):
        index = len(result)
        result.append(None)
        if "leaf_value" in node:
            result[index] = node["leaf_value"]
        else:
            assert node["decision_type"] == "<=" and node["missing_type"] in ("None", "NaN")
            left = flatten(node["left_child"], result)
            right = flatten(node["right_child"], result)
            result[index] = [node["split_feature"], node["threshold"], node["default_left"], left, right]
        return index

    dump = model.booster_.dump_model(num_iteration=model.best_iter_)
    trees = []
    for tree in dump["tree_info"]:
        flat = []
        flatten(tree["tree_structure"], flat)
        trees.append(flat)
    payload = dict(names=dump["feature_names"], trees=trees, fsrs=W.tolist(),
                   span=model.span, prior=model.agg_.prior)
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/tmp/axiom-karl-data/reviews.parquet"))
    fitted, holdout, lookup, t0 = build(parser.parse_args().source)
    print(f"Fitting on {fitted.user_id.nunique()} learners", flush=True)
    model = GBDT(**FROZEN_KWARGS).fit(fitted)
    cards, target, activity = select_case(model, holdout, lookup, t0)
    dest = Path(__file__).resolve().parents[1] / "src/app/dashboard/demo/replay.json"
    payload = dict(fitLearners=int(fitted.user_id.nunique()),
                   target=target, reviewLimit=DEMO_REVIEWS_PER_CARD, cards=cards, activity=activity)
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    export_booster(model, dest.parents[3] / "lib" / "review-booster.json")
    print(f"Exported {sum(day['remembered'] + day['missed'] for day in activity):,} recorded reviews over {len(activity)} active days.")
    print([(card["word"], card["plan"]["interval"],
            card["plan"]["missed"]["interval"], card["plan"]["remembered"]["interval"]) for card in cards])


if __name__ == "__main__":
    main()
