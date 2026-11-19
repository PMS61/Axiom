"""Export the Axiom review replay. Run from the repo root with the report's Python dependencies."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fsrs_benchmark import duolingo, split
from fsrs_benchmark.models.gbdt import FROZEN_KWARGS, GBDT

DAYS = 21
DEMO_REVIEWS_PER_CARD = 3
FIT_USERS = 6000
TARGET = 0.90
DAY_SECONDS = duolingo.SECONDS_PER_DAY
SPANISH = dict(
    agua="water", pan="bread", leche="milk", manzana="apple", casa="house",
    gato="cat", perro="dog", caballo="horse", pájaro="bird", pez="fish",
    libro="book", escuela="school", mesa="table", silla="chair", puerta="door",
    ventana="window", camisa="shirt", zapato="shoe", zapatos="shoes",
    naranja="orange", queso="cheese", arroz="rice", pescado="fish", pollo="chicken",
    sopa="soup", café="coffee", vino="wine", mantequilla="butter", azúcar="sugar",
    cebolla="onion", tomate="tomato", carne="meat", fruta="fruit", huevo="egg",
    plato="plate", cuchara="spoon", botella="bottle", taza="cup", limón="lemon",
    amarillo="yellow", verde="green", azul="blue", rojo="red", blanco="white",
    familia="family", hermano="brother", hermana="sister", padre="father",
    madre="mother", mujer="woman", hombre="man", niña="girl", niño="boy",
    trabajo="work", comida="food", vestido="dress", pantalones="trousers",
    ciudad="city", tren="train", museo="museum", parque="park", playa="beach",
    reloj="clock", llave="key", flor="flower", árbol="tree", jardín="garden",
    montaña="mountain", río="river", luna="moon", sol="sun", estrella="star",
)


def candidate_labels(source, candidates, counts):
    cols = ["user_id", "lexeme_id", "lexeme_string", "learning_language", "ui_language"]
    taken = []
    for chunk in pd.read_csv(source, chunksize=duolingo.DEFAULT_CHUNKSIZE, usecols=cols):
        taken.append(chunk[chunk.user_id.isin(candidates)].drop_duplicates())
    lexemes = pd.concat(taken).drop_duplicates(["user_id", "lexeme_id"])
    lexemes = lexemes.sort_values(["user_id", "lexeme_id"])
    lexemes["card_id"] = lexemes.groupby("user_id").cumcount()
    lexemes["word"] = lexemes.lexeme_string.str.split("/").str[0]
    lexemes["learner"] = pd.Index(counts.user_id).get_indexer(lexemes.user_id)
    lexemes = lexemes[
        (lexemes.learning_language == "es")
        & (lexemes.ui_language == "en")
        & lexemes.word.isin(SPANISH)
    ]
    return lexemes.groupby("learner").apply(
        lambda group: dict(zip(group.card_id, group.word)), include_groups=False
    ).to_dict()


def future_review(history, last, ts, t0, recalled):
    future = last.copy().reset_index(drop=True)
    future["ts"] = ts
    future["day_offset"] = (ts - t0) // DAY_SECONDS
    future["elapsed_days"] = future.day_offset - last.day_offset.to_numpy()
    future["dt_days"] = (ts - last.ts.to_numpy()) / DAY_SECONDS
    future["review_idx"] = int(history.review_idx.max()) + 1 + np.arange(len(last))
    future["event_id"] = future.user_id * split.EVENT_ID_STRIDE + future.review_idx
    return future.assign(
        is_first=False, is_train=False, scored=True, y=int(recalled), rating=3 if recalled else 1
    )


def continuation(model, history, card, scores, t0, depth, target=TARGET):
    """Forecast both possible answers when this word is reviewed at its due date."""
    days = interval(scores, target)
    node = dict(interval=days, recall=float(round(scores[days - 1], 6)))
    if depth == 0:
        return node

    last = history[history.card_id == card].tail(1)
    due = int(history.ts.max()) + node["interval"] * DAY_SECONDS
    for option, recalled in (("missed", False), ("remembered", True)):
        answer = future_review(history, last, due, t0, recalled)
        next_history = pd.concat([history, answer], ignore_index=True)
        forecast = project(model, next_history, [card], t0)[card]
        node[option] = continuation(model, next_history, card, forecast, t0, depth - 1, target)
    return node


def project(model, history, cards, t0):
    """Give each candidate day the same real prefix, with no intervening practice."""
    anchor = int(history.ts.max())
    last = history[history.card_id.isin(cards)].groupby("card_id").tail(1)
    batches = []
    for delay in range(1, DAYS + 1):
        future = future_review(history, last, anchor + delay * DAY_SECONDS, t0, recalled=False)
        batches.append(pd.concat([history.assign(scored=False), future], ignore_index=True))
    # Different dates are independent scenarios. Distinct temporary user keys keep their
    # history kernels apart in one batch. The frozen model excludes the user-ID prior.
    for idx, batch in enumerate(batches):
        batch["user_id"] = idx
        batch["event_id"] = idx * split.EVENT_ID_STRIDE + batch.review_idx
    events = pd.concat(batches, ignore_index=True)
    scores = model.predict(events).to_numpy().reshape(DAYS, len(last)).T
    return {int(card): row for card, row in zip(last.card_id, scores)}


def interval(scores, target=TARGET):
    return next((max(1, idx) for idx, p in enumerate(scores) if p < target), DAYS)


def pick_case(model, example_events, labels, t0):
    best = None
    for uid, history in example_events.groupby("user_id"):
        words = labels.get(uid, {})
        if not words:
            continue
        for anchor in sorted(history.ts.unique())[-12:]:
            prefix = history[history.ts <= anchor]
            known = prefix[prefix.card_id.isin(words)]
            latest = known.groupby("card_id").agg(sessions=("ts", "size"), last=("ts", "max"))
            cards = latest.index[(latest.sessions >= 3) & (latest["last"] >= anchor - 5 * DAY_SECONDS)].tolist()
            if len({words[card] for card in cards}) < 3:
                continue
            scores = project(model, prefix, cards, t0)
            due = {card: interval(scores[card]) for card in cards}
            early_cards = [card for card in cards if due[card] <= 4 and scores[card][6] < TARGET]
            late_cards = [card for card in cards if due[card] >= 10 and scores[card][6] >= TARGET]
            for early in early_cards:
                for late in late_cards:
                    if words[early] == words[late]:
                        continue
                    alternatives = [card for card in cards if words[card] not in {words[early], words[late]}
                                    and due[early] < due[card] < due[late]]
                    if not alternatives:
                        continue
                    mid = min(alternatives, key=lambda card: abs(due[card] - 7))
                    missed = 1 - known.loc[known.card_id == early, "y"].iloc[-1]
                    streak = known.loc[known.card_id == late, "y"].tail(3).all()
                    order = [early, mid, late]
                    relaxed = [interval(scores[card], 0.88) for card in order]
                    strict = [interval(scores[card], 0.92) for card in order]
                    quality = (
                        scores[late][6] - scores[early][6]
                        + 0.25 * missed + 0.1 * streak
                        + 0.35 * (due[late] < DAYS)
                        + 0.15 * (len(set(relaxed)) > 1)
                        + 0.15 * (len(set(strict)) > 1)
                        + 0.1 * (relaxed[0] > due[early])
                        + 0.1 * (strict[2] < due[late])
                        - 0.02 * abs(due[mid] - 7)
                    )
                    if best is None or quality > best[0]:
                        best = (quality, [early, mid, late], scores, prefix, words)
    if best is None:
        raise RuntimeError("No three-card contrast found in the sampled histories")
    _, order, scores, prefix, words = best
    anchor = int(prefix.ts.max())
    cards = []
    for card in order:
        trace = prefix[prefix.card_id == card]
        cards.append({
            "word": words[card],
            "meaning": SPANISH[words[card]],
            "practice": [
                {"daysAgo": (anchor - int(row.ts)) // DAY_SECONDS, "recalled": bool(row.y)}
                for row in trace.itertuples()
            ],
            "forecast": np.round(scores[card], 6).tolist(),
            "plan": continuation(model, prefix, card, scores[card], t0, DEMO_REVIEWS_PER_CARD),
        })
    return cards, len(prefix)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(
        "data/duolingo/settles.acl16.learning_traces.13m.csv.gz"
    ))
    args = parser.parse_args()
    source = args.source
    counts, t0 = duolingo.scan(source, source.parent / "duolingo_counts.parquet")
    pool = counts[counts.n_rows.between(650, 3200)]
    candidates = duolingo.sample_users(pool, 96, seed=17, min_rows=1)
    fit_users = duolingo.sample_users(
        counts[~counts.user_id.isin(candidates)], FIT_USERS, seed=0, min_rows=1
    )

    print("Reading practice logs", flush=True)
    raw = duolingo.load_users(source, fit_users + candidates, counts, t0,
                              with_lexeme=True, with_ts=True)
    candidate_ids = pd.Index(counts.user_id).get_indexer(candidates)
    is_replay = raw.user_id.isin(candidate_ids)
    example_events = split.make_split(raw[is_replay], max_cards=10**9)
    fit_events = split.make_split(raw[~is_replay], max_cards=10**9)
    del raw
    print(f"Fitting report configuration on {FIT_USERS:,} sampled learners", flush=True)
    model = GBDT(**FROZEN_KWARGS).fit(fit_events)
    print("Projecting review dates from real practice histories", flush=True)
    labels = candidate_labels(source, candidates, counts)
    cards, history_size = pick_case(model, example_events, labels, t0)
    dest = Path(__file__).resolve().parents[1] / "src/app/learn/replay.json"
    payload = dict(fitLearners=FIT_USERS, historySize=history_size, target=TARGET,
                   reviewLimit=DEMO_REVIEWS_PER_CARD, cards=cards)
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(dest)


if __name__ == "__main__":
    main()
