"""Synthetic event tables and Duolingo-format csv files for the Phase 5 model and runner tests."""
import numpy as np
import pandas as pd

from fsrs_benchmark.data import LOAD_COLUMNS
from fsrs_benchmark.split import make_split

CSV_COLUMNS = [
    "p_recall", "timestamp", "delta", "user_id", "learning_language", "ui_language",
    "lexeme_id", "lexeme_string", "history_seen", "history_correct", "session_seen",
    "session_correct",
]
T = 1_600_000_000


def loader_frame(seed=0, n_users=3, n_cards=4, reviews=10):
    """A data.load_user-shaped frame: each card has a first review, then gaps of 1-5 days."""
    rng = np.random.default_rng(seed)
    frames = []
    for user in range(n_users):
        rows = []
        for card in range(n_cards):
            day = int(rng.integers(0, 10))
            for k in range(reviews):
                gap = 0 if k == 0 else int(rng.integers(1, 6))
                day += gap
                rating = int(rng.choice([1, 3]))
                rows.append(dict(
                    user_id=user, card_id=card, day_offset=day, rating=rating,
                    state=0 if k == 0 else 2, elapsed_days=-1 if k == 0 else gap,
                    raw_elapsed_days=gap, dt_days=np.nan if k == 0 else gap + 0.5,
                    y=int(rating > 1), is_first=k == 0,
                ))
        rows.sort(key=lambda r: (r["day_offset"], r["card_id"]))
        for review_idx, row in enumerate(rows):
            row["review_idx"] = review_idx
        frames.append(pd.DataFrame(rows))
    frame = pd.concat(frames, ignore_index=True)
    return frame[LOAD_COLUMNS].astype({"user_id": np.int64, "card_id": np.int64, "review_idx": np.int64,
                                       "day_offset": np.int64, "rating": np.int64, "state": np.int64,
                                       "elapsed_days": np.int64, "raw_elapsed_days": np.int64,
                                       "dt_days": float, "y": np.int64, "is_first": bool})


def synthetic_events(seed=0, n_users=3, n_cards=4, reviews=10):
    """The event table make_split builds from loader_frame (is_train, scored, event_id, ...)."""
    return make_split(loader_frame(seed, n_users, n_cards, reviews), max_cards=300, seed=0)


def write_duolingo_csv(path, seed=0, n_users=3, n_lexemes=4, reviews=10):
    """A gzipped csv in the real Duolingo column layout, with user ids and lexemes as strings."""
    rng = np.random.default_rng(seed)
    rows = []
    for user in range(n_users):
        for lex in range(n_lexemes):
            for _ in range(reviews):
                day = int(rng.integers(0, 60))
                rows.append(dict(
                    p_recall=round(float(rng.uniform(0.05, 0.99)), 3),
                    timestamp=T + day * 86400 + int(rng.integers(0, 86400)),
                    delta=int(rng.integers(0, 90)) * 86400, user_id=f"u{user}",
                    learning_language="de", ui_language="en", lexeme_id=f"lex{lex}",
                    lexeme_string="w", history_seen=1, history_correct=1, session_seen=1,
                    session_correct=1,
                ))
    pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(path, index=False, compression="gzip")
    return path
