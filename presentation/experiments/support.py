"""Terminal output shared by the presentation experiments."""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def table(headers, rows):
    rows = [tuple(map(str, row)) for row in rows]
    widths = [max(map(len, column)) for column in zip(headers, *rows)]

    def line(row):
        return "  ".join(value.ljust(width) for value, width in zip(row, widths)).rstrip()

    print(line(headers))
    print(line(["-" * width for width in widths]))
    for row in rows:
        print(line(row))


def scores(rows):
    table(
        ("Model / variant", "Reviews", "Log loss ↓", "Brier ↓", "AUC ↑", "ECE ↓"),
        (
            (row["model"], f"{int(row['n']):,}",
             *(f"{float(row[key]):.5f}" if row[key] else "—"
               for key in ("log_loss", "brier", "auc", "ece")))
            for row in sorted(rows, key=lambda row: float(row["log_loss"]))
        ),
    )


def show_saved(title, path, *, overall=False):
    print(f"{title}\nSaved results from {path.relative_to(ROOT)}\n")
    rows = load_rows(path)
    scores([row for row in rows if row["scope"] == "overall"] if overall else rows)


def score_predictions(events, predictions):
    from fsrs_benchmark.metrics import summarize

    actual = events.loc[events["scored"]].set_index("event_id")["y"]
    if not predictions.index.is_unique or set(actual.index) != set(predictions.index):
        raise ValueError("The model must predict every scored review exactly once.")
    return summarize(actual.loc[predictions.index].to_numpy(), predictions.to_numpy())
