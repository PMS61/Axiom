"""Tests for paper/scripts/make_tables.py: run it on paper/data and check cells against the CSVs.

The tables are written to a temporary folder, so the committed paper/tables/ is not touched.
"""
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "paper" / "data"
sys.path.insert(0, str(REPO / "paper" / "scripts"))

import make_tables  # noqa: E402


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    out = tmp_path_factory.mktemp("tables")
    make_tables.write_all(DATA, out)
    return {name: (out / name).read_text() for name in ("overall.tex", "paired.tex", "seeds.tex")}


def _cells(text, label):
    """Cells of the table row whose first cell starts with label. Bold markup is removed."""
    line = next(ln for ln in text.splitlines() if ln.startswith(label + " &"))
    body = line.split("\\\\")[0]
    cells = [c.strip() for c in body.split("&")]
    return [re.sub(r"^\\textbf\{(.*)\}$", r"\1", c) for c in cells]


def _csv_value(name, model, column):
    df = pd.read_csv(DATA / name).set_index("model")
    return float(df.loc[model, column])


def test_overall_cells_match_csv(tables):
    # Const. global, Anki log loss: round to 3 dp.
    cells = _cells(tables["overall.tex"], "Const. global (floor)")
    assert cells[1] == f"{round(_csv_value('anki_overall.csv', 'const_global', 'log_loss'), 3):.3f}"
    # FSRS-6 opt., Duolingo ECE.
    cells = _cells(tables["overall.tex"], "FSRS-6 opt.")
    assert cells[8] == f"{round(_csv_value('duolingo_overall.csv', 'fsrs_opt', 'ece'), 3):.3f}"
    # GBDT, Duolingo log loss: the best non-constant value, so it is bold.
    cells = _cells(tables["overall.tex"], "GBDT")
    assert cells[5] == f"{round(_csv_value('duolingo_overall.csv', 'gbdt', 'log_loss'), 3):.3f}"
    assert "\\textbf{" + cells[5] + "}" in tables["overall.tex"]


def test_paired_cell_matches_csv(tables):
    # GBDT minus FSRS-6 opt. on Anki, log loss: the point estimate in the first diff column.
    cells = _cells(tables["paired.tex"], "GBDT")
    pt = pd.read_csv(DATA / "anki_paired.csv")
    row = pt[(pt["model"] == "gbdt") & (pt["reference"] == "fsrs_opt") & (pt["metric"] == "log_loss")].iloc[0]
    diff = round(float(row["diff"]), 3)
    assert diff < 0
    assert cells[1].startswith(f"$-$" + f"{abs(diff):.3f} [")


def test_seed_cell_matches_csv(tables):
    # FSRS-6 opt., seed 1 log loss.
    cells = _cells(tables["seeds.tex"], "FSRS-6 opt.")
    assert cells[2] == f"{round(_csv_value('anki_seeds.csv', 'fsrs_opt', 'log_loss_seed1'), 3):.3f}"
