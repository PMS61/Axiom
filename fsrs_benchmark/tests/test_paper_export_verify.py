"""Verifier failure-case tests for paper_export.py (not covered by test_paper_export.py)."""
import numpy as np
import pandas as pd
import pytest

from fsrs_benchmark import paper_export as pe
from fsrs_benchmark.tests.test_paper_export import _make_run, MODELS


def test_unknown_dataset_name_raises(tmp_path):
    run = _make_run(tmp_path / "results" / "x")
    with pytest.raises(ValueError, match="--name"):
        pe.export(run, "nope", tmp_path / "paper" / "data")
    assert not (tmp_path / "paper" / "FACTS.md").exists()


def test_missing_constant_row_raises_and_writes_no_facts(tmp_path):
    run = _make_run(tmp_path / "results" / "paper_anki", models=[m for m in MODELS if m != "const_user_smooth"])
    with pytest.raises(ValueError, match="const_user_smooth"):
        pe.export(run, "anki", tmp_path / "paper" / "data")
    assert not (tmp_path / "paper" / "FACTS.md").exists()


def test_missing_reference_pair_raises(tmp_path):
    run = _make_run(tmp_path / "results" / "paper_anki")
    pt = pd.read_csv(run / "paired.csv")
    pt = pt[~((pt["reference"] == "fsrs_opt") & (pt["model"] == "ours") & (pt["metric"] == "ece"))]
    pt.to_csv(run / "paired.csv", index=False)
    with pytest.raises(ValueError, match="ours"):
        pe.export(run, "anki", tmp_path / "paper" / "data")


def test_nan_metric_prints_na_not_nan(tmp_path):
    run = _make_run(tmp_path / "results" / "paper_anki")
    mt = pd.read_csv(run / "metrics.csv")
    mt.loc[(mt["model"] == "gbdt") & (mt["scope"] == "overall"), "auc"] = np.nan
    mt.to_csv(run / "metrics.csv", index=False)
    text = pe.export(run, "anki", tmp_path / "paper" / "data").read_text()
    gbdt_line = [l for l in text.splitlines() if l.startswith("- GBDT (LightGBM on history features): log loss")][0]
    body = gbdt_line.split(" (source")[0]
    assert "AUC n/a" in body and "nan" not in body.lower()


def test_existing_facts_without_markers_is_kept_and_section_appended(tmp_path):
    run = _make_run(tmp_path / "results" / "paper_anki")
    facts = tmp_path / "paper" / "FACTS.md"
    facts.parent.mkdir(parents=True)
    facts.write_text("# Hand notes\n\nkeep me\n")
    pe.export(run, "anki", tmp_path / "paper" / "data")
    text = facts.read_text()
    assert text.startswith("# Hand notes") and "keep me" in text and "## Anki" in text


def test_rerun_is_idempotent(tmp_path):
    run = _make_run(tmp_path / "results" / "paper_anki")
    facts = pe.export(run, "anki", tmp_path / "paper" / "data")
    first = facts.read_text()
    pe.export(run, "anki", tmp_path / "paper" / "data")
    assert facts.read_text() == first
