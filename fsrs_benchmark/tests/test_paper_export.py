"""Tests for paper_export.py, using small synthetic report folders (metrics/paired/user_wins CSVs)."""
import numpy as np
import pandas as pd
import pytest

from fsrs_benchmark import paper_export as pe
from fsrs_benchmark import report

MODELS = ["const_global", "const_user", "const_user_smooth", "fsrs_opt", "ours", "gbdt"]


def _metric_row(model, scope, bin_label, base):
    return {"model": model, "scope": scope, "bin": bin_label, "n": 100 + len(model),
            "recall": 0.3, "log_loss": base, "brier": base / 2, "auc": 0.7,
            "ece": base / 10, "rmse_bins": 0.05, "mean_bias": 0.001}


def _make_run(folder, models=MODELS, base=None, with_gbdt=True):
    """Write metrics.csv, paired.csv and user_wins.csv as report.py does, with known numbers."""
    folder.mkdir(parents=True, exist_ok=True)
    base = base or {m: 0.30 + 0.01 * i for i, m in enumerate(models)}
    rows = [_metric_row(m, "overall", "", base[m]) for m in models]
    for label, _, _ in report.GAP_BINS:
        rows += [_metric_row(m, "gap", label, base[m] + 0.001) for m in models]
    pd.DataFrame(rows).to_csv(folder / "metrics.csv", index=False)
    paired = []
    for ref in ["fsrs_opt", "ours"]:
        for m in models:
            if m == ref:
                continue
            for metric in report.PAIR_METRICS:
                paired.append({"reference": ref, "model": m, "metric": metric,
                               "diff": -0.0123456 if m == "gbdt" else 0.5,
                               "lo": -0.02, "hi": -0.001 if m == "gbdt" else 0.6,
                               "ci_excludes_zero": m == "gbdt"})
    if not with_gbdt:
        paired = [r for r in paired if r["model"] != "gbdt"]
    pd.DataFrame(paired, columns=report.PAIRED_COLUMNS).to_csv(folder / "paired.csv", index=False)
    wins = [{"reference": "fsrs_opt", "model": m, "users": 40, "users_lower_log_loss": 25 if m == "gbdt" else 10,
             "share": (25 if m == "gbdt" else 10) / 40} for m in models if m != "fsrs_opt"]
    pd.DataFrame(wins, columns=report.WINS_COLUMNS).to_csv(folder / "user_wins.csv", index=False)
    return folder


def test_overall_table_has_display_names_and_rounding(tmp_path):
    run = _make_run(tmp_path / "run")
    mt = pd.read_csv(run / "metrics.csv")
    out = pe.overall_table(mt)
    assert list(out.columns[:3]) == ["display", "model", "n"]
    assert len(out) == len(MODELS)
    row = out[out["model"] == "gbdt"].iloc[0]
    assert row["display"] == report.DISPLAY["gbdt"]
    assert row["log_loss"] == round(0.30 + 0.01 * MODELS.index("gbdt"), 4)


def test_rounding_is_four_decimals(tmp_path):
    mt = pd.DataFrame([_metric_row("gbdt", "overall", "", 0.123456789)])
    out = pe.overall_table(mt)
    assert out["log_loss"].iloc[0] == 0.1235
    assert out["brier"].iloc[0] == round(0.123456789 / 2, 4)


def test_gaps_table_has_one_row_per_model_and_bin(tmp_path):
    mt = pd.read_csv(_make_run(tmp_path / "run") / "metrics.csv")
    out = pe.gaps_table(mt)
    assert len(out) == len(MODELS) * len(report.GAP_BINS)
    assert set(out["bin"]) == {label for label, _, _ in report.GAP_BINS}
    assert "display" in out.columns


def test_paired_and_wins_tables_carry_display_names(tmp_path):
    run = _make_run(tmp_path / "run")
    pt = pe.paired_table(pd.read_csv(run / "paired.csv"))
    gb = pt[(pt["model"] == "gbdt") & (pt["reference"] == "fsrs_opt") & (pt["metric"] == "log_loss")].iloc[0]
    assert gb["model_display"] == report.DISPLAY["gbdt"]
    assert gb["reference_display"] == report.DISPLAY["fsrs_opt"]
    assert gb["diff"] == -0.0123
    uw = pe.wins_table(pd.read_csv(run / "user_wins.csv"))
    assert {"reference_display", "model_display", "share"} <= set(uw.columns)
    assert uw[uw["model"] == "gbdt"]["share"].iloc[0] == 0.625


def test_export_writes_four_csvs_and_facts(tmp_path):
    run = _make_run(tmp_path / "results" / "paper_anki")
    out_dir = tmp_path / "paper" / "data"
    facts = pe.export(run, "anki", out_dir)
    for suffix in ["overall", "gaps", "paired", "wins"]:
        assert (out_dir / f"anki_{suffix}.csv").exists()
    assert facts == tmp_path / "paper" / "FACTS.md"
    text = facts.read_text()
    assert "## Anki" in text


def test_facts_lines_all_end_with_a_source(tmp_path):
    run = _make_run(tmp_path / "results" / "paper_anki")
    facts = pe.export(run, "anki", tmp_path / "paper" / "data")
    bullets = [line for line in facts.read_text().splitlines() if line.startswith("- ")]
    assert bullets, "no bullet lines written"
    for line in bullets:
        assert line.endswith(")") and "(source: " in line, line


def test_facts_has_constants_gbdt_and_v1_diffs(tmp_path):
    run = _make_run(tmp_path / "results" / "paper_anki")
    text = pe.export(run, "anki", tmp_path / "paper" / "data").read_text()
    for constant in ["Constant (global)", "Constant (per user)", "Constant (per user, smoothed)"]:
        assert f"- {constant}: log loss" in text
    # hand-checked: gbdt minus fsrs_opt, log loss diff -0.0123456 rounds to -0.0123, CI [-0.02, -0.001]
    assert "GBDT (LightGBM on history features) minus FSRS-6 (optimised), log loss: diff -0.0123, " \
           "95% CI [-0.0200, -0.0010], CI excludes 0 (source: " in text
    assert "Ours (dynamic, learned λ) minus FSRS-6 (optimised), log loss" in text
    assert "lower log loss for 25 of 40 users" in text


def test_facts_source_points_at_the_run_folder(tmp_path):
    run = _make_run(tmp_path / "results" / "paper_anki")
    text = pe.export(run, "anki", tmp_path / "paper" / "data").read_text()
    assert f"(source: {run.as_posix()}/metrics.csv)" in text
    assert f"(source: {run.as_posix()}/paired.csv)" in text


def test_second_dataset_is_added_and_rerun_replaces_not_duplicates(tmp_path):
    out_dir = tmp_path / "paper" / "data"
    anki = _make_run(tmp_path / "results" / "paper_anki")
    duo_models = ["const_global", "const_user", "const_user_smooth", "fsrs_opt", "ours_gpu", "gbdt"]
    duo = _make_run(tmp_path / "results" / "paper_duolingo", models=duo_models,
                    base={m: 0.25 + 0.01 * i for i, m in enumerate(duo_models)})
    pe.export(anki, "anki", out_dir)
    facts = pe.export(duo, "duolingo", out_dir)
    pe.export(anki, "anki", out_dir)  # re-run: must replace, not add a second copy
    text = facts.read_text()
    assert text.count("## Anki") == 1
    assert text.count("## Duolingo") == 1
    assert text.count("<!-- facts:anki:start -->") == 1
    # the duolingo section must still hold its own numbers after the anki re-run
    assert "- FSRS-6 (optimised): log loss 0.2800" in text  # fixture: 0.25 + 0.01 * index 3


def test_missing_headline_row_raises_instead_of_writing_partial_facts(tmp_path):
    run = _make_run(tmp_path / "results" / "paper_anki", with_gbdt=False)
    with pytest.raises(ValueError, match="gbdt"):
        pe.export(run, "anki", tmp_path / "paper" / "data")
    assert not (tmp_path / "paper" / "FACTS.md").exists()


def test_duolingo_uses_ours_gpu_as_the_v1_model(tmp_path):
    models = ["const_global", "const_user", "const_user_smooth", "fsrs_opt", "ours_gpu", "gbdt"]
    run = _make_run(tmp_path / "results" / "paper_duolingo", models=models)
    text = pe.export(run, "duolingo", tmp_path / "paper" / "data").read_text()
    assert "Ours (GPU port) minus FSRS-6 (optimised), log loss" in text


def test_nan_values_are_written_as_n_a(tmp_path):
    assert pe._fmt(float("nan")) == "n/a"
    assert pe._fmt(None) == "n/a"
    assert pe._fmt(0.5) == "0.5000"


def test_unknown_name_raises(tmp_path):
    run = _make_run(tmp_path / "run")
    with pytest.raises(ValueError, match="--name"):
        pe.export(run, "fable", tmp_path / "paper" / "data")


def test_main_cli_runs(tmp_path, capsys):
    run = _make_run(tmp_path / "results" / "paper_anki")
    out = tmp_path / "paper" / "data"
    assert pe.main(["--run_dir", str(run), "--name", "anki", "--out", str(out)]) == 0
    assert (out / "anki_overall.csv").exists()
    assert (tmp_path / "paper" / "FACTS.md").exists()
