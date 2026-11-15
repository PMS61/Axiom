"""Tests for paper_export.py, using small synthetic report folders (metrics/paired/user_wins CSVs)."""
import json

import numpy as np
import pandas as pd
import pytest

from fsrs_benchmark import paper_export as pe
from fsrs_benchmark import report

MODELS = ["const_global", "const_user", "const_user_smooth", "fsrs_opt", "ours", "ours_v3", "gbdt"]
REFS = ["fsrs_opt", "ours", "ours_gpu", "const_user_smooth"]


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
    for ref in [r for r in REFS if r in models]:
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
    wins = [{"reference": ref, "model": m, "users": 40, "users_lower_log_loss": 25 if m == "gbdt" else 10,
             "share": (25 if m == "gbdt" else 10) / 40}
            for ref in [r for r in REFS if r in models] for m in models if m != ref]
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
    duo_models = ["const_global", "const_user", "const_user_smooth", "fsrs_opt", "ours_gpu", "ours_v3", "gbdt"]
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
    models = ["const_global", "const_user", "const_user_smooth", "fsrs_opt", "ours_gpu", "ours_v3", "gbdt"]
    run = _make_run(tmp_path / "results" / "paper_duolingo", models=models)
    text = pe.export(run, "duolingo", tmp_path / "paper" / "data").read_text()
    assert "Ours (GPU port) minus FSRS-6 (optimised), log loss" in text
    assert "Ours (GPU port) minus Constant (per user, smoothed), log loss" in text


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


def test_smoothed_constant_lines_for_v1_gbdt_fsrs_and_v3(tmp_path):
    run = _make_run(tmp_path / "results" / "paper_anki")
    text = pe.export(run, "anki", tmp_path / "paper" / "data").read_text()
    section = text.split("Paired log-loss differences vs Constant (per user, smoothed)", 1)[1]
    section = section.split("Per-user wins on log loss against Constant (per user, smoothed)", 1)[0]
    for name in ["Ours (dynamic, learned λ)", "GBDT (LightGBM on history features)",
                 "FSRS-6 (optimised)", "Ours v3 (v2 + user-scale penalty, lag, card effect)"]:
        assert f"- {name} minus Constant (per user, smoothed), log loss: diff " in section, name
    # GBDT fixture diff -0.0123456 -> -0.0123, CI [-0.02, -0.001]
    assert "GBDT (LightGBM on history features) minus Constant (per user, smoothed), log loss: diff -0.0123, " \
           "95% CI [-0.0200, -0.0010], CI excludes 0 (source: " in text
    wins = text.split("Per-user wins on log loss against Constant (per user, smoothed):", 1)[1]
    assert "- GBDT (LightGBM on history features): lower log loss for 25 of 40 users, share 0.6250" in wins
    assert "(source: " in wins


def test_smoothed_block_missing_row_raises(tmp_path):
    run = _make_run(tmp_path / "results" / "paper_anki")
    pt = pd.read_csv(run / "paired.csv")
    pt = pt[~((pt["reference"] == "const_user_smooth") & (pt["model"] == "ours_v3"))]
    pt.to_csv(run / "paired.csv", index=False)
    with pytest.raises(ValueError, match="ours_v3"):
        pe.export(run, "anki", tmp_path / "paper" / "data")


# ---- robustness mode (seed runs) ----

SEED_MODELS = ["const_global", "const_user", "const_user_smooth", "fsrs_opt", "ours_v3", "gbdt"]


def _seed_run(folder, seed, base, gbdt_diff, models=SEED_MODELS):
    """A synthetic seed run: report CSVs with known log losses plus run_info.json with the seed."""
    run = _make_run(folder, models=models, base=base)
    (run / "run_info.json").write_text(json.dumps({"args": {"seed": seed}}))
    pt = pd.read_csv(run / "paired.csv")
    mask = (pt["model"] == "gbdt") & (pt["reference"] == "fsrs_opt") & (pt["metric"] == "log_loss")
    pt.loc[mask, "diff"] = gbdt_diff
    pt.loc[mask, "lo"] = gbdt_diff - 0.01
    pt.loc[mask, "hi"] = gbdt_diff + 0.01
    pt.loc[mask, "ci_excludes_zero"] = True
    pt.to_csv(run / "paired.csv", index=False)
    return run


def _three_seeds(tmp_path):
    base0 = {"const_global": 0.45, "const_user": 0.43, "const_user_smooth": 0.43, "fsrs_opt": 0.38,
             "ours": 0.42, "ours_v3": 0.378, "gbdt": 0.358}
    base1 = {"const_global": 0.46, "const_user": 0.44, "const_user_smooth": 0.44, "fsrs_opt": 0.39,
             "ours_v3": 0.37, "gbdt": 0.35}
    base2 = {"const_global": 0.47, "const_user": 0.42, "const_user_smooth": 0.42, "fsrs_opt": 0.40,
             "ours_v3": 0.38, "gbdt": 0.36}
    seed0 = _make_run(tmp_path / "results" / "full", base=base0)
    (seed0 / "run_info.json").write_text(json.dumps({"args": {"seed": 0}}))
    pt = pd.read_csv(seed0 / "paired.csv")
    mask = (pt["model"] == "gbdt") & (pt["reference"] == "fsrs_opt") & (pt["metric"] == "log_loss")
    pt.loc[mask, ["diff", "lo", "hi"]] = [-0.021, -0.028, -0.014]
    pt.loc[mask, "ci_excludes_zero"] = True
    pt.to_csv(seed0 / "paired.csv", index=False)
    seed1 = _seed_run(tmp_path / "results" / "seed1", 1, base1, gbdt_diff=-0.02, models=SEED_MODELS)
    seed2 = _seed_run(tmp_path / "results" / "seed2", 2, base2, gbdt_diff=-0.03, models=SEED_MODELS)
    return [seed0, seed1, seed2]


def test_robustness_table_keeps_only_models_in_every_run(tmp_path):
    runs = _three_seeds(tmp_path)
    table, _, seeds = pe.robustness_tables(runs)
    assert seeds == [0, 1, 2]
    assert "ours" not in set(table["model"])  # v1 exists only in the final run
    assert set(table["model"]) == set(SEED_MODELS)


def test_robustness_mean_min_max_range_by_hand(tmp_path):
    runs = _three_seeds(tmp_path)
    table, _, _ = pe.robustness_tables(runs)
    row = table[table["model"] == "gbdt"].iloc[0]
    assert row["log_loss_seed0"] == 0.358 and row["log_loss_seed1"] == 0.35 and row["log_loss_seed2"] == 0.36
    assert row["mean"] == round((0.358 + 0.35 + 0.36) / 3, 4)
    assert row["min"] == 0.35 and row["max"] == 0.36  # seed 1 is the lowest, seed 2 the highest
    assert row["range"] == round(0.36 - 0.35, 4)


def test_robustness_gbdt_diff_per_seed(tmp_path):
    runs = _three_seeds(tmp_path)
    _, diffs, _ = pe.robustness_tables(runs)
    got = dict(zip(diffs["seed"], diffs["diff"]))
    assert got == {0: -0.021, 1: -0.02, 2: -0.03}
    assert diffs.loc[diffs["seed"] == 1, "lo"].iloc[0] == pytest.approx(-0.03)
    assert diffs["ci_excludes_zero"].all()


def test_robustness_writes_csv_and_facts_section_with_sources(tmp_path):
    runs = _three_seeds(tmp_path)
    out = tmp_path / "paper" / "data"
    facts = pe.export_robustness(runs, "anki_seeds", out)
    assert (out / "anki_seeds.csv").exists() and (out / "anki_seeds_gbdt_diff.csv").exists()
    text = facts.read_text()
    section = text.split("<!-- facts:anki_seeds:start -->", 1)[1].split("<!-- facts:anki_seeds:end -->", 1)[0]
    assert "## Anki robustness: seeds 0, 1, 2" in section
    assert "- GBDT (LightGBM on history features): seed 0 0.3580, seed 1 0.3500, seed 2 0.3600" in section
    assert "- seed 1: diff -0.0200, 95% CI [-0.0300, -0.0100], CI excludes 0" in section
    assert "Ours (dynamic" not in section
    bullets = [l for l in section.splitlines() if l.startswith("- ")]
    assert bullets and all("(source: " in l and l.endswith(")") for l in bullets)


def test_robustness_section_coexists_with_dataset_sections_and_rerun_replaces(tmp_path):
    runs = _three_seeds(tmp_path)
    out = tmp_path / "paper" / "data"
    single = _make_run(tmp_path / "results" / "paper_anki")
    pe.export(single, "anki", out)
    pe.export_robustness(runs, "anki_seeds", out)
    facts = pe.export_robustness(runs, "anki_seeds", out)
    text = facts.read_text()
    assert text.count("<!-- facts:anki_seeds:start -->") == 1
    assert text.count("## Anki (") == 1 and "## Anki robustness" in text


def test_robustness_rejects_duplicate_seeds_and_single_run(tmp_path):
    runs = _three_seeds(tmp_path)
    dup = _seed_run(tmp_path / "results" / "dup", 1, {"fsrs_opt": 0.4, "gbdt": 0.36, "ours_v3": 0.37,
                                                      "const_global": 0.4, "const_user": 0.4,
                                                      "const_user_smooth": 0.4}, gbdt_diff=-0.01)
    with pytest.raises(ValueError, match="duplicate seeds"):
        pe.robustness_tables([runs[0], runs[1], dup])
    with pytest.raises(ValueError, match="at least two"):
        pe.robustness_tables([runs[0]])


def test_robustness_missing_gbdt_diff_raises(tmp_path):
    runs = _three_seeds(tmp_path)
    pt = pd.read_csv(runs[2] / "paired.csv")
    pt = pt[~((pt["model"] == "gbdt") & (pt["reference"] == "fsrs_opt") & (pt["metric"] == "log_loss"))]
    pt.to_csv(runs[2] / "paired.csv", index=False)
    with pytest.raises(ValueError, match="gbdt"):
        pe.robustness_tables(runs)


def test_robustness_cli(tmp_path):
    runs = _three_seeds(tmp_path)
    out = tmp_path / "paper" / "data"
    argv = ["--robustness", *[str(r) for r in runs], "--name", "anki_seeds", "--out", str(out)]
    assert pe.main(argv) == 0
    assert (out / "anki_seeds.csv").exists()
    assert (tmp_path / "paper" / "FACTS.md").read_text().count("## Anki robustness") == 1


def test_single_run_cli_still_requires_run_dir(tmp_path):
    with pytest.raises(SystemExit):
        pe.main(["--name", "anki", "--out", str(tmp_path)])
