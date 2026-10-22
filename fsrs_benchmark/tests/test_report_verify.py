"""Verifier checks for Task 3.2 (fsrs_benchmark/report.py).

Reads the already generated outputs in results/timing_c300 (the coordinator's input, not
rebuilt here, to keep the CPU light while the full run is going) and checks them against
fsrs_benchmark.metrics. Synthetic tests build small reports in a temp folder.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fsrs_benchmark import metrics, report
from fsrs_benchmark.tests.test_report import _synthetic_preds

RUN = Path("/home/amal/fyp/Axiom/results/timing_c300")
OUTPUTS = ["metrics.csv", "paired.csv", "users.csv", "user_wins.csv", "reliability.png",
           "gapbins_logloss.png", "report.md"]


def _need_run():
    missing = [n for n in ["predictions.parquet", *OUTPUTS] if not (RUN / n).exists()]
    if missing:
        pytest.skip(f"results/timing_c300 is missing {missing}")


# ---- Gap bins: integer-day edges, exact partition ----

@pytest.mark.parametrize("day,label", [(1, "1d"), (2, "2-3d"), (3, "2-3d"), (4, "4-14d"),
                                       (14, "4-14d"), (15, "15-60d"), (60, "15-60d"),
                                       (61, "61d+"), (400, "61d+")])
def test_each_integer_gap_lands_in_its_bin(day, label):
    preds = _synthetic_preds([day])
    mt = report.metrics_table(preds, ["fsrs_opt"])
    gap = mt[(mt["scope"] == "gap") & (mt["n"] > 0)]
    assert list(gap["bin"]) == [label]


def test_gap_bins_partition_every_scored_event():
    rng = np.random.default_rng(1)
    elapsed = rng.integers(1, 400, 2000)
    preds = _synthetic_preds(elapsed)
    mt = report.metrics_table(preds, ["fsrs_opt", "ours"])
    gap = mt[mt["scope"] == "gap"]
    for m in ["fsrs_opt", "ours"]:
        assert gap[gap["model"] == m]["n"].sum() == len(preds)


def test_real_run_gap_bins_sum_to_scored_events():
    _need_run()
    preds = pd.read_parquet(RUN / "predictions.parquet")
    mt = pd.read_csv(RUN / "metrics.csv")
    gap = mt[(mt["scope"] == "gap") & (mt["model"] == "fsrs_opt")]
    assert gap["n"].sum() == len(preds) == 2920
    assert preds["elapsed_days"].min() >= 1


# ---- Paired table: model minus reference, bootstrap settings, CI flag ----

def test_paired_diff_sign_is_model_minus_reference():
    preds = _synthetic_preds(np.ones(300, int) * 2)
    pt = report.paired_table(preds, ["fsrs_opt", "ours"])
    row = pt[(pt["reference"] == "fsrs_opt") & (pt["model"] == "ours") & (pt["metric"] == "log_loss")].iloc[0]
    expected = metrics.log_loss(preds["y"], preds["ours"]) - metrics.log_loss(preds["y"], preds["fsrs_opt"])
    assert row["diff"] == pytest.approx(expected, abs=1e-12)


def test_paired_uses_1000_boot_seed_0():
    preds = _synthetic_preds(np.ones(300, int) * 2)
    pt = report.paired_table(preds, ["fsrs_opt", "ours"])
    row = pt[(pt["reference"] == "fsrs_opt") & (pt["model"] == "ours") & (pt["metric"] == "brier")].iloc[0]
    direct = metrics.paired_bootstrap(preds["user_id"].to_numpy(), preds["y"].to_numpy(),
                                      preds["ours"].to_numpy(), preds["fsrs_opt"].to_numpy(),
                                      "brier", n_boot=1000, seed=0)
    assert row["lo"] == pytest.approx(direct["lo"], abs=1e-12)
    assert row["hi"] == pytest.approx(direct["hi"], abs=1e-12)


def test_ci_excludes_zero_flag_matches_interval():
    _need_run()
    pt = pd.read_csv(RUN / "paired.csv")
    flag = (pt["lo"] > 0) | (pt["hi"] < 0)
    assert (pt["ci_excludes_zero"].astype(bool) == flag).all()


def test_report_marks_exactly_the_excluding_rows():
    _need_run()
    pt = pd.read_csv(RUN / "paired.csv")
    text = (RUN / "report.md").read_text()
    table = text.split("## Paired differences")[1].split("## Per-user wins")[0]
    rows = [l for l in table.splitlines() if l.startswith("| ") and "95% CI" not in l and "---" not in l]
    assert len(rows) == len(pt)
    starred = [l for l in rows if l.rstrip().endswith("*  |") or re.search(r"\] \*\s*\|$", l)]
    assert len(starred) == int(pt["ci_excludes_zero"].sum())


# ---- user wins ----

def test_real_run_user_wins_match_direct_count():
    _need_run()
    preds = pd.read_parquet(RUN / "predictions.parquet")
    uw = pd.read_csv(RUN / "user_wins.csv")
    row = uw[(uw["model"] == "ours") & (uw["reference"] == "fsrs_opt")].iloc[0]
    wins = sum(metrics.log_loss(s["y"], s["ours"]) < metrics.log_loss(s["y"], s["fsrs_opt"])
               for _, s in preds.groupby("user_id"))
    assert row["users_lower_log_loss"] == wins
    assert row["users"] == preds["user_id"].nunique()
    assert row["share"] == pytest.approx(wins / preds["user_id"].nunique())


# ---- Numbers in metrics.csv ----

def test_real_run_overall_and_bin_numbers_match_metrics():
    _need_run()
    preds = pd.read_parquet(RUN / "predictions.parquet")
    mt = pd.read_csv(RUN / "metrics.csv")
    row = mt[(mt["model"] == "fsrs_opt") & (mt["scope"] == "overall")].iloc[0]
    assert row["auc"] == pytest.approx(metrics.auc(preds["y"], preds["fsrs_opt"]), abs=1e-9)
    assert row["mean_bias"] == pytest.approx(float(np.mean(preds["fsrs_opt"]) - np.mean(preds["y"])), abs=1e-9)
    sub = preds[(preds["elapsed_days"] >= 61)]
    row = mt[(mt["model"] == "ours") & (mt["scope"] == "gap") & (mt["bin"] == "61d+")].iloc[0]
    assert row["log_loss"] == pytest.approx(metrics.log_loss(sub["y"], sub["ours"]), abs=1e-9)


# ---- Bold-best logic (on the table renderer) ----

def test_bold_min_max_and_closest_zero():
    rows = [["A", 0.2, 0.9, -0.3], ["B", 0.5, 0.8, 0.1]]
    text = report._table(["m", "loss", "auc", "bias"], rows,
                         bold_cols={1: "min", 2: "max", 3: "closest0"})
    assert "**0.200**" in text and "**0.900**" in text and "**0.100**" in text
    assert "**0.500**" not in text and "**0.800**" not in text and "**-0.300**" not in text


def test_bold_marks_every_tied_best_cell():
    """Two models tied for the lowest value are both best, so both should be bold."""
    text = report._table(["m", "loss"], [["A", 0.3], ["B", 0.3], ["C", 0.5]], bold_cols={1: "min"})
    assert text.count("**0.300**") == 2


def test_real_report_bold_is_on_the_best_overall_cell():
    _need_run()
    mt = pd.read_csv(RUN / "metrics.csv")
    text = (RUN / "report.md").read_text()
    overall = text.split("## Overall")[1].split("## Gap bins")[0]
    fsrs_opt_line = [l for l in overall.splitlines() if l.startswith("| FSRS-6 (optimised)")][0]
    best_ll = mt[(mt["scope"] == "overall")]["log_loss"].min()
    assert f"**{best_ll:.3f}**" in fsrs_opt_line


# ---- report.md content ----

def test_report_states_sign_convention_and_flag_legend():
    _need_run()
    text = (RUN / "report.md").read_text()
    assert "model minus reference" in text
    assert "* CI excludes 0." in text
    assert "95% user-cluster bootstrap" in text


def test_report_contains_config_and_git_info():
    _need_run()
    text = (RUN / "report.md").read_text()
    assert "Git commit: " in text
    assert "Config: n_users=5, max_cards=300, seed=0, iters=200, burn=100" in text


def test_real_pngs_are_real_images():
    _need_run()
    import matplotlib.image as mpimg
    for name in ["reliability.png", "gapbins_logloss.png"]:
        img = mpimg.imread(RUN / name)
        assert img.shape[0] > 300 and img.shape[1] > 300, name


def test_report_without_reference_omits_paired_and_wins(tmp_path):
    preds = _synthetic_preds([1, 5, 20, 70, 2, 9]).drop(columns=["ours"])
    preds = preds.rename(columns={"fsrs_opt": "fsrs_default"})
    preds["hlr"] = preds["fsrs_default"]
    preds.to_parquet(tmp_path / "predictions.parquet", index=False)
    out = report.build_report(tmp_path)
    text = (out / "report.md").read_text()
    assert "No reference model in this run" in text
    # Intended behaviour: paired.csv and user_wins.csv always carry their header, with zero rows.
    for name in ["paired.csv", "user_wins.csv"]:
        df = pd.read_csv(out / name)
        assert df.empty and len(df.columns) > 0, name
