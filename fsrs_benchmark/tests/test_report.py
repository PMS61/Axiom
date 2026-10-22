"""Tests for the report generator. Numbers are recomputed directly with fsrs_benchmark.metrics."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fsrs_benchmark import metrics, report

REAL_RUN = Path("/home/amal/fyp/Axiom/results/timing_c300")


def _synthetic_preds(elapsed):
    rng = np.random.default_rng(0)
    n = len(elapsed)
    y = rng.integers(0, 2, n)
    base = {"event_id": np.arange(n, dtype=np.int64), "user_id": np.arange(n) % 3,
            "card_id": np.arange(n), "y": y, "elapsed_days": np.asarray(elapsed, np.int64),
            "day_offset": np.zeros(n, np.int64)}
    for m in ["fsrs_opt", "ours"]:
        base[m] = np.clip(rng.uniform(0.05, 0.95, n), 0.01, 0.99)
    return pd.DataFrame(base)


def test_gap_bins_use_integer_day_edges(tmp_path):
    elapsed = [1, 2, 3, 4, 14, 15, 60, 61, 500]
    preds = _synthetic_preds(elapsed)
    preds.to_parquet(tmp_path / "predictions.parquet", index=False)
    out = report.build_report(tmp_path)
    mt = pd.read_csv(out / "metrics.csv")
    counts = mt[(mt["model"] == "fsrs_opt") & (mt["scope"] == "gap")].set_index("bin")["n"].to_dict()
    assert counts == {"1d": 1, "2-3d": 2, "4-14d": 2, "15-60d": 2, "61d+": 2}


def test_synthetic_outputs_all_written(tmp_path):
    _synthetic_preds([1, 5, 20, 70, 2, 9]).to_parquet(tmp_path / "predictions.parquet", index=False)
    out = report.build_report(tmp_path)
    for name in ["metrics.csv", "paired.csv", "users.csv", "user_wins.csv",
                 "reliability.png", "gapbins_logloss.png", "report.md"]:
        assert (out / name).exists(), name
    text = (out / "report.md").read_text()
    assert "## Overall" in text and "## Paired differences" in text


@pytest.fixture(scope="module")
def real_report(tmp_path_factory):
    if not (REAL_RUN / "predictions.parquet").exists():
        pytest.skip("results/timing_c300 is not available")
    out = tmp_path_factory.mktemp("report")
    report.build_report(REAL_RUN, out)
    return out


def test_real_run_writes_all_files(real_report):
    for name in ["metrics.csv", "paired.csv", "users.csv", "user_wins.csv",
                 "reliability.png", "gapbins_logloss.png", "report.md"]:
        assert (real_report / name).exists(), name


def test_metrics_csv_matches_direct_computation(real_report):
    preds = pd.read_parquet(REAL_RUN / "predictions.parquet")
    mt = pd.read_csv(real_report / "metrics.csv")

    row = mt[(mt["model"] == "ours") & (mt["scope"] == "overall")].iloc[0]
    assert row["log_loss"] == pytest.approx(metrics.log_loss(preds["y"], preds["ours"]), abs=1e-9)
    assert row["ece"] == pytest.approx(metrics.ece(preds["y"], preds["ours"]), abs=1e-9)

    sub = preds[(preds["elapsed_days"] >= 4) & (preds["elapsed_days"] <= 14)]
    row = mt[(mt["model"] == "fsrs_opt") & (mt["scope"] == "gap") & (mt["bin"] == "4-14d")].iloc[0]
    assert row["n"] == len(sub)
    assert row["brier"] == pytest.approx(metrics.brier(sub["y"], sub["fsrs_opt"]), abs=1e-9)


def test_paired_difference_matches_direct_computation(real_report):
    preds = pd.read_parquet(REAL_RUN / "predictions.parquet")
    pt = pd.read_csv(real_report / "paired.csv")
    row = pt[(pt["reference"] == "fsrs_opt") & (pt["model"] == "ours") & (pt["metric"] == "log_loss")].iloc[0]
    expected = metrics.log_loss(preds["y"], preds["ours"]) - metrics.log_loss(preds["y"], preds["fsrs_opt"])
    assert row["diff"] == pytest.approx(expected, abs=1e-9)
    assert row["lo"] <= row["diff"] <= row["hi"]


def test_report_renders_mcmc_diagnostics_when_present(tmp_path):
    preds = _synthetic_preds([1, 5, 20, 70, 2, 9])
    preds.to_parquet(tmp_path / "predictions.parquet", index=False)
    diag = {"lam": {"mean": 0.4, "lo": 0.2, "hi": 0.6, "ess": 123.0, "rhat": 1.01},
            "sigma2": {"mean": 0.9, "lo": 0.5, "hi": 1.4, "ess": 98.0, "rhat": 1.02},
            "n_samples": 50, "mh_acc_rate": 0.31}
    (tmp_path / "diagnostics_ours.json").write_text(json.dumps(diag))
    out = report.build_report(tmp_path)
    text = (out / "report.md").read_text()
    assert "## MCMC diagnostics" in text
    assert "split-R-hat computed on a single chain" in text
    assert "| lam | 0.400 | [0.200, 0.600] | 123 | 1.010 |" in text
    assert "Ours (+κ)" not in text.split("## MCMC diagnostics")[1].split("## Figures")[0]


def _run_info(models):
    return {"counts": {"users": 3, "cards": 6, "events": 6, "train": 3, "test": 3, "scored": 6},
            "users_zero_train": [], "git": {"commit": "abc", "dirty": False},
            "args": {"n_users": 3, "max_cards": 6, "seed": 0, "iters": 10, "burn": 5},
            "models": models}


def test_resumed_model_without_timing_shows_na_and_note(tmp_path):
    _synthetic_preds([1, 5, 20, 70, 2, 9]).to_parquet(tmp_path / "predictions.parquet", index=False)
    info = _run_info({"fsrs_opt": {"resumed": True},
                      "ours": {"fit_seconds": 12.5, "predict_seconds": 0.75, "resumed": False}})
    (tmp_path / "run_info.json").write_text(json.dumps(info))
    out = report.build_report(tmp_path)
    text = (out / "report.md").read_text()
    assert "| FSRS-6 (optimised) | n/a | n/a |" in text
    assert "| Ours (dynamic, learned λ) | 12.500 | 0.750 |" in text
    assert ("n/a: model reused from an interrupted earlier run via --resume; "
            "its timing was not recorded.") in text
    assert "fallback users (defaults kept): n/a" in text


def test_fully_timed_run_has_no_na_note(tmp_path):
    _synthetic_preds([1, 5, 20, 70, 2, 9]).to_parquet(tmp_path / "predictions.parquet", index=False)
    info = _run_info({"fsrs_opt": {"fit_seconds": 1.0, "predict_seconds": 2.0},
                      "ours": {"fit_seconds": 3.0, "predict_seconds": 4.0}})
    (tmp_path / "run_info.json").write_text(json.dumps(info))
    out = report.build_report(tmp_path)
    text = (out / "report.md").read_text()
    assert "| FSRS-6 (optimised) | 1.000 | 2.000 |" in text
    assert "timing was not recorded" not in text
