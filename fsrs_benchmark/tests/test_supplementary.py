"""Tests for fsrs_benchmark.supplementary: hand-checked numbers on a tiny table, and a check that
the committed full run's supplementary.json is reproduced from its local parquet files (skipped
when those files are not present)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fsrs_benchmark import supplementary as supp
from fsrs_benchmark.models.fsrs_model import MIN_INFORMATIVE_TRAIN

REPO = Path("/home/amal/fyp/Axiom")


def _events(n_informative_user1, n_user2=0):
    """User 1 has n_informative_user1 train rows that count; user 2 has n_user2 of them."""
    rows = []
    eid = 0
    for uid, n in [(1, n_informative_user1), (2, n_user2)]:
        for i in range(n + 1):  # first row is a first review, the rest are inter-day
            rows.append(dict(event_id=eid, user_id=uid, card_id=1, is_train=True, is_first=(i == 0),
                             elapsed_days=(-1.0 if i == 0 else 2.0), scored=False, y=1, dt_days=2.0))
            eid += 1
    return pd.DataFrame(rows)


def test_fallback_counts_users_below_threshold():
    events = _events(MIN_INFORMATIVE_TRAIN - 1, n_user2=MIN_INFORMATIVE_TRAIN + 10)
    out = supp.fallback_summary(events)
    assert out["n_fallback"] == 1
    assert out["fallback_users"] == {"1": MIN_INFORMATIVE_TRAIN - 1}
    assert out["n_other_users"] == 1
    assert out["median_informative_other_users"] == MIN_INFORMATIVE_TRAIN + 10


def test_fallback_counts_user_with_no_train_rows_as_zero():
    events = _events(MIN_INFORMATIVE_TRAIN + 5)
    extra = events.iloc[:1].copy()
    extra["user_id"], extra["is_train"], extra["event_id"] = 3, False, 10**6
    out = supp.fallback_summary(pd.concat([events, extra], ignore_index=True))
    # user 2 (one first review, no informative rows) and user 3 (no train rows) both count as zero
    assert out["n_users"] == 3 and out["fallback_users"] == {"2": 0, "3": 0}


def test_ours_vs_kappa_correlation_and_max_diff():
    preds = pd.DataFrame({"ours": [0.2, 0.4, 0.6, 0.8], "ours_kappa": [0.2, 0.5, 0.6, 0.7]})
    out = supp.ours_vs_kappa(preds)
    assert out["n_events"] == 4
    assert out["max_abs_diff"] == pytest.approx(0.1)
    assert out["pearson_r"] == pytest.approx(np.corrcoef(preds["ours"], preds["ours_kappa"])[0, 1], abs=1e-4)


def test_reliability_bins_skip_empty_and_clip_p_one():
    preds = pd.DataFrame({"ours": [0.15, 0.17, 0.95, 1.0], "y": [1, 0, 1, 1]})
    bins = supp.reliability_bins(preds)
    assert [b["bin"] for b in bins] == [1, 9]
    assert bins[0] == {"bin": 1, "n": 2, "mean_p": 0.16, "mean_y": 0.5}
    assert bins[1] == {"bin": 9, "n": 2, "mean_p": 0.975, "mean_y": 1.0}


def test_regenerated_values_match_committed_full_run(tmp_path):
    run_dir = REPO / "results" / "full_u100_c1000"
    if not (run_dir / "events.parquet").exists() or not (run_dir / "predictions.parquet").exists():
        pytest.skip("committed full run parquet files not present")
    out = tmp_path / "supplementary.json"
    supp.main(["--run_dir", str(run_dir), "--out", str(out)])
    new = json.loads(out.read_text())
    old = json.loads((run_dir / "supplementary.json").read_text())
    for key in ("fsrs_opt_fallback", "ours_vs_kappa", "ours_reliability_bins"):
        assert new[key] == old[key], key
