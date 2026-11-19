"""Tests for export_demo_data.py. Run with:

  cd /home/amal/fyp/Axiom && uv run --quiet --with pandas --with pyarrow --with pytest pytest client/scripts/test_export_demo_data.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_demo_data as ex  # noqa: E402


@pytest.fixture(scope="module")
def seed():
    return ex.build_seed()


@pytest.fixture(scope="module")
def preds():
    return pd.read_parquet(ex.RESULTS / "predictions.parquet")


def test_top_level_keys(seed):
    assert set(seed) == {"overall", "gapBins", "reliability", "userWins", "cards"}


def test_overall_models_and_fields(seed):
    assert set(seed["overall"]) == {"fsrs_default", "fsrs_opt", "hlr", "ours", "ours_v3", "gbdt"}
    for stats in seed["overall"].values():
        assert set(stats) == {"logLoss", "auc", "ece", "brier"}


def test_overall_log_loss_values(seed):
    assert seed["overall"]["gbdt"]["logLoss"] == pytest.approx(0.3584, abs=1e-3)
    assert seed["overall"]["fsrs_opt"]["logLoss"] == pytest.approx(0.3794, abs=1e-3)


def test_log_loss_matches_raw_predictions(seed, preds):
    y = preds["y"].to_numpy(float)
    for model in ("gbdt", "fsrs_opt"):
        q = np.clip(preds[model].to_numpy(float), ex.EPS, 1 - ex.EPS)
        ll = -(y * np.log(q) + (1 - y) * np.log(1 - q)).mean()
        assert seed["overall"][model]["logLoss"] == pytest.approx(ll, abs=1e-4)


def test_gap_bins(seed):
    assert set(seed["gapBins"]) == {"fsrs_opt", "gbdt"}
    for model in ("fsrs_opt", "gbdt"):
        bins = seed["gapBins"][model]
        assert len(bins) == 5
        assert sum(b["n"] for b in bins) == 95400
        assert all(set(b) == {"bin", "n", "logLoss"} for b in bins)


def test_reliability_bins_sum_to_all_reviews(seed):
    for model in ("fsrs_opt", "gbdt"):
        bins = seed["reliability"][model]
        assert sum(b["n"] for b in bins) == 95400
        assert all(b["n"] > 0 for b in bins)
        assert all(0 <= b["binLow"] < b["binHigh"] <= 1 for b in bins)
        assert all(set(b) == {"binLow", "binHigh", "meanPred", "meanObs", "n"} for b in bins)


def test_user_wins(seed):
    assert seed["userWins"]["total"] == 100
    assert seed["userWins"]["count"] == 82
    diffs = seed["userWins"]["diffs"]
    assert len(diffs) == 100
    assert diffs == sorted(diffs)


def test_cards(seed):
    cards = seed["cards"]
    assert len(cards) == 4
    assert [c["label"] for c in cards] == ["Card A", "Card B", "Card C", "Card D"]
    for card in cards:
        scored = [r for r in card["reviews"] if r["scored"]]
        assert len(scored) >= 6
        assert card["nScored"] == len(scored)
        assert {r["y"] for r in scored} == {0, 1}
        for r in scored:
            assert 0 < r["pFsrs"] < 1
            assert 0 < r["pOurs"] < 1
            assert "pFsrs" in r and "pOurs" in r
        for r in card["reviews"]:
            assert set(r) >= {"review", "day", "rating", "y", "scored"}


def test_written_file_is_small_and_parses(tmp_path):
    out = ex.main(tmp_path / "results.json")
    assert out.stat().st_size < 300 * 1024
    data = json.loads(out.read_text(encoding="utf-8"))
    assert set(data) == {"overall", "gapBins", "reliability", "userWins", "cards"}


def test_export_is_deterministic(tmp_path):
    a = ex.main(tmp_path / "a.json").read_bytes()
    b = ex.main(tmp_path / "b.json").read_bytes()
    assert a == b


def test_reliability_mean_pred_inside_bin(seed):
    for model in ("fsrs_opt", "gbdt"):
        for b in seed["reliability"][model]:
            assert b["binLow"] - 1e-4 <= b["meanPred"] <= b["binHigh"] + 1e-4
            assert 0 <= b["meanObs"] <= 1


def test_user_wins_count_matches_diffs(seed):
    diffs = seed["userWins"]["diffs"]
    assert sum(d < 0 for d in diffs) == seed["userWins"]["count"]


def test_card_predictions_match_parquet(seed, preds):
    keys = ex.pick_cards(preds)
    assert len(keys) == len(seed["cards"])
    for (uid, cid), card in zip(keys, seed["cards"]):
        pr = preds[(preds["user_id"] == uid) & (preds["card_id"] == cid)].set_index("event_id")
        ev = pd.read_parquet(ex.RESULTS / "events.parquet")
        ev = ev[(ev["user_id"] == uid) & (ev["card_id"] == cid)].sort_values("review_idx")
        scored_rows = [r for r, (_, e) in zip(card["reviews"], ev.iterrows()) if r["scored"]]
        scored_ev = ev[ev["scored"]]
        assert len(scored_rows) == len(scored_ev) == len(pr)
        for r, (_, e) in zip(card["reviews"], ev.iterrows()):
            assert r["review"] == int(e["review_idx"]) and r["day"] == int(e["day_offset"])
            assert r["y"] == int(e["y"]) and r["rating"] == int(e["rating"])
            if r["scored"]:
                p = pr.loc[int(e["event_id"])]
                assert r["pFsrs"] == pytest.approx(p["fsrs_opt"], abs=1e-4)
                assert r["pOurs"] == pytest.approx(p["gbdt"], abs=1e-4)


def test_card_histories_are_time_ordered(seed):
    for card in seed["cards"]:
        reviews = card["reviews"]
        assert [r["review"] for r in reviews] == sorted(r["review"] for r in reviews)
        assert [r["day"] for r in reviews] == sorted(r["day"] for r in reviews)


def test_honest_card_is_where_fsrs_wins(seed):
    def card_gap(card):
        scored = [r for r in card["reviews"] if r["scored"]]
        y = np.array([r["y"] for r in scored], float)
        f = ex.log_loss_vec(y, np.array([r["pFsrs"] for r in scored])).mean()
        g = ex.log_loss_vec(y, np.array([r["pOurs"] for r in scored])).mean()
        return f - g

    honest = seed["cards"][3]
    assert honest["pickedFor"] == "fsrs_wins"
    assert card_gap(honest) < 0
    for top in seed["cards"][:3]:
        assert top["pickedFor"] == "top"
        assert card_gap(top) > 0


def test_no_raw_ids_in_output(tmp_path):
    text = ex.main(tmp_path / "r.json").read_text(encoding="utf-8")
    for token in ('"user_id"', '"card_id"', '"event_id"', '"userId"', '"cardId"'):
        assert token not in text
