"""Tests for the constant-recall baselines in report.py (const_global, const_user).

The toy numbers are computed by hand in the comments, so the tests check the rule itself,
not the code that implements it.
"""
import numpy as np
import pandas as pd
import pytest

from fsrs_benchmark import metrics, report


def _events():
    """Toy event table. Columns are the ones make_split writes that the constants read."""
    rows = [
        # user 1: train rows that count = (y=1, ed=2), (y=0, ed=5) -> rate 1/2
        dict(user_id=1, is_train=True, is_first=False, elapsed_days=2, y=1),
        dict(user_id=1, is_train=True, is_first=False, elapsed_days=5, y=0),
        # user 1: train rows that must NOT count: a first review, and elapsed_days < 1
        dict(user_id=1, is_train=True, is_first=True, elapsed_days=-1, y=0),
        dict(user_id=1, is_train=True, is_first=False, elapsed_days=0, y=0),
        # user 1: test rows (labels deliberately extreme; they must not matter)
        dict(user_id=1, is_train=False, is_first=False, elapsed_days=3, y=0),
        dict(user_id=1, is_train=False, is_first=False, elapsed_days=3, y=0),
        # user 2: train rows that count = (1, 3), (1, 4), (1, 1) -> rate 1.0
        dict(user_id=2, is_train=True, is_first=False, elapsed_days=3, y=1),
        dict(user_id=2, is_train=True, is_first=False, elapsed_days=4, y=1),
        dict(user_id=2, is_train=True, is_first=False, elapsed_days=1, y=1),
        # user 2: a train first review with y=0 that must NOT count (it would give 0.75)
        dict(user_id=2, is_train=True, is_first=True, elapsed_days=-1, y=0),
        dict(user_id=2, is_train=False, is_first=False, elapsed_days=2, y=0),
        # user 3: test rows only, so no train rows at all -> falls back to global
        dict(user_id=3, is_train=False, is_first=False, elapsed_days=7, y=1),
    ]
    df = pd.DataFrame(rows)
    df["is_first"] = df["is_first"].astype(bool)
    df["is_train"] = df["is_train"].astype(bool)
    df["elapsed_days"] = df["elapsed_days"].astype(np.int64)
    df["user_id"] = df["user_id"].astype(np.int64)
    df["y"] = df["y"].astype(np.int64)
    return df


def _scored_preds(events):
    """Predictions for the test rows only, as run.py writes them (user_id, y, elapsed_days)."""
    test = events[~events["is_train"]]
    out = pd.DataFrame({"event_id": np.arange(len(test), dtype=np.int64),
                        "user_id": test["user_id"].to_numpy(np.int64),
                        "card_id": np.arange(len(test), dtype=np.int64),
                        "y": test["y"].to_numpy(np.int64),
                        "elapsed_days": test["elapsed_days"].to_numpy(np.int64),
                        "day_offset": np.zeros(len(test), np.int64)})
    for m in ["fsrs_opt", "ours"]:
        out[m] = 0.5
    return out


def test_hand_computed_rates():
    events = _events()
    preds = _scored_preds(events)
    out = report.constant_predictions(preds, events)
    # global: (1 + 0 + 1 + 1 + 1) / 5 = 0.8
    assert np.allclose(out["const_global"], 0.8)
    # per user: user 1 = 1/2, user 2 = 3/3, user 3 falls back to 0.8
    expected_user = preds["user_id"].map({1: 0.5, 2: 1.0, 3: 0.8}).to_numpy(float)
    assert np.allclose(out["const_user"], expected_user)
    assert len(out["const_global"]) == len(preds)


def test_test_labels_never_change_the_rates():
    events = _events()
    base = report.constant_predictions(_scored_preds(events), events)
    flipped = events.copy()
    flipped.loc[~flipped["is_train"], "y"] = 1 - flipped.loc[~flipped["is_train"], "y"]
    flipped_preds = _scored_preds(flipped)
    after = report.constant_predictions(flipped_preds, flipped)
    assert np.allclose(after["const_global"], base["const_global"])
    assert np.allclose(after["const_user"], base["const_user"])


def test_first_reviews_and_same_day_train_rows_are_excluded():
    events = _events()
    preds = _scored_preds(events)
    out = report.constant_predictions(preds, events)
    # If the train first review (y=0) of user 2 were counted, user 2 would be 3/4 = 0.75.
    u2 = preds["user_id"].to_numpy() == 2
    assert np.allclose(out["const_user"][u2], 1.0)
    # If user 1's elapsed_days=0 row (y=0) were counted, user 1 would be 1/3, not 1/2.
    u1 = preds["user_id"].to_numpy() == 1
    assert np.allclose(out["const_user"][u1], 0.5)


def test_user_without_train_rows_falls_back_to_global():
    events = _events()
    preds = _scored_preds(events)
    out = report.constant_predictions(preds, events)
    u3 = preds["user_id"].to_numpy() == 3
    assert np.allclose(out["const_user"][u3], out["const_global"][u3])
    assert np.allclose(out["const_user"][u3], 0.8)


def test_no_rate_rows_raises():
    events = _events()
    events["is_train"] = False
    with pytest.raises(ValueError):
        report.constant_predictions(_scored_preds(_events()), events)


def test_constants_are_first_in_model_order():
    events = _events()
    preds = report.add_constant_models(_scored_preds(events), events)
    assert report.model_columns(preds)[:2] == ["const_global", "const_user"]
    assert report.DISPLAY["const_global"] == "Constant (global)"
    assert report.DISPLAY["const_user"] == "Constant (per user)"


def test_build_report_includes_constants_everywhere(tmp_path):
    events = _events()
    events.to_parquet(tmp_path / "events.parquet", index=False)
    preds = _scored_preds(events)
    preds["ours"] = np.linspace(0.2, 0.9, len(preds))
    preds.to_parquet(tmp_path / "predictions.parquet", index=False)
    out = report.build_report(tmp_path)

    mt = pd.read_csv(out / "metrics.csv")
    overall = mt[(mt["model"] == "const_global") & (mt["scope"] == "overall")].iloc[0]
    assert overall["n"] == len(preds)
    assert overall["log_loss"] == pytest.approx(metrics.log_loss(preds["y"], np.full(len(preds), 0.8)), abs=1e-9)
    assert set(mt[mt["scope"] == "gap"]["model"]) >= {"const_global", "const_user"}

    pt = pd.read_csv(out / "paired.csv")
    pairs = set(zip(pt["reference"], pt["model"]))
    assert ("ours", "const_global") in pairs and ("fsrs_opt", "const_user") in pairs

    uw = pd.read_csv(out / "user_wins.csv")
    assert {"const_global", "const_user"} <= set(uw["model"])

    text = (out / "report.md").read_text()
    assert "Constant (global)" in text and "Constant (per user)" in text


def test_no_events_file_means_no_constants(tmp_path):
    preds = _scored_preds(_events())
    preds.to_parquet(tmp_path / "predictions.parquet", index=False)
    out = report.build_report(tmp_path)
    mt = pd.read_csv(out / "metrics.csv")
    assert not mt["model"].str.startswith("const_").any()
