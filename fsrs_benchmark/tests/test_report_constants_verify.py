"""Verifier tests for the constant-recall baselines (failure cases not covered by test_report_constants.py)."""
import numpy as np
import pandas as pd
import pytest

from fsrs_benchmark import report


def _events(rows):
    df = pd.DataFrame(rows)
    df["event_id"] = np.arange(len(df), dtype=np.int64)
    df["is_first"] = df["is_first"].astype(bool)
    df["is_train"] = df["is_train"].astype(bool)
    if df["elapsed_days"].isna().any():
        df["elapsed_days"] = df["elapsed_days"].astype(float)
    else:
        df["elapsed_days"] = df["elapsed_days"].astype(np.int64)
    df["user_id"] = df["user_id"].astype(np.int64)
    df["y"] = df["y"].astype(np.int64)
    return df


def _preds(user_ids, ys=None):
    """Predictions for constant_predictions only (it does not join to events, so ids need not match)."""
    n = len(user_ids)
    out = pd.DataFrame({"event_id": np.arange(n, dtype=np.int64),
                        "user_id": np.asarray(user_ids, np.int64),
                        "card_id": np.arange(n, dtype=np.int64),
                        "y": np.asarray(ys if ys is not None else [1] * n, np.int64),
                        "elapsed_days": np.full(n, 3, np.int64),
                        "day_offset": np.zeros(n, np.int64)})
    out["fsrs_opt"] = 0.5
    return out


def _matching_preds(events):
    """Predictions for the test rows of events, copied from them (so the events check passes)."""
    test = events[~events["is_train"]]
    return pd.DataFrame({"event_id": test["event_id"].to_numpy(np.int64),
                         "user_id": test["user_id"].to_numpy(np.int64),
                         "card_id": test["event_id"].to_numpy(np.int64),
                         "y": test["y"].to_numpy(np.int64),
                         "elapsed_days": test["elapsed_days"].to_numpy(np.int64),
                         "day_offset": np.zeros(len(test), np.int64),
                         "fsrs_opt": 0.5})


BASE = [
    dict(user_id=1, is_train=True, is_first=False, elapsed_days=2, y=1),
    dict(user_id=1, is_train=True, is_first=False, elapsed_days=5, y=0),
    dict(user_id=2, is_train=True, is_first=False, elapsed_days=3, y=1),
]


def test_nan_elapsed_days_is_excluded_not_crash():
    # A NaN gap must not count as a rate row (NaN >= 1 is False). Its y=0 would lower user 1 to 1/3.
    rows = BASE + [dict(user_id=1, is_train=True, is_first=False, elapsed_days=np.nan, y=0)]
    events = _events(rows)
    out = report.constant_predictions(_preds([1, 2]), events)
    assert np.allclose(out["const_user"], [0.5, 1.0])
    assert np.isclose(out["const_global"][0], 2 / 3)


def test_user_only_with_non_train_rows_falls_back_even_if_others_have_rows():
    rows = BASE + [dict(user_id=9, is_train=False, is_first=False, elapsed_days=4, y=0),
                   dict(user_id=9, is_train=False, is_first=False, elapsed_days=6, y=1)]
    events = _events(rows)
    out = report.constant_predictions(_preds([9, 1]), events)
    g = out["const_global"][0]
    assert np.isclose(g, 2 / 3)
    assert np.isclose(out["const_user"][0], g)


def test_user_whose_train_rows_all_fail_the_filter_falls_back():
    rows = BASE + [dict(user_id=5, is_train=True, is_first=True, elapsed_days=-1, y=0),
                   dict(user_id=5, is_train=True, is_first=False, elapsed_days=0, y=0)]
    events = _events(rows)
    out = report.constant_predictions(_preds([5]), events)
    assert np.isclose(out["const_user"][0], out["const_global"][0])


def test_output_follows_prediction_row_order():
    events = _events(BASE + [dict(user_id=2, is_train=True, is_first=False, elapsed_days=1, y=0)])
    # user 1 rate 0.5, user 2 rate 0.5 (1 of 2)... make them differ: user 2 has y=1 and y=0 -> 0.5
    # so add one more user-2 positive to make it 2/3
    events = _events(BASE + [dict(user_id=2, is_train=True, is_first=False, elapsed_days=1, y=0),
                             dict(user_id=2, is_train=True, is_first=False, elapsed_days=2, y=1)])
    fwd = _preds([1, 2, 1, 2])
    rev = fwd.iloc[::-1].reset_index(drop=True)
    a = report.constant_predictions(fwd, events)["const_user"]
    b = report.constant_predictions(rev, events)["const_user"]
    assert np.allclose(a[::-1], b)
    assert np.isclose(a[1], 2 / 3)


def test_empty_predictions_return_empty_arrays():
    events = _events(BASE)
    out = report.constant_predictions(_preds([]), events)
    assert len(out["const_global"]) == 0 and len(out["const_user"]) == 0


def test_add_constant_models_does_not_mutate_input():
    events = _events(BASE + [dict(user_id=1, is_train=False, is_first=False, elapsed_days=3, y=0)])
    preds = _matching_preds(events)
    before = preds.copy()
    report.add_constant_models(preds, events)
    pd.testing.assert_frame_equal(preds, before)


def test_build_report_without_train_rows_raises(tmp_path):
    events = _events(BASE)
    events["is_train"] = False
    events.to_parquet(tmp_path / "events.parquet", index=False)
    _matching_preds(events).to_parquet(tmp_path / "predictions.parquet", index=False)
    with pytest.raises(ValueError, match="no train rows"):
        report.build_report(tmp_path)
