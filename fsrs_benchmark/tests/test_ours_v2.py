"""Tests for validation split (no leakage into the fit set) and OursV2 (one-step-ahead, complete)."""
import numpy as np
import pandas as pd
import pytest

from fsrs_benchmark import split
from fsrs_benchmark.experiments.validation import make_validation_split
from fsrs_benchmark.models.ours_v2 import OursV2


def _raw(n_users=4, n_cards=25, reviews=14, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(1, n_users + 1):
        for c in range(n_cards):
            day = int(rng.integers(0, 40))
            for k in range(reviews):
                gap = 0 if k == 0 else int(rng.integers(0, 8))
                day += gap
                r = int(rng.integers(1, 5))
                rows.append(dict(user_id=u, card_id=c, day_offset=day, rating=r,
                                 state=0 if k == 0 else 2, elapsed_days=-1 if k == 0 else gap,
                                 dt_days=np.nan if k == 0 else float(gap), y=int(r > 1), is_first=k == 0))
    df = pd.DataFrame(rows).sort_values(["user_id", "day_offset"], kind="stable").reset_index(drop=True)
    df["review_idx"] = df.groupby("user_id").cumcount()
    return df


@pytest.fixture(scope="module")
def events():
    return split.make_split(_raw(), max_cards=1000, seed=0)


def test_validation_split_no_leak(events):
    val = make_validation_split(events)
    test_ids = set(events.loc[~events["is_train"], "event_id"])
    assert test_ids and not (test_ids & set(val["event_id"]))      # test rows are gone
    assert val["is_train"].sum() > 0 and val["scored"].sum() > 0
    for _, u in val.groupby("user_id"):
        fit, sc = u[u["is_train"]], u[u["scored"]]
        assert fit["day_offset"].max() < u["val_cutoff_day"].iloc[0] <= sc["day_offset"].min()
        assert (u["day_offset"] < u["cutoff_day"]).all()           # only original train days
        assert not sc["is_first"].any() and (sc["elapsed_days"] >= 1).all()
    assert not (val["is_train"] & val["scored"]).any()


def test_fit_ignores_validation_and_test_rows(events):
    val = make_validation_split(events)
    a = OursV2(epochs=2).fit(val)
    mutated = val.copy()
    late = ~mutated["is_train"]
    mutated.loc[late, "rating"] = 1
    mutated.loc[late, "y"] = 0
    b = OursV2(epochs=2).fit(mutated)
    np.testing.assert_allclose(a.theta.detach().numpy(), b.theta.detach().numpy())
    np.testing.assert_allclose(a.z.detach().numpy(), b.z.detach().numpy())


def test_predict_complete_and_valid(events):
    val = make_validation_split(events)
    m = OursV2(epochs=2).fit(val)
    p = m.predict(val)
    assert set(p.index) == set(val.loc[val["scored"], "event_id"])
    assert p.between(1e-6, 1 - 1e-6).all() and p.index.is_unique and np.isfinite(p).all()


@pytest.mark.parametrize("kwargs", [dict(), dict(process_noise=True, calibrate=True)])
def test_no_leakage_in_prediction(events, kwargs):
    val = make_validation_split(events)
    m = OursV2(epochs=1, **kwargs).fit(val)
    base = m.predict(val)
    rng = np.random.default_rng(1)
    for eid in rng.choice(base.index.to_numpy(), size=6, replace=False):
        row = val[val["event_id"] == eid].iloc[0]
        mod = val.copy()
        same = (mod["user_id"] == row["user_id"]) & (mod["card_id"] == row["card_id"]) & (mod["review_idx"] >= row["review_idx"])
        # change this event's own outcome and everything later on the card
        mod.loc[same, "rating"] = np.where(mod.loc[same, "rating"] == 1, 4, 1)
        mod.loc[same, "y"] = (mod.loc[same, "rating"] > 1).astype(int)
        after = m.predict(mod)
        assert after.loc[eid] == pytest.approx(base.loc[eid], abs=1e-6)
        earlier = same & (mod["review_idx"] > row["review_idx"])
        # other cards unaffected entirely
        other = val[(val["user_id"] == row["user_id"]) & (val["card_id"] != row["card_id"]) & val["scored"]]["event_id"]
        np.testing.assert_allclose(after.loc[other].to_numpy(), base.loc[other].to_numpy(), atol=1e-6)


def test_unseen_user_gets_predictions(events):
    val = make_validation_split(events)
    m = OursV2(epochs=1).fit(val[val["user_id"] != 1])
    assert len(m.predict(val)) == int(val["scored"].sum())


def test_run_registers_ours_v2(events):
    import argparse
    from fsrs_benchmark import run as run_mod
    from fsrs_benchmark.models.ours_v2 import FROZEN_KWARGS
    assert "ours_v2" in run_mod.OPT_IN_MODELS and "ours_v2" not in run_mod.ALL_MODELS
    args = argparse.Namespace(seed=0, n_users=2, max_cards=300, iters=5, burn=2, dataset="anki")
    model = run_mod.make_model("ours_v2", args)
    assert isinstance(model, OursV2) and model.epochs == FROZEN_KWARGS["epochs"]
    cfg = run_mod.model_config("ours_v2", args, np.array([1, 2], np.int64))
    assert cfg["hyperparams"]["epochs"] == FROZEN_KWARGS["epochs"] and "iters" not in cfg
    model.epochs = 1
    p = model.fit(events).predict(events)
    assert set(p.index) == set(events.loc[events["scored"], "event_id"])
    assert run_mod.model_details("ours_v2", model)["hyperparams"]["process_noise"] is False


def test_run_end_to_end_ours_v2(tmp_path):
    import json
    from fsrs_benchmark import run as run_mod
    from fsrs_benchmark.tests._synthetic_events import write_duolingo_csv
    csv = write_duolingo_csv(tmp_path / "traces.csv.gz")
    out_dir = tmp_path / "results"
    run_mod.main(["--dataset", "duolingo", "--duolingo_path", str(csv), "--n_users", "0",
                  "--min_user_rows", "10", "--models", "ours_v2", "--max_cards", "300",
                  "--seed", "0", "--run_name", "v2", "--out_dir", str(out_dir)])
    run_dir = out_dir / "v2"
    side = json.loads((run_dir / "pred_ours_v2.json").read_text())
    assert side["details"]["hyperparams"]["epochs"] == 20 and side["details"]["fit_seconds"] >= 0
    preds = pd.read_parquet(run_dir / "predictions.parquet")
    assert "ours_v2" in preds.columns and preds["ours_v2"].between(1e-6, 1 - 1e-6).all()
    assert "ours_v2" in json.loads((run_dir / "run_info.json").read_text())["models"]


def test_report_runs_with_ours_v2_and_unknown_model(tmp_path):
    from fsrs_benchmark import run as run_mod
    from fsrs_benchmark.report import build_report
    from fsrs_benchmark.tests._synthetic_events import write_duolingo_csv
    csv = write_duolingo_csv(tmp_path / "traces.csv.gz")
    out_dir = tmp_path / "results"
    run_mod.main(["--dataset", "duolingo", "--duolingo_path", str(csv), "--n_users", "0",
                  "--min_user_rows", "10", "--models", "fsrs_default,ours_v2", "--max_cards", "300",
                  "--seed", "0", "--run_name", "r", "--out_dir", str(out_dir)])
    run_dir = out_dir / "r"
    from fsrs_benchmark import report as rep
    assert rep.MARKERS["brand_new"] and rep.COLOURS["brand_new"] and rep.DISPLAY["brand_new"] == "brand_new"
    report_dir = build_report(run_dir)
    assert (report_dir / "reliability.png").exists()
