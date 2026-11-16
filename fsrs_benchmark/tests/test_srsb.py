"""Tests for the srs-benchmark comparison (Phase 2.2): past-only features, TimeSeriesSplit test rows,
the srs-benchmark jsonl format, the README reproduction, and the paired bootstrap."""
import json

import numpy as np
import pandas as pd
import pytest

from fsrs_benchmark.srsb import compare as C
from fsrs_benchmark.srsb import run_gbdt as R


def _revlog(n_cards=120, reviews_per_card=12, seed=0, start_day=0):
    """A synthetic srs-benchmark revlog: one row per review, file order = chronological by day.

    Each card has growing intervals (a few distinct values) and mostly good grades, so srs-benchmark's
    outlier filter keeps rows. Same-day reviews are not generated (srs-benchmark drops them from scoring).
    """
    rng = np.random.default_rng(seed)
    events = []
    for c in range(n_cards):
        day = start_day + c
        for k in range(reviews_per_card):
            if k > 0:
                interval = max(1, int(round(1.8 ** k)))   # few distinct intervals, so no outlier removal
                day += interval
            rating = int(rng.integers(1, 5)) if k == 0 else int(rng.choice([1, 2, 3, 4], p=[0.05, 0.1, 0.6, 0.25]))
            events.append(dict(card_id=1000 + c, day_offset=day, rating=rating, state=0, duration=1000))
    raw = pd.DataFrame(events).sort_values(["day_offset", "card_id"], kind="stable").reset_index(drop=True)
    raw["elapsed_days"] = -1
    raw["elapsed_seconds"] = -1
    last = {}
    for idx, row in raw.iterrows():
        c = row["card_id"]
        if c in last:
            raw.at[idx, "elapsed_days"] = int(row["day_offset"] - last[c])
            raw.at[idx, "elapsed_seconds"] = raw.at[idx, "elapsed_days"] * 86400
        last[c] = row["day_offset"]
    return raw[["card_id", "day_offset", "rating", "state", "duration", "elapsed_days", "elapsed_seconds"]]


def _features_equal(a, b, cols=None):
    cols = list(a.columns) if cols is None else cols
    for c in cols:
        assert np.array_equal(a[c].to_numpy(float), b[c].to_numpy(float), equal_nan=True), c


# ---- past-only features ------------------------------------------------------------------------

def test_features_ignore_later_labels():
    raw = _revlog(seed=1)
    base = R.past_only_features(raw, 128)
    k = 70                                       # a late row
    changed = raw.copy()
    changed.at[k, "rating"] = 1 if raw.at[k, "rating"] != 1 else 4
    other = R.past_only_features(changed, 128)
    earlier = base.index[base.index <= k]        # event ids are review_th = file position + 1
    later_changed = base.index[base.index > k]
    _features_equal(base.loc[earlier], other.loc[earlier])
    # the change must be visible somewhere after it, otherwise the test proves nothing
    assert not np.array_equal(base.loc[later_changed].to_numpy(float),
                              other.loc[later_changed].to_numpy(float), equal_nan=True)


def test_features_are_prefix_invariant():
    raw = _revlog(seed=2)
    full = R.past_only_features(raw, 128)
    k = 50
    prefix = R.past_only_features(raw.iloc[:k], 128)
    _features_equal(full.loc[prefix.index], prefix)


def test_review_stream_drops_like_srs_benchmark():
    raw = _revlog(n_cards=2, reviews_per_card=10, seed=3)
    raw.loc[3, "rating"] = 0                     # invalid rating: dropped before history
    ev = R.review_stream(raw, max_card_reviews=4)
    assert 4 not in ev.index                     # review_th 4 (file position 3) is gone
    per_card = ev.groupby("card_id").size()
    assert (per_card <= 4).all()
    assert set(ev["y"].unique()) <= {0, 1}
    assert ev.loc[ev["is_first"]].shape[0] == 2  # one first review per card


def test_features_have_frozen_columns():
    feats = R.past_only_features(_revlog(seed=4), 128)
    assert list(feats.columns) == R.FEATURE_COLS
    assert "dt_days" not in R.FEATURE_COLS and "lexeme" not in R.FEATURE_COLS


# ---- TimeSeriesSplit test rows -----------------------------------------------------------------

def test_test_rows_are_timeseriessplit_test_folds():
    from sklearn.model_selection import TimeSeriesSplit

    n = 101
    mask = R.test_row_mask(n, 5)
    expected = np.zeros(n, bool)
    for _, te in TimeSeriesSplit(n_splits=5).split(np.arange(n)):
        expected[te] = True
    assert np.array_equal(mask, expected)
    first_chunk = n - 5 * (n // 6)               # the first chunk is train-only
    assert not mask[:first_chunk].any()
    assert mask[first_chunk:].all()


# ---- end to end on a tiny synthetic data directory ----------------------------------------------

@pytest.fixture(scope="module")
def tiny_data(tmp_path_factory):
    root = tmp_path_factory.mktemp("srsb_data")
    for uid in range(1, 6):
        d = root / "revlogs" / f"user_id={uid}"
        d.mkdir(parents=True)
        _revlog(seed=uid).to_parquet(d / "data.parquet")
    return root


def test_driver_writes_srsb_jsonl(tiny_data, tmp_path):
    out = tmp_path / "GBDT-test.jsonl"
    summary = R.run([1, 2, 3], [4, 5], tiny_data, out, max_rounds=20, patience=5)
    lines = [json.loads(x) for x in out.read_text().splitlines() if x.strip()]
    assert summary["eval_users"] == len(lines) == 2
    assert [x["user"] for x in lines] == [4, 5]
    for x in lines:
        assert set(x) == {"metrics", "user", "size"}
        assert {"LogLoss", "RMSE(bins)", "AUC", "RMSE", "ICI", "MBE"} <= set(x["metrics"])
        assert 0 < x["size"] < 1320          # fewer than the user's 1320 rows (first chunk is train-only)
        assert 0 <= x["metrics"]["LogLoss"] < 5
    # sizes are the TimeSeriesSplit test rows of srs-benchmark's own dataset for that user
    S = R._srs_modules()
    cfg = S["Config"](S["create_parser"]().parse_args(["--algo", "FSRS-6", "--short", "--data", str(tiny_data)]))
    loader = S["UserDataLoader"](cfg)
    for x in lines:
        ds = loader.load_user_data(x["user"])
        assert x["size"] == int(R.test_row_mask(len(ds), cfg.n_splits).sum())


def test_holdout_users_are_train_users_and_stable():
    users = list(range(10, 40))
    hold = R.holdout_users(users)
    assert hold == R.holdout_users(list(reversed(users)))   # order independent
    assert len(hold) == 3 and set(hold) <= set(users)
    assert R.holdout_users([7]) == []


# ---- compare.py --------------------------------------------------------------------------------

def test_weighted_mean_toy():
    assert C.weighted_mean([0.2, 0.4], [1, 3]) == pytest.approx(0.35)


def test_paired_bootstrap_constant_and_toy():
    flat = C.paired_bootstrap([0.05] * 6, [3, 1, 4, 1, 5, 9], n_boot=200, ci=0.99, seed=0)
    assert flat["lo"] == pytest.approx(0.05) and flat["hi"] == pytest.approx(0.05)
    d = [1.0, 0.0]
    b = C.paired_bootstrap(d, [1, 9], n_boot=1000, ci=0.99, seed=0)
    assert b["mean"] == pytest.approx(0.1)
    assert b["lo"] <= b["mean"] <= b["hi"]
    assert b == C.paired_bootstrap(d, [1, 9], n_boot=1000, ci=0.99, seed=0)  # deterministic
    assert b["n_users"] == 2 and b["n_boot"] == 1000


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for user, size, ll in rows:
            f.write(json.dumps({"metrics": {"LogLoss": ll, "RMSE(bins)": 0.1, "AUC": 0.7},
                                "user": user, "size": size}) + "\n")


def _toy_result_dir(tmp_path):
    """Every README file gets two users: sizes 1 and 3, LogLoss README -0.01 and README +0.01.
    The unweighted mean is then the README value, and the size-weighted mean is not."""
    rd = tmp_path / "result"
    rd.mkdir()
    for name, (fname, readme) in C.README_WITHOUT_SAME_DAY.items():
        _write_jsonl(rd / f"{fname}.jsonl", [(1, 1, round(readme - 0.01, 4)), (2, 3, round(readme + 0.01, 4))])
    return rd


def test_readme_check_reproduces_unweighted_means(tmp_path):
    rd = _toy_result_dir(tmp_path)
    rows = {r["model"]: r for r in C.readme_check(rd)}
    assert all(r["match"] for r in rows.values())
    assert rows["FSRS-6"]["unweighted"] == pytest.approx(0.3460)
    assert rows["FSRS-6"]["weighted"] == pytest.approx(0.3510)


def test_compare_pairs_on_users_with_matching_sizes(tmp_path):
    rd = _toy_result_dir(tmp_path)
    ours = tmp_path / "ours.jsonl"
    users = [1, 2, 3, 4]
    sizes = {1: 5, 2: 7, 3: 9, 4: 11}
    _write_jsonl(ours, [(u, sizes[u], 0.25) for u in users])
    # the published FSRS-6 file has a different size for user 3: that user must be excluded
    fsrs = [{"metrics": {"LogLoss": 0.3, "RMSE(bins)": 0.1, "AUC": 0.7}, "user": u, "size": s}
            for u, s in [(1, 5), (2, 7), (3, 99), (4, 11)]]
    (rd / "FSRS-6-short.jsonl").write_text("\n".join(json.dumps(x) for x in fsrs) + "\n")
    # other published files must cover the same users with matching sizes
    for name, (fname, readme) in C.README_WITHOUT_SAME_DAY.items():
        if fname == "FSRS-6-short":
            continue
        _write_jsonl(rd / f"{fname}.jsonl", [(u, sizes[u], readme) for u in users])
    res = C.compare(ours, rd, users=None, ci=0.99, n_boot=200, seed=0)
    assert res["size_match"]["mismatched"] == [3]
    assert res["paired_users"] == 3
    diff = res["bootstrap"]["GBDT - FSRS-6"]
    assert diff["mean"] == pytest.approx(0.25 - 0.3)
    assert diff["lo"] <= diff["mean"] <= diff["hi"]
    sub = C.compare(ours, rd, users={1, 2}, ci=0.99, n_boot=50, seed=0)
    assert sub["paired_users"] == 2


# ---- unequal user sets (task 2.3 regression): a published file missing some users must not crash --------

def _all_model_files(rd, users_by_file, sizes, ll=0.3):
    for name, (fname, _) in C.README_WITHOUT_SAME_DAY.items():
        us = users_by_file.get(fname, sorted(sizes))
        _write_jsonl(rd / f"{fname}.jsonl", [(u, sizes[u], ll) for u in us])


def test_compare_handles_unequal_user_sets(tmp_path):
    rd = tmp_path / "result"
    rd.mkdir()
    sizes = {1: 5, 2: 7, 3: 9, 4: 11}
    ours = tmp_path / "ours.jsonl"
    _write_jsonl(ours, [(u, sizes[u], 0.25) for u in sizes])
    # FSRS-6 lacks user 4; GRU lacks user 1; AVG lacks user 3 and has user 4 with a different size
    _all_model_files(rd, {"FSRS-6-short": [1, 2, 3], "GRU-short-secs-equalize_test_with_non_secs": [2, 3, 4],
                          "AVG": [1, 2]}, sizes)
    _write_jsonl(rd / "AVG.jsonl", [(1, 5, 0.4), (2, 7, 0.4), (4, 99, 0.4)])
    res = C.compare(ours, rd, n_boot=100, seed=0)          # must not raise
    per = res["per_model"]
    assert per["FSRS-6"]["n_users"] == 3 and per["FSRS-6"]["n_ours_only"] == 1
    assert per["GRU"]["n_users"] == 3 and per["GRU"]["n_ours_only"] == 1
    assert per["AVG"]["n_users"] == 2                       # user 4 has a different size, so it is not paired
    assert res["paired_users"] == 1                         # main table: users in ALL models = {2}
    main = {r["model"]: r for r in res["table"]}
    assert main["GBDT (ours)"]["n_users"] == 1 and main["FSRS-6"]["n_users"] == 1
    assert main["GBDT (ours)"]["LogLoss_u"] == pytest.approx(0.25)
    assert set(res["bootstrap"]) == {f"GBDT - {m}" for m in C.BOOTSTRAP_BASELINES}
    assert res["size_match"]["compared"] == 3 and res["size_match"]["matching"] == 3   # FSRS-6 has users 1-3


def test_compare_refuses_when_no_common_users(tmp_path):
    rd = tmp_path / "result"
    rd.mkdir()
    ours = tmp_path / "ours.jsonl"
    _write_jsonl(ours, [(1, 5, 0.25)])
    _all_model_files(rd, {}, {1: 9})                       # sizes differ from ours everywhere
    with pytest.raises(ValueError, match="no users are common"):
        C.compare(ours, rd, n_boot=10, seed=0)


def test_compare_handles_missing_auc(tmp_path):
    """Users with one class have AUC null: they drop out of the AUC mean only, with aligned sizes."""
    rd = tmp_path / "result"
    rd.mkdir()
    sizes = {1: 1, 2: 3}
    _all_model_files(rd, {}, sizes)
    ours = tmp_path / "ours.jsonl"
    lines = [{"metrics": {"LogLoss": 0.25, "RMSE(bins)": 0.1, "AUC": None if u == 1 else 0.7}, "user": u,
              "size": sizes[u]} for u in (1, 2)]
    ours.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    res = C.compare(ours, rd, n_boot=50, seed=0)
    gbdt = res["table"][0]
    assert gbdt["AUC_n"] == 1
    assert gbdt["AUC_w"] == pytest.approx(0.7)


# ---- paper writer (task 2.3): srsb.csv, srsb_paired.csv and the FACTS block -------------------------

def test_export_srsb_writes_tables_and_facts(tmp_path):
    from fsrs_benchmark import paper_export as PE

    rd = tmp_path / "result"
    rd.mkdir()
    _all_model_files(rd, {}, {1: 1, 2: 3})
    run = tmp_path / "srsb"
    run.mkdir()
    _write_jsonl(run / "GBDT-short.jsonl", [(1, 1, 0.25), (2, 3, 0.25)])
    (run / "run_info.json").write_text(json.dumps(dict(train_users=450, holdout_users=50, fit_rows=17060060,
                                                      best_iter=2000, eval_users=2)))
    out = tmp_path / "paper" / "data"
    facts = PE.export_srsb(run, out, n_boot=50, ci=0.99, seed=0, result_dir=rd)
    import pandas as pd
    main = pd.read_csv(out / "srsb.csv")
    assert list(main["model"])[0] == "GBDT (ours)" and len(main) == 12
    assert {"LogLoss_unweighted", "LogLoss_weighted", "AUC_unweighted", "AUC_weighted"} <= set(main.columns)
    paired = pd.read_csv(out / "srsb_paired.csv")
    assert set(paired["comparison"]) == {f"GBDT - {m}" for m in C.BOOTSTRAP_BASELINES}
    assert (paired["ci_level"] == 0.99).all()
    text = facts.read_text()
    block = text.split("<!-- facts:srsb:start -->")[1].split("<!-- facts:srsb:end -->")[0]
    assert "17,060,060" in block and "best_iter: 2000" in block and "450 fit + 50 holdout" in block
    assert "Size match vs FSRS-6-short" in block and "GBDT - FSRS-6" in block
    # every line of the block ends with its source file
    for line in block.splitlines():
        if line.startswith("- "):
            assert "source:" in line, line
    # re-running replaces the block instead of appending
    PE.export_srsb(run, out, n_boot=50, ci=0.99, seed=0, result_dir=rd)
    assert facts.read_text().count("<!-- facts:srsb:start -->") == 1
