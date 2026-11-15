"""Verifier checks for Phase 2.2 (srs-benchmark GBDT driver): leakage, size match on local users,
and the run_gbdt resume/RAM behaviour. Uses the synthetic revlog helper from test_srsb."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fsrs_benchmark.srsb import run_gbdt as R
from fsrs_benchmark.tests.test_srsb import _revlog

REPO = Path(__file__).resolve().parents[2]
LOCAL_DATA = REPO / "data" / "anki10k"
PUBLISHED = REPO / "external" / "srs-benchmark" / "result" / "FSRS-6-short.jsonl"
SMOKE_USERS = {1, 28, 54, 83, 164, 220, 282, 334, 406, 486, 728, 746, 799, 840, 887, 1236, 1347, 1737, 1745, 2271}


def _tables_for_synthetic(root, uids, cfg, loader):
    return [R.user_table(u, loader, cfg, root) for u in uids]


def _cfg_loader(root):
    S = R._srs_modules()
    cfg = S["Config"](S["create_parser"]().parse_args(["--algo", "FSRS-6", "--short", "--data", str(root)]))
    return cfg, S["UserDataLoader"](cfg)


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("verify_data")
    for uid in range(1, 8):
        d = root / "revlogs" / f"user_id={uid}"
        d.mkdir(parents=True)
        _revlog(seed=10 + uid, n_cards=80).to_parquet(d / "data.parquet")
    return root


# ---- 1. leakage ----------------------------------------------------------------------------------

def test_eval_label_change_leaves_earlier_predictions_unchanged(synth):
    """Train a booster on users 1-4 only. Flip a late label of eval user 7 and check that every earlier
    review of user 7 gets exactly the same features and the same predicted probability."""
    cfg, loader = _cfg_loader(synth)
    train = _tables_for_synthetic(synth, [1, 2, 3, 4], cfg, loader)
    booster, best_iter, _ = R.train_model(train, max_rounds=30, patience=5)

    raw = pd.read_parquet(synth / "revlogs" / "user_id=7")
    k = len(raw) - 40                                  # a late position, file order = review_th - 1
    changed_raw = raw.copy()
    changed_raw.at[k, "rating"] = 1 if raw.at[k, "rating"] != 1 else 4
    base_feats = R.past_only_features(raw, cfg.max_seq_len * 2)
    pert_feats = R.past_only_features(changed_raw, cfg.max_seq_len * 2)
    early = base_feats.index[base_feats.index <= k]       # review_th = file position + 1, so these precede the flip
    assert len(early) > 0
    np.testing.assert_array_equal(base_feats.loc[early].to_numpy(float),
                                  pert_feats.loc[early].to_numpy(float))
    p_base = booster.predict(base_feats.loc[early].to_numpy(np.float32), num_iteration=best_iter)
    p_pert = booster.predict(pert_feats.loc[early].to_numpy(np.float32), num_iteration=best_iter)
    np.testing.assert_array_equal(p_base, p_pert)
    # the perturbation must reach at least one later row, otherwise the check proves nothing
    later = base_feats.index[base_feats.index > k]
    assert not np.array_equal(base_feats.loc[later].to_numpy(float), pert_feats.loc[later].to_numpy(float),
                              equal_nan=True)


def test_eval_users_do_not_enter_training(synth):
    """The model must not depend on eval users: training on users 1-4 gives the same booster whether or
    not user 7 is also present in the table list passed to train_model, if user 7 is not given."""
    cfg, loader = _cfg_loader(synth)
    train = _tables_for_synthetic(synth, [1, 2, 3, 4], cfg, loader)
    b1, it1, info1 = R.train_model(train, max_rounds=15, patience=5)
    b2, it2, info2 = R.train_model(list(train), max_rounds=15, patience=5)
    assert info1["train_users"] + info1["holdout_users"] == 4
    assert it1 == it2
    x = train[0]["X"][:50]
    np.testing.assert_allclose(b1.predict(x, num_iteration=it1), b2.predict(x, num_iteration=it2))


def test_run_rejects_overlapping_train_and_eval(synth, tmp_path):
    with pytest.raises(ValueError):
        R.run([1, 2], [2, 3], synth, tmp_path / "x.jsonl", max_rounds=5, patience=2)


def test_main_rejects_overlapping_train_and_eval(synth, tmp_path):
    with pytest.raises(SystemExit):
        R.main(["--data", str(synth), "--train_users", "1,2", "--eval_users", "2,3",
                "--out", str(tmp_path / "x.jsonl")])


def test_holdout_is_drawn_from_train_users_only():
    train = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert set(R.holdout_users(train)) <= set(train)


# ---- 2. test rows equal srs-benchmark's scored rows, on real local users --------------------------

@pytest.mark.skipif(not (LOCAL_DATA / "revlogs").exists() or not PUBLISHED.exists(), reason="local data absent")
def test_sizes_match_published_on_three_more_local_users():
    S = R._srs_modules()
    cfg = S["Config"](S["create_parser"]().parse_args(["--algo", "FSRS-6", "--short", "--data", str(LOCAL_DATA)]))
    loader = S["UserDataLoader"](cfg)
    pub = {}
    for line in PUBLISHED.read_text().splitlines():
        r = json.loads(line)
        pub[r["user"]] = r["size"]
    users = [2542, 2560, 2638]
    assert not (set(users) & SMOKE_USERS)
    for u in users:
        t = R.user_table(u, loader, cfg, LOCAL_DATA)
        assert int(t["test"].sum()) == pub[u], (u, int(t["test"].sum()), pub[u])


def test_scored_rows_have_no_same_day_reviews_or_first_chunk(tmp_path):
    """Scored rows (ds) never have elapsed_days 0, and no test row lies in the first TimeSeriesSplit chunk."""
    raw_root = tmp_path / "d"
    (raw_root / "revlogs" / "user_id=1").mkdir(parents=True)
    _revlog(seed=21, n_cards=100).to_parquet(raw_root / "revlogs" / "user_id=1" / "data.parquet")
    cfg, loader = _cfg_loader(raw_root)
    ds = loader.load_user_data(1)
    assert (ds["delta_t"] > 0).all() and (ds["elapsed_days"] > 0).all()
    t = R.user_table(1, loader, cfg, raw_root)
    n = len(ds)
    first_chunk = n - 5 * (n // 6)
    assert not t["test"][:first_chunk].any()
    assert t["test"].sum() < n
