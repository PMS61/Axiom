"""Tests for fsrs_benchmark/srsb/resid_gru.py (Phase 6, task 6.6): guards, the leak rules (row t never sees y or
anything after t), zero head equals the GBDT, chunked carry equals the full pass, and a tiny end-to-end run on
local users above 1000 (never the test set 1-1000)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from fsrs_benchmark.srsb import resid_gru as RG
from fsrs_benchmark.srsb import run_gbdt as R
from fsrs_benchmark.srsb import val_plus as V

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "anki10k"


def _random_booster(path: Path):
    """A small LightGBM on the 78 GBDT features (random labels), saved as <path>.model.txt + .model.json."""
    import lightgbm as lgb
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, len(RG.COLS))).astype(np.float32)
    y = (rng.random(600) < 0.7).astype(int)
    b = lgb.train(dict(objective="binary", num_leaves=7, verbose=-1, min_data_in_leaf=5, seed=0),
                  lgb.Dataset(X, y), 10)
    model_txt = Path(path)
    b.save_model(str(model_txt))
    Path(str(model_txt).replace(".model.txt", ".model.json")).write_text(
        json.dumps({"best_iter": 10}), encoding="utf-8")
    return model_txt, b


def _synthetic_rows(n=300, seed=1):
    rng = np.random.default_rng(seed)
    m = rng.normal(0, 2, n)
    y = (rng.random(n) < 0.8).astype(float)
    card = rng.integers(0, 40, n)
    known = rng.normal(size=(n, RG.KNOWN_COLS)).astype(np.float32)
    return m, y, card, known


# ---- guards -----------------------------------------------------------------------------------------

def test_guard_test_set_and_validation_and_gbdt_in_sample():
    model = Path("/nonexistent/none.model.txt")          # no run_info next to it: the 2001-2300 range only
    with pytest.raises(ValueError, match="test set"):
        RG.check_users([5, 2301], [8001], model)
    with pytest.raises(ValueError, match="validation"):
        RG.check_users([8001, 2301], [8002], model)
    with pytest.raises(ValueError, match="in-sample"):
        RG.check_users([2301], [2002], model)            # 2002 was a GBDT train user
    with pytest.raises(ValueError, match="in-sample"):
        RG.check_users([2301], [2300], model)            # boundary of the GBDT train range
    RG.check_users([2301, 2500], [8001, 8100], model)    # the planned sets pass
    RG.check_users([1236], [1737], model)                # local users above 1000, outside the GBDT range


def test_gbdt_train_users_reads_run_info(tmp_path):
    model = tmp_path / "m.model.txt"
    (tmp_path / "m.run_info.json").write_text(json.dumps({"train_users": [3000, 3001]}), encoding="utf-8")
    users = RG.gbdt_train_users(model)
    assert {3000, 3001} <= users and 2001 in users and 2300 in users and 2301 not in users


# ---- leak rules at the input level ------------------------------------------------------------------

def test_inputs_row_t_uses_no_label_at_or_after_t():
    m, y, card, known = _synthetic_rows()
    X = RG.build_inputs(m, y, card, known)
    for t in (0, 1, 17, 150, 299):
        y2 = y.copy()
        y2[t:] = 1.0 - y2[t:]                             # change every label from t on
        X2 = RG.build_inputs(m, y2, card, known)
        assert np.array_equal(X[:t + 1], X2[:t + 1]), f"input changed at or before row {t}"


def test_inputs_row_t_uses_no_margin_after_t():
    m, y, card, known = _synthetic_rows()
    X = RG.build_inputs(m, y, card, known)
    t = 120
    m2 = m.copy()
    m2[t + 1:] += 5.0
    X2 = RG.build_inputs(m2, y, card, known)
    assert np.array_equal(X[:t + 1], X2[:t + 1])


def test_inputs_card_lag_is_previous_row_of_the_same_card():
    m = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([1.0, 0.0, 1.0, 1.0])
    card = np.array([7, 8, 7, 7])
    known = np.zeros((4, RG.KNOWN_COLS), np.float32)
    X = RG.build_inputs(m, y, card, known)
    # layout: known (KNOWN_COLS), m, prev y, prev m, has, card residual, log gap
    has, res, gap = X[:, RG.KNOWN_COLS + 3], X[:, RG.KNOWN_COLS + 4], X[:, RG.KNOWN_COLS + 5]
    assert has.tolist() == [0.0, 0.0, 1.0, 1.0]
    p0 = 1 / (1 + np.exp(-m[0]))
    assert res[2] == pytest.approx(1.0 - p0, rel=1e-5)     # row 2 sees card 7's row 0
    assert gap[2] == pytest.approx(np.log1p(2), rel=1e-6)
    assert res[3] == pytest.approx(1.0 - 1 / (1 + np.exp(-m[2])), rel=1e-5)


# ---- model ------------------------------------------------------------------------------------------

def _random_head_model(seed=0):
    torch.manual_seed(seed)
    model = RG.ResidGRU(RG.K_IN, 16)
    with torch.no_grad():
        model.head.weight.normal_(0, 0.5)
        model.head.bias.fill_(0.1)
    return model


def test_model_output_at_t_does_not_depend_on_later_inputs():
    m, y, card, known = _synthetic_rows(200)
    X = RG.build_inputs(m, y, card, known)
    model = _random_head_model()
    full, _ = model(torch.from_numpy(X)[None])
    t = 90
    y2 = y.copy()
    y2[t:] = 1.0 - y2[t:]
    X2 = RG.build_inputs(m, y2, card, known)
    d2, _ = model(torch.from_numpy(X2)[None])
    assert torch.allclose(full[0, :t + 1], d2[0, :t + 1], atol=1e-6)
    assert not torch.allclose(full[0, t + 1:], d2[0, t + 1:])   # the change does reach later rows


def test_zero_head_equals_gbdt():
    m, y, card, known = _synthetic_rows(120)
    X = RG.build_inputs(m, y, card, known)
    model = RG.ResidGRU(RG.K_IN, 16)                       # fresh: head is zero
    delta = RG.predict_delta(model, X)
    assert np.all(delta == 0.0)
    assert np.allclose(RG.probs(m + delta), RG.probs(m), atol=0, rtol=0)
    assert np.allclose(RG.probs(m), 1 / (1 + np.exp(-m)), atol=1e-9)


def test_zero_head_equals_real_booster_predictions(tmp_path):
    model_txt, booster = _random_booster(tmp_path / "g.model.txt")
    rng = np.random.default_rng(3)
    Xf = rng.normal(size=(150, len(RG.COLS))).astype(np.float32)
    raw = booster.predict(Xf, raw_score=True, num_iteration=10)
    plain = booster.predict(Xf, num_iteration=10)
    assert np.allclose(RG.probs(raw), np.clip(plain, RG.G.P_CLIP, 1 - RG.G.P_CLIP), atol=1e-9)
    X = RG.build_inputs(raw, (rng.random(150) < 0.5).astype(float), rng.integers(0, 9, 150),
                        np.zeros((150, RG.KNOWN_COLS), np.float32))
    delta = RG.predict_delta(RG.ResidGRU(RG.K_IN, 16), X)
    assert np.allclose(RG.probs(raw + delta), np.clip(plain, RG.G.P_CLIP, 1 - RG.G.P_CLIP), atol=1e-6)


def test_chunked_pass_with_carried_state_equals_full_pass():
    m, y, card, known = _synthetic_rows(230)
    X = torch.from_numpy(RG.build_inputs(m, y, card, known))
    model = _random_head_model(seed=2)
    full, _ = model(X[None])
    h = torch.zeros(1, 1, 16)
    parts = []
    for s in range(0, len(X), 64):
        d, h = model(X[None, s:s + 64], h)
        parts.append(d)
    assert torch.allclose(torch.cat(parts, dim=1)[0], full[0], atol=1e-6)


def test_training_moves_the_head_and_lowers_loss():
    m, y, card, known = _synthetic_rows(400, seed=5)
    X = RG.build_inputs(m, y, card, known)
    users = [dict(X=X, m=m.astype(np.float32), y=y.astype(np.float32), loss=np.ones(len(y), bool))]
    model = _random_head_model(seed=4)
    before = model.head.weight.detach().clone()
    hist = RG.train_model(model, users, epochs=3, lr=1e-2, seed=0, chunk=64, log=lambda *a: None)
    assert not torch.equal(before, model.head.weight.detach())
    assert hist[-1] < hist[0]


# ---- end to end -------------------------------------------------------------------------------------

def _local_user_ok(u):
    return (DATA / "revlogs" / f"user_id={u}" / "data.parquet").exists()


def test_leak_end_to_end_future_labels_and_durations_do_not_change_past(tmp_path):
    """Local user above 1000: change rating (y) and duration of every raw row from position K on. The GBDT margins,
    the GRU inputs and the GRU outputs of all rows before K must not change."""
    if not _local_user_ok(1236):
        pytest.skip("local data for user 1236 not present")
    uid, K = 1236, 1500
    _, booster = _random_booster(tmp_path / "g.model.txt")
    S = R._srs_modules()
    cfg = S["Config"](S["create_parser"]().parse_args(["--algo", "FSRS-6", "--short", "--data", str(DATA)]))
    loader = S["UserDataLoader"](cfg)
    raw = pd.read_parquet(DATA / "revlogs" / f"user_id={uid}").iloc[:3000].reset_index(drop=True)
    ds_all = loader.load_user_data(uid)
    ds = ds_all[ds_all["review_th"] <= len(raw)].reset_index(drop=True)
    assert len(ds) > 100

    raw2 = raw.copy()
    later = raw2.index >= K
    rating = raw2["rating"].to_numpy().copy()
    flip = later & np.isin(rating, [1, 3])
    rating[flip] = np.where(rating[flip] == 1, 3, 1)      # y changes for these rows (1 <-> 3 keeps the row)
    raw2["rating"] = rating
    raw2.loc[later, "duration"] = raw2.loc[later, "duration"] + 12345

    model = _random_head_model(seed=7)
    a = RG.prepare_user_raw(raw, ds, cfg, None, None, booster, 10, check_y=False)
    b = RG.prepare_user_raw(raw2, ds, cfg, None, None, booster, 10, check_y=False)
    st_idx = R.review_stream(raw, cfg.max_seq_len * 2).index.to_numpy()
    k = int((st_idx <= K).sum())                          # stream rows at or before raw position K
    assert 50 < k < len(st_idx)
    assert not np.array_equal(a["y"][k:], b["y"][k:])     # the future really changed
    assert np.array_equal(a["m"][:k], b["m"][:k])
    assert np.array_equal(a["X"][:k], b["X"][:k])
    da = RG.predict_delta(model, a["X"])
    db = RG.predict_delta(model, b["X"])
    assert np.allclose(da[:k], db[:k], atol=1e-6)
    assert np.allclose(RG.probs(a["m"] + da)[:k], RG.probs(b["m"] + db)[:k], atol=1e-6)


# ---- tiny end-to-end run on three local users above 1000 --------------------------------------------

def test_tiny_end_to_end_run_writes_outputs(tmp_path):
    users = [1236, 1347, 1737]
    if not all(_local_user_ok(u) for u in users):
        pytest.skip("need three local users above 1000")
    gbdt_txt, _ = _random_booster(tmp_path / "gbdt.model.txt")
    out = tmp_path / "gru.jsonl"
    info = RG.run([1236, 1347], [1737], DATA, out, gbdt_model=gbdt_txt, epochs=1, hidden=16, lr=1e-3,
                  result_dir=tmp_path, log=lambda *a: None)
    lines = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert [r["user"] for r in lines] == [1737]
    assert lines[0]["size"] > 0
    assert {"RMSE", "LogLoss", "RMSE(bins)", "ICI", "AUC", "MBE"} <= set(lines[0]["metrics"])
    gb = [json.loads(x) for x in Path(str(out) + ".gbdt_only.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["user"] for r in gb] == [1737] and gb[0]["size"] == lines[0]["size"]
    assert Path(str(out) + ".model.pt").exists()
    saved = json.loads(Path(str(out) + ".run_info.json").read_text(encoding="utf-8"))
    assert len(saved["train_loss_per_epoch"]) == 1 and saved["eval_lines"] == 1
    assert info["features"] == RG.COLS
    assert set(saved["comparisons"]) == {f"{a} | {b}" for a in ("GBDT + residual GRU", "GBDT alone (reference)")
                                         for b in ("8001-8050", "8051-8100", "all")}


def test_run_refuses_gbdt_train_users(tmp_path):
    gbdt_txt, _ = _random_booster(tmp_path / "gbdt.model.txt")
    with pytest.raises(ValueError, match="in-sample"):
        RG.run([2002], [1737], DATA, tmp_path / "x.jsonl", gbdt_model=gbdt_txt, result_dir=tmp_path)
    assert not (tmp_path / "x.jsonl").exists()
