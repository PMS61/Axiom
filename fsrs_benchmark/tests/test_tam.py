"""Tests for fsrs_benchmark/srsb/tam.py (Phase 6, task 6.4): past-only context (leakage), shapes and a
forward/backward pass on a real local user, the zero-initialised head (output = FSRS prior at init), and a
short training run that lowers the training log loss. Real data: local users > 1000 only (never 1-1000)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from fsrs_benchmark.srsb import tam as T

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "anki10k"
REAL_USER = 1236                  # local, outside 1-1000

needs_data = pytest.mark.skipif(not (DATA / "revlogs" / f"user_id={REAL_USER}").exists(),
                                reason="local anki10k data not present")


def _small_local_users(n: int) -> list[int]:
    """The n smallest local users above 1000 (keeps the training test fast)."""
    rev = DATA / "revlogs"
    uids = [int(p.name.split("=")[1]) for p in rev.iterdir() if p.name.startswith("user_id=")]
    uids = [u for u in uids if u > 1000]
    uids.sort(key=lambda u: (rev / f"user_id={u}").stat().st_size)
    return uids[:n]


def _perturb(raw: pd.DataFrame, t0: int) -> pd.DataFrame:
    """Change rating (valid ratings only) and duration of every raw row from position t0 on."""
    r = raw.reset_index(drop=True).copy()
    later = np.arange(len(r)) >= t0
    valid = r["rating"].isin([1, 2, 3, 4]).to_numpy()
    m = later & valid
    r.loc[m, "rating"] = (r.loc[m, "rating"] % 4) + 1
    r.loc[later, "duration"] = r.loc[later, "duration"] + 12345
    return r


def test_context_index_is_past_only_and_deduplicated():
    rng = np.random.default_rng(0)
    card = rng.integers(0, 6, 400)
    K, M = 20, 5
    ctx = T.context_index(card, K=K, M=M)
    assert ctx.shape == (400, K + M)
    for p in range(400):
        row = ctx[p][ctx[p] >= 0]
        assert (row < p).all(), f"row {p} sees itself or the future"
        assert len(np.unique(row)) == len(row), f"row {p} has duplicates"
        user_part = ctx[p, :K]
        assert set(user_part[user_part >= 0]) == set(range(max(0, p - K), p))
        card_part = ctx[p, K:][ctx[p, K:] >= 0]
        assert (card[card_part] == card[p]).all()
        assert not np.isin(card_part, user_part).any()
        assert len(card_part) <= M
        # every same-card review before the user window is in the card part, up to M of them
        older = np.flatnonzero(card[:max(0, p - K)] == card[p])
        assert len(card_part) == min(M, len(older))


@needs_data
def test_leakage_rating_and_duration_of_later_rows_do_not_change_earlier_outputs(tmp_path):
    cfg, loader = T._ctx(DATA)
    ds = loader.load_user_data(REAL_USER)
    raw = pd.read_parquet(DATA / "revlogs" / f"user_id={REAL_USER}")
    cards = T._read_opt(DATA / "cards" / f"user_id={REAL_USER}")
    decks = T._read_opt(DATA / "decks" / f"user_id={REAL_USER}")
    u0 = T.arrays_from_raw(raw, cfg, cards, decks, ds)

    t0 = 6000                                  # raw position; the changed rows have review_th >= t0 + 1
    u1 = T.arrays_from_raw(_perturb(raw, t0), cfg, cards, decks, None)
    assert len(u1["day"]) == len(u0["day"])
    k = int(np.searchsorted(u0["review_th"], t0 + 1, side="right"))   # rows with review_th <= t0 + 1
    assert 0 < k < len(u0["day"]) - 30

    scaler = T.QueryScaler.fit([u0])
    pool0 = T.build_pool([u0], scaler, rows="all")
    pool1 = T.build_pool([u1], scaler, rows="all")
    rng = np.random.default_rng(1)
    early = np.unique(np.concatenate([np.arange(max(0, k - 60), k), rng.choice(k, 200, replace=False)]))
    late = np.arange(k, k + 30)

    torch.manual_seed(0)
    model = T.TAM(2 * len(T.QUERY_COLS))
    with torch.no_grad():                      # a zero head would make every output the FSRS prior
        model.head[2].weight.normal_(0.0, 0.1)
    p0 = T.predict(model, pool0, early)
    p1 = T.predict(model, pool1, early)
    assert np.allclose(p0, p1, atol=1e-6), f"max diff {np.abs(p0 - p1).max()}"
    # sanity: the perturbation does reach the model for later rows, so the check above is not vacuous
    q0 = T.predict(model, pool0, late)
    q1 = T.predict(model, pool1, late)
    assert np.abs(q0 - q1).max() > 1e-4


@needs_data
def test_shapes_and_forward_backward_on_real_user(tmp_path):
    u = T.build_user_tensors(REAL_USER, DATA, cache_dir=tmp_path)
    n = len(u["day"])
    assert n > 1000 and u["tok"].shape == (n, T.TOK_DIM)
    assert u["xq"].shape == (n, len(T.QUERY_COLS)) and u["fr"].shape == (n,)
    assert u["scored"].sum() > 0 and (u["test"] <= u["scored"]).all()
    # the cache round-trips
    again = T.build_user_tensors(REAL_USER, DATA, cache_dir=tmp_path)
    assert all(np.array_equal(u[k], again[k], equal_nan=(u[k].dtype.kind == "f")) for k in u)

    scaler = T.QueryScaler.fit([u])
    pool = T.build_pool([u], scaler, rows="scored")
    model = T.TAM(2 * len(T.QUERY_COLS))
    n_params = sum(p.numel() for p in model.parameters())
    assert 0.5e6 < n_params < 0.75e6, n_params

    rows = np.arange(32)
    b = T.make_batch(pool, rows)
    L = T.K_USER + T.M_CARD + 1
    assert b["tok"].shape == (32, L, T.TOK_DIM) and b["ids"].shape == (32, L, 4)
    assert b["valid"].shape == (32, L) and b["q"].shape == (32, 2 * len(T.QUERY_COLS))
    assert b["valid"][:, -1].all()             # the query is always present
    logits = model(b)
    assert logits.shape == (32,) and torch.isfinite(logits).all()
    loss = T.bce(logits, b["y"])
    loss.backward()
    assert torch.isfinite(loss)
    assert model.head[2].weight.grad.abs().sum() > 0


@needs_data
def test_init_output_is_close_to_fsrs_prior(tmp_path):
    u = T.build_user_tensors(REAL_USER, DATA, cache_dir=tmp_path)
    scaler = T.QueryScaler.fit([u])
    pool = T.build_pool([u], scaler, rows="scored")
    torch.manual_seed(0)
    model = T.TAM(2 * len(T.QUERY_COLS))
    p = T.predict(model, pool, np.arange(64))
    prior = np.clip(u["fr"][u["scored"]][:64], T.P_CLIP, 1 - T.P_CLIP)
    assert np.allclose(p, prior, atol=1e-5), np.abs(p - prior).max()


@needs_data
def test_training_lowers_train_log_loss(tmp_path):
    users = [T.build_user_tensors(u, DATA, cache_dir=tmp_path) for u in _small_local_users(5)]
    scaler = T.QueryScaler.fit(users)
    pool = T.build_pool(users, scaler, rows="scored")
    eval_rows = np.random.default_rng(2).choice(len(pool), 512, replace=False)
    torch.manual_seed(0)
    model = T.TAM(2 * len(T.QUERY_COLS))
    before = T.mean_log_loss(model, pool, eval_rows)
    losses = T.train(model, pool, steps=200, batch_size=8, lr=1e-3, seed=0)
    after = T.mean_log_loss(model, pool, eval_rows)
    assert all(np.isfinite(losses))
    assert after < before, (before, after)
