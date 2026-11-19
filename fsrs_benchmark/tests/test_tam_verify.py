"""Verifier checks for Phase 6.4 steps 1-3 (srsb/tam.py): relation flags, context = exactly the rows the
design names (own row, later rows and non-context rows cannot change a target's output), query token
invisible to context tokens, scored/test masks equal to srs-benchmark's user_table, zero-initialised head
gives f_R on every scored row, padding is inert, short histories and first reviews give finite outputs,
and users 1-1000 are refused. Real data: local user 1236 (outside 1-1000)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from fsrs_benchmark.srsb import run_gbdt as R
from fsrs_benchmark.srsb import tam as T

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "anki10k"
UID = 1236
needs_data = pytest.mark.skipif(not (DATA / "revlogs" / f"user_id={UID}").exists(),
                                reason="local anki10k data not present")


@pytest.fixture(scope="module")
def user(tmp_path_factory):
    cache = tmp_path_factory.mktemp("tam_verify_cache")
    u = T.build_user_tensors(UID, DATA, cache_dir=cache)
    scaler = T.QueryScaler.fit([u])
    pool = T.build_pool([u], scaler, rows="all")
    return u, pool


def _model(seed=0, nonzero_head=True):
    torch.manual_seed(seed)
    m = T.TAM(2 * len(T.QUERY_COLS))
    if nonzero_head:
        with torch.no_grad():
            m.head[2].weight.normal_(0.0, 0.1)
    m.eval()
    return m


def _logits(model, pool, rows):
    with torch.no_grad():
        return model(T.make_batch(pool, np.asarray(rows))).numpy()


# ---- relation flags ------------------------------------------------------------------------------

def test_relation_codes_card_note_deck_preset_other():
    # id tuples (card, note, deck, preset); -1 means unknown and never matches
    a = torch.tensor([[1, 10, 100, 7]] * 6)
    b = torch.tensor([
        [1, 10, 100, 7],     # same card -> 0 (also same note/deck/preset: finest wins)
        [2, 10, 100, 7],     # same note, deck, preset -> note (1): list order card > note > deck > preset
        [2, 11, 100, 7],     # same deck and preset -> 2 (deck is finer than preset)
        [2, 11, 101, 7],     # same preset only -> 3
        [2, 11, 101, 8],     # nothing -> 4
        [-1, -1, -1, -1],    # unknown ids never match -> 4
    ])
    got = T.relation(a, b).tolist()
    assert got == [0, 1, 2, 3, 4, 4]
    note_only = torch.tensor([[2, 10, 101, 8]])
    assert T.relation(torch.tensor([[1, 10, 100, 7]]), note_only).tolist() == [1]


# ---- scored / test masks against srs-benchmark ----------------------------------------------------

@needs_data
def test_scored_and_test_masks_equal_user_table(user):
    u, _ = user
    cfg, loader = T._ctx(DATA)
    ut = R.user_table(UID, loader, cfg, DATA)
    ds_keys = loader.load_user_data(UID)["review_th"].to_numpy(np.int64)
    review_th = np.asarray(u["review_th"])
    pos = np.searchsorted(review_th, ds_keys)
    assert np.array_equal(review_th[pos], ds_keys)
    assert u["scored"].sum() == len(ds_keys) and u["scored"][pos].all()
    assert np.array_equal(u["test"][pos], ut["test"])
    assert int(u["test"].sum()) == int(ut["test"].sum())
    assert (u["test"] <= u["scored"]).all()
    assert np.array_equal(u["y"][pos].astype(np.int64), ut["y"])


# ---- context: exactly the design's rows -------------------------------------------------------------

def test_perturbing_non_context_rows_does_not_change_output(user):
    u, pool = user
    model = _model()
    ctx_all = T.context_index(u["card"])
    rng = np.random.default_rng(3)
    n = len(u["day"])
    targets = rng.choice(np.arange(400, n - 1), size=6, replace=False)
    for p in targets:
        ctx = set(int(c) for c in ctx_all[p] if c >= 0)
        far = [j for j in range(0, p) if j not in ctx and j != p]
        base = _logits(model, pool, [p])
        # token inputs of rows outside the context (earlier, not in the window, not on this card)
        mod = pool.tok.copy()
        for j in rng.choice(far, size=min(40, len(far)), replace=False):
            mod[j, :] = rng.random(T.TOK_DIM).astype(np.float32)
        saved = pool.tok
        pool.tok = mod
        try:
            other = _logits(model, pool, [p])
        finally:
            pool.tok = saved
        assert np.array_equal(base, other), f"target {p}: a non-context row changed the output"


def test_perturbing_context_row_changes_output(user):
    """Non-vacuity: the previous row (always in the window) must matter."""
    u, pool = user
    model = _model()
    p = 500
    base = _logits(model, pool, [p])
    mod = pool.tok.copy()
    mod[p - 1, :] = 0.0
    mod[p - 1, 0] = 1.0
    saved = pool.tok
    pool.tok = mod
    try:
        other = _logits(model, pool, [p])
    finally:
        pool.tok = saved
    assert not np.array_equal(base, other)


def test_target_own_rating_duration_and_label_are_not_inputs(user):
    """Changing the target's own token (rating, duration, state, first flag) must not change its logit."""
    u, pool = user
    model = _model()
    rng = np.random.default_rng(5)
    for p in rng.choice(np.arange(50, len(u["day"]) - 1), size=8, replace=False):
        base = _logits(model, pool, [p])
        mod = pool.tok.copy()
        mod[p, :] = rng.random(T.TOK_DIM).astype(np.float32)
        saved = pool.tok
        pool.tok = mod
        try:
            other = _logits(model, pool, [p])
        finally:
            pool.tok = saved
        assert np.array_equal(base, other), f"row {p} sees its own token"


def test_query_token_is_invisible_to_context_positions(user):
    """Hook the block outputs: changing the query features of the target leaves every context position's
    hidden state unchanged, in every layer."""
    u, pool = user
    model = _model()
    p = 700
    b = T.make_batch(pool, np.array([p]))
    states = []

    def run(batch):
        states.clear()
        hooks = [blk.register_forward_hook(lambda m, i, o: states.append(o.detach().clone()))
                 for blk in model.blocks]
        with torch.no_grad():
            model(batch)
        for h in hooks:
            h.remove()
        return [s[:, :-1].clone() for s in states]

    s0 = run(b)
    b2 = dict(b)
    b2["q"] = b["q"] + 5.0
    s1 = run(b2)
    for l, (x, y) in enumerate(zip(s0, s1)):
        assert torch.equal(x, y), f"layer {l}: a context position sees the query"


# ---- zero head: output is the FSRS prior -----------------------------------------------------------

def test_zero_head_gives_f_r_on_every_scored_row(user):
    u, pool = user
    model = _model(nonzero_head=False)
    rows = np.flatnonzero(u["scored"])
    rows = rows[:400]
    p = 1.0 / (1.0 + np.exp(-_logits(model, pool, rows)))
    prior = np.clip(u["fr"][rows], T.P_CLIP, 1 - T.P_CLIP)
    assert np.allclose(p, prior, atol=1e-5), np.abs(p - prior).max()


# ---- padding, short histories, NaN ------------------------------------------------------------------

def test_padding_contents_are_inert(user):
    u, pool = user
    model = _model()
    rows = np.array([0, 1, 2, 3, 5, 40])            # empty or very short histories
    base = _logits(model, pool, rows)
    N = pool.n
    saved = (pool.tok, pool.ids, pool.day)
    pool.tok = pool.tok.copy()
    pool.ids = pool.ids.copy()
    pool.day = pool.day.copy()
    pool.tok[N] = 3.0
    pool.ids[N] = 7
    pool.day[N] = 1e4
    try:
        other = _logits(model, pool, rows)
    finally:
        pool.tok, pool.ids, pool.day = saved
    assert np.array_equal(base, other)


def test_short_histories_are_padded_and_outputs_finite(user):
    u, pool = user
    model = _model()
    rows = np.arange(0, 300)                         # first reviews and the first K rows of the stream
    b = T.make_batch(pool, rows)
    valid = b["valid"].numpy()
    assert valid[0, :-1].sum() == 0                  # row 0 has no context at all
    assert valid[3, :-1].sum() == 3                  # row 3 sees rows 0..2
    assert valid[:, -1].all()
    logits = _logits(model, pool, rows)
    assert np.isfinite(logits).all()
    assert all(np.isfinite(v.numpy()).all() for v in b.values())


def test_no_nan_in_query_or_token_inputs(user):
    u, pool = user
    assert np.isfinite(pool.q).all() and np.isfinite(pool.tok).all() and np.isfinite(pool.day).all()
    assert np.isfinite(pool.fr).all()


# ---- guard -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("uid", [1, 500, 1000])
def test_test_set_users_are_refused(uid):
    with pytest.raises(ValueError, match="final test set"):
        T.build_user_tensors(uid, DATA)
