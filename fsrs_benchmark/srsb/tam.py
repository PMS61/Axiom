"""TAM: time-aware attention memory for srs-benchmark (Phase 6, task 6.4). Design: paper/docs/SRSB_ARCH.md.

Data. A user's review stream (run_gbdt.review_stream, rows in review_th order) becomes per-row arrays. Every
row of the stream is a token (what a past review contributes). A row is a target when it is scored (the ds
rows of srs-benchmark's loader). Targets see only the rows strictly before them (see the context rule below).

Time. The only time axis in the data is day_offset (whole days), so relative time is in days. Same-day
reviews therefore have relative time 0.

Context of target t: the user's last K_USER reviews before t, plus the card's last M_CARD reviews before t
(the card's reviews are dropped when they already fall inside the user window: deduplicated). The current
row's rating and duration are never in its own context: the context is strictly earlier rows only.

Model. Context tokens and a query token (the target's query features) go through 3 pre-norm transformer
layers (d=128, 4 heads). Attention logits get a learned per-head decay a_h * log1p(relative days) and a
learned relation bias per (layer, head, relation). Relations: 0 same card, 1 same note, 2 same deck,
3 same preset, 4 other. Context tokens cannot see the query token; the query token sees all context.
Output: p = sigmoid(logit(f_R) + MLP(query output)); the MLP's last layer is zero at init, so p = f_R at
step 0.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from fsrs_benchmark.models import gbdt as G
from fsrs_benchmark.srsb import run_gbdt as R
from fsrs_benchmark.srsb.features_plus import PLUS_COLS, plus_features

K_USER = 256
M_CARD = 32
QUERY_COLS = R.FEATURE_COLS + PLUS_COLS
TOK_DIM = 12            # rating one-hot 4, log1p duration, log1p el_frac, is_first, state one-hot 5
N_REL = 5               # 0 same card, 1 same note, 2 same deck, 3 same preset, 4 other
REL_OTHER = 4
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 3
D_FFN = 512
DEFAULT_FR = 0.9        # used where the FSRS prior is NaN (first reviews)
P_CLIP = 1e-6
TEST_LO, TEST_HI = 1, 1000   # srs-benchmark users 1-1000 are the final test set and never used here

_CTX: dict = {}


def _ctx(data_dir: Path):
    """srs-benchmark Config and UserDataLoader for one data directory (built once)."""
    key = str(Path(data_dir).resolve())
    if key not in _CTX:
        R._init_worker(str(data_dir))
        _CTX[key] = (R._W["cfg"], R._W["loader"])
    return _CTX[key]


def _int_ids(values) -> np.ndarray:
    """Float ids (NaN when unknown) to int64 with -1 for unknown."""
    a = np.asarray(values, dtype=float)
    return np.where(np.isnan(a), -1, a).astype(np.int64)


def _map_attr(keys: np.ndarray, table: pd.DataFrame | None, key: str, val: str) -> np.ndarray:
    """table[val] looked up by table[key] == keys (float, NaN where unknown or keys == -1)."""
    if table is None:
        return np.full(len(keys), np.nan)
    m = table.drop_duplicates(key).set_index(key)[val]
    return m.reindex(keys).to_numpy(float)


def arrays_from_raw(raw: pd.DataFrame, cfg, cards: pd.DataFrame | None = None,
                    decks: pd.DataFrame | None = None, ds: pd.DataFrame | None = None) -> dict:
    """Per-row arrays for one user's whole review stream (review_th order).

    ds: srs-benchmark's scored rows for the user (UserDataLoader output). When given, scored marks its rows
    and test marks the TimeSeriesSplit test rows; labels must agree. When None, no row is scored.
    """
    st = R.review_stream(raw, cfg.max_seq_len * 2)
    n = len(st)
    ar = np.arange(n)
    card = st["card_id"].to_numpy(np.int64)
    rating = st["rating"].to_numpy(np.int64)
    state = st["state"].to_numpy(np.int64).clip(0, 4)
    day = st["day_offset"].to_numpy(np.int64)
    dur = st["duration"].to_numpy(float)
    secs = st["elapsed_seconds"].to_numpy(float)
    first = st["is_first"].to_numpy(bool)

    deck = _int_ids(_map_attr(card, cards, "card_id", "deck_id"))
    note = _int_ids(_map_attr(card, cards, "card_id", "note_id"))
    preset = _int_ids(_map_attr(deck, decks, "deck_id", "preset_id"))

    el = np.where(~first & (secs >= 0), secs / 86400.0, np.nan)
    tok = np.zeros((n, TOK_DIM), np.float32)
    tok[ar, rating - 1] = 1.0
    tok[:, 4] = np.log1p(np.clip(dur, 0, None))
    tok[:, 5] = np.log1p(np.nan_to_num(el))
    tok[:, 6] = first
    tok[ar, 7 + state] = 1.0

    base = G.base_features(st, span=20, user_group="row")
    pf = plus_features(st, cards, decks)
    xq = pd.concat([base[R.FEATURE_COLS], pf[PLUS_COLS]], axis=1).to_numpy(np.float32)
    fr = base["f_R"].to_numpy(float)
    fr = np.where(np.isnan(fr), DEFAULT_FR, fr).astype(np.float32)

    y = (rating >= 2).astype(np.int8)
    scored = np.zeros(n, bool)
    test = np.zeros(n, bool)
    meta_gap = np.zeros(n, np.float64)
    meta_i = np.zeros(n, np.float64)
    meta_rmse = np.zeros(n, np.float64)
    if ds is not None:
        keys = ds["review_th"].to_numpy(np.int64)
        review_th = st.index.to_numpy(np.int64)
        pos = np.searchsorted(review_th, keys)
        if (pos >= n).any() or not np.array_equal(review_th[pos], keys):
            raise RuntimeError("scored rows missing from the review stream")
        if not np.array_equal(y[pos].astype(np.int64), ds["y"].to_numpy(np.int64)):
            raise RuntimeError("labels differ between the review stream and srs-benchmark")
        scored[pos] = True
        test[pos[R.test_row_mask(len(ds), cfg.n_splits)]] = True
        meta_gap[pos] = ds["elapsed_days"].to_numpy(np.float64)
        meta_i[pos] = ds["i"].to_numpy(np.float64)
        meta_rmse[pos] = ds["rmse_bins_lapse"].to_numpy(np.float64)

    return dict(review_th=st.index.to_numpy(np.int64), day=day, card=card, note=note, deck=deck,
                preset=preset, tok=tok, xq=xq, fr=fr, y=y, scored=scored, test=test,
                meta_gap=meta_gap, meta_i=meta_i, meta_rmse=meta_rmse)


def build_user_tensors(uid: int, data_dir, cache_dir=None) -> dict:
    """Per-row arrays for one local user (see arrays_from_raw). With cache_dir, reads or writes
    <cache_dir>/user_<uid>.npz. Users 1-1000 are refused: they are the final test set."""
    if TEST_LO <= uid <= TEST_HI:
        raise ValueError(f"user {uid} is in the final test set (1-1000) and must not be used here")
    data_dir = Path(data_dir)
    path = Path(cache_dir) / f"user_{uid}.npz" if cache_dir is not None else None
    if path is not None and path.exists():
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    cfg, loader = _ctx(data_dir)
    ds = loader.load_user_data(uid)
    raw = pd.read_parquet(data_dir / "revlogs" / f"user_id={uid}")
    cards = _read_opt(data_dir / "cards" / f"user_id={uid}")
    decks = _read_opt(data_dir / "decks" / f"user_id={uid}")
    arrs = arrays_from_raw(raw, cfg, cards, decks, ds)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrs)
    return arrs


def _read_opt(path: Path) -> pd.DataFrame | None:
    return pd.read_parquet(path) if path.exists() else None


class QueryScaler:
    """Column mean and std of the query features (fit on training scored rows). A NaN becomes the mean and
    a missing-indicator column is appended, so the query vector has 2 * len(QUERY_COLS) entries."""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    @classmethod
    def fit(cls, users: list[dict], max_rows_per_user: int = 2000) -> "QueryScaler":
        parts = []
        for u in users:
            idx = np.flatnonzero(u["scored"])
            idx = idx[:: max(1, len(idx) // max_rows_per_user)]
            parts.append(np.asarray(u["xq"][idx], np.float32))
        X = np.concatenate(parts)
        with np.errstate(all="ignore"):
            mean = np.nanmean(X, axis=0)
            std = np.nanstd(X, axis=0)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0)
        return cls(mean, std)

    def transform(self, xq: np.ndarray) -> np.ndarray:
        miss = np.isnan(xq)
        xf = np.where(miss, self.mean, xq)
        z = np.clip((xf - self.mean) / self.std, -10.0, 10.0)
        return np.concatenate([z, miss.astype(np.float32)], axis=1).astype(np.float32)


def dense_ids(a: np.ndarray) -> np.ndarray:
    """Per-user dense int32 ids (relations only compare ids within one user). Unknown (<0) stays -1."""
    a = np.asarray(a, np.int64)
    out = np.full(len(a), -1, np.int32)
    m = a >= 0
    if m.any():
        out[m] = np.unique(a[m], return_inverse=True)[1].astype(np.int32)
    return out


def context_index(card: np.ndarray, K: int = K_USER, M: int = M_CARD, rows: np.ndarray | None = None) -> np.ndarray:
    """(len(rows), K+M) int32 local positions of each target row's context, -1 for padding (all rows by
    default). Columns 0..K-1: the user's window, positions p-K .. p-1. Columns K..K+M-1: the card's previous
    reviews before p, most recent first, skipping any that lie inside the user window (deduplicated)."""
    n = len(card)
    p = np.arange(n, dtype=np.int64) if rows is None else np.asarray(rows, np.int64)
    out = np.full((len(p), K + M), -1, np.int64)
    j = p[:, None] - K + np.arange(K)[None, :]
    out[:, :K] = np.where(j >= 0, j, -1)
    prev = pd.Series(np.arange(n)).groupby(card).shift(1).to_numpy(float)
    prev = np.where(np.isnan(prev), -1, prev).astype(np.int64)
    cur = prev[p]
    for m in range(M):
        ok = cur >= 0
        out[:, K + m] = np.where(ok & (cur < p - K), cur, -1)
        cur = np.where(ok, prev[np.maximum(cur, 0)], -1)
    return out.astype(np.int32)


class Pool:
    """Several users' rows in one global table. Row g of the table is the stream row of some user. The dummy
    row n (last) is all zeros with ids -1; padding in context indices points there."""

    def __init__(self, tok, ids, day, q, fr_logit, ctx, tgt, y, n):
        self.tok, self.ids, self.day, self.q, self.fr = tok, ids, day, q, fr_logit
        self.ctx, self.tgt, self.y, self.n = ctx, tgt, y, n

    def __len__(self):
        return len(self.tgt)


def build_pool(users: list[dict], scaler: QueryScaler, rows: str = "scored") -> Pool:
    """Pool over the users' scored rows (rows="scored"), their test rows (rows="test") or all their stream
    rows (rows="all"). Tokens are kept for every row of each user (context); query features and FSRS priors
    are kept for the target rows only. Host storage: tokens and queries float16, ids int32."""
    toks, ids, days, qs, frs, ctxs, tgts, ys = [], [], [], [], [], [], [], []
    off = 0
    for u in users:
        n = len(u["day"])
        if rows == "scored":
            sel = np.flatnonzero(u["scored"])
        elif rows == "test":
            sel = np.flatnonzero(u["test"])
        else:
            sel = np.arange(n)
        ctx = context_index(u["card"], rows=sel).astype(np.int64)
        ctxs.append(np.where(ctx >= 0, ctx + off, -1).astype(np.int32))
        tgts.append(sel + off)
        ys.append(u["y"][sel])
        toks.append(np.asarray(u["tok"], np.float16))
        ids.append(np.stack([dense_ids(u["card"]), dense_ids(u["note"]), dense_ids(u["deck"]),
                             dense_ids(u["preset"])], axis=1))
        days.append(np.asarray(u["day"], np.float32))
        qs.append(scaler.transform(np.asarray(u["xq"][sel], np.float32)).astype(np.float16))
        frs.append(np.asarray(u["fr"][sel], np.float32))
        off += n
    N = off
    width = scaler.mean.shape[0] * 2
    fr = np.clip(np.concatenate(frs) if frs else np.zeros(0, np.float32), P_CLIP, 1 - P_CLIP)
    ctx = np.concatenate(ctxs) if ctxs else np.zeros((0, K_USER + M_CARD), np.int32)
    return Pool(
        tok=np.concatenate(toks + [np.zeros((1, TOK_DIM), np.float16)]),
        ids=np.concatenate(ids + [np.full((1, 4), -1, np.int32)]),
        day=np.concatenate(days + [np.zeros(1, np.float32)]),
        q=np.concatenate(qs) if qs else np.zeros((0, width), np.float16),
        fr_logit=np.log(fr / (1 - fr)).astype(np.float32),
        ctx=np.where(ctx < 0, N, ctx).astype(np.int32),
        tgt=np.concatenate(tgts).astype(np.int64) if tgts else np.zeros(0, np.int64),
        y=np.concatenate(ys).astype(np.float32) if ys else np.zeros(0, np.float32),
        n=N,
    )


class TorchPool:
    """A Pool with its arrays on one device (tokens, query features and context indices stay resident, so a
    batch is only index gathers). Built once per run from a numpy Pool."""

    def __init__(self, pool: Pool, device):
        self.device = device
        self.tok = torch.as_tensor(pool.tok, device=device).float()     # fp16 on host, fp32 on device
        self.ids = torch.as_tensor(pool.ids, device=device).long()
        self.day = torch.as_tensor(pool.day, device=device)
        self.q = torch.as_tensor(pool.q, device=device).float()
        self.fr = torch.as_tensor(pool.fr, device=device)
        self.ctx = torch.as_tensor(pool.ctx, device=device)          # int32, converted per batch
        self.tgt = torch.as_tensor(pool.tgt, device=device)
        self.y = torch.as_tensor(pool.y, device=device)
        self.y_np = pool.y
        self.n = pool.n

    def __len__(self):
        return len(self.tgt)


def make_batch_t(tp: TorchPool, rows: torch.Tensor) -> dict:
    """Device version of make_batch: rows is a long tensor of pool rows."""
    g = tp.tgt[rows]
    idx = torch.cat([tp.ctx[rows].long(), g[:, None]], dim=1)
    return dict(tok=tp.tok[idx], ids=tp.ids[idx], day=tp.day[idx], valid=idx != tp.n,
                q=tp.q[rows], fr=tp.fr[rows], y=tp.y[rows])


def _as_torch(pool, device) -> TorchPool:
    return pool if isinstance(pool, TorchPool) else TorchPool(pool, device)


def make_batch(pool: Pool, rows: np.ndarray) -> dict:
    """Tensors for the targets at pool rows `rows`. The last token of each sequence is the query."""
    g = pool.tgt[rows]
    idx = np.concatenate([pool.ctx[rows].astype(np.int64), g[:, None]], axis=1)
    return dict(
        tok=torch.from_numpy(pool.tok[idx].astype(np.float32)),
        ids=torch.from_numpy(pool.ids[idx].astype(np.int64)),
        day=torch.from_numpy(pool.day[idx]),
        valid=torch.from_numpy(idx != pool.n),
        q=torch.from_numpy(pool.q[rows].astype(np.float32)),
        fr=torch.from_numpy(pool.fr[rows]),
        y=torch.from_numpy(pool.y[rows]),
    )


def relation(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Relation code (see module doc) between id tuples a and b (last dim 4: card, note, deck, preset)."""
    same = [(a[..., k] == b[..., k]) & (a[..., k] >= 0) for k in range(4)]
    code = torch.full(same[0].shape, REL_OTHER, dtype=torch.long, device=a.device)
    for k in (3, 2, 1, 0):                  # the finest match wins: card (0) is applied last
        code = torch.where(same[k], torch.full_like(code, k), code)
    return code


class _Block(nn.Module):
    def __init__(self, d: int, heads: int, dff: int):
        super().__init__()
        self.h = heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, dff), nn.GELU(), nn.Linear(dff, d))

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        dh = D // self.h
        q, k, v = self.qkv(self.ln1(x)).view(B, L, 3, self.h, dh).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)              # scale 1/sqrt(dh) inside
        x = x + self.proj(o.transpose(1, 2).reshape(B, L, D))
        return x + self.ff(self.ln2(x))


class TAM(nn.Module):
    def __init__(self, n_query: int, d: int = D_MODEL, heads: int = N_HEADS, layers: int = N_LAYERS,
                 dff: int = D_FFN):
        super().__init__()
        self.tok_proj = nn.Linear(TOK_DIM, d)
        self.q_proj = nn.Linear(n_query, d)
        self.rel_emb = nn.Embedding(N_REL, d)
        self.rt_proj = nn.Linear(1, d)
        self.blocks = nn.ModuleList(_Block(d, heads, dff) for _ in range(layers))
        self.decay = nn.Parameter(torch.full((layers, heads), -0.5))
        self.rel_bias = nn.Parameter(torch.zeros(layers, heads, N_REL))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        nn.init.zeros_(self.head[2].weight)
        nn.init.zeros_(self.head[2].bias)

    def forward(self, b: dict) -> torch.Tensor:
        """Logits [B] of the targets in batch b (see make_batch)."""
        ids, day, valid = b["ids"], b["day"], b["valid"]
        B, L = valid.shape
        rel_t = relation(ids, ids[:, -1:, :])                                   # [B, L]
        rdays_t = torch.log1p((day[:, -1:] - day).clamp_min(0.0))              # [B, L]
        x_tok = self.tok_proj(b["tok"]) + self.rel_emb(rel_t) + self.rt_proj(rdays_t.unsqueeze(-1))
        x_q = self.q_proj(b["q"]) + self.rel_emb(rel_t[:, -1]) + self.rt_proj(rdays_t[:, -1:])
        x = torch.cat([x_tok[:, :-1], x_q.unsqueeze(1)], dim=1)

        rel_p = relation(ids.unsqueeze(2), ids.unsqueeze(1))                    # [B, L, L]
        rel_oh = F.one_hot(rel_p, N_REL).to(x.dtype)                           # [B, L, L, 5]
        dt = torch.log1p((day.unsqueeze(2) - day.unsqueeze(1)).abs())          # [B, L, L]
        allow = valid[:, None, None, :].expand(B, 1, L, L).clone()
        allow[:, :, :-1, -1] = False                  # context rows cannot see the query token
        add_mask = torch.zeros(B, 1, L, L, dtype=x.dtype, device=x.device).masked_fill(~allow, -1e4)
        for l, blk in enumerate(self.blocks):
            rel_term = torch.einsum("blmk,hk->bhlm", rel_oh, self.rel_bias[l])  # [B, H, L, L]
            bias = self.decay[l].view(1, -1, 1, 1) * dt.unsqueeze(1) + rel_term + add_mask
            x = blk(x, bias)
        out = self.head(self.ln(x[:, -1])).squeeze(-1)
        return b["fr"] + out


def bce(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, y)


@torch.no_grad()
def predict(model: TAM, pool, rows=None, batch_size: int = 128) -> np.ndarray:
    """Probabilities for the pool rows (all rows by default), in eval mode. pool: Pool or TorchPool."""
    model.eval()
    device = next(model.parameters()).device
    tp = _as_torch(pool, device)
    if rows is None:
        rows_t = torch.arange(len(tp), device=device)
    else:
        rows_t = torch.as_tensor(np.asarray(rows, dtype=np.int64), device=device)
    out = [torch.sigmoid(model(make_batch_t(tp, rows_t[s:s + batch_size]))).float().cpu().numpy()
           for s in range(0, len(rows_t), batch_size)]
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def mean_log_loss(model: TAM, pool, rows: np.ndarray, batch_size: int = 128) -> float:
    p = np.clip(predict(model, pool, rows, batch_size).astype(np.float64), P_CLIP, 1 - P_CLIP)
    y = (pool.y_np if isinstance(pool, TorchPool) else pool.y)[np.asarray(rows)].astype(np.float64)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def train(model: TAM, pool, steps: int, batch_size: int = 32, lr: float = 1e-3, seed: int = 0) -> list:
    """AdamW on uniformly sampled scored targets of the pool. Returns the per-step training losses."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = next(model.parameters()).device
    tp = _as_torch(pool, device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    losses = []
    for _ in range(steps):
        rows = torch.as_tensor(rng.integers(0, len(tp), batch_size), device=device)
        b = make_batch_t(tp, rows)
        loss = bce(model(b), b["y"])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    model.eval()
    return losses
