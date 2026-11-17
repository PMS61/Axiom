"""Per-user live residual correction on top of the frozen pooled GBDT (Phase 6, task 6.6). See paper/docs/SRSB_BEAT.md.

The GBDT (val_plus features, LightGBM raw margin m_t) gives every review of a user's stream a logit. A small GRU
reads the user's stream in review_th order and outputs a correction delta_t, so that

    p_t = sigmoid(m_t + delta_t),      delta_t = head(GRU(inputs_1..inputs_t))

The head starts at zero, so p equals the GBDT at the start.

Inputs of row t (K = 15, see build_inputs):
  known before the answer: log1p-ish elapsed fraction, same_day, log1p(card review count), is_first,
                           state one-hot (5), and the GBDT margin m_t itself;
  earlier rows only:       previous row's y and m (any card), this card's previous residual y - sigmoid(m) and
                           log1p(rows since this card's previous review), and a has-previous flag.
Row t never sees y_t, and never sees anything after t (the leak test checks this).

The stream is the user's FULL stream (review_stream: same-day and first reviews included, as history). Labels and
masks are the srs-benchmark scored rows (the ds rows, which exclude same-day and first reviews): the loss is on the
scored rows that are not first reviews, and the metrics use the TimeSeriesSplit test rows, as in val_plus.

Out-of-sample GBDT margins: the GBDT was trained on users 2001-2300, so those (and users 1-1000, the test set, and
validation users 8001-8300 when training) may not be used here. Train users in the project plan are 2301-2500;
eval users are 8001-8100.

Training: users are batched (by length, to limit padding); each batch is run in chunks of CHUNK rows with the GRU
state carried across chunks and detached (truncated BPTT). Loss: log loss on the labeled non-first rows. AdamW.
Fixed epoch count, no early stopping on validation (model choice is frozen before the 8001-8100 scoring).

Output (srs-benchmark jsonl for the eval users' test rows, same as val_plus):
  <out>                  {"metrics", "user", "size"} per eval user, GRU-corrected p
  <out>.gbdt_only.jsonl  the same users with the GBDT alone (reference)
  <out>.model.pt         GRU state dict
  <out>.run_info.json    settings, users, skipped users, losses, comparisons

Usage (repo root):
  .venv/bin/python -m fsrs_benchmark.srsb.resid_gru --data data/anki10k \
      --train_users 2301-2500 --eval_users 8001-8100 --out results/srsb_beat/resid_gru.jsonl \
      [--epochs 3 --hidden 64 --lr 1e-3 --gbdt_model results/srsb_beat/sweep_a_l63.jsonl.model.txt --device cpu]
"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from fsrs_benchmark import data as D
from fsrs_benchmark.models import gbdt as G
from fsrs_benchmark.srsb import compare as C
from fsrs_benchmark.srsb import run_gbdt as R
from fsrs_benchmark.srsb import val_plus as V
from fsrs_benchmark.srsb.features_plus import plus_features

REPO = Path(__file__).resolve().parents[2]
DEFAULT_GBDT = REPO / "results" / "srsb_beat" / "sweep_a_l63.jsonl.model.txt"
GBDT_TRAIN_RANGE = (2001, 2300)          # the GBDT's own train users: their margins are in-sample
CHUNK = 512                              # truncated BPTT chunk, rows
BATCH_USERS = 8                          # users trained together (sorted by length)
HIDDEN = 64
LR = 1e-3
EPOCHS = 3
CLIP_MARGIN = 15.0                       # margins are clipped to +-CLIP_MARGIN before scaling by 1/5
MARGIN_SCALE = 5.0
KNOWN_COLS = 9                           # log el_frac, same_day, log1p card_n, is_first, state one-hot (5)
LAG_COLS = 6                             # m_t, prev y, prev m, has card prev, card prev residual, log gap
K_IN = KNOWN_COLS + LAG_COLS
COLS = V.group_columns(None)             # the 78 GBDT features, in the order the sweep model was trained on


# ---- guards ---------------------------------------------------------------------------------------

def gbdt_train_users(model_path) -> set:
    """Users the GBDT was trained on: GBDT_TRAIN_RANGE, plus the train list of its run_info when present."""
    users = set(range(GBDT_TRAIN_RANGE[0], GBDT_TRAIN_RANGE[1] + 1))
    info = Path(str(model_path).replace(".model.txt", ".run_info.json"))
    if info.is_file():
        users |= {int(u) for u in json.loads(info.read_text(encoding="utf-8")).get("train_users", [])}
    return users


def check_users(train_uids, eval_uids, gbdt_model=DEFAULT_GBDT):
    """val_plus's guards (test set 1-1000, validation users 8001-8300 not for training, no overlap), plus: no
    train or eval user may be one the GBDT was trained on (its margins would be in-sample)."""
    V.check_users(train_uids, eval_uids, final=False)
    bad = sorted((set(int(u) for u in train_uids) | set(int(u) for u in eval_uids)) & gbdt_train_users(gbdt_model))
    if bad:
        raise ValueError(f"GBDT training users have in-sample margins and may not be used: {bad[:10]}")


# ---- inputs -----------------------------------------------------------------------------------------

def known_inputs(feats: pd.DataFrame, st: pd.DataFrame) -> np.ndarray:
    """Known-before-answer inputs of each row (KNOWN_COLS), from the GBDT feature frame and the stream."""
    le = np.nan_to_num(feats["log_el_frac"].to_numpy(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    sd = np.nan_to_num(feats["same_day"].to_numpy(np.float64), nan=0.0)
    cn = np.log1p(np.clip(np.nan_to_num(feats["card_n"].to_numpy(np.float64), nan=0.0), 0.0, None))
    first = st["is_first"].to_numpy(np.float64)
    state = np.clip(st["state"].to_numpy(np.int64), 0, 4)
    return np.column_stack([le, sd, cn, first, np.eye(5)[state]]).astype(np.float32)


def build_inputs(m, y, card, known) -> np.ndarray:
    """(n, K_IN) GRU inputs. Row t uses m_t, known_t, and y, m, card and rows of index < t only."""
    m = np.asarray(m, np.float64)
    y = np.asarray(y, np.float64)
    n = len(m)
    pos = np.arange(n)
    mc = np.clip(m, -CLIP_MARGIN, CLIP_MARGIN) / MARGIN_SCALE
    prev_y = np.r_[0.0, y[:-1]]
    prev_m = np.r_[0.0, mc[:-1]]
    prev = pd.Series(pos).groupby(np.asarray(card)).shift(1).to_numpy(np.float64)
    has = ~np.isnan(prev)
    pi = np.where(has, prev, 0).astype(np.int64)
    p = 1.0 / (1.0 + np.exp(-m))
    card_res = np.where(has, (y - p)[pi], 0.0)
    gap = np.where(has, np.log1p(pos - pi), 0.0)
    X = np.column_stack([known, mc, prev_y, prev_m, has.astype(np.float64), card_res, gap])
    return X.astype(np.float32)


# ---- per-user data ----------------------------------------------------------------------------------

def prepare_user_raw(raw, ds, cfg, cards, decks, booster, best_iter, check_y=True) -> dict:
    """All per-row arrays of one user from its raw revlog frame (stream order = review_th order).
    ds is the srs-benchmark scored frame (membership, labels, test mask, meta). Margins come from the GBDT on the
    val_plus features of the FULL stream, so the same-day and first rows are inputs too."""
    st = R.review_stream(raw, cfg.max_seq_len * 2)
    feats = pd.concat([G.base_features(st, span=20, user_group="row")[R.FEATURE_COLS],
                       plus_features(st, cards, decks)], axis=1)
    idx = st.index.to_numpy()
    keys = ds["review_th"].to_numpy()
    pos = np.searchsorted(idx, keys)
    if keys.size and (pos.max() >= len(idx) or np.any(idx[np.minimum(pos, len(idx) - 1)] != keys)):
        raise RuntimeError("scored rows missing from the stream")
    if check_y and not np.array_equal(st["y"].to_numpy()[pos], ds["y"].to_numpy()):
        raise RuntimeError("stream labels differ from the srs-benchmark labels on the scored rows")
    scored = np.zeros(len(st), bool)
    scored[pos] = True
    X_all = feats[COLS].to_numpy(np.float32)
    m = booster.predict(X_all, raw_score=True, num_iteration=best_iter).astype(np.float64)
    y = st["y"].to_numpy(np.float64)
    X = build_inputs(m, y, st["card_id"].to_numpy(np.int64), known_inputs(feats, st))
    return dict(
        X=X, m=m, y=y, loss=scored & ~st["is_first"].to_numpy(bool),
        pos=pos, test=R.test_row_mask(len(ds), cfg.n_splits), y_ds=ds["y"].to_numpy(np.float64),
        meta=pd.DataFrame({"y": ds["y"].to_numpy(np.float64),
                           "elapsed_days": ds["elapsed_days"].to_numpy(np.float64),
                           "i": ds["i"].to_numpy(np.float64),
                           "rmse_bins_lapse": ds["rmse_bins_lapse"].to_numpy(np.float64)}),
    )


def prepare_user(uid, loader, cfg, data_dir: Path, booster, best_iter) -> dict:
    ds = loader.load_user_data(uid)
    raw = pd.read_parquet(data_dir / "revlogs" / f"user_id={uid}")
    # Side tables are loaded exactly as val_plus._full_table does, so the margins match the sweep GBDT.
    cards = V._read_opt(data_dir / "cards" / f"user_id={uid}")
    decks = V._read_opt(data_dir / "decks" / f"user_id={uid}")
    return prepare_user_raw(raw, ds, cfg, cards, decks, booster, best_iter)


# ---- model ------------------------------------------------------------------------------------------

class ResidGRU(nn.Module):
    """Linear input -> 1-layer GRU -> linear head. The head starts at zero, so the output is the GBDT alone."""

    def __init__(self, k_in: int = K_IN, hidden: int = HIDDEN):
        super().__init__()
        self.inp = nn.Linear(k_in, hidden)
        self.gru = nn.GRU(hidden, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, h0=None):
        """x (B, L, K) -> delta (B, L), final state (1, B, H)."""
        h, hn = self.gru(self.inp(x), h0)
        return self.head(h).squeeze(-1), hn


def predict_delta(model: ResidGRU, X: np.ndarray, device="cpu") -> np.ndarray:
    model.eval()
    with torch.no_grad():
        delta, _ = model(torch.from_numpy(X).unsqueeze(0).to(device))
    return delta[0].cpu().numpy().astype(np.float64)


def probs(z) -> np.ndarray:
    z = np.asarray(z, np.float64)
    return np.clip(1.0 / (1.0 + np.exp(-z)), G.P_CLIP, 1 - G.P_CLIP)


def train_model(model: ResidGRU, users: list, epochs: int, lr: float, seed: int = 0, device="cpu",
                batch_users: int = BATCH_USERS, chunk: int = CHUNK, log=print) -> list:
    """Truncated BPTT over chunks of `chunk` rows, GRU state carried across chunks and detached. Users are batched
    by length. Returns the mean training loss per epoch."""
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    by_len = sorted(range(len(users)), key=lambda k: len(users[k]["y"]))
    batches = [by_len[i:i + batch_users] for i in range(0, len(by_len), batch_users)]
    hist = []
    for ep in range(epochs):
        tot, n_tot, t0 = 0.0, 0.0, time.time()
        for bi in rng.permutation(len(batches)):
            ub = [users[k] for k in batches[bi]]
            T = max(len(u["y"]) for u in ub)
            B = len(ub)
            h = torch.zeros(1, B, model.gru.hidden_size, device=device)
            for s in range(0, T, chunk):
                e = min(s + chunk, T)
                L = e - s
                X = np.zeros((B, L, K_IN), np.float32)
                M = np.zeros((B, L), np.float32)
                Y = np.zeros((B, L), np.float32)
                W = np.zeros((B, L), bool)
                for b, u in enumerate(ub):
                    n = len(u["y"])
                    if s >= n:
                        continue
                    k = min(e, n) - s
                    X[b, :k] = u["X"][s:s + k]
                    M[b, :k] = u["m"][s:s + k]
                    Y[b, :k] = u["y"][s:s + k]
                    W[b, :k] = u["loss"][s:s + k]
                nw = int(W.sum())
                delta, h_new = model(torch.from_numpy(X).to(device), h)
                if nw:
                    z = torch.from_numpy(M).to(device) + delta
                    wt = torch.from_numpy(W).to(device)
                    loss = F.binary_cross_entropy_with_logits(
                        z[wt], torch.from_numpy(Y).to(device)[wt], reduction="sum") / nw
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    tot += float(loss.item()) * nw
                    n_tot += nw
                h = h_new.detach()
        hist.append(tot / max(n_tot, 1.0))
        log(f"epoch {ep + 1}/{epochs}: train log loss {hist[-1]:.5f} "
            f"({len(users)} users, {int(n_tot)} labeled rows, {time.time() - t0:.0f}s)")
    return hist


# ---- evaluation --------------------------------------------------------------------------------------

def eval_user(model: ResidGRU, u: dict, device="cpu"):
    """(metrics GRU, metrics GBDT only, size) on the user's test rows, in srs-benchmark's format."""
    delta = predict_delta(model, u["X"], device)
    sel = u["pos"][u["test"]]
    y = u["y_ds"][u["test"]]
    meta = u["meta"][u["test"]].reset_index(drop=True)
    p = probs(u["m"] + delta)[sel]
    p0 = probs(u["m"])[sel]
    return R.user_metrics(y, p, meta), R.user_metrics(y, p0, meta), int(len(y))


def _write_lines(path: Path, rows: dict):
    with open(path, "w", encoding="utf-8") as f:
        for uid in sorted(rows):
            f.write(json.dumps(rows[uid], ensure_ascii=False) + "\n")


def _subset(src: Path, lo: int, hi: int, dst: Path) -> Path:
    keep = {}
    for line in src.read_text(encoding="utf-8").splitlines():
        if line.strip():
            obj = json.loads(line)
            if lo <= obj["user"] <= hi:
                keep[obj["user"]] = obj
    _write_lines(dst, keep)
    return dst


def print_comparisons(gru_path: Path, gbdt_path: Path, result_dir=C.DEFAULT_RESULT_DIR) -> dict:
    """val_plus's comparison (vs RWKV-Instant, RWKV-Curve, FSRS-6, paired 99% CI) for all eval users and for
    users 8001-8050 and 8051-8100 separately, for the GRU-corrected model and for the GBDT alone."""
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for tag, label, src in (("gru", "GBDT + residual GRU", gru_path),
                                ("gbdt", "GBDT alone (reference)", gbdt_path)):
            for part, (lo, hi) in (("8001-8050", (8001, 8050)), ("8051-8100", (8051, 8100)),
                                   ("all", (0, 10 ** 9))):
                f = _subset(src, lo, hi, tmp / f"{tag}_{part}.jsonl")
                cmp = V.compare_published(f, result_dir)
                print(f"--- {label}, users {part} ---")
                V._print_comparison(cmp)
                out[f"{label} | {part}"] = cmp
    return out


def run(train_uids, eval_uids, data_dir, out, gbdt_model=DEFAULT_GBDT, epochs=EPOCHS, hidden=HIDDEN, lr=LR,
        seed=0, device="cpu", result_dir=C.DEFAULT_RESULT_DIR, log=print) -> dict:
    """Train the GRU on train_uids' out-of-sample GBDT margins, score eval_uids' test rows, write the jsonl and
    run_info. Returns run_info."""
    check_users(train_uids, eval_uids, gbdt_model)
    import lightgbm as lgb

    t0 = time.time()
    torch.manual_seed(seed)
    data_dir, out = Path(data_dir), Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    gbdt_model = Path(gbdt_model)
    booster = lgb.Booster(model_file=str(gbdt_model))
    mj = Path(str(gbdt_model).replace(".model.txt", ".model.json"))
    best_iter = json.loads(mj.read_text(encoding="utf-8"))["best_iter"] if mj.is_file() else booster.current_iteration()
    if booster.num_feature() != len(COLS):
        raise RuntimeError(f"GBDT has {booster.num_feature()} features, expected {len(COLS)}")
    S = R._srs_modules()
    cfg = S["Config"](S["create_parser"]().parse_args(["--algo", "FSRS-6", "--short", "--data", str(data_dir)]))
    if cfg.get_evaluation_file_name() != "FSRS-6-short":
        raise RuntimeError(f"unexpected srs-benchmark setting {cfg.get_evaluation_file_name()}")
    loader = S["UserDataLoader"](cfg)

    skipped = []
    train_data = []
    for uid in sorted(set(int(u) for u in train_uids)):
        try:
            u = prepare_user(uid, loader, cfg, data_dir, booster, best_iter)
        except Exception as e:                       # srs-benchmark skips these users too
            skipped.append((uid, f"{type(e).__name__}: {str(e)[:100]}"))
            continue
        if u["loss"].any():
            train_data.append(u)
        else:
            skipped.append((uid, "no labeled non-first rows"))
    log(f"train users: {len(train_data)} usable, {len(skipped)} skipped so far; "
        f"stream rows {sum(len(u['y']) for u in train_data)}")
    if not train_data:
        raise RuntimeError("no usable train users")

    model = ResidGRU(K_IN, hidden)
    hist = train_model(model, train_data, epochs, lr, seed=seed, device=device, log=log)
    torch.save(model.state_dict(), Path(str(out) + ".model.pt"))
    del train_data

    gru_rows, gbdt_rows, eval_info = {}, {}, {}
    for uid in sorted(set(int(u) for u in eval_uids)):
        try:
            u = prepare_user(uid, loader, cfg, data_dir, booster, best_iter)
        except Exception as e:
            skipped.append((uid, f"{type(e).__name__}: {str(e)[:100]}"))
            continue
        if not u["test"].any():
            skipped.append((uid, "no test rows"))
            continue
        mg, m0, size = eval_user(model, u, device)
        gru_rows[uid] = {"metrics": mg, "user": uid, "size": size}
        gbdt_rows[uid] = {"metrics": m0, "user": uid, "size": size}
    _write_lines(out, gru_rows)
    _write_lines(Path(str(out) + ".gbdt_only.jsonl"), gbdt_rows)
    log(f"eval users written: {len(gru_rows)}; skipped {len(skipped)}")

    cmp = print_comparisons(out, Path(str(out) + ".gbdt_only.jsonl"), result_dir)
    run_info = dict(
        gbdt_model=str(gbdt_model), best_iter=int(best_iter), data=str(data_dir), out=str(out),
        train_users=sorted(set(int(u) for u in train_uids)), eval_users=sorted(set(int(u) for u in eval_uids)),
        epochs=epochs, hidden=hidden, lr=lr, seed=seed, device=device, chunk=CHUNK, batch_users=BATCH_USERS,
        features=COLS, k_in=K_IN, train_loss_per_epoch=[round(x, 6) for x in hist],
        eval_lines=len(gru_rows), skipped=skipped, elapsed_min=round((time.time() - t0) / 60, 2),
        comparisons={k: v for k, v in cmp.items()},
    )
    Path(str(out) + ".run_info.json").write_text(json.dumps(run_info, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in run_info.items() if k not in ("comparisons", "features")},
                     indent=1, default=str))
    return run_info


def main(argv=None):
    ap = argparse.ArgumentParser(description="Per-user residual GRU on the pooled GBDT (SRS-Beat task 6.6).")
    ap.add_argument("--data", default="data/anki10k", help="dir with revlogs/, cards/, decks/")
    ap.add_argument("--train_users", required=True, help="ids (comma separated) or a file of ids")
    ap.add_argument("--eval_users", required=True, help="ids (comma separated) or a file of ids")
    ap.add_argument("--out", required=True, help="output jsonl (plus .gbdt_only.jsonl, .model.pt, .run_info.json)")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--hidden", type=int, default=HIDDEN)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--gbdt_model", default=str(DEFAULT_GBDT), help="LightGBM .model.txt (with .model.json, .run_info.json)")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--download", action="store_true",
                    help="download revlogs and side tables of the requested users from HF first")
    args = ap.parse_args(argv)

    train_uids = R.load_ids(args.train_users)
    eval_uids = R.load_ids(args.eval_users)
    check_users(train_uids, eval_uids, args.gbdt_model)      # before any download
    if args.download:
        want = sorted(set(train_uids) | set(eval_uids))
        _, missing_rev = D.download_revlogs(want, args.data)
        side = D.download_side_tables(want, args.data)
        print(f"download: revlogs missing {missing_rev}; side-table files missing {len(side['missing'])}")
    return run(train_uids, eval_uids, args.data, args.out, gbdt_model=args.gbdt_model, epochs=args.epochs,
               hidden=args.hidden, lr=args.lr, seed=args.seed, device=args.device)


if __name__ == "__main__":
    main(sys.argv[1:])
