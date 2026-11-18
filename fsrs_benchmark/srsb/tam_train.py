"""TAM train and eval CLI (Phase 6, task 6.4, step 4). Design: paper/docs/SRSB_ARCH.md; model: srsb/tam.py.

Train on the train users' scored rows (all scored rows are targets). Early stopping on holdout users taken
from the train users (run_gbdt.holdout_users). AdamW with a cosine schedule. Eval users: their TimeSeriesSplit
test rows, scored as srs-benchmark does for FSRS-6-short.

Outputs (resumable; a finished user is never rescored):
  <out>                    srs-benchmark style jsonl, one line per eval user {"metrics", "user", "size"}
  <out>.preds.parquet      per-row predictions of the eval users' test rows: uid, review_th, p
  <out>.ckpt.pt            best checkpoint (by holdout log loss; last epoch when there is no holdout)
  <out>.run_info.json      args, params, best epoch, history, skipped users, comparison summary
  <out>.preds/             per-user prediction parts (internal, resume)

Guards (val_plus.check_users, plus): users 1-1000 are the test set, and users 8001-8300 (validation) may not
be train users. Both are refused unless --final (task 6.5 only). Train and eval must not overlap.

Usage (repo root):
  .venv/bin/python -m fsrs_benchmark.srsb.tam_train --data data/anki10k --train_users T.txt \
      --eval_users V.txt --out result/tam/run1.jsonl --epochs 10 --batch 256 --device auto --amp
"""
from __future__ import annotations

import argparse
import json
import gc
import math
import multiprocessing as mp
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from fsrs_benchmark import data as D
from fsrs_benchmark.srsb import compare as C
from fsrs_benchmark.srsb import run_gbdt as R
from fsrs_benchmark.srsb import tam as T
from fsrs_benchmark.srsb import val_plus as V

VAL_LO, VAL_HI = 8001, 8300            # validation users: never train users unless final
DEFAULT_EPOCHS = 10
DEFAULT_BATCH = 256
DEFAULT_LR = 1e-3
DEFAULT_PATIENCE = 3
WEIGHT_DECAY = 0.01
TRAIN_KEEP_ROWS = 60000       # train/holdout users keep only their last N stream rows (context is 256+32 back)
N_PRINT_ERRORS = 5


def peak_rss_mb() -> float:
    """Peak resident set size of this process in MB (Linux reports KB)."""
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)


def check_users(train_uids, eval_uids, final: bool = False) -> None:
    """val_plus guards (1-1000, overlap, empty sides), plus: no validation user (8001-8300) as a train user."""
    V.check_users(train_uids, eval_uids, final)
    if not final:
        bad = sorted(u for u in train_uids if VAL_LO <= int(u) <= VAL_HI)
        if bad:
            raise ValueError(f"users {VAL_LO}-{VAL_HI} are validation users and may not be train users "
                             f"(only --final lifts this): {bad[:10]}")


def pick_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def slim_train(u: dict, keep: int = TRAIN_KEEP_ROWS) -> dict:
    """Keep only the last `keep` stream rows of a training user, in compact dtypes: tokens and queries
    float16, ids dense int32, day int32. Context only looks back K_USER + M_CARD rows, so the kept rows
    give every kept target its full context; the dropped early rows are never needed for them."""
    n = len(u["day"])
    sl = slice(max(0, n - keep), n)
    return dict(
        review_th=u["review_th"][sl], day=u["day"][sl].astype(np.int32),
        card=T.dense_ids(u["card"][sl]), note=T.dense_ids(u["note"][sl]),
        deck=T.dense_ids(u["deck"][sl]), preset=T.dense_ids(u["preset"][sl]),
        tok=u["tok"][sl].astype(np.float16),
        xq=np.clip(u["xq"][sl], -6e4, 6e4).astype(np.float16),   # float16 saturates beyond 65504
        fr=u["fr"][sl], y=u["y"][sl], scored=u["scored"][sl], test=u["test"][sl],
    )


def _build_one(uid: int, data_dir: str, cache_dir, keep):
    try:
        arrs = T.build_user_tensors(uid, data_dir, cache_dir)
        return uid, (arrs if keep is None else slim_train(arrs, keep)), None
    except Exception as e:                 # srs-benchmark skips such users too
        return uid, None, f"{type(e).__name__}: {str(e)[:100]}"


def build_users(uids, data_dir: Path, cache_dir: Path | None, workers: int, keep=None):
    """Per-user arrays (slimmed when keep is set) for uids; fork pool when workers > 1.
    Returns (dict uid -> arrays, skipped list of (uid, error))."""
    args = [(int(u), str(data_dir), None if cache_dir is None else str(cache_dir), keep) for u in uids]
    if workers > 1 and len(args) > 1:
        with mp.get_context("fork").Pool(workers) as pool:
            results = pool.starmap(_build_one, args, chunksize=1)
    else:
        results = [_build_one(*a) for a in args]
    users, skipped = {}, []
    for uid, arrs, err in results:
        if err is None:
            users[uid] = arrs
        else:
            skipped.append((uid, err))
    return users, skipped


def report_skipped(label: str, skipped) -> None:
    print(f"{label}: {len(skipped)} skipped", flush=True)
    for uid, err in skipped[:N_PRINT_ERRORS]:
        print(f"  skipped user {uid}: {err}", flush=True)


def subsample_targets(u: dict, k: int | None, rng: np.random.Generator) -> dict:
    """The user's arrays with at most k scored rows kept as training targets (seeded). Tokens are unchanged."""
    if k is None:
        return u
    idx = np.flatnonzero(u["scored"])
    if len(idx) <= k:
        return u
    keep = np.zeros(len(u["scored"]), bool)
    keep[np.sort(rng.choice(idx, k, replace=False))] = True
    return {**u, "scored": keep}


@torch.no_grad()
def _loss_on(model, tp: T.TorchPool, batch: int) -> float:
    """Mean log loss over all rows of a TorchPool."""
    rows = np.arange(len(tp))
    p = np.clip(T.predict(model, tp, rows, batch).astype(np.float64), T.P_CLIP, 1 - T.P_CLIP)
    y = tp.y_np.astype(np.float64)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def save_ckpt(model, scaler: T.QueryScaler, path: Path) -> None:
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(dict(model=state, scaler_mean=torch.from_numpy(scaler.mean),
                    scaler_std=torch.from_numpy(scaler.std)), path)


def load_model(path: Path, device, scaler: T.QueryScaler | None = None):
    """Model from a checkpoint. New checkpoints carry the scaler; for a legacy bare state dict the scaler
    argument (fit on the training users) is used."""
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict) and "model" in obj:
        state = obj["model"]
        scaler = T.QueryScaler(obj["scaler_mean"].numpy(), obj["scaler_std"].numpy())
    else:
        state = obj
    model = T.TAM(2 * len(T.QUERY_COLS)).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, scaler


def train_model(train_arrs: list[dict], hold_arrs: list[dict], args, device, ckpt: Path):
    """Train with AdamW + cosine; keep the best epoch by holdout log loss. Returns (model, info)."""
    scaler = T.QueryScaler.fit(train_arrs)
    rng = np.random.default_rng(args.seed)
    sub = [subsample_targets(u, args.max_targets_per_user, rng) for u in train_arrs]
    tp = T.TorchPool(T.build_pool(sub, scaler, "scored"), device)
    hold_sub = [subsample_targets(u, args.max_targets_per_user, rng) for u in hold_arrs]
    hp = T.TorchPool(T.build_pool(hold_sub, scaler, "scored"), device) if hold_arrs else None
    n = len(tp)
    if n == 0:
        raise RuntimeError("no scored training rows")

    torch.manual_seed(args.seed)
    model = T.TAM(2 * len(T.QUERY_COLS)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = max(1, math.ceil(n / args.batch))
    total = args.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1.0 + math.cos(math.pi * min(s, total) / total)))
    use_amp = bool(args.amp and device.type == "cuda")
    amp_scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    gen = torch.Generator().manual_seed(args.seed)

    history, best, best_ep, bad = [], math.inf, 0, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        order = torch.randperm(n, generator=gen).to(device)
        tot, cnt = 0.0, 0
        for s in range(steps_per_epoch):
            rows = order[s * args.batch:(s + 1) * args.batch]
            b = T.make_batch_t(tp, rows)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(b)
            loss = T.bce(logits.float(), b["y"])
            opt.zero_grad(set_to_none=True)
            amp_scaler.scale(loss).backward()
            amp_scaler.step(opt)
            amp_scaler.update()
            sched.step()
            tot += float(loss) * len(rows)
            cnt += len(rows)
        train_ll = tot / cnt
        hold_ll = _loss_on(model, hp, args.batch) if hp is not None else None
        rec = dict(epoch=ep, train_log_loss=round(train_ll, 6),
                   holdout_log_loss=None if hold_ll is None else round(hold_ll, 6),
                   seconds=round(time.time() - t0, 1), peak_rss_mb=peak_rss_mb())
        history.append(rec)
        print(json.dumps(rec), flush=True)
        score = hold_ll if hold_ll is not None else train_ll
        if score < best:
            best, best_ep, bad = score, ep, 0
            save_ckpt(model, scaler, ckpt)
        else:
            bad += 1
            if hp is not None and bad >= args.patience:
                break
        if hp is None:
            best_ep = ep
    model, _ = load_model(ckpt, device, scaler)
    info = dict(params=n_params, n_train_targets=int(n), train_users=len(train_arrs),
                holdout_users=len(hold_arrs), best_epoch=best_ep, best_score=round(best, 6),
                history=history)
    return model, scaler, info


def eval_users(model, scaler, uids, data_dir: Path, cache_dir, out: Path, device, batch: int) -> dict:
    """Score each eval user's test rows, one user at a time (built, scored, freed), appending one jsonl line
    per user (resumable). Per-row predictions go to <out>.preds/<uid>.parquet and are combined at the end.
    Returns the users that could not be built and those with no test rows."""
    parts = Path(str(out) + ".preds")
    parts.mkdir(parents=True, exist_ok=True)
    done = R._load_done(out)
    todo = [u for u in sorted(uids) if u not in done]
    skipped, no_test = [], []
    ll_sum, rows_sum = 0.0, 0
    t_start = time.time()
    with open(out, "a", encoding="utf-8") as f:
        for i, uid in enumerate(todo, 1):
            t0 = time.time()
            try:
                u = T.build_user_tensors(uid, str(data_dir), cache_dir)
            except Exception as e:
                skipped.append((uid, f"{type(e).__name__}: {str(e)[:100]}"))
                print(f"eval {i}/{len(todo)} user {uid}: skipped ({skipped[-1][1]})", flush=True)
                continue
            test_pos = np.flatnonzero(u["test"])
            if len(test_pos) == 0:
                no_test.append(uid)
                del u
                continue
            pool = T.build_pool([u], scaler, rows="test")
            p = np.clip(T.predict(model, pool, None, batch).astype(np.float64), T.P_CLIP, 1 - T.P_CLIP)
            y = u["y"][test_pos].astype(np.float64)
            ll_sum += float(-np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
            rows_sum += len(y)
            meta = pd.DataFrame({"y": y, "elapsed_days": u["meta_gap"][test_pos],
                                 "i": u["meta_i"][test_pos], "rmse_bins_lapse": u["meta_rmse"][test_pos]})
            line = {"metrics": R.user_metrics(y, p, meta), "user": int(uid), "size": int(len(y))}
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
            f.flush()
            pd.DataFrame({"uid": np.full(len(y), uid, np.int64), "review_th": u["review_th"][test_pos],
                          "p": p}).to_parquet(parts / f"{uid}.parquet", index=False)
            print(f"eval {i}/{len(todo)} user {uid}: {len(y)} test rows, {time.time() - t0:.1f}s, "
                  f"running LogLoss {ll_sum / rows_sum:.5f} (elapsed {time.time() - t_start:.0f}s)", flush=True)
            del u, pool, p, y, meta
            gc.collect()
        f.flush()
    R._sort_file(out)
    finished = sorted(R._load_done(out))
    parts_found = [parts / f"{uid}.parquet" for uid in finished if (parts / f"{uid}.parquet").exists()]
    if parts_found:
        pd.concat([pd.read_parquet(x) for x in parts_found], ignore_index=True).to_parquet(
            Path(str(out) + ".preds.parquet"), index=False)
    return dict(eval_skipped=skipped, eval_skipped_no_test=no_test)


def run(args) -> dict:
    train_uids = R.load_ids(args.train_users)
    eval_uids = R.load_ids(args.eval_users)
    check_users(train_uids, eval_uids, args.final)
    data_dir = Path(args.data)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache) if args.cache else data_dir / "tam_cache"
    if args.download:
        want = sorted(set(train_uids) | set(eval_uids))
        _, missing_rev = D.download_revlogs(want, str(data_dir))
        side = D.download_side_tables(want, str(data_dir))
        print(f"download: revlogs missing {missing_rev}; side-table files missing {len(side['missing'])}")
    device = pick_device(args.device)
    print(f"device {device}, amp {bool(args.amp and device.type == 'cuda')}", flush=True)

    eval_batch = args.eval_batch or (512 if device.type == "cuda" else args.batch)
    ckpt = Path(str(out) + ".ckpt.pt")
    info_path = Path(str(out) + ".run_info.json")
    resume = ckpt.exists() and info_path.exists() and "train" in json.loads(info_path.read_text(encoding="utf-8"))
    if args.eval_only and not ckpt.exists():
        raise SystemExit(f"--eval_only: no checkpoint at {ckpt}")

    hold_uids = R.holdout_users(train_uids)
    hold_set = set(hold_uids)
    fit_uids = [u for u in train_uids if u not in hold_set]
    skipped = []
    if args.eval_only or resume:
        # scaler comes from the checkpoint; train users are only needed for a legacy checkpoint
        model, scaler = load_model(ckpt, device)
        info = json.loads(info_path.read_text(encoding="utf-8"))["train"] if info_path.exists() else {}
        if scaler is None:
            users, skipped = build_users(sorted(set(train_uids)), data_dir, cache, args.workers, keep=TRAIN_KEEP_ROWS)
            scaler = T.QueryScaler.fit([users[u] for u in fit_uids if u in users])
            del users
        print(f"loaded checkpoint {ckpt} (no training)", flush=True)
    else:
        users, skipped = build_users(sorted(set(train_uids)), data_dir, cache, args.workers, keep=TRAIN_KEEP_ROWS)
        report_skipped("train users", skipped)
        fit_arrs = [users[u] for u in fit_uids if u in users]
        hold_arrs = [users[u] for u in hold_uids if u in users]
        if not fit_arrs:
            raise SystemExit(f"no train users loaded ({len(skipped)} skipped); see the errors above")
        print(f"loaded {len(fit_arrs)} train + {len(hold_arrs)} holdout users, peak RSS {peak_rss_mb()} MB",
              flush=True)
        model, scaler, info = train_model(fit_arrs, hold_arrs, args, device, ckpt)
        # write the training part now: a restart after this point reloads the checkpoint, not retrains
        info_path.write_text(json.dumps(dict(train=info, args=vars(args)), indent=1, default=str),
                             encoding="utf-8")
        del users, fit_arrs, hold_arrs
        gc.collect()

    ev = eval_users(model, scaler, eval_uids, data_dir, cache, out, device, eval_batch)
    report_skipped("eval users", ev["eval_skipped"])
    cmp = V.compare_published(out)
    V._print_comparison(cmp)
    run_info = dict(args=vars(args), train=info, train_users=fit_uids, holdout_users=hold_uids,
                    skipped_users=[dict(user=u, error=e) for u, e in skipped],
                    eval_skipped=[dict(user=u, error=e) for u, e in ev["eval_skipped"]],
                    eval_skipped_no_test=ev["eval_skipped_no_test"],
                    peak_rss_mb=peak_rss_mb(), comparison=cmp, out=str(out))
    info_path.write_text(json.dumps(run_info, indent=1, default=str), encoding="utf-8")
    print(f"peak RSS {peak_rss_mb()} MB", flush=True)
    return run_info


def main(argv=None):
    ap = argparse.ArgumentParser(description="TAM train and eval (task 6.4).")
    ap.add_argument("--data", default="data/anki10k", help="dir with revlogs/, cards/, decks/")
    ap.add_argument("--train_users", required=True, help="ids (comma separated) or a file of ids")
    ap.add_argument("--eval_users", required=True, help="ids (comma separated) or a file of ids")
    ap.add_argument("--out", required=True, help="output jsonl (resumable)")
    ap.add_argument("--cache", default=None, help="per-user npz cache dir (default <data>/tam_cache)")
    ap.add_argument("--download", action="store_true", help="download revlogs and side tables first")
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--patience", type=int, default=DEFAULT_PATIENCE, help="epochs without holdout gain")
    ap.add_argument("--max_targets_per_user", type=int, default=None,
                    help="train on at most N scored rows per train user (seeded)")
    ap.add_argument("--device", default="auto", help="auto (cuda if available), cpu or cuda")
    ap.add_argument("--amp", action="store_true", help="fp16 autocast on cuda")
    ap.add_argument("--workers", type=int, default=1, help="parallel user featurization processes")
    ap.add_argument("--eval_batch", type=int, default=None,
                    help="targets per scoring batch (default 512 on cuda, --batch on cpu)")
    ap.add_argument("--eval_only", action="store_true", help="load the checkpoint and score; no training")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--final", action="store_true", help="allow users 1-1000 and 8001-8300 (task 6.5 only)")
    args = ap.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
