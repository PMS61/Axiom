"""Speed and size benchmark: our final model vs RWKV (srs-benchmark RWKV-P setting). Timing and size only, no accuracy.

Ours   : features (review_stream + gbdt.base_features + features_plus.plus_features) -> LightGBM raw margin
         (sweep_a_l63_side, best_iter from its run_info) -> per-user online layer (online_adapt.online_kernel,
         variant B, lr 0.003). Batch per user: featurization, GBDT and online layer are timed separately and
         divided by the user's review count.
RWKV   : srs-benchmark's RWKV-Instant / RWKV-P setting, run as an RNN one review at a time, exactly as
         rwkv/run_as_rnn.py does (imm_predict + process_row per review). Imported from external/srs-benchmark,
         which is never modified. Only the per-review loop is timed; reading the revlogs/cards/decks and the
         checkpoint is not. The RNN path uses pure torch (rwkv_ops.single_timestep), so CPU works.

Each model runs in its own child process so that peak RSS (ru_maxrss) is per model. A warm-up runs before
the timed passes (numba compile, LightGBM/torch init). Users 1-1000 are the test set and are refused here
(online_adapt.check_users).

Usage (repo root):
  .venv/bin/python -m fsrs_benchmark.srsb.speed_bench --users 8002,8019 --data data/anki10k \
      --device cpu --threads 4 --out results/srsb_beat/speed/speed_local.txt
  Writes the table to --out and the JSON next to it (same stem, .json).
"""
import argparse
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fsrs_benchmark.models import gbdt as G
from fsrs_benchmark.srsb import online_adapt as OA
from fsrs_benchmark.srsb import run_gbdt as R
from fsrs_benchmark.srsb import val_plus as V
from fsrs_benchmark.srsb import ensure_srsb_on_path
from fsrs_benchmark.srsb.features_plus import plus_features

DEFAULT_MODEL = "results/srsb_beat/sweep_a_l63_side.jsonl.model.txt"
DEFAULT_RUN_INFO = "results/srsb_beat/sweep_a_l63_side.jsonl.run_info.json"
DEFAULT_CKPT = "external/srs-benchmark/pretrain/RWKV_trained_on_5000_10000.pth"
VARIANT, LR = "B", 0.003                       # the frozen final online layer


def peak_rss_mb():
    """Peak resident set size of this process so far (ru_maxrss is KiB on Linux)."""
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)


# ---------------------------------------------------------------- ours

def gbdt_stats(model_path, best_iter):
    """Trees used at inference (first best_iter), leaves, and the numbers the predictor reads: per internal node
    split_feature, threshold, decision_type, left_child, right_child; per leaf leaf_value."""
    n_trees = leaves = 0
    with open(model_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("num_leaves="):
                if n_trees >= best_iter:
                    break
                leaves += int(line.split("=")[1])
                n_trees += 1
    internal = leaves - n_trees
    return dict(trees=n_trees, leaves=leaves, internal_nodes=internal,
                numbers=5 * internal + leaves, file_bytes=os.path.getsize(model_path))


def ours_run(args, uids):
    import lightgbm as lgb
    import numba

    numba.set_num_threads(args.threads)
    data_dir = Path(args.data)
    info = json.loads(Path(args.run_info).read_text(encoding="utf-8"))
    names, best_iter = list(info["features"]), int(info["best_iter"])
    booster = lgb.Booster(model_file=str(args.model))
    V._init(str(data_dir), names)
    cfg = V._W["cfg"]
    w0 = OA.w0_for(VARIANT)

    # untimed inputs: raw tables for every user
    inputs = {}
    for u in uids:
        raw = pd.read_parquet(data_dir / "revlogs" / f"user_id={u}")
        if args.max_reviews:
            raw = raw.iloc[:args.max_reviews]
        cards = V._read_opt(data_dir / "cards" / f"user_id={u}")
        decks = V._read_opt(data_dir / "decks" / f"user_id={u}")
        inputs[u] = (raw, cards, decks)

    def one_user(u, collect):
        raw, cards, decks = inputs[u]
        t0 = time.perf_counter()
        st = R.review_stream(raw, cfg.max_seq_len * 2)                          # featurization
        base = G.base_features(st, span=20, user_group="row")[R.FEATURE_COLS]
        feats = pd.concat([base, plus_features(st, cards, decks)], axis=1)
        X = feats[names].to_numpy(np.float32)
        t1 = time.perf_counter()
        margin = booster.predict(X, raw_score=True, num_iteration=best_iter, num_threads=args.threads)  # GBDT
        t2 = time.perf_counter()
        extra = {c: feats[c].to_numpy(np.float64) for c in OA.EXTRA}           # online layer
        labelled = st["card_id"].duplicated().to_numpy()                     # every non-first review
        y = (st["rating"].to_numpy() > 1).astype(np.float64)
        Xd = OA.design(margin.astype(np.float64), extra, VARIANT)
        OA.online_kernel(Xd, y, labelled, w0, LR, OA.LAMBDA)
        t3 = time.perf_counter()
        n = len(st)
        if collect:
            return dict(user=int(u), reviews=int(n), featurize_us=(t1 - t0) / n * 1e6,
                        gbdt_us=(t2 - t1) / n * 1e6, online_us=(t3 - t2) / n * 1e6,
                        total_us=(t3 - t0) / n * 1e6)
        return None

    one_user(uids[0], collect=False)                                            # warm-up (numba, lightgbm)
    one_user(uids[0], collect=False)
    rows = []
    for u in uids:
        best = None
        for _ in range(args.repeats):
            r = one_user(u, collect=True)
            if best is None or r["total_us"] < best["total_us"]:
                best = r
        rows.append(best)
    online_params = len(w0)
    return dict(per_user=rows, peak_rss_mb=peak_rss_mb(),
                size=dict(gbdt=gbdt_stats(args.model, best_iter), online_numbers=online_params,
                          features=len(names), feature_cols_base=len(R.FEATURE_COLS)))


# ---------------------------------------------------------------- RWKV

def rwkv_user_frames(data_dir, uid, max_reviews=0):
    """run_as_rnn.run()'s input: revlogs with review_th, merged with cards (note_id, deck_id) and decks
    (preset_id). Untimed. max_reviews > 0 keeps only the first reviews (quick local runs)."""
    df = pd.read_parquet(data_dir / "revlogs" / f"user_id={uid}")
    if max_reviews:
        df = df.iloc[:max_reviews]
    df_cards = pd.read_parquet(data_dir / "cards", filters=[("user_id", "=", uid)])
    df_cards.drop(columns=["user_id"], inplace=True)
    df_decks = pd.read_parquet(data_dir / "decks", filters=[("user_id", "=", uid)])
    df_decks.drop(columns=["user_id", "parent_id"], inplace=True)
    df = df.merge(df_cards, on="card_id", how="left", validate="many_to_one")
    df = df.merge(df_decks, on="deck_id", how="left", validate="many_to_one")
    df["review_th"] = range(1, df.shape[0] + 1)
    return df


def rwkv_run(args, uids):
    import torch
    from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG
    from rwkv.run_as_rnn import RNNProcess

    dev = torch.device(args.device)
    torch.set_num_threads(args.threads)
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    n_params = int(sum(v.numel() for v in sd.values()))
    n_tensors = len(sd)
    del sd
    data_dir = Path(args.data)
    frames = {u: rwkv_user_frames(data_dir, u, args.max_reviews) for u in uids}

    def one_user(u, limit=None):
        df = frames[u] if limit is None else frames[u].iloc[:limit]
        proc = RNNProcess(path=args.ckpt, device=dev, dtype=torch.float32, config=DEFAULT_ANKI_RWKV_CONFIG)
        curves = {}
        t0 = time.perf_counter()
        for _, row in df.iterrows():
            card_id = row["card_id"]
            if card_id in curves:
                proc.predict_func(curves[card_id], row["elapsed_seconds"])
            imm_info = row.copy()
            imm_info.drop(columns=["rating", "duration"], inplace=True)
            proc.imm_predict(imm_info)
            curves[card_id] = proc.process_row(row)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        n = len(df)
        return dict(user=int(u), reviews=int(n), total_us=(t1 - t0) / n * 1e6)

    one_user(uids[0], limit=300)                                               # warm-up (torch init)
    rows = []
    for u in uids:
        best = None
        for _ in range(args.repeats):
            r = one_user(u)
            if best is None or r["total_us"] < best["total_us"]:
                best = r
        rows.append(best)
    return dict(per_user=rows, peak_rss_mb=peak_rss_mb(), device=str(dev),
                size=dict(params=n_params, tensors=n_tensors, file_bytes=os.path.getsize(args.ckpt)))


# ---------------------------------------------------------------- driver

def _child(args, part, uids, part_out):
    ensure_srsb_on_path()
    if part == "ours":
        res = ours_run(args, uids)
    else:
        res = rwkv_run(args, uids)
    Path(part_out).write_text(json.dumps(res, default=float), encoding="utf-8")


def _spawn(args, part, uids, outdir):
    fd, part_out = tempfile.mkstemp(suffix=f".{part}.json", dir=str(outdir))
    os.close(fd)
    cmd = [sys.executable, "-m", "fsrs_benchmark.srsb.speed_bench", "--part", part, "--part_out", part_out,
           "--users", ",".join(str(u) for u in uids), "--data", args.data, "--device", args.device,
           "--threads", str(args.threads), "--repeats", str(args.repeats), "--model", args.model,
           "--max_reviews", str(args.max_reviews),
           "--run_info", args.run_info, "--ckpt", args.ckpt]
    try:
        subprocess.run(cmd, check=True)
        return json.loads(Path(part_out).read_text(encoding="utf-8"))
    finally:
        Path(part_out).unlink(missing_ok=True)


def _summary(per_user, key):
    total_t = sum(r[key] * r["reviews"] for r in per_user)
    total_n = sum(r["reviews"] for r in per_user)
    return total_t / total_n, total_n


def _table(res, uids):
    lines = []
    o, r = res.get("ours"), res.get("rwkv")
    if o:
        lines.append("OURS (features + LightGBM + online layer), us per review, batch per user")
        lines.append(f"{'user':>6} {'reviews':>8} {'featurize':>10} {'GBDT':>8} {'online':>8} {'total':>8}")
        for row in o["per_user"]:
            lines.append(f"{row['user']:>6} {row['reviews']:>8} {row['featurize_us']:>10.1f} "
                         f"{row['gbdt_us']:>8.2f} {row['online_us']:>8.2f} {row['total_us']:>8.1f}")
        f, n = _summary(o["per_user"], "featurize_us")
        g, _ = _summary(o["per_user"], "gbdt_us")
        on, _ = _summary(o["per_user"], "online_us")
        t, _ = _summary(o["per_user"], "total_us")
        lines.append(f"{'ALL':>6} {n:>8} {f:>10.1f} {g:>8.2f} {on:>8.2f} {t:>8.1f}")
        s = o["size"]
        lines.append(f"  size: GBDT {s['gbdt']['trees']} trees, {s['gbdt']['leaves']} leaves, "
                     f"{s['gbdt']['numbers']} stored numbers, {s['gbdt']['file_bytes']/1e6:.2f} MB file; "
                     f"online layer {s['online_numbers']} weights; {s['features']} GBDT features")
        lines.append(f"  peak RSS: {o['peak_rss_mb']} MB")
    if r:
        lines.append("")
        lines.append(f"RWKV-P (RNN, one review at a time), device={r['device']}, us per review")
        lines.append(f"{'user':>6} {'reviews':>8} {'total':>10}")
        for row in r["per_user"]:
            lines.append(f"{row['user']:>6} {row['reviews']:>8} {row['total_us']:>10.1f}")
        t, n = _summary(r["per_user"], "total_us")
        lines.append(f"{'ALL':>6} {n:>8} {t:>10.1f}")
        s = r["size"]
        lines.append(f"  size: {s['params']} parameters ({s['tensors']} tensors), "
                     f"{s['file_bytes']/1e6:.2f} MB checkpoint")
        lines.append(f"  peak RSS: {r['peak_rss_mb']} MB")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Speed and size: ours (GBDT + online layer) vs RWKV-P (RNN).")
    ap.add_argument("--users", default=None, help="ids (comma/space separated), a text file or a JSON list")
    ap.add_argument("--data", default="data/anki10k")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="RWKV device (ours is CPU)")
    ap.add_argument("--threads", type=int, default=4, help="torch, numba and LightGBM threads")
    ap.add_argument("--repeats", type=int, default=1, help="timed passes per user; the fastest is kept")
    ap.add_argument("--parts", default="ours,rwkv", help="which models to run: ours, rwkv")
    ap.add_argument("--max_reviews", type=int, default=0,
                    help="0 = all reviews of each user; N > 0 = only the first N revlog rows (quick local runs)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--run_info", default=DEFAULT_RUN_INFO)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--out", default="results/srsb_beat/speed/speed_bench.txt",
                    help="table written here; JSON next to it (same stem, .json)")
    # internal: child process mode
    ap.add_argument("--part", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--part_out", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.part is not None:                                       # child: one model, one process
        uids = R.load_ids(args.users)
        OA.check_users(uids, final=False)
        _child(args, args.part, uids, args.part_out)
        return None

    if not args.users:
        ap.error("--users is required")
    uids = R.load_ids(args.users)
    if not uids:
        ap.error("no users given")
    OA.check_users(uids, final=False)
    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            print("cuda requested but torch.cuda.is_available() is False; RWKV runs on cpu", flush=True)
            args.device = "cpu"
    parts = [p.strip() for p in args.parts.split(",") if p.strip()]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json_path = out.with_suffix(".json")
    if json_path == out:
        ap.error("--out must not end in .json (the JSON is written next to it)")
    res = {}
    for part in parts:
        res[part] = _spawn(args, part, uids, out.parent)
    text = _table(res, uids)
    print(text, flush=True)
    out.write_text(text + "\n", encoding="utf-8")
    payload = dict(users=uids, data=str(args.data), device=args.device, threads=args.threads,
                   repeats=args.repeats, max_reviews=args.max_reviews, **res)
    json_path.write_text(json.dumps(payload, indent=1, default=float), encoding="utf-8")
    return payload


if __name__ == "__main__":
    main()
