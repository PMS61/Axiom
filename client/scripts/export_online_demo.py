"""Export the frozen GBDT + online-layer results (srs-benchmark users 1-1000) for the /demo route.

Writes client/src/lib/demo/seed/online.json. Reads only; fsrs_benchmark/ is imported, never modified.
The per-card replay reproduces final_user() in fsrs_benchmark/srsb/online_adapt.py for three test users
(variant B, lr 0.003, model sweep_a_l63_side) and checks each replayed user's LogLoss against
results/srsb_beat/final/online_test1000.jsonl.

Run from the repo root:
  .venv/bin/python client/scripts/export_online_demo.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from fsrs_benchmark.srsb import online_adapt as OA  # noqa: E402
from fsrs_benchmark.srsb import run_gbdt as R  # noqa: E402
from fsrs_benchmark.srsb import val_plus as V  # noqa: E402

DEFAULT_OUT = REPO / "client" / "src" / "lib" / "demo" / "seed" / "online.json"
MODEL = REPO / "results/srsb_beat/sweep_a_l63_side.jsonl.model.txt"
RUN_INFO = REPO / "results/srsb_beat/sweep_a_l63_side.jsonl.run_info.json"
DATA = REPO / "data/anki10k"
FINAL_DIR = REPO / "results/srsb_beat/final"
ONLINE_JSONL = FINAL_DIR / "online_test1000.jsonl"
GBDT_JSONL = FINAL_DIR / "gbdt_test1000.jsonl"
COMPARISON = FINAL_DIR / "online_test1000.jsonl.comparison.json"
FSRS_JSONL = REPO / "external/srs-benchmark/result/FSRS-6-short.jsonl"

REPLAY_USERS = (994, 574, 203)
CARD_LABELS = ("Card A", "Card B", "Card C", "Card D")
MIN_TEST = 8
EPS = 1e-6
PROB_LO, PROB_HI = 1e-4, 1 - 1e-4
TOL = 1e-4
FSRS_NOTE = "FSRS-6 with default weights (per-card replay); headline 0.338 is per-user fitted FSRS-6"


def r6(x: float) -> float:
    return round(float(x), 6)


def r4_prob(x: float) -> float:
    return round(min(max(float(x), PROB_LO), PROB_HI), 4)


def ll_rows(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    q = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(q) + (1 - y) * np.log(1 - q))


def load_jsonl(path: Path) -> dict[int, dict]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[int(r["user"])] = {"size": int(r["size"]), "logLoss": float(r["metrics"]["LogLoss"])}
    return out


def replay_user(uid: int, booster, names, best_iter, loader, cfg) -> dict:
    """Frozen final prediction for one user, mirroring online_adapt.final_user()."""
    from sklearn.metrics import log_loss

    data_dir = DATA
    arr = OA.featurized_arrays(uid, booster, names, best_iter, loader, cfg, data_dir)
    t = arr["test"]
    ds = arr["ds"]
    variant, lr = OA.FINAL_VARIANT, OA.FINAL_LR
    Xd = OA.design(arr["margin"], arr["extra"], variant)
    p_full = OA.online_kernel(Xd, arr["y"], arr["labelled"], OA.w0_for(variant), lr, OA.LAMBDA)
    pos = arr["pos"][t]
    p_online = np.clip(p_full[arr["pos"]][t], OA.P_CLIP, 1 - OA.P_CLIP)
    p_gbdt = np.clip(1.0 / (1.0 + np.exp(-arr["margin"][pos])), OA.P_CLIP, 1 - OA.P_CLIP)
    p_fsrs = arr["ctx"]["f_R"][pos]
    if not np.all(np.isfinite(p_fsrs)):
        raise RuntimeError(f"user {uid}: non-finite f_R on test rows")
    p_fsrs = np.clip(p_fsrs, OA.P_CLIP, 1 - OA.P_CLIP)
    y = arr["y_eval"][t]
    if not np.isfinite(p_online).all():
        raise RuntimeError(f"user {uid}: NaN online predictions on test rows")
    rows = pd.DataFrame({
        "review_th": arr["keys"][t].astype(np.int64),
        "card_id": ds["card_id"].to_numpy(np.int64)[t],
        "day": ds["day_offset"].to_numpy(np.int64)[t],
        "rating": ds["rating"].to_numpy(np.int64)[t],
        "y": y.astype(np.int64),
        "pOnline": p_online,
        "pGbdt": p_gbdt,
        "pFsrs": p_fsrs,
    })
    raw = pd.read_parquet(Path(data_dir) / "revlogs" / f"user_id={uid}")
    st = R.review_stream(raw, cfg.max_seq_len * 2)
    hist = pd.DataFrame({
        "review_th": st["event_id"].to_numpy(np.int64),
        "card_id": st["card_id"].to_numpy(np.int64),
        "day": st["day_offset"].to_numpy(np.int64),
        "rating": st["rating"].to_numpy(np.int64),
        "y": st["y"].to_numpy(np.int64),
    })
    return dict(
        size=int(t.sum()),
        logLoss=float(log_loss(y, p_online, labels=[0, 1])),
        gbdtLogLoss=float(log_loss(y, p_gbdt, labels=[0, 1])),
        rows=rows,
        hist=hist,
    )


def pick_cards(replay: dict[int, dict]) -> list[tuple[int, int, str]]:
    pool = []
    for uid in REPLAY_USERS:
        rows = replay[uid]["rows"]
        for cid, g in rows.groupby("card_id"):
            if len(g) < MIN_TEST or g["y"].nunique() < 2:
                continue
            y = g["y"].to_numpy(float)
            fsrs_ll = ll_rows(y, g["pFsrs"].to_numpy(float)).mean()
            on_ll = ll_rows(y, g["pOnline"].to_numpy(float)).mean()
            pool.append((float(fsrs_ll - on_ll), int(uid), int(cid)))
    if len(pool) < 4:
        raise RuntimeError(f"only {len(pool)} candidate cards")
    pool.sort(key=lambda t: (-t[0], t[1], t[2]))
    top = pool[:3]
    honest = min(pool, key=lambda t: (t[0], t[1], t[2]))
    if honest in top:
        raise RuntimeError("honesty card collides with a top card")
    return [(u, c, "top") for _, u, c in top] + [(honest[1], honest[2], "fsrs_wins")]


def card_payload(label: str, picked: str, uid: int, cid: int, replay: dict[int, dict]) -> dict:
    rep = replay[uid]
    rows = rep["rows"]
    test = rows[rows["card_id"] == cid].set_index("review_th")
    hist = rep["hist"]
    hist = hist[hist["card_id"] == cid].sort_values("review_th")
    reviews = []
    for i, h in enumerate(hist.itertuples(index=False), 1):
        item = {"review": i, "day": int(h.day), "rating": int(h.rating), "y": int(h.y)}
        if int(h.review_th) in test.index:
            tr = test.loc[int(h.review_th)]
            item["scored"] = True
            item["pOnline"] = r4_prob(tr["pOnline"])
            item["pGbdt"] = r4_prob(tr["pGbdt"])
            item["pFsrsDefault"] = r4_prob(tr["pFsrs"])
        else:
            item["scored"] = False
        reviews.append(item)
    return {
        "label": label,
        "pickedFor": picked,
        "nScored": int(len(test)),
        "reviews": reviews,
    }


def build_online_seed() -> dict:
    import lightgbm as lgb

    os.chdir(REPO)
    info = json.loads(RUN_INFO.read_text(encoding="utf-8"))
    names, best_iter = info["features"], int(info["best_iter"])
    booster = lgb.Booster(model_file=str(MODEL))
    V._init(str(DATA), names)
    loader, cfg = V._W["loader"], V._W["cfg"]

    online = load_jsonl(ONLINE_JSONL)
    gbdt = load_jsonl(GBDT_JSONL)
    fsrs = load_jsonl(FSRS_JSONL)
    comparison = json.loads(COMPARISON.read_text(encoding="utf-8"))

    replay = {}
    replay_check = []
    for uid in REPLAY_USERS:
        rep = replay_user(uid, booster, names, best_iter, loader, cfg)
        replay[uid] = rep
        if abs(rep["logLoss"] - online[uid]["logLoss"]) > TOL:
            raise RuntimeError(f"user {uid}: replay LogLoss {rep['logLoss']} != {online[uid]['logLoss']}")
        if abs(rep["gbdtLogLoss"] - gbdt[uid]["logLoss"]) > TOL:
            raise RuntimeError(f"user {uid}: replay GBDT LogLoss {rep['gbdtLogLoss']} != {gbdt[uid]['logLoss']}")
        if rep["size"] != online[uid]["size"]:
            raise RuntimeError(f"user {uid}: replay size {rep['size']} != {online[uid]['size']}")
        replay_check.append({
            "user": uid,
            "size": rep["size"],
            "logLoss": r6(rep["logLoss"]),
            "recordedLogLoss": r6(online[uid]["logLoss"]),
            "gbdtLogLoss": r6(rep["gbdtLogLoss"]),
            "recordedGbdtLogLoss": r6(gbdt[uid]["logLoss"]),
        })

    picks = pick_cards(replay)
    cards = [card_payload(CARD_LABELS[i], picked, u, c, replay) for i, (u, c, picked) in enumerate(picks)]

    users = sorted(online)
    per_user_raw = [(u, online[u]["size"], online[u]["logLoss"], gbdt[u]["logLoss"], fsrs[u]["logLoss"]) for u in users]
    wins_fsrs = sum(1 for _, _, o, _, f in per_user_raw if o < f)
    wins_gbdt = sum(1 for _, _, o, g, _ in per_user_raw if o < g)
    per_user = [
        {"user": u, "size": s, "ours": r6(o), "gbdt": r6(g), "fsrs6": r6(f)}
        for u, s, o, g, f in per_user_raw
    ]

    return {
        "final": {
            "table": comparison["table"],
            "bootstrap": comparison["bootstrap"],
            "nUsers": comparison["n_users"],
            "ci": comparison["ci"],
        },
        "perUser": per_user,
        "winsVsFsrs6": wins_fsrs,
        "winsVsGbdt": wins_gbdt,
        "replayCheck": replay_check,
        "cards": cards,
        "fsrsNote": FSRS_NOTE,
    }


def _nan_to_none(obj):
    """srs-benchmark writes NaN for FSRS-6 AUC; strict JSON (and JSON.parse) needs null."""
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    return obj


def main(out_path: Path = DEFAULT_OUT) -> Path:
    seed = _nan_to_none(build_online_seed())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(seed, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    out_path.write_text(text + "\n", encoding="utf-8")
    print(f"wrote {out_path} ({len(text.encode('utf-8')) / 1024:.1f} KB)", file=sys.stderr)
    return out_path


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT)
