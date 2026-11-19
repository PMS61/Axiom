"""Export real Axiom results into the seed JSON used by the /demo route.

Reads results/full_u100_c1000_final/ (predictions.parquet, events.parquet,
metrics.csv, user_wins.csv) and writes client/src/lib/demo/seed/results.json.

Output keys:
  overall      {model: {logLoss, auc, ece, brier}} for the six demo models
  gapBins      {model: [{bin, n, logLoss}]} for fsrs_opt and gbdt
  reliability  {model: [{binLow, binHigh, meanPred, meanObs, n}]} (10 bins)
  userWins     {count, total, diffs} gbdt vs fsrs_opt per-user log loss
  cards        [{label, nScored, pickedFor, reviews: [...]}] (4 anonymised cards)

Run from the repo root:
  uv run --quiet --with pandas --with pyarrow python client/scripts/export_demo_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "full_u100_c1000_final"
DEFAULT_OUT = REPO / "client" / "src" / "lib" / "demo" / "seed" / "results.json"

OVERALL_MODELS = ["fsrs_default", "fsrs_opt", "hlr", "ours", "ours_v3", "gbdt"]
GAP_MODELS = ["fsrs_opt", "gbdt"]
RELIABILITY_MODELS = ["fsrs_opt", "gbdt"]
CARD_LABELS = ["Card A", "Card B", "Card C", "Card D"]
EPS = 1e-6
MIN_SCORED = 6


def r4(x: float) -> float:
    return round(float(x), 4)


def log_loss_vec(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    q = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(q) + (1 - y) * np.log(1 - q))


def build_overall(metrics: pd.DataFrame) -> dict:
    rows = metrics[(metrics["scope"] == "overall") & metrics["model"].isin(OVERALL_MODELS)]
    out = {}
    for model in OVERALL_MODELS:
        row = rows[rows["model"] == model]
        if len(row) != 1:
            raise ValueError(f"expected one overall row for {model}, got {len(row)}")
        row = row.iloc[0]
        out[model] = {
            "logLoss": r4(row["log_loss"]),
            "auc": r4(row["auc"]),
            "ece": r4(row["ece"]),
            "brier": r4(row["brier"]),
        }
    return out


def build_gap_bins(metrics: pd.DataFrame) -> dict:
    out = {}
    for model in GAP_MODELS:
        rows = metrics[(metrics["scope"] == "gap") & (metrics["model"] == model)]
        if rows.empty:
            raise ValueError(f"no gap rows for {model}")
        out[model] = [
            {"bin": str(r["bin"]), "n": int(r["n"]), "logLoss": r4(r["log_loss"])}
            for _, r in rows.iterrows()
        ]
    return out


def build_reliability(preds: pd.DataFrame) -> dict:
    out = {}
    for model in RELIABILITY_MODELS:
        p = preds[model].to_numpy(dtype=float)
        y = preds["y"].to_numpy(dtype=float)
        idx = np.minimum((p * 10).astype(int), 9)
        bins = []
        for b in range(10):
            mask = idx == b
            n = int(mask.sum())
            if n == 0:
                continue
            bins.append(
                {
                    "binLow": r4(b / 10),
                    "binHigh": r4((b + 1) / 10),
                    "meanPred": r4(p[mask].mean()),
                    "meanObs": r4(y[mask].mean()),
                    "n": n,
                }
            )
        out[model] = bins
    return out


def build_user_wins(preds: pd.DataFrame) -> dict:
    per_user = preds.assign(
        ll_gbdt=log_loss_vec(preds["y"].to_numpy(float), preds["gbdt"].to_numpy(float)),
        ll_fsrs=log_loss_vec(preds["y"].to_numpy(float), preds["fsrs_opt"].to_numpy(float)),
    ).groupby("user_id")[["ll_gbdt", "ll_fsrs"]].mean()
    diff = per_user["ll_gbdt"] - per_user["ll_fsrs"]
    return {
        "count": int((per_user["ll_gbdt"] < per_user["ll_fsrs"]).sum()),
        "total": int(len(per_user)),
        "diffs": sorted(r4(d) for d in diff.to_numpy()),
    }


def pick_cards(preds: pd.DataFrame) -> list[tuple[int, int]]:
    """Return 4 (user_id, card_id) keys: top 3 where gbdt wins most, then the one where fsrs_opt wins most."""
    rows = []
    for (uid, cid), g in preds.groupby(["user_id", "card_id"]):
        if len(g) < MIN_SCORED or g["y"].nunique() < 2:
            continue
        y = g["y"].to_numpy(float)
        fsrs_ll = log_loss_vec(y, g["fsrs_opt"].to_numpy(float)).mean()
        gbdt_ll = log_loss_vec(y, g["gbdt"].to_numpy(float)).mean()
        rows.append((fsrs_ll - gbdt_ll, int(uid), int(cid)))
    # Sort by diff desc, then ids for determinism.
    rows.sort(key=lambda t: (-t[0], t[1], t[2]))
    top = rows[:3]
    honest = min(rows, key=lambda t: (t[0], t[1], t[2]))
    if any((honest[1], honest[2]) == (t[1], t[2]) for t in top):
        raise ValueError("honesty card collides with a top card; not enough candidates")
    return [(t[1], t[2]) for t in top] + [(honest[1], honest[2])]


def build_cards(preds: pd.DataFrame, events: pd.DataFrame) -> list[dict]:
    keys = pick_cards(preds)
    scored = events[events["scored"]]
    out = []
    for i, (uid, cid) in enumerate(keys):
        ev = events[(events["user_id"] == uid) & (events["card_id"] == cid)].sort_values("review_idx")
        pr = preds[(preds["user_id"] == uid) & (preds["card_id"] == cid)].set_index("event_id")
        reviews = []
        for _, e in ev.iterrows():
            row = {
                "review": int(e["review_idx"]),
                "day": int(e["day_offset"]),
                "rating": int(e["rating"]),
                "y": int(e["y"]),
                "scored": bool(e["scored"]),
            }
            if bool(e["scored"]):
                p = pr.loc[int(e["event_id"])]
                row["pFsrs"] = r4(min(max(p["fsrs_opt"], 1e-4), 1 - 1e-4))
                row["pOurs"] = r4(min(max(p["gbdt"], 1e-4), 1 - 1e-4))
            reviews.append(row)
        n_scored = int(scored[(scored["user_id"] == uid) & (scored["card_id"] == cid)].shape[0])
        out.append(
            {
                "label": CARD_LABELS[i],
                "nScored": n_scored,
                "pickedFor": "top" if i < 3 else "fsrs_wins",
                "reviews": reviews,
            }
        )
    return out


def build_seed() -> dict:
    metrics = pd.read_csv(RESULTS / "metrics.csv", keep_default_na=False, na_values=[""])
    metrics["bin"] = metrics["bin"].astype("string")
    preds = pd.read_parquet(RESULTS / "predictions.parquet")
    events = pd.read_parquet(RESULTS / "events.parquet")
    return {
        "overall": build_overall(metrics),
        "gapBins": build_gap_bins(metrics),
        "reliability": build_reliability(preds),
        "userWins": build_user_wins(preds),
        "cards": build_cards(preds, events),
    }


def main(out_path: Path = DEFAULT_OUT) -> Path:
    seed = build_seed()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(seed, separators=(",", ":"), ensure_ascii=False)
    out_path.write_text(text + "\n", encoding="utf-8")
    print(f"wrote {out_path} ({len(text.encode('utf-8')) / 1024:.1f} KB)", file=sys.stderr)
    return out_path


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    main(out)
