"""Compare our srs-benchmark-format jsonl with the published srs-benchmark results (Phase 2.2).

(a) --check-readme: reproduces the README "Without same-day reviews" table. The README's
    means are UNWEIGHTED means over users (srs-benchmark evaluate.py uses scale="users"),
    not review-weighted. The size-weighted means are printed next to them for reference.
(b) --ours <jsonl> [--users <ids>]: on the users that are in our file and in the published
    files, with the same `size`, prints a table of means for our GBDT and the published
    models, the size-match count, and paired user-cluster bootstrap CIs (1000 resamples,
    weighted by size) for GBDT minus FSRS-6 and GBDT minus RWKV.

Usage (from the repo root):
  .venv/bin/python -m fsrs_benchmark.srsb.compare --check-readme
  .venv/bin/python -m fsrs_benchmark.srsb.compare --ours result/GBDT.jsonl --users ids.txt
"""
import argparse
import json
from pathlib import Path

import numpy as np

from fsrs_benchmark.srsb import SRSB_DIR

DEFAULT_RESULT_DIR = SRSB_DIR / "result"

# README "Without same-day reviews" table -> srs-benchmark result file (result/<name>.jsonl).
# Each README value is reproduced as the unweighted mean of that file's per-user LogLoss.
README_WITHOUT_SAME_DAY = {
    # display name: (result file, README LogLoss)
    "RWKV-Instant": ("RWKV-P", 0.2773),
    "RWKV-Curve": ("RWKV", 0.3193),
    "GRU": ("GRU-short-secs-equalize_test_with_non_secs", 0.3328),
    "LSTM": ("LSTM-short-secs-duration-equalize_test_with_non_secs", 0.3332),
    "MOVING-AVG": ("MOVING-AVG", 0.3369),
    "FSRS-7 recency": ("FSRS-7-short-secs-recency-equalize_test_with_non_secs", 0.3370),
    "Logistic Regression": ("LogisticRegression-short-secs-recency-equalize_test_with_non_secs", 0.3393),
    "FSRS-6": ("FSRS-6-short", 0.3460),
    "DASH": ("DASH", 0.3682),
    "AVG": ("AVG", 0.3945),
    "HLR": ("HLR", 0.4694),
}
FSRS6_FILE = "FSRS-6-short"
RWKV_FILES = {"RWKV-Instant": "RWKV-P", "RWKV-Curve": "RWKV"}
TABLE_METRICS = ("LogLoss", "RMSE(bins)", "AUC")


def load_rows(path) -> dict[int, dict]:
    """{user: {"size": n, "metrics": {...}}} from a srs-benchmark style jsonl."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            out[int(r["user"])] = {"size": int(r["size"]), "metrics": r["metrics"]}
    return out


def weighted_mean(values, sizes) -> float:
    """Size-weighted mean (the same formula as srs-benchmark's weighted_avg_and_std)."""
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(sizes, dtype=np.float64)
    return float(np.sum(v * w) / np.sum(w))


def _column(rows: dict, users, metric):
    vals, sizes = [], []
    for u in users:
        m = rows[u]["metrics"].get(metric)
        if m is None or not np.isfinite(m):      # AUC is null (or NaN) when a user has one class only
            continue
        vals.append(m)
        sizes.append(rows[u]["size"])
    return vals, sizes


def readme_check(result_dir=DEFAULT_RESULT_DIR) -> list[dict]:
    """For each README row: our reproduction (unweighted LogLoss mean, all users) against the README value."""
    result_dir = Path(result_dir)
    out = []
    for name, (fname, readme) in README_WITHOUT_SAME_DAY.items():
        rows = load_rows(result_dir / f"{fname}.jsonl")
        users = sorted(rows)
        vals, sizes = _column(rows, users, "LogLoss")
        unweighted = float(np.mean(vals))
        out.append(dict(model=name, file=fname, n_users=len(users), readme=readme,
                        unweighted=round(unweighted, 4), weighted=round(weighted_mean(vals, sizes), 4),
                        match=round(unweighted, 4) == readme))
    return out


def size_match(ours: dict, reference: dict, users) -> tuple[int, int, list[int]]:
    """(matching, compared, mismatched user ids) of `size` between ours and a reference file."""
    compared = [u for u in users if u in reference]
    bad = [u for u in compared if reference[u]["size"] != ours[u]["size"]]
    return len(compared) - len(bad), len(compared), bad


def paired_bootstrap(diff, sizes, n_boot: int = 1000, ci: float = 0.99, seed: int = 0) -> dict:
    """User-cluster bootstrap of the size-weighted mean of per-user differences.

    Users are resampled with replacement (n_boot times). Each resample's statistic is the mean of diff
    weighted by size. The CI is the central `ci` percentile interval.
    """
    d = np.asarray(diff, dtype=np.float64)
    w = np.asarray(sizes, dtype=np.float64)
    n = len(d)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    ws = w[idx]
    stats = (ws * d[idx]).sum(axis=1) / ws.sum(axis=1)
    lo, hi = np.percentile(stats, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])
    return dict(mean=weighted_mean(d, w), lo=float(lo), hi=float(hi), ci=ci, n_boot=n_boot, n_users=n)


def parse_users(spec):
    if spec is None:
        return None
    p = Path(spec)
    if p.exists():
        text = p.read_text(encoding="utf-8")
        if p.suffix == ".json":
            return {int(x) for x in json.loads(text)}
        return {int(x) for x in text.replace(",", " ").split() if x.strip()}
    return {int(x) for x in spec.replace(",", " ").split() if x.strip()}


LISTED_MODELS = tuple(README_WITHOUT_SAME_DAY)          # the 11 published models of the README table
BOOTSTRAP_BASELINES = ("FSRS-6", "RWKV-Instant", "RWKV-Curve", "GRU", "AVG", "MOVING-AVG")
MAIN_METRICS = ("LogLoss", "RMSE(bins)", "AUC")


def _same_size_users(ours, rows, users):
    """Users present in `rows` whose `size` equals ours (paired test rows)."""
    return sorted(u for u in users if u in rows and rows[u]["size"] == ours[u]["size"])


def compare(ours_path, result_dir=DEFAULT_RESULT_DIR, users=None, ci: float = 0.99,
            n_boot: int = 1000, seed: int = 0) -> dict:
    """Compare our jsonl with the published files.

    Per model: its own common users (in ours and in the model file, same size), reported as n_users.
    Main table: users common to GBDT and ALL listed models (same size), unweighted and size-weighted means.
    Bootstrap: GBDT minus each baseline on the main common users, user-cluster, size-weighted.
    Size match: how many of our users have the same `size` as FSRS-6-short.
    """
    result_dir = Path(result_dir)
    ours = load_rows(ours_path)
    subset = set(ours) if users is None else (set(ours) & set(users))
    fsrs6 = load_rows(result_dir / f"{FSRS6_FILE}.jsonl")
    size_ok, size_cmp, size_bad = size_match(ours, fsrs6, sorted(subset))
    models = {name: load_rows(result_dir / f"{fname}.jsonl") for name, (fname, _) in README_WITHOUT_SAME_DAY.items()}

    per_model = {}
    for name, rows in models.items():
        own = _same_size_users(ours, rows, subset)
        per_model[name] = dict(file=README_WITHOUT_SAME_DAY[name][0], n_users=len(own),
                               n_ours_only=len(subset) - len(own),
                               LogLoss_u=round(float(np.mean(_column(rows, own, "LogLoss")[0])), 4)
                               if own else None,
                               LogLoss_w=round(weighted_mean(*_column(rows, own, "LogLoss")), 4) if own else None)

    common = [u for u in _same_size_users(ours, ours, subset)
              if all(u in m and m[u]["size"] == ours[u]["size"] for m in models.values())]
    if not common:
        raise ValueError("no users are common to GBDT and all listed models with the same size")
    sizes = [ours[u]["size"] for u in common]

    table = []
    rows_for = {"GBDT (ours)": ours, **models}
    for name, rows in rows_for.items():
        fname = "ours" if name == "GBDT (ours)" else README_WITHOUT_SAME_DAY[name][0]
        row = dict(model=name, file=fname, n_users=len(common))
        for metric in MAIN_METRICS:
            vals, sz = _column(rows, common, metric)     # values and sizes stay aligned (AUC can be None)
            row[f"{metric}_w"] = round(weighted_mean(vals, sz), 4) if vals else None
            row[f"{metric}_n"] = len(vals)
            row[f"{metric}_u"] = round(float(np.mean(vals)), 4) if vals else None
        table.append(row)

    boots = {}
    for ref in BOOTSTRAP_BASELINES:
        diff = [ours[u]["metrics"]["LogLoss"] - models[ref][u]["metrics"]["LogLoss"] for u in common]
        boots[f"GBDT - {ref}"] = paired_bootstrap(diff, sizes, n_boot=n_boot, ci=ci, seed=seed)

    return dict(size_match=dict(matching=size_ok, compared=size_cmp, mismatched=size_bad[:20]),
                paired_users=len(common), per_model=per_model, table=table, bootstrap=boots,
                ci=ci, n_boot=n_boot)


def _print_readme(rows):
    print("README 'Without same-day reviews' LogLoss (unweighted mean over users, as README)")
    print(f"{'model':22s} {'file':55s} {'users':>6s} {'README':>7s} {'ours':>7s} {'match':>5s} {'size-w':>7s}")
    for r in rows:
        print(f"{r['model']:22s} {r['file']:55s} {r['n_users']:6d} {r['readme']:7.4f} "
              f"{r['unweighted']:7.4f} {str(r['match']):>5s} {r['weighted']:7.4f}")


def _print_compare(res):
    sm = res["size_match"]
    print(f"size match vs {FSRS6_FILE}.jsonl: {sm['matching']} of {sm['compared']} users "
          f"(mismatched ids, first 20: {sm['mismatched']})")
    print(f"main table: users common to GBDT and all {len(LISTED_MODELS)} listed models, same size: "
          f"{res['paired_users']}")
    print(f"{'model':22s} {'file':55s} {'n':>6s} {'LL(u)':>7s} {'LL(w)':>7s} {'RMSEb(u)':>9s} {'RMSEb(w)':>9s} "
          f"{'AUC(u)':>7s} {'AUC(w)':>7s}")
    for r in res["table"]:
        print(f"{r['model']:22s} {r['file']:55s} {r['n_users']:6d} {r['LogLoss_u']:7.4f} {r['LogLoss_w']:7.4f} "
              f"{r['RMSE(bins)_u']:9.4f} {r['RMSE(bins)_w']:9.4f} {r['AUC_u']:7.4f} {r['AUC_w']:7.4f}")
    print("per-model common users (in ours and in the model file, same size):")
    for name, pm in res["per_model"].items():
        print(f"  {name:22s} n={pm['n_users']:5d} ours-only={pm['n_ours_only']:4d} "
              f"LL(u)={pm['LogLoss_u']} LL(w)={pm['LogLoss_w']}")
    print(f"paired LogLoss difference on the main common users, size-weighted, {int(res['ci'] * 100)}% "
          f"user-cluster bootstrap ({res['n_boot']} resamples):")
    for label, b in res["bootstrap"].items():
        excl = "excludes 0" if (b["lo"] > 0 or b["hi"] < 0) else "includes 0"
        print(f"  {label:22s} {b['mean']:+.4f}  [{b['lo']:+.4f}, {b['hi']:+.4f}]  {excl}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Compare with srs-benchmark published results.")
    ap.add_argument("--result-dir", default=str(DEFAULT_RESULT_DIR))
    ap.add_argument("--check-readme", action="store_true")
    ap.add_argument("--ours", default=None, help="our srs-benchmark style jsonl")
    ap.add_argument("--users", default=None, help="user ids (comma separated) or a file; default all users in --ours")
    ap.add_argument("--ci", type=float, default=0.99)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.check_readme:
        _print_readme(readme_check(args.result_dir))
    if args.ours:
        res = compare(args.ours, args.result_dir, parse_users(args.users), args.ci, args.boot, args.seed)
        _print_compare(res)
    return 0


if __name__ == "__main__":
    main()
