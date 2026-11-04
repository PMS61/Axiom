"""
run_stacking.py -- Phase 7 task 7.5: stacking / blending of GBDT, v3, FSRS-opt (and v2) on VALIDATION only.

  python -m fsrs_benchmark.experiments.run_stacking
Every validation prediction is out-of-fold: 5 user folds inside the validation set, the stacker is fitted on 4
folds and predicts the 5th. Weights for the test run (frozen) are fitted on ALL validation rows (`fit_full`).
Writes results/phase7_validation/stack_{anki,duolingo}.csv and the "Task 7.5: stacking" section of
docs/reports/phase-7-experiments.md.
"""
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from fsrs_benchmark.experiments.prepare_data import CACHE as P6_CACHE
from fsrs_benchmark.experiments.run_validation import ci_vs, evaluate
from fsrs_benchmark.experiments.run_validation_gbdt import build
from fsrs_benchmark.experiments.validation import make_validation_split
from fsrs_benchmark.models.gbdt import _fold_of
from fsrs_benchmark.metrics import log_loss

OUT = "results/phase7_validation"
LOG = "docs/reports/phase-7-experiments.md"
GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
CLIP = 1e-6


def logit(p):
    p = np.clip(np.asarray(p, float), CLIP, 1 - CLIP)
    return np.log(p / (1 - p))


def design(sc, preds, names, extra):
    cols = [logit(preds[n]) for n in names]
    if extra:
        cols += [np.log1p(sc["elapsed_days"].to_numpy(float)), np.log1p(sc["review_idx"].to_numpy(float))]
    return np.column_stack(cols)


class _LR:
    """Plain logistic regression (tiny L2 on the weights, none on the intercept), L-BFGS."""

    def __init__(self, X, y, l2=1e-6):
        mu, sd = X.mean(0), X.std(0) + 1e-9
        Z = np.column_stack([(X - mu) / sd, np.ones(len(X))])

        def f(w):
            z = Z @ w
            loss = np.sum(np.logaddexp(0, z) - y * z) + l2 * np.sum(w[:-1] ** 2)
            g = Z.T @ (1 / (1 + np.exp(-z)) - y) + 2 * l2 * np.r_[w[:-1], 0]
            return loss, g
        w = minimize(f, np.zeros(Z.shape[1]), jac=True, method="L-BFGS-B").x
        self.coef_ = (w[:-1] / sd)[None]
        self.intercept_ = np.array([w[-1] - np.sum(w[:-1] * mu / sd)])

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-(X @ self.coef_[0] + self.intercept_[0])))
        return np.column_stack([1 - p, p])


def fit_lr(X, y):
    return _LR(X, np.asarray(y, float))


def blend_w(p_a, p_b, y):
    return float(GRID[int(np.argmin([log_loss(y, w * p_a + (1 - w) * p_b) for w in GRID]))])


SPECS = {   # name -> (kind, component names, extra features)
    "stack3": ("lr", ["gbdt", "v3", "opt"], False),
    "stack4": ("lr", ["gbdt", "v3", "opt", "v2"], False),
    "stack3x": ("lr", ["gbdt", "v3", "opt"], True),
    "stack4x": ("lr", ["gbdt", "v3", "opt", "v2"], True),
    "stack_gv": ("lr", ["gbdt", "v3"], False),
    "blend_gv": ("blend", ["gbdt", "v3"], False),
}


def oof(sc, preds, kind, names, extra, folds):
    y = sc["y"].to_numpy()
    out = np.zeros(len(sc))
    for k in range(5):
        te, tr = folds == k, folds != k
        if kind == "lr":
            X = design(sc, preds, names, extra)
            out[te] = fit_lr(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        else:
            a, b = (np.asarray(preds[n]) for n in names)
            w = blend_w(a[tr], b[tr], y[tr])
            out[te] = w * a[te] + (1 - w) * b[te]
    return np.clip(out, CLIP, 1 - CLIP)


def fit_full(sc, preds, kind, names, extra):
    """Frozen weights fitted on ALL validation rows (for the test run)."""
    y = sc["y"].to_numpy()
    if kind == "lr":
        m = fit_lr(design(sc, preds, names, extra), y)
        return {"kind": "lr", "names": names, "extra": extra, "coef": m.coef_[0].tolist(), "intercept": float(m.intercept_[0])}
    a, b = (np.asarray(preds[n]) for n in names)
    return {"kind": "blend", "names": names, "w": blend_w(a, b, y)}


def main():
    res, frozen = {}, {}
    for d in ("anki", "duolingo"):
        ev = make_validation_split(build(d))
        sc = ev[ev["scored"]].set_index("event_id").sort_index()
        load = lambda p: pd.read_parquet(p)["p"].reindex(sc.index)
        preds = {"gbdt": load(f"{OUT}/pred_gbdt_{d}.parquet"), "v3": load(f"{OUT}/pred_v3_{d}.parquet"),
                 "opt": load(next(P6_CACHE.glob(f"{d}_val_fsrs_opt_*.parquet"))),
                 "v2": load(f"results/phase6_validation/pred_v2_{d}.parquet")}
        assert all(p.notna().all() for p in preds.values())
        folds = _fold_of(sc["user_id"].to_numpy(np.int64) + 12345)
        rows = {}
        for name, (kind, names, extra) in SPECS.items():
            p = pd.Series(oof(sc, preds, kind, names, extra, folds), index=sc.index)
            rows[name] = p
            frozen.setdefault(d, {})[name] = fit_full(sc, preds, kind, names, extra)
        table = []
        for name, p in {"gbdt": preds["gbdt"], "v3": preds["v3"], **rows}.items():
            r = {"model": name, **evaluate(ev, p)}
            for ref in ("gbdt", "opt"):
                if name == ref:
                    continue
                ci = ci_vs(ev, p, preds[ref])
                r.update({f"d_vs_{ref}": ci["diff"], f"lo_{ref}": ci["lo"], f"hi_{ref}": ci["hi"]})
            table.append(r)
        t = pd.DataFrame(table)
        t.to_csv(f"{OUT}/stack_{d}.csv", index=False)
        res[d] = t
        print(d); print(t[["model", "log_loss", "auc", "d_vs_gbdt", "lo_gbdt", "hi_gbdt"]].round(4).to_string(index=False))
    # decision: a stacker must beat GBDT on BOTH datasets with a CI excluding 0; pick lowest mean log loss among those
    ok = []
    for name in SPECS:
        a, b = (res[d].set_index("model").loc[name] for d in ("anki", "duolingo"))
        if a["hi_gbdt"] < 0 and b["hi_gbdt"] < 0:
            ok.append((name, (a["log_loss"] + b["log_loss"]) / 2))
    ok.sort(key=lambda x: x[1])
    decision = ok[0][0] if ok else "gbdt"
    print("DECISION:", decision, ok)
    json.dump({"decision": decision, "weights": frozen}, open(f"{OUT}/stack_weights.json", "w"), indent=1)
    lines = ["\n## Task 7.5: stacking\n",
             "Validation only. Stackers are fitted cross-fitted by user (5 user folds inside the validation set; every prediction is out-of-fold) on the logits of "
             "GBDT (frozen), ours_v3 (q2), FSRS-opt, and v2 (stack4*); `x` adds log1p(elapsed_days) and log1p(review_idx); `blend_gv` is a convex blend of GBDT and v3 with w on a 0.05 grid chosen on the 4 training folds. "
             "CIs are user-cluster bootstrap (1000 reps) of the log-loss difference; negative = better than the reference.\n"]
    for d in res:
        lines.append(f"\n**{d}**\n\n| model | log_loss | auc | ece | dLL vs GBDT [95% CI] | dLL vs FSRS-opt [95% CI] |\n|---|---|---|---|---|---|")
        for _, r in res[d].iterrows():
            g = f"{r['d_vs_gbdt']:+.4f} [{r['lo_gbdt']:+.4f}, {r['hi_gbdt']:+.4f}]" if "d_vs_gbdt" in r and pd.notna(r["d_vs_gbdt"]) else "-"
            o = f"{r['d_vs_opt']:+.4f} [{r['lo_opt']:+.4f}, {r['hi_opt']:+.4f}]" if pd.notna(r.get("d_vs_opt")) else "-"
            lines.append(f"| {r['model']} | {r['log_loss']:.5f} | {r['auc']:.4f} | {r['ece']:.4f} | {g} | {o} |")
    lines.append(f"\n**Decision** (rule: a stacker is used only if it beats GBDT alone on BOTH datasets with a CI excluding 0; lowest mean log loss among those): "
                 f"**{decision}**." + ("" if ok else " No stacker qualifies, so GBDT alone is the final pick.") + "\n")
    open(LOG, "a").write("\n".join(lines))


if __name__ == "__main__":
    main()
