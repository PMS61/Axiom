"""Report generator: predictions.parquet + run_info.json -> metrics, paired CIs, figures, report.md.

Why this module: every number in the write-up must come from the same functions the tests
check (fsrs_benchmark.metrics), and the report must be reproducible from a run folder alone.
It writes only derived files, so it can be re-run on any finished run without touching the
predictions.

Usage: .venv/bin/python -m fsrs_benchmark.report --run_dir results/<run>

Writes into run_dir (or --out_dir):
  metrics.csv     overall and per gap-bin metrics, one row per (model, scope)
  paired.csv      model minus reference differences with 95% user-cluster bootstrap CIs
  users.csv       per-user n, log loss and AUC for each model
  user_wins.csv   share of users where one model has the lower log loss than the reference
  reliability.png reliability diagram with a bin-count panel
  gapbins_logloss.png log loss per gap bin, with n per bin
  report.md       the human-readable report
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fsrs_benchmark import metrics  # noqa: E402

ALL_MODELS = ["fsrs_default", "fsrs_opt", "hlr", "ours", "ours_kappa"]
DISPLAY = {
    "fsrs_default": "FSRS-6 (default)",
    "fsrs_opt": "FSRS-6 (optimised)",
    "hlr": "HLR (Settles & Meeder)",
    "ours": "Ours (dynamic, learned λ)",
    "ours_kappa": "Ours (+κ)",
}
# Okabe-Ito colourblind-safe palette, one colour per model.
COLOURS = {
    "fsrs_default": "#E69F00",
    "fsrs_opt": "#0072B2",
    "hlr": "#009E73",
    "ours": "#D55E00",
    "ours_kappa": "#CC79A7",
}
ID_COLUMNS = ["event_id", "user_id", "card_id", "y", "elapsed_days", "day_offset"]
# Scored events have elapsed_days >= 1, so the first bin is exactly one day.
GAP_BINS = [("1d", 1, 1), ("2-3d", 2, 3), ("4-14d", 4, 14), ("15-60d", 15, 60), ("61d+", 61, None)]
PAIR_METRICS = ["log_loss", "brier", "auc", "ece"]
PAIRED_COLUMNS = ["reference", "model", "metric", "diff", "lo", "hi", "ci_excludes_zero"]
WINS_COLUMNS = ["reference", "model", "users", "users_lower_log_loss", "share"]
MARKERS = {"fsrs_default": "o", "fsrs_opt": "s", "hlr": "^", "ours": "D", "ours_kappa": "v"}
SPARSE_N = 30  # bins with fewer events are drawn faded
REFERENCES = ["fsrs_opt", "ours"]
N_BOOT = 1000
BOOT_SEED = 0


def _safe(fn, *args):
    """Metric value, or NaN when it is undefined (for example AUC with one class)."""
    try:
        return float(fn(*args))
    except ValueError:
        return float("nan")


def _metric_row(y, p):
    """All scalar metrics for one set of predictions. AUC is NaN if only one class is present."""
    row = {"n": int(len(y)), "recall": float(np.mean(y))}
    row["log_loss"] = _safe(metrics.log_loss, y, p)
    row["brier"] = _safe(metrics.brier, y, p)
    row["auc"] = _safe(metrics.auc, y, p)
    row["ece"] = _safe(metrics.ece, y, p)
    row["mean_bias"] = _safe(metrics.mean_bias, y, p)
    row["rmse_bins"] = _safe(metrics.rmse_bins, y, p)
    return row


def model_columns(preds):
    """Model columns present in the predictions, in the canonical order."""
    present = [c for c in preds.columns if c not in ID_COLUMNS]
    unknown = [c for c in present if c not in ALL_MODELS]
    if unknown:
        raise ValueError(f"unknown model columns in predictions: {unknown}")
    return [m for m in ALL_MODELS if m in present]


def metrics_table(preds, models):
    rows = []
    for m in models:
        rows.append({"model": m, "scope": "overall", "bin": "", **_metric_row(preds["y"], preds[m])})
    for label, lo, hi in GAP_BINS:
        in_bin = preds["elapsed_days"] >= lo
        if hi is not None:
            in_bin &= preds["elapsed_days"] <= hi
        sub = preds[in_bin]
        for m in models:
            rows.append({"model": m, "scope": "gap", "bin": label, **_metric_row(sub["y"], sub[m])})
    return pd.DataFrame(rows)


def paired_table(preds, models):
    rows = []
    for ref in REFERENCES:
        if ref not in models:
            continue
        for m in models:
            if m == ref:
                continue
            for metric in PAIR_METRICS:
                res = metrics.paired_bootstrap(
                    preds["user_id"].to_numpy(), preds["y"].to_numpy(),
                    preds[m].to_numpy(), preds[ref].to_numpy(),
                    metric, n_boot=N_BOOT, seed=BOOT_SEED)
                excludes_zero = bool(np.isfinite(res["lo"]) and (res["lo"] > 0 or res["hi"] < 0))
                rows.append({"reference": ref, "model": m, "metric": metric,
                             "diff": res["diff"], "lo": res["lo"], "hi": res["hi"],
                             "ci_excludes_zero": excludes_zero})
    return pd.DataFrame(rows, columns=PAIRED_COLUMNS)


def per_user_table(preds, models):
    rows = []
    for uid, sub in preds.groupby("user_id", sort=True):
        row = {"user_id": int(uid), "n": int(len(sub))}
        for m in models:
            row[f"{m}_log_loss"] = _safe(metrics.log_loss, sub["y"], sub[m])
            row[f"{m}_auc"] = _safe(metrics.auc, sub["y"], sub[m])
        rows.append(row)
    return pd.DataFrame(rows)


def user_wins(users, models):
    rows = []
    for ref in REFERENCES:
        if ref not in models:
            continue
        for m in models:
            if m == ref:
                continue
            a = users[f"{m}_log_loss"].to_numpy(float)
            b = users[f"{ref}_log_loss"].to_numpy(float)
            ok = np.isfinite(a) & np.isfinite(b)
            wins = int(np.sum(a[ok] < b[ok]))
            rows.append({"reference": ref, "model": m, "users": int(ok.sum()),
                         "users_lower_log_loss": wins,
                         "share": wins / ok.sum() if ok.sum() else float("nan")})
    return pd.DataFrame(rows, columns=WINS_COLUMNS)


def plot_reliability(preds, models, path):
    fig, (ax, ax_hist) = plt.subplots(2, 1, figsize=(7, 8.5), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 1.3]})
    ax.plot([0, 1], [0, 1], color="#666666", linestyle="--", linewidth=1, label="perfect calibration")
    width = 0.1 * 0.8 / max(len(models), 1)  # each bin is 0.1 wide; 5 models share 80% of it
    edges = np.linspace(0, 1, 11)
    for i, m in enumerate(models):
        rc = metrics.reliability_curve(preds["y"], preds[m], n_bins=10)
        shown = rc[rc["n"] > 0]
        ax.plot(shown["mean_p"], shown["mean_y"], marker=MARKERS[m], markersize=6,
                color=COLOURS[m], linewidth=1.6, label=DISPLAY[m], alpha=0.9)
        # Marker area scales with bin size; bins under SPARSE_N events are drawn hollow and faded.
        biggest = max(rc["n"].max(), 1)
        for _, b in shown.iterrows():
            sparse = b["n"] < SPARSE_N
            ax.scatter([b["mean_p"]], [b["mean_y"]], s=20 + 180 * np.sqrt(b["n"] / biggest),
                       marker=MARKERS[m], facecolors="none" if sparse else COLOURS[m],
                       edgecolors=COLOURS[m], alpha=0.35 if sparse else 0.9, linewidths=1.2, zorder=3)
        centres = (edges[:-1] + edges[1:]) / 2 + (i - (len(models) - 1) / 2) * width
        ax_hist.bar(centres, rc["n"], width=width, color=COLOURS[m], label=DISPLAY[m])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("observed recall rate (mean y in bin)", fontsize=11)
    ax.set_title("Reliability: scored events, 10 equal-width bins", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.grid(alpha=0.3)
    # Inside the top panel, lower right: the diagonal runs bottom-left to top-right, so this corner is empty.
    ax.legend(loc="lower right", fontsize=9, frameon=True, framealpha=0.9)
    fig.text(0.5, 0.005, f"Marker area is proportional to the bin's event count. Hollow, faded markers: "
             f"fewer than {SPARSE_N} events.", ha="center", fontsize=8, color="#444444")
    ax_hist.set_xlabel("predicted recall probability p", fontsize=11)
    ax_hist.set_ylabel("events per bin", fontsize=11)
    ax_hist.tick_params(labelsize=10)
    ax_hist.grid(alpha=0.3, axis="y")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_gap_logloss(table, models, path):
    bins = [label for label, _, _ in GAP_BINS]
    gap = table[table["scope"] == "gap"]
    n_per_bin = gap[gap["model"] == models[0]].set_index("bin")["n"]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    width = 0.8 / max(len(models), 1)
    x = np.arange(len(bins))
    for i, m in enumerate(models):
        vals = gap[gap["model"] == m].set_index("bin").reindex(bins)["log_loss"].to_numpy(float)
        ax.bar(x + (i - (len(models) - 1) / 2) * width, vals, width=width,
               color=COLOURS[m], label=DISPLAY[m])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\nn={int(n_per_bin.get(b, 0))}" for b in bins], fontsize=10)
    ax.set_xlabel("gap since the card's previous review", fontsize=11)
    ax.set_ylabel("log loss (lower is better)", fontsize=11)
    ax.set_title("Log loss by gap bin", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fmt(x, digits=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.{digits}f}"


def _table(header, rows, bold_cols=None):
    """Markdown table. bold_cols maps column index -> 'min', 'max' or 'closest0' to bold the best cell."""
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    bold_cols = bold_cols or {}
    best = {}  # column -> set of row indices holding the best value (ties included)
    for col, rule in bold_cols.items():
        vals = [(r[col], i) for i, r in enumerate(rows) if isinstance(r[col], float) and np.isfinite(r[col])]
        if vals:
            key = {"min": lambda v: v, "max": lambda v: -v, "closest0": lambda v: abs(v)}[rule]
            top = min(key(v) for v, _ in vals)
            best[col] = {i for v, i in vals if key(v) == top}
    for i, r in enumerate(rows):
        cells = []
        for col, val in enumerate(r):
            if isinstance(val, str):
                text = val
            elif isinstance(val, (int, np.integer)):
                text = str(int(val))
            else:
                text = _fmt(val)
            if i in best.get(col, ()):
                text = f"**{text}**"
            cells.append(text)
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


MCMC_VARIANTS = [("ours", "Ours (dynamic, learned λ)"), ("ours_kappa", "Ours (+κ)")]


def mcmc_section(run_dir):
    """Markdown lines for the MCMC diagnostics of each fitted ours variant (empty if none)."""
    lines = []
    for key, name in MCMC_VARIANTS:
        path = run_dir / f"diagnostics_{key}.json"
        if not path.exists():
            continue
        diag = json.loads(path.read_text())
        if not lines:
            lines += ["## MCMC diagnostics", "",
                      "R-hat here is split-R-hat computed on a single chain (no between-chain comparison). "
                      "ESS is the effective sample size of the kept draws.", ""]
        params = [k for k, v in diag.items() if isinstance(v, dict)]
        rows = [[p, diag[p]["mean"], f"[{_fmt(diag[p]['lo'])}, {_fmt(diag[p]['hi'])}]",
                 None if diag[p]["ess"] is None else int(round(diag[p]["ess"])), diag[p]["rhat"]]
                for p in params]
        lines += [f"### {name}", "",
                  f"Kept draws: {diag['n_samples']}; lambda Metropolis-Hastings acceptance: mean after "
                  f"burn-in {_fmt(diag.get('mh_acc_rate_mean_postburn'))}, mean over all sweeps "
                  f"{_fmt(diag.get('mh_acc_rate'))}. {diag.get('notes', '')}", "",
                  _table(["Param", "Mean", "95% interval", "ESS", "R-hat"], rows), ""]
    return lines


def write_report(run_dir, out_dir, preds, info, models, mt, pt, uw):
    lines = [f"# Results: {run_dir.name}", ""]

    lines += ["## Data", ""]
    if info:
        c = info["counts"]
        scored_recall = float(preds["y"].mean())
        fb = info["models"].get("fsrs_opt", {}).get("fallback_count")
        lines += [
            f"- Users: {c['users']}, cards: {c['cards']}, events: {c['events']} "
            f"({c['train']} train, {c['test']} test)",
            f"- Scored events: {c['scored']}, recall rate {scored_recall:.3f}",
            f"- FSRS-6 optimised fallback users (defaults kept): "
            f"{fb if fb is not None else 'n/a'}",
            f"- Users with zero train rows: {len(info.get('users_zero_train', []))}",
            f"- Git commit: {info['git']['commit']} (dirty: {info['git']['dirty']})",
            f"- Config: n_users={info['args']['n_users']}, max_cards={info['args']['max_cards']}, "
            f"seed={info['args']['seed']}, iters={info['args']['iters']}, burn={info['args']['burn']}",
            "",
            "Per-model timings (seconds):",
            "",
            _table(["Model", "Fit", "Predict"],
                   [[DISPLAY[m], info["models"][m].get("fit_seconds"), info["models"][m].get("predict_seconds")]
                    for m in models if m in info["models"]]),
            "",
        ]
        untimed = [m for m in models if m in info["models"]
                   and ("fit_seconds" not in info["models"][m] or "predict_seconds" not in info["models"][m])]
        if untimed:
            lines += ["n/a: model reused from an interrupted earlier run via --resume; "
                      "its timing was not recorded.", ""]
    else:
        lines += ["run_info.json not found; data summary omitted.", ""]

    lines += ["## Overall", ""]
    overall = mt[mt["scope"] == "overall"].set_index("model").reindex(models)
    header = ["Model", "n", "Recall", "Log loss", "Brier", "AUC", "ECE", "RMSE (bins)", "Mean bias"]
    rows = [[DISPLAY[m], int(overall.loc[m, "n"]), float(overall.loc[m, "recall"]),
             float(overall.loc[m, "log_loss"]), float(overall.loc[m, "brier"]),
             float(overall.loc[m, "auc"]), float(overall.loc[m, "ece"]),
             float(overall.loc[m, "rmse_bins"]), float(overall.loc[m, "mean_bias"])] for m in models]
    lines += [_table(header, rows, bold_cols={3: "min", 4: "min", 5: "max", 6: "min", 7: "min", 8: "closest0"}),
              "", "Best value per column in bold. Mean bias is closest to zero.", ""]

    lines += ["## Gap bins (days since the card's previous review)", ""]
    for label, _, _ in GAP_BINS:
        sub = mt[(mt["scope"] == "gap") & (mt["bin"] == label)].set_index("model").reindex(models)
        if sub["n"].isna().all() or int(sub["n"].max()) == 0:
            lines += [f"### {label}", "", "No scored events in this bin.", ""]
            continue
        lines += [f"### {label} (n = {int(sub['n'].iloc[0])})", ""]
        rows = [[DISPLAY[m], float(sub.loc[m, "recall"]), float(sub.loc[m, "log_loss"]),
                 float(sub.loc[m, "brier"]), float(sub.loc[m, "auc"]), float(sub.loc[m, "ece"]),
                 float(sub.loc[m, "mean_bias"])] for m in models]
        lines += [_table(["Model", "Recall", "Log loss", "Brier", "AUC", "ECE", "Mean bias"], rows,
                         bold_cols={2: "min", 3: "min", 4: "max", 5: "min", 6: "closest0"}), ""]
    lines += ["![Log loss by gap bin](gapbins_logloss.png)", ""]

    lines += ["## Paired differences (model minus reference), 95% user-cluster bootstrap", "",
              f"Bootstrap: {N_BOOT} user-cluster resamples with replacement, seed {BOOT_SEED}, "
              "95% percentile CI, diff = model minus reference.", ""]
    if pt.empty:
        lines += ["No reference model in this run; paired table omitted.", ""]
    else:
        rows = [[DISPLAY[r.model] + " - " + DISPLAY[r.reference], r.metric, float(r.diff),
                 f"[{_fmt(r.lo)}, {_fmt(r.hi)}]" + (" *" if r.ci_excludes_zero else "")]
                for r in pt.itertuples()]
        lines += [_table(["Comparison", "Metric", "Diff", "95% CI"], rows), "",
                  "* CI excludes 0.", ""]

    lines += ["## Per-user wins on log loss", ""]
    if uw.empty:
        lines += ["No reference model in this run.", ""]
    else:
        rows = [[DISPLAY[r.model] + " vs " + DISPLAY[r.reference], int(r.users),
                 int(r.users_lower_log_loss), float(r.share)] for r in uw.itertuples()]
        lines += [_table(["Comparison", "Users", "Users with lower log loss", "Share"], rows), "",
                  "Share = users with lower log loss for the model / users with a finite log loss for both.", ""]

    lines += mcmc_section(run_dir)
    lines += ["## Figures", "", "![Reliability](reliability.png)", ""]
    (out_dir / "report.md").write_text("\n".join(lines))


def build_report(run_dir, out_dir=None):
    run_dir = Path(run_dir)
    out_dir = Path(out_dir) if out_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    preds = pd.read_parquet(run_dir / "predictions.parquet")
    info_path = run_dir / "run_info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else None
    models = model_columns(preds)
    if not models:
        raise ValueError("predictions.parquet has no model columns")

    mt = metrics_table(preds, models)
    pt = paired_table(preds, models)
    users = per_user_table(preds, models)
    uw = user_wins(users, models)

    mt.to_csv(out_dir / "metrics.csv", index=False)
    pt.to_csv(out_dir / "paired.csv", index=False)
    users.to_csv(out_dir / "users.csv", index=False)
    uw.to_csv(out_dir / "user_wins.csv", index=False)
    plot_reliability(preds, models, out_dir / "reliability.png")
    plot_gap_logloss(mt, models, out_dir / "gapbins_logloss.png")
    write_report(run_dir, out_dir, preds, info, models, mt, pt, uw)
    return out_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description="Write metrics, figures and report.md for a run.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--out_dir", default=None, help="defaults to run_dir")
    args = parser.parse_args(argv)
    out = build_report(args.run_dir, args.out_dir)
    print(f"wrote report to {out / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
