"""
regenerate_reliability.py  –  Step 6: Regenerate reliability diagrams from saved CSVs.

Since predictions.parquet is git-ignored, we reconstruct the reliability curves
from the committed metrics.csv (which contains mean_p and mean_y per bin from the
original predictions.parquet) and cross-check the stored reliability.png.

We also generate a fresh standalone reliability diagram from the stored
reliability_curve data embedded in supplementary.json.
"""
import sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/vaishvi/Desktop/FYP/Axiom")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN_DIR = "/Users/vaishvi/Desktop/FYP/Axiom/results/full_u100_c1000"
OUT_DIR = "/Users/vaishvi/Desktop/FYP/Axiom/scratch"

# ── Load supplementary.json for the 'ours' reliability bins ──────────────────
with open(f"{RUN_DIR}/supplementary.json") as f:
    supp = json.load(f)

print("=== Contents of supplementary.json ===")
for k, v in supp.items():
    if isinstance(v, dict) or isinstance(v, list):
        print(f"  {k}: (complex)")
    else:
        print(f"  {k}: {v}")

# ── Reconstruct reliability diagram from metrics.csv + known ECE ─────────────
# The reliability curve per-bin data is in supplementary.json under 'ours_reliability'
ours_rc = supp.get("ours_reliability", None)
print(f"\n  ours_reliability key: {'present' if ours_rc else 'MISSING'}")

# ── Generate a gap-bin log-loss comparison figure from metrics.csv ─────────────
mt = pd.read_csv(f"{RUN_DIR}/metrics.csv")

MODELS = ["fsrs_default", "fsrs_opt", "hlr", "ours", "ours_kappa"]
DISPLAY = {
    "fsrs_default": "FSRS-6 (default)",
    "fsrs_opt":     "FSRS-6 (opt.)",
    "hlr":          "HLR",
    "ours":         "Ours",
    "ours_kappa":   "Ours (+κ)",
}
COLOURS = {
    "fsrs_default": "#E69F00",
    "fsrs_opt":     "#0072B2",
    "hlr":          "#009E73",
    "ours":         "#D55E00",
    "ours_kappa":   "#CC79A7",
}
GAP_BINS = ["1d", "2-3d", "4-14d", "15-60d", "61d+"]

gap_data = mt[mt["scope"] == "gap"].copy()
n_per_bin = gap_data[gap_data["model"] == "fsrs_default"].set_index("bin")["n"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: log loss by gap bin
ax = axes[0]
x = np.arange(len(GAP_BINS))
w = 0.8 / len(MODELS)
for i, m in enumerate(MODELS):
    vals = gap_data[gap_data["model"] == m].set_index("bin").reindex(GAP_BINS)["log_loss"].to_numpy(float)
    bars = ax.bar(x + (i - (len(MODELS) - 1)/2)*w, vals, width=w, color=COLOURS[m], label=DISPLAY[m])
ax.set_xticks(x)
ax.set_xticklabels([f"{b}\n(n={int(n_per_bin.get(b,0))})" for b in GAP_BINS], fontsize=9)
ax.set_ylabel("Log loss (lower is better)", fontsize=10)
ax.set_title("Log loss by gap bin (recomputed from metrics.csv)", fontsize=11)
ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
ax.grid(alpha=0.3, axis="y")

# Right: AUC by gap bin
ax = axes[1]
for i, m in enumerate(MODELS):
    vals = gap_data[gap_data["model"] == m].set_index("bin").reindex(GAP_BINS)["auc"].to_numpy(float)
    ax.bar(x + (i - (len(MODELS) - 1)/2)*w, vals, width=w, color=COLOURS[m], label=DISPLAY[m])
ax.set_xticks(x)
ax.set_xticklabels([f"{b}" for b in GAP_BINS], fontsize=9)
ax.set_ylabel("AUC (higher is better)", fontsize=10)
ax.set_title("AUC by gap bin", fontsize=11)
ax.set_ylim(0.4, 0.9)
ax.axhline(0.5, color="#666", linestyle="--", linewidth=0.8, alpha=0.6, label="random (AUC=0.5)")
ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
ax.grid(alpha=0.3, axis="y")

fig.tight_layout()
out_path = f"{OUT_DIR}/gap_metrics_recomputed.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved gap metrics figure: {out_path}")

# ── Reliability curve from supplementary.json ─────────────────────────────────
if ours_rc:
    fig2, (ax_rc, ax_hist) = plt.subplots(2, 1, figsize=(7, 8), sharex=True,
                                           gridspec_kw={"height_ratios": [3, 1.3]})
    ax_rc.plot([0, 1], [0, 1], "--", color="#666", linewidth=1, label="perfect calibration")
    
    edges = np.linspace(0, 1, 11)
    mid = (edges[:-1] + edges[1:]) / 2
    
    if isinstance(ours_rc, list):
        rc_df = pd.DataFrame(ours_rc)
        print(f"\n  ours_reliability columns: {list(rc_df.columns)}")
        if "mean_p" in rc_df.columns and "mean_y" in rc_df.columns:
            shown = rc_df[rc_df["n"] > 0]
            ax_rc.plot(shown["mean_p"], shown["mean_y"], "D-", color=COLOURS["ours"],
                       markersize=6, linewidth=1.6, label="Ours (from supplementary.json)", alpha=0.9)
            ax_hist.bar(mid, rc_df["n"], width=0.1*0.8, color=COLOURS["ours"])
    
    ax_rc.set_xlim(0, 1); ax_rc.set_ylim(0, 1)
    ax_rc.set_ylabel("Observed recall rate (mean y in bin)", fontsize=10)
    ax_rc.set_title("Reliability diagram – Ours model (from supplementary.json)", fontsize=11)
    ax_rc.legend(fontsize=9); ax_rc.grid(alpha=0.3)
    ax_hist.set_xlabel("Predicted recall probability p", fontsize=10)
    ax_hist.set_ylabel("Events per bin", fontsize=10); ax_hist.grid(alpha=0.3, axis="y")
    fig2.tight_layout()
    out2 = f"{OUT_DIR}/reliability_ours_recomputed.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved reliability diagram: {out2}")

print("\nDone.")

