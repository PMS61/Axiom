"""Two-panel log-loss bar chart for Summary_Report.tex (HLR, FSRS-6 default/optimised, v1 MCMC, GBDT).

Reads the overall rows of the two final report metrics files:
  results/full_u100_c1000_final/metrics.csv   (Anki, model 'ours' = v1)
  results/duolingo_final/metrics.csv          (Duolingo, model 'ours_gpu' = v1)
Writes summary_log_loss.pdf next to this script.
Run: /home/amal/fyp/Axiom/.venv/bin/python docs/report/make_summary_figure.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE.parent.parent / "results"
SETS = [("Anki (100 users, 95,400 scored reviews)", RES / "full_u100_c1000_final" / "metrics.csv", "ours"),
        ("Duolingo (all users, 1,189,355 scored reviews)", RES / "duolingo_final" / "metrics.csv", "ours_gpu")]
LABELS = ["HLR", "FSRS-6 (default)", "FSRS-6 (optimised)", "v1 (Bayesian MCMC)", "GBDT"]
COLORS = ["#c00000", "#5b9bd5", "#1f4e79", "#2e7d32", "#e65100"]

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e5e5e5", "axes.axisbelow": True, "pdf.fonttype": 42})
fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7))
for ax, (title, path, v1) in zip(axes, SETS):
    m = pd.read_csv(path)
    m = m[m.scope == "overall"].set_index("model")
    keys = ["hlr", "fsrs_default", "fsrs_opt", v1, "gbdt"]
    vals = [float(m.loc[k, "log_loss"]) for k in keys]
    y = np.arange(len(keys))[::-1]
    ax.barh(y, vals, color=COLORS, edgecolor="#333333", linewidth=0.4, height=0.65)
    for yi, v in zip(y, vals):
        ax.text(v, yi, f" {v:.3f}", va="center", fontsize=7.5)
    ax.set_yticks(y)
    ax.set_yticklabels(LABELS if ax is axes[0] else [""] * 5, fontsize=8)
    ax.set_xlim(0, max(vals) * 1.2)
    ax.set_xlabel("Log loss (lower is better)")
    ax.set_title(title, fontsize=8)
    ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(HERE / "summary_log_loss.pdf")
print("wrote", HERE / "summary_log_loss.pdf")
