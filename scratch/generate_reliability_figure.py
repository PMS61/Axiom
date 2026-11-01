"""
generate_reliability_figure.py
Plot the exact reliability curve for 'Ours' using the recorded bins in supplementary.json
and save to scratch/reliability_ours_validated.png.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

with open("results/full_u100_c1000/supplementary.json") as f:
    supp = json.load(f)

bins = supp["ours_reliability_bins"]
mean_p = [b["mean_p"] for b in bins]
mean_y = [b["mean_y"] for b in bins]
n = [b["n"] for b in bins]

fig, (ax_rc, ax_hist) = plt.subplots(2, 1, figsize=(6.5, 7.5), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 1.2]})

ax_rc.plot([0, 1], [0, 1], "--", color="#666666", linewidth=1.2, label="Perfect calibration")
ax_rc.plot(mean_p, mean_y, "D-", color="#D55E00", linewidth=1.8, markersize=6, label="Ours (Dynamic, learned λ)")

# Scaling marker sizes proportional to sqrt(n)
max_n = max(n)
for mp, my, cnt in zip(mean_p, mean_y, n):
    ax_rc.scatter([mp], [my], s=25 + 160 * np.sqrt(cnt / max_n), color="#D55E00", alpha=0.8, zorder=3)

ax_rc.set_xlim(0, 1)
ax_rc.set_ylim(0, 1)
ax_rc.set_ylabel("Observed recall rate (mean y in bin)", fontsize=11)
ax_rc.set_title("Reliability Diagram: Ours (Full 100-User Benchmark)", fontsize=12)
ax_rc.grid(alpha=0.3)
ax_rc.legend(loc="lower right", fontsize=9, framealpha=0.9)

# Bar chart of event counts
ax_hist.bar(mean_p, n, width=0.06, color="#D55E00", alpha=0.7)
ax_hist.set_xlabel("Predicted recall probability p", fontsize=11)
ax_hist.set_ylabel("Events per bin", fontsize=11)
ax_hist.set_yscale("log")
ax_hist.set_ylabel("Events (log scale)", fontsize=10)
ax_hist.grid(alpha=0.3, axis="y")

fig.tight_layout()
fig.savefig("scratch/reliability_ours_validated.png", dpi=150)
plt.close(fig)
print("Successfully generated scratch/reliability_ours_validated.png")

