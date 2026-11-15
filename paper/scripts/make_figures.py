"""Write the paper's figures (PDF) and the reliability data (CSV). Nothing is typed by hand.

Outputs:
  paper/figures/gaps_{anki,duolingo}.pdf        log loss by gap bin (from paper/data/*_gaps.csv)
  paper/figures/reliability_{anki,duolingo}.pdf reliability diagrams (from the predictions parquet)
  paper/data/{anki,duolingo}_reliability.csv    the binned reliability data behind those diagrams

Usage (from the repo root):
  .venv/bin/python paper/scripts/make_figures.py
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
REPO = PAPER.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

from fsrs_benchmark import metrics, report  # noqa: E402
from make_tables import SHORT, V1_KEY  # noqa: E402

DATASETS = ("anki", "duolingo")
RUN_DIR = {"anki": REPO / "results" / "paper_anki", "duolingo": REPO / "results" / "paper_duolingo"}
GAP_BINS = ["1d", "2-3d", "4-14d", "15-60d", "61d+"]
N_BINS = 10
# Reliability bins with fewer reviews than this are kept in the CSV but not drawn.
MIN_BIN_N = 30
# Okabe-Ito colour-blind-safe palette, one colour and marker per series.
STYLE = {
    "fsrs_opt": {"color": "#0072B2", "marker": "o", "ls": "-"},
    "v1": {"color": "#E69F00", "marker": "s", "ls": "-"},
    "ours_v3": {"color": "#009E73", "marker": "^", "ls": "-"},
    "gbdt": {"color": "#CC79A7", "marker": "D", "ls": "-"},
    "const_user_smooth": {"color": "#555555", "marker": "x", "ls": "--"},
}
# Single-column ACM/EDM width is 3.3 in. Keep fonts readable at that size.
WIDTH = 3.3
HEIGHT_GAPS = 2.5
HEIGHT_REL = 3.0
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 7.5, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})


def series_key(style_key, dataset):
    """Map a style key ("v1" is per dataset) to the model key used in the data files."""
    return V1_KEY[dataset] if style_key == "v1" else style_key


def gaps_data(data_dir, dataset):
    return pd.read_csv(Path(data_dir) / f"{dataset}_gaps.csv")


def plot_gaps(data_dir, dataset, out_path):
    df = gaps_data(data_dir, dataset)
    # Only bins with scored events. Duolingo has none beyond 11 days, so its 15-60d and 61d+ bins are empty.
    bins = [b for b in GAP_BINS if (df[df["bin"] == b]["n"] > 0).any()]
    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT_GAPS))
    x = np.arange(len(bins))
    for style_key in STYLE:
        key = series_key(style_key, dataset)
        sub = df[df["model"] == key].set_index("bin").reindex(bins)
        st = STYLE[style_key]
        ax.plot(x, sub["log_loss"].to_numpy(float), color=st["color"], marker=st["marker"],
                linestyle=st["ls"], markersize=3.5, linewidth=1.1, label=SHORT[key])
    ax.set_xticks(x)
    ax.set_xticklabels(bins)
    ax.set_xlabel("Gap since previous review")
    ax.set_ylabel("Log loss")
    ax.grid(axis="y", color="#dddddd", linewidth=0.5)
    # Headroom above the data for a two-column legend that does not cover any line.
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.45 * (hi - lo))
    ax.legend(frameon=False, ncol=2, loc="upper center", columnspacing=0.8, handlelength=1.8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def reliability_frame(run_dir):
    """Binned reliability data for the four series, from predictions.parquet and events.parquet."""
    preds = pd.read_parquet(Path(run_dir) / "predictions.parquet")
    events = pd.read_parquet(Path(run_dir) / "events.parquet")
    preds["const_user_smooth"] = report.constant_predictions(preds, events)["const_user_smooth"]
    return preds


def reliability_table(run_dir, dataset):
    preds = reliability_frame(run_dir)
    rows = []
    for style_key in STYLE:
        key = series_key(style_key, dataset)
        rc = metrics.reliability_curve(preds["y"].to_numpy(), preds[key].to_numpy(float), n_bins=N_BINS)
        rc = pd.DataFrame(rc)
        rc.insert(0, "model", key)
        rc.insert(1, "display", SHORT[key])
        rows.append(rc)
    return pd.concat(rows, ignore_index=True)


def plot_reliability(table, dataset, out_path):
    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT_REL))
    ax.plot([0, 1], [0, 1], color="#999999", linewidth=0.8, linestyle=":", label="Perfect")
    for style_key in STYLE:
        key = series_key(style_key, dataset)
        sub = table[(table["model"] == key) & (table["n"] >= MIN_BIN_N)]
        st = STYLE[style_key]
        ax.plot(sub["mean_p"], sub["mean_y"], color=st["color"], marker=st["marker"],
                linestyle=st["ls"], markersize=3.5, linewidth=1.0, label=SHORT[key])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted recall (bin mean)")
    ax.set_ylabel("Observed recall")
    ax.set_aspect("equal")
    ax.grid(color="#eeeeee", linewidth=0.5)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_all(data_dir, fig_dir, run_dirs=RUN_DIR):
    fig_dir, data_dir = Path(fig_dir), Path(data_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for ds in DATASETS:
        plot_gaps(data_dir, ds, fig_dir / f"gaps_{ds}.pdf")
        table = reliability_table(run_dirs[ds], ds)
        table.to_csv(data_dir / f"{ds}_reliability.csv", index=False, float_format="%.6f")
        plot_reliability(table, ds, fig_dir / f"reliability_{ds}.pdf")
        written += [fig_dir / f"gaps_{ds}.pdf", fig_dir / f"reliability_{ds}.pdf",
                    data_dir / f"{ds}_reliability.csv"]
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(PAPER / "data"))
    ap.add_argument("--figures", default=str(PAPER / "figures"))
    args = ap.parse_args(argv)
    for p in write_all(args.data, args.figures):
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
