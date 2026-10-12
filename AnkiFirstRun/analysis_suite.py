"""
analysis_suite.py

Tests the claim "the same time-aware model handles short- and long-gap forgetting".
Run it ONCE PER DATASET (same code, same settings), then compare the outputs.

    python analysis_suite.py --data duolingo --pickle processed_sequences.pkl --iters 2000 --burn 1000
    python analysis_suite.py --data anki --anki_root <path> --n_users 100 --iters 2000 --burn 1000
    python analysis_suite.py --data synthetic --iters 400 --burn 200          # quick check

What it does (numbers match your task list)
  1+2  GAP BINS: splits held-out events by the gap before them (days), reports per bin:
       n, recall rate, log loss, calibration bias (mean prediction - observed rate), ECE, AUC,
       for the baselines and every Dynamic variant; plus the PAIRED log-loss / AUC difference
       Dynamic - LagLogit+dt with a user-clustered bootstrap CI. If the latent state helps at
       short gaps and not at long ones, the difference is negative (good) at short gaps and
       ~0 at long gaps.
  3    LAMBDA ABLATION: lambda learned vs fixed ~0 (no decay) vs fixed large (no memory),
       with paired CIs.
  4    KAPPA: lets the decay RATE slow down with successful practice:
       rate_t = lambda * exp(-kappa * log(1 + successes so far)); kappa > 0 means "memory gets
       more durable with practice" (the property HLR and FSRS have and the basic model lacks).
  5    LAMBDA REPORT: lambda, kappa, 1/lambda time scale, ESS and R-hat, plus data summary.

Everything is fit on the SAME train/test split. Report ALL of it, wins and losses.
Bins with too few events or forgetting events are skipped and listed (small bins are noisy).
"""
import argparse
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import metrics as M
from baselines import LaggedLogistic
from hlr_baseline_v2 import HalfLifeRegressionV2
from dynamic_retention_dt import DynamicRetentionModelDT
from common import add_data_args, load

VARIANTS = {
    'learned': ('Dyn learned-lambda', dict()),
    'small': ('Dyn lambda~0 (no decay)', dict(fixed_lam=1e-3)),
    'large': ('Dyn lambda large (no memory)', dict(fixed_lam=5.0)),
    'kappa': ('Dyn +kappa (decay slows)', dict(use_kappa=True)),
}
LAG = 'LagLogit+dt'
HLR = 'HLR (paper loss)'


def fit_dyn(train, kw, seed, iters, burn):
    m = DynamicRetentionModelDT(train, rng=np.random.default_rng(seed), **kw)
    m.run(n_iter=iters, burn_in=burn, thin=4)
    return m


def pretty_edges(edges):
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        out.append(f"{a:g}-{b:g}d" if np.isfinite(b) else f"{a:g}d+")
    return out


def main():
    ap = argparse.ArgumentParser()
    add_data_args(ap)
    ap.add_argument('--variants', default='learned,small,large,kappa')
    ap.add_argument('--iters', type=int, default=2000)
    ap.add_argument('--burn', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--bins', default='0,1,3,14,60',
                    help='gap bin edges in DAYS (last bin is open-ended). On Duolingo you can add '
                         'short bins, e.g. 0,0.25,1,3,14,60')
    ap.add_argument('--min_n', type=int, default=300)
    ap.add_argument('--min_forgot', type=int, default=30)
    ap.add_argument('--n_boot', type=int, default=300)
    ap.add_argument('--out', default='suite')
    a = ap.parse_args()

    train, test = load(a, a.seed)
    print(f"train seqs {len(train)}, test seqs {len(test)}")

    # ------------------------------------------------------------ fit
    flats, dyn_models = {}, {}
    print("Fitting lagged logistic + dt, HLR ...")
    flats[LAG] = M.flatten(test, LaggedLogistic(use_dt=True).fit(train).predict(test))
    flats[HLR] = M.flatten(test, HalfLifeRegressionV2(alpha=0.01).fit(train).predict(test))
    chosen = [v.strip() for v in a.variants.split(',')]
    for v in chosen:
        name, kw = VARIANTS[v]
        print(f"Fitting {name} ({a.iters} iters) ...")
        m = fit_dyn(train, kw, a.seed, a.iters, a.burn)
        dyn_models[name] = m
        flats[name] = M.flatten(test, m.predict(test))
    ref = flats[LAG]
    for n, f in flats.items():
        assert np.array_equal(f['y'], ref['y']), "models disagree on test events"
    y, users = ref['y'], ref['user']

    # Flatten dt across all test sequences directly from test_seqs
    dt = np.concatenate([np.asarray(s['dt'], float) for s in test])

    # ------------------------------------------------------------ data summary
    print("\n=== DATA SUMMARY ===")
    print(f"test events {len(y)}, overall recall {y.mean():.3f}, "
          f"median gap {np.median(dt):.2f} d, 90th pct gap {np.percentile(dt, 90):.2f} d, "
          f"max gap {dt.max():.1f} d")

    # ------------------------------------------------------------ 1+2: gap bins
    edges = [float(x) for x in a.bins.split(',')] + [np.inf]
    labels = pretty_edges(edges)
    dyn_names = [VARIANTS[v][0] for v in chosen]
    rows, kept = [], []
    print("\n=== GAP BINS (log loss / calibration bias = mean p - recall rate / AUC) ===")
    hdr = f"{'bin':<10}{'n':>7}{'recall':>8}  " + "".join(f"{n[:22]:>24}" for n in flats)
    print(hdr)
    for b, lab in enumerate(labels):
        m = (dt >= edges[b]) & (dt < edges[b + 1])
        n = int(m.sum()); nf = int((y[m] == 0).sum())
        if n < a.min_n or nf < a.min_forgot:
            print(f"{lab:<10}{n:>7}   (skipped: needs >= {a.min_n} events and >= {a.min_forgot} forgets, has {nf})")
            continue
        kept.append((b, lab, m))
        cells = []
        for name, f in flats.items():
            p = f['p'][m]
            ll, bias, au, ec = M.logloss(y[m], p), p.mean() - y[m].mean(), M.auc(y[m], p), M.ece(y[m], p)
            cells.append(f"LL {ll:.3f} b{bias:+.3f} A{au:.3f}")
            rows.append(dict(bin=lab, n=n, recall=y[m].mean(), model=name, logloss=ll,
                             bias=bias, auc=au, ece=ec))
        print(f"{lab:<10}{n:>7}{y[m].mean():>8.3f}  " + "".join(f"{c:>24}" for c in cells))

    print(f"\n=== PAIRED DIFFERENCE vs {LAG} per bin (negative LogLoss diff / positive AUC diff = Dynamic better) ===")
    gain = {n: [] for n in dyn_names}
    for name in dyn_names:
        print(f"-- {name}")
        for b, lab, m in kept:
            dl, ll_lo, ll_hi = M.paired_bootstrap_diff(M.logloss, y[m], flats[name]['p'][m], flats[LAG]['p'][m],
                                                       users[m], a.n_boot)
            da, a_lo, a_hi = M.paired_bootstrap_diff(M.auc, y[m], flats[name]['p'][m], flats[LAG]['p'][m],
                                                     users[m], a.n_boot)
            gain[name].append((lab, dl, ll_lo, ll_hi))
            sig = '*' if (ll_hi < 0 or ll_lo > 0) else ' '
            print(f"   {lab:<10} dLogLoss {dl:+.4f} [{ll_lo:+.4f}, {ll_hi:+.4f}]{sig}   dAUC {da:+.4f} [{a_lo:+.4f}, {a_hi:+.4f}]")
    print("   (* = CI excludes 0. Small bins have wide CIs; do not over-read a single bin.)")

    # ------------------------------------------------------------ 3: lambda ablation (overall)
    L = VARIANTS['learned'][0]
    if L in flats:
        print("\n=== LAMBDA ABLATION (overall, paired vs learned lambda; negative dLogLoss = fixed is BETTER) ===")
        for v in ('small', 'large'):
            nm = VARIANTS[v][0]
            if nm in flats:
                dl, lo, hi = M.paired_bootstrap_diff(M.logloss, y, flats[nm]['p'], flats[L]['p'], users, a.n_boot)
                da, alo, ahi = M.paired_bootstrap_diff(M.auc, y, flats[nm]['p'], flats[L]['p'], users, a.n_boot)
                print(f"  {nm:<30} dLogLoss {dl:+.4f} [{lo:+.4f}, {hi:+.4f}]   dAUC {da:+.4f} [{alo:+.4f}, {ahi:+.4f}]")
        print("  Learned lambda 'beats both' only if both rows have POSITIVE dLogLoss with CIs above 0.")

    # ------------------------------------------------------------ 4: kappa (overall)
    K = VARIANTS['kappa'][0]
    if K in flats and L in flats:
        dl, lo, hi = M.paired_bootstrap_diff(M.logloss, y, flats[K]['p'], flats[L]['p'], users, a.n_boot)
        da, alo, ahi = M.paired_bootstrap_diff(M.auc, y, flats[K]['p'], flats[L]['p'], users, a.n_boot)
        print(f"\n=== KAPPA vs learned-lambda (overall): dLogLoss {dl:+.4f} [{lo:+.4f}, {hi:+.4f}]   "
              f"dAUC {da:+.4f} [{alo:+.4f}, {ahi:+.4f}]")

    # ------------------------------------------------------------ 5: lambda report
    print("\n=== LAMBDA REPORT (same code, this dataset) ===")
    for name, m in dyn_models.items():
        print(f"-- {name}")
        m.print_summary()
        lam = m.samples['lam']
        print(f"   time scale 1/lambda = {1 / lam.mean():.1f} days (lambda 95% CI "
              f"{np.percentile(lam, 2.5):.3f}-{np.percentile(lam, 97.5):.3f}); "
              f"ESS<100 or R-hat>1.1 means run longer before trusting the interval")

    # ------------------------------------------------------------ outputs
    with open(f"{a.out}_gapbins.csv", 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['bin', 'n', 'recall', 'model', 'logloss', 'bias', 'auc', 'ece'])
        w.writeheader(); w.writerows(rows)
    if kept:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
        xs = np.arange(len(kept)); labs = [k[1] for k in kept]
        for name in flats:
            ys_ = [r['logloss'] for r in rows if r['model'] == name]
            ax[0].plot(xs, ys_, 'o-', label=name)
        ax[0].set_xticks(xs); ax[0].set_xticklabels(labs); ax[0].set_xlabel("gap before the reviewed event")
        ax[0].set_ylabel("log loss (lower is better)"); ax[0].legend(fontsize=7); ax[0].set_title("Log loss by gap")
        for k, name in enumerate(dyn_names):
            d = np.array([g[1] for g in gain[name]]); lo = np.array([g[2] for g in gain[name]])
            hi = np.array([g[3] for g in gain[name]])
            ax[1].errorbar(xs + 0.07 * k, d, yerr=[d - lo, hi - d], fmt='o-', capsize=3, label=name)
        ax[1].axhline(0, color='k', lw=1)
        ax[1].set_xticks(xs); ax[1].set_xticklabels(labs); ax[1].set_xlabel("gap before the reviewed event")
        ax[1].set_ylabel(f"Delta log loss vs {LAG}  (below 0 = Dynamic better)")
        ax[1].legend(fontsize=7); ax[1].set_title("Where does the latent state help?")
        fig.tight_layout(); fig.savefig(f"{a.out}_gapbins.png", dpi=150)
        print(f"\nSaved {a.out}_gapbins.csv and {a.out}_gapbins.png")
    M.reliability_figure({n: flats[n] for n in [LAG, HLR] + dyn_names[:2]}, f"{a.out}_reliability.png")


if __name__ == "__main__":
    main()