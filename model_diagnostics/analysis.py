"""Rank-based multi-chain diagnostics, proper scoring and artifact generation."""
import csv
import json
from pathlib import Path
import arviz as az
import numpy as np
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def scalar_draws(samples, variant):
    out = {}
    for key, values in samples.items():
        if key == 'kappa' and variant in ('no_practice', 'no_dynamics'):
            continue
        if variant == 'no_hierarchy' and key in ('alpha', 'beta', 'gamma'):
            continue
        if values.ndim == 2:
            out[key] = values
        else:
            for index in range(values.shape[2]):
                out[f'{key}[{index}]'] = values[:, :, index]
    if variant != 'no_hierarchy':
        out['population_level'] = samples['b0'] + samples['alpha'].mean(axis=2) + samples['beta'].mean(axis=2)
        for key in ('alpha', 'beta'):
            centered = samples[key] - samples[key].mean(axis=2, keepdims=True)
            for i in range(centered.shape[2]):
                out[f'{key}_centered[{i}]'] = centered[:, :, i]
        out['gamma_mean'] = samples['gamma'].mean(axis=2)
    return out


def diagnostics(draws, seconds):
    rows = []
    for name, values in draws.items():
        rhat = float(az.rhat(values, method='rank'))
        bulk = float(az.ess(values, method='bulk'))
        tail = float(az.ess(values, method='tail'))
        mcse = float(az.mcse(values, method='mean'))
        lo, hi = np.quantile(values, [.025, .975])
        passed = bool(np.isfinite([rhat, bulk, tail, mcse]).all()
                      and rhat < 1.01 and min(bulk, tail) >= 100 * len(values))
        rows.append(dict(parameter=name, mean=float(values.mean()), sd=float(values.std(ddof=1)),
                         low=float(lo), high=float(hi), rhat=rhat, ess_bulk=bulk,
                         ess_tail=tail, mcse_mean=mcse, ess_per_cpu_second=bulk / max(seconds, 1e-9),
                         passed=passed))
    return rows


def metrics(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    positives, negatives = int(y.sum()), int((1 - y).sum())
    auc = ((rankdata(p)[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives)
           if positives and negatives else None)
    return dict(log_loss=float(np.mean(-y * np.log(p) - (1 - y) * np.log(1 - p))),
                brier=float(np.mean((y - p) ** 2)), mae=float(np.mean(np.abs(y - p))),
                auc=float(auc) if auc is not None else None,
                calibration_bias=float(p.mean() - y.mean()), test_events=len(y))


def paired_bootstrap(y, full, ablated, users, seed):
    def loss(p):
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -y * np.log(p) - (1 - y) * np.log1p(-p)
    delta = loss(ablated) - loss(full)
    unique = np.unique(users)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(1000):
        selected = rng.choice(unique, len(unique), replace=True)
        sampled = np.concatenate([delta[users == u] for u in selected])
        means.append(sampled.mean())
    lo, hi = np.quantile(means, [.025, .975])
    return dict(delta_log_loss=float(delta.mean()), low=float(lo), high=float(hi), clusters=len(unique))


def truth_values(truth):
    out = {k: float(v) for k, v in truth.items() if np.ndim(v) == 0}
    out['population_level'] = truth['b0'] + float(np.mean(truth['alpha'])) + float(np.mean(truth['beta']))
    out['gamma_mean'] = float(np.mean(truth['gamma']))
    for k in ('alpha', 'beta', 'gamma'):
        for i, value in enumerate(truth[k]):
            out[f'{k}[{i}]'] = float(value)
            if k != 'gamma':
                out[f'{k}_centered[{i}]'] = float(value - np.mean(truth[k]))
    return out


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def atomic_json(path, data):
    def clean(x):
        if isinstance(x, dict): return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)): return [clean(v) for v in x]
        if isinstance(x, np.ndarray): return clean(x.tolist())
        if isinstance(x, np.generic): return clean(x.item())
        if isinstance(x, float) and not np.isfinite(x): return None
        return x
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(clean(data), indent=2, allow_nan=False))
    temp.replace(path)


def plots(out, prefix, draws, warmups):
    keys = [k for k in ('lam', 'kappa', 'sigma2', 'population_level', 'g0') if k in draws]
    fig, axes = plt.subplots(len(keys), 2, figsize=(11, 2.3 * len(keys)), squeeze=False)
    for row, key in enumerate(keys):
        warm = warmups.get(key)
        for c, values in enumerate(draws[key]):
            color = f'C{c}'
            if warm is not None:
                trace = np.concatenate([warm[c], values]); offset = len(warm[c])
            else:
                trace = values; offset = 0
            axes[row, 0].plot(trace, color=color, alpha=.7, lw=.55, label=f'chain {c+1}')
            centered = values - values.mean()
            acf = np.correlate(centered, centered, mode='full')[len(values)-1:]
            if acf[0] > 0: acf /= acf[0]
            axes[row, 1].plot(acf[:101], color=color, lw=.9)
        if offset: axes[row, 0].axvline(offset, color='black', ls='--', lw=.7)
        axes[row, 0].set_ylabel(key); axes[row, 0].set_xlabel('Iteration (warmup before dashed line)')
        axes[row, 1].set_xlabel('Post-warmup lag'); axes[row, 1].set_ylim(-.2, 1)
    axes[0, 0].legend(fontsize=8, ncol=4)
    fig.suptitle(f'{prefix}: chain traces and autocorrelation', fontsize=12)
    fig.tight_layout(); fig.savefig(out / f'{prefix}-trace.png', dpi=140); plt.close(fig)
    rank_axes = az.plot_rank(az.from_dict(posterior={k: draws[k] for k in keys}), kind='bars', show=False)
    plt.gcf().suptitle(f'{prefix}: pooled rank histograms', y=1.02)
    plt.gcf().savefig(out / f'{prefix}-rank.png', dpi=140, bbox_inches='tight'); plt.close('all')
    if 'lam' in draws and 'kappa' in draws:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        for c in range(len(draws['lam'])):
            axes[0].scatter(np.log(draws['lam'][c, ::5]), draws['kappa'][c, ::5], s=3, alpha=.3)
            axes[1].scatter(np.log(draws['lam'][c, ::5]), np.log(draws['sigma2'][c, ::5]), s=3, alpha=.3)
        axes[0].set(xlabel='log lambda', ylabel='kappa')
        axes[1].set(xlabel='log lambda', ylabel='log diffusion variance')
        fig.suptitle(f'{prefix}: posterior dependence'); fig.tight_layout()
        fig.savefig(out / f'{prefix}-pairs.png', dpi=140); plt.close(fig)


def overview_plots(out, results, recovery):
    names = [f"{r['scenario']} / {r['replicate']} / {r['variant']}" for r in results]
    fig, axes = plt.subplots(1, 3, figsize=(13, max(4, len(names)*.27)))
    for ax, key, label in zip(axes, ['log_loss', 'max_rhat', 'fit_cpu_seconds'],
                              ['Held-out log loss ↓', 'Maximum rank R̂ ↓', 'Total chain CPU seconds ↓']):
        ax.barh(names, [r[key] for r in results]); ax.set_title(label, fontsize=10); ax.invert_yaxis()
    axes[1].axvline(1.01, color='red', ls='--', lw=1)
    fig.tight_layout(); fig.savefig(out / 'comparison.png', dpi=150); plt.close(fig)
    rows = [r for r in recovery if r['parameter'] in ('lam', 'kappa', 'sigma2', 'population_level') and r['variant'] == 'full']
    if rows:
        fig, axes = plt.subplots(1, 4, figsize=(13, max(3, len(rows) / 4 * .4)), squeeze=False)
        for ax, parameter in zip(axes[0], ('lam', 'kappa', 'sigma2', 'population_level')):
            selected = [r for r in rows if r['parameter'] == parameter]
            for j, r in enumerate(selected):
                color = 'C0' if r['passed'] else 'C3'
                ax.plot([r['low'], r['high']], [j, j], color=color)
                ax.plot(r['mean'], j, 'o', color=color, ms=4)
                ax.plot(r['truth'], j, '|', color='black', ms=12)
            ax.set_yticks(range(len(selected)), [f"{r['scenario']} / {r['replicate']}" for r in selected], fontsize=8)
            ax.set_title(parameter); ax.grid(axis='x', alpha=.2)
        fig.suptitle('95% recovery intervals; black = truth, red = failed diagnostic gate')
        fig.tight_layout(); fig.savefig(out / 'recovery.png', dpi=150); plt.close(fig)
