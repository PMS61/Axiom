"""Compare matched sampler studies without overwriting any source artifacts.

python -m model_diagnostics.compare --runs .task3-runs/task3-remedy-* --out DIR
Pass explicit run directories (not files), including ordinary and stress studies.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .analysis import atomic_json, write_csv


def compare(runs, out, saved_reference=None):
    if out.exists():
        raise ValueError('Comparison output already exists; use a new directory.')
    studies = [(p, json.loads((p / 'summary.json').read_text())) for p in runs]
    baselines = {}
    datasets = {}
    settings = {}
    match_fields = ('iterations', 'warmup', 'chains', 'seed', 'users', 'items', 'events', 'holdout', 'mh_steps')
    for path, study in studies:
        config = study['config']
        for row in study['results']:
            key = row['scenario'], row['replicate'], row['variant']
            dataset = path / f'{key[0]}-{key[1]}-dataset.json'
            digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
            signature = tuple(config.get(k, 5 if k == 'mh_steps' else None) for k in match_fields)
            if key in datasets and (digest != datasets[key] or signature != settings[key]):
                raise ValueError(f'Unmatched data/settings for {key}')
            datasets[key], settings[key] = digest, signature
            if config['sampler'] == 'reference':
                if key in baselines: raise ValueError(f'Duplicate reference {key}')
                baselines[key] = (path, study, row)
    results, parameters = [], []
    for path, study in studies:
        sampler = study['config']['sampler']
        for row in study['results']:
            key = row['scenario'], row['replicate'], row['variant']
            _, refstudy, reference = baselines[key]
            diag = [r for r in study['diagnostics'] if (r['scenario'], r['replicate'], r['variant']) == key]
            refdiag = {r['parameter']: r for r in refstudy['diagnostics'] if (r['scenario'], r['replicate'], r['variant']) == key}
            recovered = [r for r in study['recovery'] if (r['scenario'], r['replicate'], r['variant']) == key and r['matched']]
            results.append(dict(run=path.name, sampler=sampler, **row,
                                min_bulk_ess_per_cpu_second=row['min_ess_bulk']/row['fit_cpu_seconds'],
                                cpu_ratio_to_reference=row['fit_cpu_seconds']/reference['fit_cpu_seconds'],
                                delta_log_loss=row['log_loss']-reference['log_loss'],
                                delta_brier=row['brier']-reference['brier'],
                                matched_recovery_parameters=len(recovered),
                                recovery_rmse=float(np.sqrt(np.mean([r['squared_error'] for r in recovered]))) if recovered else None,
                                recovery_covered=sum(r['covered'] for r in recovered),
                                recovery_passing=sum(r['passed'] for r in recovered)))
            for r in diag:
                ref = refdiag[r['parameter']]
                denom = np.hypot(r['mcse_mean'], ref['mcse_mean'])
                parameters.append(dict(sampler=sampler, **r,
                                       reference_mean=ref['mean'], mean_difference=r['mean']-ref['mean'],
                                       difference_over_combined_mcse=(r['mean']-ref['mean'])/denom if denom else None,
                                       ess_cpu_ratio=r['ess_per_cpu_second']/ref['ess_per_cpu_second']))
    preservation = []
    if saved_reference:
        path, _, _ = baselines[('full', 0, 'full')]
        for variant in ('full', 'no_dynamics'):
            filename = f'full-0-{variant}-draws.npz'
            old, new = np.load(saved_reference/filename), np.load(path/filename)
            preservation.append(dict(variant=variant, saved_reference=str(saved_reference),
                                     exact_draw_equality=set(old.files) == set(new.files) and all(np.array_equal(old[k], new[k]) for k in old.files)))
    out.mkdir(parents=True)
    write_csv(out/'sampler-comparison.csv', results)
    write_csv(out/'parameter-comparison.csv', parameters)
    atomic_json(out/'comparison.json', dict(results=results, parameters=parameters, saved_reference_checks=preservation,
                                           inputs=[str(p) for p in runs], dataset_sha256={str(k): v for k, v in datasets.items()}))
    import matplotlib.pyplot as plt
    keys = list(dict.fromkeys((r['scenario'], r['replicate'], r['variant']) for r in results))
    samplers = list(dict.fromkeys(r['sampler'] for r in results))
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), layout='constrained')
    fields = [('min_ess_bulk', 'Minimum bulk ESS', 400),
              ('min_bulk_ess_per_cpu_second', 'Minimum bulk ESS / CPU second', None),
              ('delta_log_loss', 'Held-out log loss minus reference', 0)]
    for ax, (field, title, threshold) in zip(axes, fields):
        for j, sampler in enumerate(samplers):
            selected = { (r['scenario'], r['replicate'], r['variant']): r for r in results if r['sampler'] == sampler }
            ax.bar(np.arange(len(keys)) + (j-(len(samplers)-1)/2)*.19,
                   [selected[k][field] for k in keys], width=.18, label=sampler)
        if threshold is not None: ax.axhline(threshold, color='black', linestyle='--', linewidth=1)
        if field != 'delta_log_loss': ax.set_yscale('log')
        ax.set_title(title)
        ax.set_xticks(np.arange(len(keys)), [f'{k[0]} {k[1]+1}\n{k[2]}' for k in keys], rotation=30, ha='right')
        ax.grid(axis='y', alpha=.2)
    axes[0].legend(fontsize=8)
    figure.suptitle('Matched sampler pilot: failed fits remain exploratory')
    figure.savefig(out/'sampler-comparison.png', dpi=160)
    plt.close(figure)
    lines = ['# Matched sampler comparison' , '',
             'Same datasets, priors, four chains, seeds, iteration/warmup and proposal budgets. '
             'Every fit uses the unchanged R-hat <1.01 and bulk/tail ESS >=400 gate. '
             'CPU is summed process time; predictive filtering remains approximate.', '',
             '| Scenario / replicate / model | Sampler | Gate | Max R-hat | Min bulk / tail ESS | Min bulk ESS / CPU s | CPU s | Log loss | Brier |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in results:
        lines.append(f"| {r['scenario']} / {r['replicate']} / {r['variant']} | {r['sampler']} | {'PASS' if r['passed'] else 'FAIL'} | {r['max_rhat']:.4f} | {r['min_ess_bulk']:.1f} / {r['min_ess_tail']:.1f} | {r['min_bulk_ess_per_cpu_second']:.2f} | {r['fit_cpu_seconds']:.1f} | {r['log_loss']:.5f} | {r['brier']:.5f} |")
    lines += ['', 'Source: ' + ', '.join(sorted({study['config']['source'] for _, study in studies})) + '. Real data have no known recovery truth.', '', '## Posterior agreement and recovery', '',
              'Parameter-comparison.csv records each posterior mean, interval, gate, MCSE, and ESS/CPU ratio '
              'against the matched reference. The difference divided by combined MCSE is a descriptive '
              'screen, not an independent-chain hypothesis test: seeds are shared and failed diagnostics '
              'make MCSE unreliable. Recovery RMSE mixes parameter scales and is descriptive only. '
              'Inspect parameter-level errors/intervals in each source recovery.csv. Few datasets cannot '
              'establish calibrated coverage. Static fits on full-generating data are deliberately not '
              'counted as matched recovery.', '', '## Saved reference preservation', '']
    for r in preservation:
        lines.append(f"- {r['variant']}: exact retained draw equality with saved longer-chain study: {r['exact_draw_equality']}.")
    lines += ['', '![Matched sampler diagnostics and predictions](sampler-comparison.png)', '', 'See experiments/TASK3_SAMPLER_REPORT.md for the conditional audit, interpretation and limitations.']
    rendered = '\n'.join(lines)+'\n'
    if all(study['config']['source'] == 'anki' for _, study in studies):
        rendered = rendered.replace('Recovery RMSE mixes parameter scales and is descriptive only. Inspect parameter-level errors/intervals in each source recovery.csv. Few datasets cannot establish calibrated coverage. Static fits on full-generating data are deliberately not counted as matched recovery.', 'No parameter recovery or coverage is computed for these real Anki observations. These are two selected cohorts, not a full-dataset efficacy evaluation.')
        rendered = rendered.replace('experiments/TASK3_SAMPLER_REPORT.md', 'experiments/TASK3_ANKI_REPORT.md')
    (out/'report.md').write_text(rendered)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--saved-reference', type=Path)
    args = parser.parse_args()
    compare(args.runs, args.out, args.saved_reference)
