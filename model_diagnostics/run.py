"""CLI/job worker: python -m model_diagnostics.run --config config.json --out DIR."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import traceback
import numpy as np
from .analysis import (atomic_json, diagnostics, metrics, overview_plots, paired_bootstrap,
                       plots, scalar_draws, truth_values, write_csv)
from .anki import ANKI_BOUNDS, load_cohort
from .data import SCENARIOS, simulate, split_suffix, validate_sequences
from .model import DiagnosticModel, VARIANTS, SAMPLERS

ROOT = Path(__file__).resolve().parents[1]


def validate_config(config):
    defaults = dict(source='synthetic', scenarios=['full'], replicates=1, chains=4,
                    iterations=2000, warmup=1000, seed=20261027, users=4, items=5,
                    events=32, holdout=.25, variants=list(VARIANTS), workers=2, sampler='reference', mh_steps=5)
    unknown = set(config) - set(defaults) - {'sequences'}
    if unknown: raise ValueError(f'Unknown settings: {sorted(unknown)}')
    c = {**defaults, **config}
    limits = dict(replicates=(1, 3), chains=(4, 4), iterations=(200, 8000),
                  warmup=(50, 6000), users=(2, 8), items=(2, 10), events=(8, 60),
                  seed=(0, 2**31-1), workers=(1, 4), mh_steps=(1, 20))
    for k, (lo, hi) in limits.items():
        if type(c[k]) is not int or not lo <= c[k] <= hi: raise ValueError(f'{k} must be an integer in [{lo}, {hi}]')
    if c['sampler'] not in SAMPLERS: raise ValueError('Unknown sampler.')
    if c['iterations'] - c['warmup'] < 100: raise ValueError('Keep at least 100 post-warmup draws per chain.')
    if type(c['holdout']) not in (int, float) or not .1 <= c['holdout'] <= .4: raise ValueError('Holdout fraction must be in [0.1, 0.4].')
    if not c['variants'] or len(set(c['variants'])) != len(c['variants']) or any(v not in VARIANTS for v in c['variants']) or 'full' not in c['variants']:
        raise ValueError('Select full plus any unique supported ablations.')
    if c['source'] == 'synthetic' and (not c['scenarios'] or len(set(c['scenarios'])) != len(c['scenarios']) or any(s not in SCENARIOS for s in c['scenarios'])):
        raise ValueError('Select unique supported synthetic scenarios.')
    if c['source'] == 'upload':
        validate_sequences(c.get('sequences'))
        c['scenarios'], c['replicates'] = ['uploaded'], 1
    elif c['source'] == 'anki':
        c['scenarios'] = ['anki']
    elif c['source'] != 'synthetic': raise ValueError('Source must be synthetic, upload or anki.')
    return c


def chain_job(train, test, variant, seed, chain, iterations, warmup, out, sampler='reference', mh_steps=5, dt_bounds=(.01, 365.)):
    model = DiagnosticModel(train, variant=variant, seed=seed, chain=chain, sampler=sampler, mh_steps=mh_steps, dt_bounds=dt_bounds)
    model.cancel_file = str(Path(out) / 'CANCEL')
    start, cpu = time.perf_counter(), time.process_time()
    model.run(n_iter=iterations, burn_in=warmup, thin=1)
    fit_wall, fit_cpu = time.perf_counter() - start, time.process_time() - cpu
    start, cpu = time.perf_counter(), time.process_time()
    pred = np.concatenate(model.predict(test))
    return dict(samples=model.samples, warmup=model.warmup, prediction=pred,
                fit_wall_seconds=fit_wall, fit_cpu_seconds=fit_cpu,
                predict_wall_seconds=time.perf_counter()-start,
                predict_cpu_seconds=time.process_time()-cpu,
                sampling=model.sampling_info, seed=seed)


def report(out, summary):
    c, results, comparisons, recovery = [summary[k] for k in ('config', 'results', 'comparisons', 'recovery')]
    lines = ['# Bayesian dynamic latent-state model: diagnostics and ablations', '',
             f"Run `{out.name}`. Source: **{c['source']}**. {c['chains']} independent chains, "
             f"{c['iterations']} iterations each; {c['warmup']} discarded warmup; no thinning.", '',
             f"Sampler: `{c['sampler']}`; {c['mh_steps']} MH proposals per decay sweep. Joint covariance and scale adapt only during warmup. Blocked effects use the exact Gaussian conditional under unchanged priors.", '',
             '## Scope and model', '',
             'This study targets `AnkiFirstRun/dynamic_retention_model_optimised.py`, the pure NumPy time-aware OU model, '
             'not the event-indexed AR(1) model used by the original recall demo. The optimized-named file actually '
             'contains the pre-Numba implementation; it is reused directly for FFBS, priors and collapsed MH.', '',
             '`mu = b0 + alpha_user + beta_item + gamma_user * previous_outcome`; '
             '`rate = lambda * exp(-kappa * log(1 + previous_successes))`; '
             '`phi = exp(-rate * gap_days)`; `q = sigma2 * (1 - phi²)/(2 * rate)`. '
             'The first state has variance 1. A Bernoulli-logistic observation measures the latent state. '
             'This is mean reversion on the logit scale, not a guarantee of monotonically decreasing recall with time.', '',
             'Full has all components. No-practice fixes kappa=0; no-hierarchy fixes ability/item effects to zero '
             'and uses one population practice slope; no-dynamics is hierarchical lagged logistic regression '
             'with the same effect priors, removing state persistence and diffusion (and hence decay parameters). '
             'The last contrast removes the entire dynamics block; it cannot isolate persistence from diffusion.', '',
             'Priors are inherited: b0 and g0 N(0,100); group effects Gaussian with inverse-gamma(2,1) variances; '
             'diffusion inverse-gamma(2,0.5); log lambda N(log(0.05),1.5²), restricted to [1e-4,20]; '
             'kappa N(0,1), restricted to [-3,3]. The harness uses the reference `polyagamma` package '
             'instead of the repository\'s truncated-series approximation. This is a deliberate sampler change, '
             'so these timings and diagnostics are not a reproduction of an earlier unrecorded run.', '',
             '## Protocol and diagnostic gate', '',
             'Each chain has its own seed and dispersed initial lambda (0.005, 0.05, 0.5, 2), diffusion '
             '(0.05, 0.2, 1, 3), kappa (-1, 0, 0.5, 1.5) and effect values. MH proposal adaptation stops '
             'at warmup. All retained draws enter diagnostics; prediction uses at most 200 evenly spaced draws '
             'per chain, with equal weight across chains. Warmup traces are retained separately. '
             'Lambda proposal acceptance and kappa sweep-move fractions are in chain-metadata.json; '
             'the latter is not an acceptance rate.', '',
             'Rank-normalized folded split R̂ < 1.01 and both bulk/tail ESS ≥ 100 × chain count '
             '(400 here) are required for every monitored scalar. Nonfinite diagnostics fail the gate. '
             'Diagnostics include all effects, learned scales, terminal latent states, and centered '
             'effect contrasts, not just decay parameters. Fixed ablated parameters are excluded. '
             'Passing does not prove convergence; failing blocks trustworthy posterior interpretation. '
             'The old within-single-chain split R̂ is not used. See '
             '[Stan diagnostics](https://mc-stan.org/learn-stan/diagnostics-warnings.html) and '
             '[ArviZ rank R̂](https://python.arviz.org/en/v0.23.4/api/generated/arviz.rhat.html).', '',
             'Each user/item sequence has a chronological held-out suffix. Every variant sees identical '
             'training and test events. Prediction is prequential: an outcome is used only after its own '
             'prediction, to update the state and practice count before later predictions. Dynamic prediction '
             'inherits the branch\'s logistic-normal approximation and PG-mean filtering; it is approximate '
             'rather than an exact held-out Bayesian filter. Log loss, Brier score, MAE, AUC and '
             'calibration bias are reported. No held-out score selects hyperparameters.', '',
             'Fit cost is summed process CPU across chains (ESS/sec uses this denominator), with summed '
             'per-chain elapsed time also reported. Parallel job wall time is recorded separately and includes '
             'analysis/plotting overhead. Prediction CPU is separate. These costs are machine-specific.', '',
             '## Results', '',
             '| Scenario / replicate | Variant | Log loss | Brier | AUC | Max R̂ | Min bulk ESS | Min tail ESS | CPU s | Gate |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in results:
        auc = f"{r['auc']:.3f}" if r['auc'] is not None else 'undefined'
        lines.append(f"| {r['scenario']} / {r['replicate']} | {r['variant']} | {r['log_loss']:.4f} | {r['brier']:.4f} | {auc} | {r['max_rhat']:.3f} | {r['min_ess_bulk']:.1f} | {r['min_ess_tail']:.1f} | {r['fit_cpu_seconds']:.1f} | {'PASS' if r['passed'] else 'FAIL'} |")
    lines += ['', '## Controlled component contrasts', '',
              'Delta = ablated log loss − full log loss: positive favors the full model. Intervals are '
              'paired user-cluster bootstrap percentiles (1,000 resamples). With few users they are '
              'unstable and do not include MCMC uncertainty or correct for multiple comparisons. '
              'Results remain exploratory when either model fails diagnostics.', '',
              '| Scenario / replicate | Ablation | Delta log loss | 95% cluster interval | Interpretation |',
              '|---|---|---:|---|---|']
    for r in comparisons:
        conclusion = ('Convergence unresolved' if not r['converged'] else
                      'Evidence for full on this sample' if r['low'] > 0 else
                      'Evidence for ablated on this sample' if r['high'] < 0 else 'Inconclusive')
        lines.append(f"| {r['scenario']} / {r['replicate']} | {r['variant']} | {r['delta_log_loss']:.4f} | [{r['low']:.4f}, {r['high']:.4f}] | {conclusion} |")
    lines += ['', '## Recovery and identifiability', '']
    if recovery:
        matched = [r for r in recovery if r['matched']]
        usable = [r for r in matched if r['passed']]
        covered = sum(r['covered'] for r in usable)
        lines += [f"Known truth is available. Matched generating/fitted variants yield {len(matched)} parameter intervals; "
                  f"{len(usable)} pass their parameter diagnostic gate; {covered} of those contain the truth. "
                  'These are dependent, few-replicate descriptive counts, not calibrated coverage or simulation-based calibration.', '',
                  'Recovery.csv contains truth, posterior mean, signed error, squared error, 95% interval, '
                  'coverage and a parameter-level gate for all recoverable parameters. Recovery-summary.csv '
                  'gives replicate bias, RMSE and coverage with separate convergence counts. Misspecified '
                  'variants are flagged and must not be interpreted as parameter recovery under their own DGP.', '']
    else:
        lines += ['Uploaded data have no known ground truth. Recovery is not computed or inferred from fit quality.', '']
    lines += ['The likelihood depends on b0+alpha+beta, so additive shifts in b0 and group effects '
              'are not separately likelihood-identified; proper priors regularize these directions without '
              'creating data information. Centered alpha/beta contrasts and population_level are reported '
              'alongside raw coefficients. Centering is post-processing, not a sampler reparameterization. '
              'User-nested item designs worsen ability/item separation; this cannot be fixed just by longer chains.', '',
              'Lambda and kappa share log(rate)=log(lambda)−kappa×log(1+past_successes). '
              'A narrow practice-count range makes them hard to separate. Diffusion and decay both '
              'control transition variance, giving additional posterior dependence. Pair plots and '
              'correlations quantify these sample-specific ridges; they do not prove structural identifiability.', '',
              '## Conclusions and next investigations', '',
              f"{sum(r['passed'] for r in results)} of {len(results)} fitted variants pass the full diagnostic gate. "
              'No component is declared empirically justified from predictive point estimates alone. '
              'A component requires a converged controlled contrast, a credible predictive improvement, '
              'and replication on representative learner data. Synthetic matched recovery tests the '
              'implementation under assumed ground truth; it does not establish real-world usefulness.', '',
              'Inspect chains for drift, poor rank overlap and long autocorrelation first. Extend warmup '
              'and retained draws using saved settings. If diagnostics remain poor, investigate blocked '
              'population/effect updates and joint decay proposals using the sampler selector. Non-centered '
              'random effects remain a separate unimplemented hypothesis. Use short-gap scenarios and multiple '
              'practice ranges to probe weak identification. Run more independent synthetic datasets '
              'and a prior-drawn SBC study before claiming calibrated inference.', '',
              'No previously observed ESS/R̂ values are reconstructed without their original chain files '
              'and exact configuration. This run is new evidence. See provenance.json for revision, '
              'source hashes, dependency versions, seed and dataset hash; config.json and dataset files '
              'allow reruns. NPZ files preserve chain/draw axes and all retained parameters.', '',
              '## Artifacts', '', '![Comparison](comparison.png)', '', '![Recovery](recovery.png)' if recovery else '', '',
              'Per-fit trace/ACF, pooled rank and posterior-pair plots are alongside CSV diagnostics, '
              'predictions, draws, recovery summaries and chain metadata. Report status is never '
              'changed to converged just because computation finished.']
    if c['source'] == 'anki':
        lines = [line.replace('Uploaded data have no known ground truth.', 'Real Anki observations have no known parameter truth.') for line in lines]
        lines[2:2] = ['Real Anki observations: fixed disjoint user cohorts from local Parquet files. The provided train/test partition is preserved (holdout setting is ignored). Exact recorded elapsed_seconds / 86400 supplies gaps, with explicit bounds [1 second, 3650 days] rather than clipping. Item IDs are user-nested. No known parameter truth or recovery claims. See selection.json artifacts for selection/exclusion counts and source hashes.', '']
    (out / 'report.md').write_text('\n'.join(lines) + '\n')


def execute(config, out):
    c = validate_config(config)
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / 'config.json', c)
    started = time.perf_counter()
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in list((ROOT / 'model_diagnostics').glob('*.py'))
              + [ROOT / 'AnkiFirstRun/dynamic_retention_model_optimised.py']}
    provenance = dict(revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                      dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()),
                      source_hashes=hashes, python=platform.python_version(), platform=platform.platform(),
                      packages={p: importlib.metadata.version(p) for p in ['numpy', 'scipy', 'matplotlib', 'arviz', 'polyagamma'] + (['pyarrow'] if c['source'] == 'anki' else [])},
                      created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), config_sha256=hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest())
    atomic_json(out / 'provenance.json', provenance)
    results, diagnostic_rows, recoveries, comparisons, metadata, correlations = [], [], [], [], [], []
    total = len(c['scenarios']) * c['replicates'] * len(c['variants']) * c['chains']
    done = 0
    atomic_json(out / 'status.json', dict(state='running', completed=0, total=total,
                                         message='Sampling independent chains (including warmup).', elapsed_seconds=0))
    with ProcessPoolExecutor(max_workers=c['workers']) as pool:
        for scenario in c['scenarios']:
            for replicate in range(c['replicates']):
                data_seed = c['seed'] + 1009 * replicate + (SCENARIOS.index(scenario) * 100003 if scenario in SCENARIOS else 0)
                if c['source'] == 'anki':
                    train, test, seqs, selection = load_cohort(c['seed'], replicate, c['users'], c['items'], c['events'])
                    truth = None
                    atomic_json(out / f'{scenario}-{replicate}-selection.json', selection)
                elif c['source'] == 'upload':
                    seqs, mapping = validate_sequences(c['sequences']); truth = None
                    atomic_json(out / 'id-mapping.json', mapping)
                else:
                    seqs, truth, states = simulate(data_seed, scenario, c['users'], c['items'], c['events'])
                    atomic_json(out / f'{scenario}-{replicate}-truth.json', dict(parameters=truth, latent_states=states))
                dataset = [{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in s.items()} for s in seqs]
                atomic_json(out / f'{scenario}-{replicate}-dataset.json', dataset)
                provenance[f'{scenario}-{replicate}-data_sha256'] = hashlib.sha256(json.dumps(dataset, sort_keys=True).encode()).hexdigest()
                atomic_json(out / 'provenance.json', provenance)
                if c['source'] != 'anki': train, test = split_suffix(seqs, c['holdout'])
                y = np.concatenate([s['r'] for s in test])
                user_ids = np.concatenate([np.full(len(s['r']), s['user']) for s in test])
                predictions, passed = {}, {}
                for variant in c['variants']:
                    prefix = f'{scenario}-{replicate}-{variant}'
                    futures = {pool.submit(chain_job, train, test, variant, data_seed + chain * 7919 + 31,
                                           chain, c['iterations'], c['warmup'], str(out), c['sampler'], c['mh_steps'], ANKI_BOUNDS if c['source'] == 'anki' else (.01, 365.)): chain for chain in range(c['chains'])}
                    chains = [None] * c['chains']
                    for future in as_completed(futures):
                        chains[futures[future]] = future.result(); done += 1
                        atomic_json(out / 'status.json', dict(state='running', completed=done, total=total,
                                                             message=f'{prefix}: {sum(x is not None for x in chains)}/{c["chains"]} chains complete',
                                                             elapsed_seconds=time.perf_counter()-started))
                    samples = {k: np.stack([ch['samples'][k] for ch in chains]) for k in chains[0]['samples']}
                    warmup = {k: np.stack([ch['warmup'][k] for ch in chains]) for k in chains[0]['warmup']}
                    np.savez_compressed(out / f'{prefix}-draws.npz', **samples)
                    np.savez_compressed(out / f'{prefix}-warmup.npz', **warmup)
                    cpu = sum(ch['fit_cpu_seconds'] for ch in chains)
                    draws = scalar_draws(samples, variant)
                    diag = diagnostics(draws, cpu)
                    passed[variant] = all(r['passed'] for r in diag)
                    for row in diag: diagnostic_rows.append(dict(scenario=scenario, replicate=replicate, variant=variant, **row))
                    prediction = np.mean([ch['prediction'] for ch in chains], axis=0)
                    predictions[variant] = prediction
                    write_csv(out / f'{prefix}-predictions.csv', [dict(event=j, user=int(user_ids[j]), outcome=int(y[j]), probability=float(p)) for j, p in enumerate(prediction)])
                    result = dict(scenario=scenario, replicate=replicate, variant=variant, **metrics(y, prediction),
                                  max_rhat=max(r['rhat'] for r in diag), min_ess_bulk=min(r['ess_bulk'] for r in diag),
                                  min_ess_tail=min(r['ess_tail'] for r in diag), passed=passed[variant], failed_parameters=sum(not r['passed'] for r in diag),
                                  fit_cpu_seconds=cpu, sum_chain_wall_seconds=sum(ch['fit_wall_seconds'] for ch in chains),
                                  prediction_cpu_seconds=sum(ch['predict_cpu_seconds'] for ch in chains))
                    results.append(result)
                    metadata.extend(dict(scenario=scenario, replicate=replicate, variant=variant, chain=j,
                                         **{k: v for k, v in ch.items() if k not in ('samples', 'warmup', 'prediction')}) for j, ch in enumerate(chains))
                    if variant == 'full':
                        for a, b in [('lam', 'kappa'), ('lam', 'sigma2'), ('b0', 'population_level')]:
                            correlations.append(dict(scenario=scenario, replicate=replicate, a=a, b=b,
                                                     correlation=float(np.corrcoef(draws[a].ravel(), draws[b].ravel())[0, 1])))
                    if truth is not None:
                        expected = truth_values(truth)
                        matched_variant = 'full' if scenario == 'short_gaps' else scenario
                        for row in diag:
                            if row['parameter'] in expected:
                                value = expected[row['parameter']]
                                recoveries.append(dict(scenario=scenario, replicate=replicate, variant=variant,
                                                       parameter=row['parameter'], truth=value, mean=row['mean'],
                                                       error=row['mean']-value, squared_error=(row['mean']-value)**2,
                                                       low=row['low'], high=row['high'], covered=row['low'] <= value <= row['high'],
                                                       passed=row['passed'], matched=variant == matched_variant))
                    plots(out, prefix, draws, warmup)
                    atomic_json(out / 'partial.json', dict(results=results, diagnostics=diagnostic_rows))
                    if (out / 'CANCEL').exists(): raise InterruptedError('Cancelled by user')
                for variant in c['variants']:
                    if variant != 'full':
                        comparisons.append(dict(scenario=scenario, replicate=replicate, variant=variant,
                                                **paired_bootstrap(y, predictions['full'], predictions[variant], user_ids, data_seed),
                                                converged=passed['full'] and passed[variant]))
    recovery_summary = []
    for scenario, variant, parameter in sorted({(r['scenario'], r['variant'], r['parameter']) for r in recoveries}):
        group = [r for r in recoveries if (r['scenario'], r['variant'], r['parameter']) == (scenario, variant, parameter)]
        recovery_summary.append(dict(scenario=scenario, variant=variant, parameter=parameter, datasets=len(group),
                                     bias=float(np.mean([r['error'] for r in group])), rmse=float(np.sqrt(np.mean([r['squared_error'] for r in group]))),
                                     coverage=float(np.mean([r['covered'] for r in group])), converged_datasets=sum(r['passed'] for r in group),
                                     matched=group[0]['matched']))
    for filename, rows in [('results.csv', results), ('diagnostics.csv', diagnostic_rows), ('recovery.csv', recoveries),
                           ('recovery-summary.csv', recovery_summary), ('ablations.csv', comparisons), ('correlations.csv', correlations)]:
        write_csv(out / filename, rows)
    atomic_json(out / 'chain-metadata.json', metadata)
    summary = dict(config={k: v for k, v in c.items() if k != 'sequences'}, results=results,
                   diagnostics=diagnostic_rows, recovery=recoveries, recovery_summary=recovery_summary,
                   comparisons=comparisons, correlations=correlations, wall_seconds=time.perf_counter()-started)
    overview_plots(out, results, recoveries)
    report(out, summary)
    atomic_json(out / 'summary.json', summary)
    atomic_json(out / 'status.json', dict(state='complete', completed=done, total=total,
                                         message='Computation complete; inspect diagnostic gates before interpretation.',
                                         elapsed_seconds=time.perf_counter()-started))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args(); out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    if (out / 'config.json').exists():
        ap.error('Output already contains a study. Use a new directory to preserve its provenance.')
    atomic_json(out / 'pid.json', os.getpid())
    try:
        execute(json.loads(args.config.read_text()), out)
    except Exception as e:
        atomic_json(out / 'status.json', dict(state='cancelled' if isinstance(e, InterruptedError) else 'failed', message=str(e)))
        traceback.print_exc()
        raise SystemExit(1) from e


if __name__ == '__main__':
    main()
