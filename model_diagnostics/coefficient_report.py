"""Curate the completed coefficient/path pilot without copying raw datasets/IDs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from .analysis import atomic_json, write_csv


def read(path):
    with path.open() as stream: return list(csv.DictReader(stream))


def summarize(source, out):
    if json.loads((source/'status.json').read_text())['state'] != 'complete':
        raise ValueError('Only complete studies may be curated.')
    if out.exists(): raise FileExistsError(out)
    results=read(source/'results.csv'); rows=read(source/'diagnostics.csv')
    metadata=json.loads((source/'chain-metadata.json').read_text())
    protocol=json.loads((source/'protocol.json').read_text())
    regressions=read(source/'baseline-regression.csv')
    if len(results)!=12 or len(metadata)!=48 or not all(r['identical']=='True' for r in regressions):
        raise ValueError('Incomplete study or failed baseline regression.')
    by_fit={r['fit']:[d for d in rows if d['fit']==r['fit']] for r in results}
    comparisons=[]; parameters=[]
    for r in results:
        if r['kernel']!='coefficient': continue
        group=by_fit[r['fit']]
        for base_kernel in ('centered','interwoven'):
            name=r['fit'].removesuffix('coefficient')+base_kernel
            base=next(v for v in results if v['fit']==name)
            old={d['parameter']:d for d in by_fit[name]}
            if set(old)!={d['parameter'] for d in group}: raise ValueError('Monitor sets differ.')
            segments={m['execution_segment'] for m in metadata if m['fit'] in (r['fit'],name)}
            if len(segments)!=1: raise ValueError('CPU comparison spans execution segments.')
            comparisons.append(dict(fit=r['fit'],baseline=name,execution_segment=next(iter(segments)),
                                    coefficient_pass=r['passed'],baseline_pass=base['passed'],
                                    ess_cpu_ratio=float(r['min_ess_per_cpu'])/float(base['min_ess_per_cpu']),
                                    min_bulk_ratio=float(r['min_ess_bulk'])/float(base['min_ess_bulk']),
                                    cpu_ratio=float(r['cpu'])/float(base['cpu']),
                                    log_loss_delta=float(r['log_loss'])-float(base['log_loss']),
                                    brier_delta=float(r['brier'])-float(base['brier'])))
            for d in group:
                b=old[d['parameter']]; se=np.hypot(float(d['mcse_mean']),float(b['mcse_mean']))
                parameters.append(dict(fit=r['fit'],baseline=name,parameter=d['parameter'],
                                       mean_difference=float(d['mean'])-float(b['mean']),combined_mcse=se,
                                       mcse_units=abs(float(d['mean'])-float(b['mean']))/se if se else None,
                                       ess_cpu_ratio=float(d['ess_per_cpu_second'])/float(b['ess_per_cpu_second']),
                                       both_parameter_gates=d['passed']=='True' and b['passed']=='True',
                                       both_fit_gates=r['passed']=='True' and base['passed']=='True'))
    # Check frozen adaptation counts and extra-move counts before publication.
    for m in metadata:
        s=m['sampling']
        if s['adaptation_draws']!=protocol['warmup']: raise ValueError('Decay adaptation count differs.')
        if s['interweave'] and s['innovation_adaptations']!=protocol['warmup']: raise ValueError('Innovation adaptation count differs.')
        if m['fit'].endswith('coefficient') and s['coefficient_moves']!=protocol['iterations']: raise ValueError('Coefficient move count differs.')
    out.mkdir(parents=True)
    for name in ('protocol.json','provenance.json','results.csv','diagnostics.csv','recovery.csv','chain-metadata.json','status.json','baseline-regression.csv'):
        shutil.copy2(source/name,out/name)
    for path in source.glob('*-predictions.csv'): shutil.copy2(path,out/path.name)
    for path in source.glob('*-coefficient-*.png'): shutil.copy2(path,out/path.name)
    write_csv(out/'matched-comparison.csv',comparisons); write_csv(out/'parameter-comparison.csv',parameters)
    write_csv(out/'failed-parameters.csv',[r for r in rows if r['passed']!='True'])
    lines=['# Coefficient/path pilot: measured results','',
           'All fits use the same full model, unchanged priors, four chains and 8000/4000 iterations/warmup, no thinning. '
           'Every one of 142 monitors must satisfy finite rank-normalized folded split Rhat <1.01 and bulk/tail ESS >=400. '
           'Real cohorts retain provisional recorded order; nested synthetic recovery conditions on fixed gaps and fixed truths, not SBC.','',
           '| Dataset / cohort | Kernel | Max Rhat | Min bulk / tail ESS | Failures / monitors | CPU s | ESS / CPU s | Log loss | Gate |',
           '|---|---|---:|---:|---:|---:|---:|---:|---|']
    for r in results:
        lines.append(f"| {r['kind']} / {int(r['cohort'])+1} | {r['kernel']} | {float(r['max_rhat']):.6f} | {float(r['min_ess_bulk']):.1f} / {float(r['min_ess_tail']):.1f} | {r['failed_parameters']} / {r['monitors']} | {float(r['cpu']):.1f} | {float(r['min_ess_per_cpu']):.3f} | {float(r['log_loss']):.5f} | {'PASS' if r['passed']=='True' else 'FAIL'} |")
    lines+=['','## Matched comparisons','',
            'Ratios use contemporary baseline reruns in the same execution segment. Minimum ESS may refer to different limiting scalars. '
            'All per-parameter ESS/CPU ratios and descriptive combined-MCSE screens are retained. No failed chains or parameters are removed. '
            'Timing is a single realization per fit, not a repeated hardware benchmark.','',
            '| New fit | Baseline | ESS / CPU ratio | CPU ratio | Delta log loss |','|---|---|---:|---:|---:|']
    for r in comparisons:
        lines.append(f"| {r['fit']} | {r['baseline'].split('-')[-1]} | {r['ess_cpu_ratio']:.2f} | {r['cpu_ratio']:.2f} | {r['log_loss_delta']:+.5f} |")
    lines+=['','## Recovery','',
            'Counts include every dependent truth summary, including failing parameters. Few fixed-truth intervals are not calibrated coverage.','',
            '| Fit | Truth-containing intervals / 142 | Parameter passes | Whole fit |','|---|---:|---:|---|']
    recovered=read(source/'recovery.csv')
    for fit in sorted({r['fit'] for r in recovered}):
        group=[r for r in recovered if r['fit']==fit]
        result=next(r for r in results if r['fit']==fit)
        lines.append(f"| {fit} | {sum(r['covered']=='True' for r in group)} / {len(group)} | {sum(r['passed']=='True' for r in group)} | {result['passed']} |")
    lines+=['','## Remaining failures of the new kernel','']
    for r in results:
        if r['kernel']!='coefficient': continue
        failed=[d for d in by_fit[r['fit']] if d['passed']!='True']
        lines.append(f"- {r['fit']}: "+(', '.join(d['parameter'] for d in failed) if failed else 'none')+'.')
    lines+=['','## Correctness and limits','',
            f"All {len(regressions)} saved parameter arrays (retained and available warmup) match the preceding baseline study exactly. "
            'Every new fit retains the complete baseline monitor set. Warmup adaptation and the expected 8000 extra coefficient moves per chain are checked. '
            'The baseline replay is a regression check, not independent data replication. Independent conditional-density and transformation tests establish the extra Gibbs target.', '',
            'Predictions use the inherited approximate prequential filter and at most 200 posterior draws per chain. '
            'All kernels target the same posterior; score differences are prediction-stability screens, not statistical model improvements or component efficacy. '
            'Nonconvergence limits interpretation of means, recovery and MCSE screens. Real chronology and early-history authenticity remain unresolved. '
            'Exact nested user/card likelihood symmetries are unchanged. Hierarchy priors and scale sensitivity are separate from Monte Carlo convergence.','',
            '![Minimum effective draws and computational efficiency](coefficient-comparison.png)','']
    (out/'report.md').write_text('\n'.join(lines))
    plot_results(out, results)
    atomic_json(out/'postprocessing.json',dict(source=str(source.resolve()),
                results_sha256=hashlib.sha256((source/'results.csv').read_bytes()).hexdigest(),
                report_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                baseline_arrays_identical=True,monitor_sets_identical=True))


def plot_results(out, results):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter
    fig,axes=plt.subplots(1,2,figsize=(13,7))
    labels=[r['fit'] for r in results]; colors=['C0' if r['passed']=='True' else 'C3' for r in results]
    axes[0].barh(labels,[float(r['min_ess_bulk']) for r in results],color=colors)
    axes[0].axvline(400,color='black',ls='--');axes[0].set_title('Minimum bulk ESS (whole-fit pass blue)')
    axes[1].barh(labels,[float(r['min_ess_per_cpu']) for r in results],color=colors);axes[1].set_title('Minimum bulk ESS per CPU second')
    for ax in axes:ax.invert_yaxis();ax.set_xscale('log');ax.tick_params(axis='y',labelsize=7);ax.grid(axis='x',alpha=.2)
    for ax, ticks in zip(axes, ([30,100,400,1000],[.2,.5,1,2,5])):
        ax.set_xticks(ticks, [str(t) for t in ticks]); ax.xaxis.set_minor_formatter(NullFormatter())
    fig.tight_layout();fig.savefig(out/'coefficient-comparison.png',dpi=150);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();summarize(args.source,args.out)


if __name__=='__main__':main()
