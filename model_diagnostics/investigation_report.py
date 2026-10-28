"""Curate a completed dynamics pilot without rewriting its raw outputs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from .analysis import atomic_json, write_csv
from .anki import ROOT


def read(path):
    with path.open() as f: return list(csv.DictReader(f))


def summarize(source,out):
    status=json.loads((source/'status.json').read_text())
    if status['state']!='complete' or status['fits']!=32:
        raise ValueError('Only a complete 32-fit pilot may be curated.')
    if out.exists(): raise ValueError('Choose a new comparison/curation directory.')
    results=read(source/'results.csv');diag=read(source/'diagnostics.csv')
    chain_metadata=json.loads((source/'chain-metadata.json').read_text())
    segments={r['fit']:{m.get('execution_segment','initial') for m in chain_metadata if m['fit']==r['fit']} for r in results}
    by_fit={r['fit']:[d for d in diag if d['fit']==r['fit']] for r in results}
    matched=[];screens=[];regression=[]
    for r in results:
        if r['interweave']!='True':continue
        centered=r['fit'].replace('-interwoven','-centered')
        base=next(x for x in results if x['fit']==centered)
        if len(segments[r['fit']])!=1 or segments[r['fit']]!=segments[centered]:
            raise ValueError(f'CPU comparison crosses execution segments: {centered}')
        matched.append(dict(execution_segment=next(iter(segments[centered])),fit=r['fit'],centered=centered,centered_pass=base['passed'],interwoven_pass=r['passed'],
                            ess_cpu_ratio=float(r['min_ess_per_cpu'])/float(base['min_ess_per_cpu']),
                            log_loss_delta=float(r['log_loss'])-float(base['log_loss']),
                            brier_delta=float(r['brier'])-float(base['brier'])))
        old={d['parameter']:d for d in by_fit[centered]}
        for d in by_fit[r['fit']]:
            b=old[d['parameter']];se=np.hypot(float(d['mcse_mean']),float(b['mcse_mean']))
            screens.append(dict(fit=r['fit'],parameter=d['parameter'],mean_difference=float(d['mean'])-float(b['mean']),
                                combined_mcse=float(se),mcse_units=abs(float(d['mean'])-float(b['mean']))/se if se>0 else None,
                                ess_cpu_ratio=float(d['ess_per_cpu_second'])/float(b['ess_per_cpu_second']),
                                both_parameter_gates=d['passed']=='True' and b['passed']=='True',
                                both_fit_gates=r['passed']=='True' and base['passed']=='True'))
    for cohort in range(2):
        a=np.load(source/f'real-{cohort}-full-centered-draws.npz')
        b=np.load(ROOT/f'.task3-runs/task3-anki-joint_blocked/anki-{cohort}-full-draws.npz')
        for key in a.files:
            regression.append(dict(cohort=cohort,parameter=key,retained_draws_identical=bool(np.array_equal(a[key],b[key]))))
    out.mkdir(parents=True)
    for name in ('protocol.json','provenance.json','results.csv','diagnostics.csv','recovery.csv','chain-metadata.json','status.json'):
        shutil.copy2(source/name,out/name)
    for name in ('resume-origin.json','previous-provenance.json'):
        if (source/name).exists():shutil.copy2(source/name,out/name)
    audit=ROOT/'.task3-runs/task3-investigation-audit'
    for name in ('audit-v3.json','nested-design-rank.json','tie-order-sensitivity.json','tie-order-cumulative.json','workspace.png'):
        if (audit/name).exists():shutil.copy2(audit/name,out/name)
    write_csv(out/'matched-comparison.csv',matched);write_csv(out/'parameter-comparison.csv',screens)
    write_csv(out/'reference-regression.csv',regression)
    # Selected figures only; all raw paths/datasets and original-ID selections stay local.
    for kind in ('real','nested_synthetic'):
        for cohort in range(2):
            for kernel in ('centered','interwoven'):
                name=f'{kind}-{cohort}-full-{kernel}-trace.png'
                shutil.copy2(source/name,out/name)
    lines=['# Dynamics investigation: measured results','',
           'Four chains per fit; 8,000 iterations / 4,000 warmup. Every monitored scalar must pass rank R-hat <1.01 and bulk/tail ESS >=400. Undefined fails. '
           'Fixed controls change the model. Real cohorts retain recorded within-day order. An alternative within-day permutation removes all timing discrepancies but does not authenticate outcome order. '
           'Synthetic fits condition on the saved nested gap schedules; these are two fixed-truth replicates, not SBC.','',
           '| Dataset / cohort | Control | Kernel | Max R-hat | Min bulk / tail ESS | ESS / CPU s | CPU s | Log loss | Gate |',
           '|---|---|---|---:|---:|---:|---:|---:|---|']
    for r in results:
        lines.append(f"| {r['kind']} / {int(r['cohort'])+1} | {r['control']} | {'interwoven' if r['interweave']=='True' else 'centered'} | {float(r['max_rhat']):.4f} | {float(r['min_ess_bulk']):.1f} / {float(r['min_ess_tail']):.1f} | {float(r['min_ess_per_cpu']):.2f} | {float(r['cpu']):.1f} | {float(r['log_loss']):.5f} | {'PASS' if r['passed']=='True' else 'FAIL'} |")
    lines+=['','## Matched computational comparisons','',
            'Ratio = interwoven minimum bulk ESS/CPU divided by centered minimum bulk ESS/CPU. The limiting parameter may differ. '
            'Parameter-comparison.csv retains every scalar ratio and MCSE difference screen. A failed whole fit prevents inferential interpretation; '
            'an MCSE screen is descriptive and does not prove target equality. Exact transformation tests independently check the target.','',
            '| Dataset / cohort / control | Centered / interwoven gate | ESS / CPU ratio | Delta log loss |',
            '|---|---|---:|---:|']
    for r in matched:lines.append(f"| {r['fit'].removesuffix('-interwoven')} | {r['centered_pass']} / {r['interwoven_pass']} | {r['ess_cpu_ratio']:.2f} | {r['log_loss_delta']:+.5f} |")
    lines+=['','## Recovery','',
            'Recovery.csv contains raw and combined parameters, all generating hierarchy variances, and terminal training states. '
            'Counts below include failed parameters and dependent scalar summaries. They are not a calibrated coverage rate.','',
            '| Fit | Truth-containing intervals / quantities | Quantities passing gate | Whole fit |',
            '|---|---:|---:|---|']
    recovery=read(source/'recovery.csv')
    for fit in sorted({r['fit'] for r in recovery}):
        group=[r for r in recovery if r['fit']==fit]
        result=next(r for r in results if r['fit']==fit)
        lines.append(f"| {fit} | {sum(r['covered']=='True' for r in group)} / {len(group)} | {sum(r['passed']=='True' for r in group)} | {result['passed']} |")
    lines+=['','## Regression and limits','',
            f"All {len(regression)} saved parameter arrays from the two centered real full fits match the preceding joint_blocked arrays exactly: {all(r['retained_draws_identical'] for r in regression)}. "
            'These repeats are target/regression controls, not additional independent real datasets. CPU varies between executions; matched ratios are checked to use the same execution segment. '
            'New user-level and within-user card summaries supplement the original diagnostic monitor set.', '',
            'Prediction uses the inherited approximate prequential filter and at most 200 retained draws per chain. '
            'Small score differences under failed diagnostics cannot establish predictive superiority or component benefit. '
            'No test outcomes tune the fixed controls or proposals. Added innovation proposals increase CPU, so iteration counts alone are not efficiency evidence. '
            'No stronger priors, parameter-support changes, dropped chains, threshold relaxation or full-dataset inference are used.','',
            '![Dynamics ladder](dynamics-ladder.png)','']
    (out/'report.md').write_text('\n'.join(lines))
    plot_ladder(out,results)
    atomic_json(out/'postprocessing.json',dict(source=str(source.resolve()),
                source_results_sha256=hashlib.sha256((source/'results.csv').read_bytes()).hexdigest(),
                report_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                reference_arrays_identical=all(r['retained_draws_identical'] for r in regression)))


def plot_ladder(out,results):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,12))
    labels=[r['fit'] for r in results]
    colors=['C0' if r['passed']=='True' else 'C3' for r in results]
    axes[0].barh(labels,[float(r['min_ess_bulk']) for r in results],color=colors)
    axes[0].axvline(400,color='black',ls='--',lw=1);axes[0].set_title('Minimum bulk ESS (whole-fit pass blue)')
    axes[1].barh(labels,[float(r['min_ess_per_cpu']) for r in results],color=colors)
    axes[1].set_title('Minimum bulk ESS per CPU second')
    for ax in axes:ax.invert_yaxis();ax.tick_params(axis='y',labelsize=7);ax.set_xscale('log');ax.grid(axis='x',alpha=.2)
    fig.tight_layout();fig.savefig(out/'dynamics-ladder.png',dpi=150);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();summarize(args.source,args.out)


if __name__=='__main__':main()
