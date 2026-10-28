"""Reproducible read-only Task 3 completion analysis and aggregate curation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from scipy.stats import invgamma
from .analysis import atomic_json, write_csv
from .anki import ROOT


def read(path):
    with path.open() as stream:return list(csv.DictReader(stream))


def hierarchy_oracle(gamma, g0, scale):
    """Known realized slopes AND population mean: exact conjugate control.

    This supplies more information than Bernoulli observations; it is a different
    observation model, not a full-model convergence result or a coverage study.
    """
    shape=2+len(gamma)/2; post_scale=scale+.5*np.sum((np.asarray(gamma)-g0)**2)
    low,high=invgamma.ppf([.025,.975],a=shape,scale=post_scale)
    return dict(shape=shape,scale=float(post_scale),mean=float(post_scale/(shape-1)),low=float(low),high=float(high))


def bottleneck(source, out):
    rows=[]; design=[]; oracle=[]
    for kind in ('real','nested_synthetic'):
        for cohort in range(2):
            stem=f'{kind}-{cohort}';fit=f'{stem}-full-coefficient'
            samples=np.load(source/f'{fit}-draws.npz')
            seqs=json.loads((source/f'{stem}-dataset.json').read_text());cuts=json.loads((source/f'{stem}-split.json').read_text())['cuts']
            c=np.concatenate([np.log1p(np.r_[0,np.cumsum(s['r'])[:-1]])[1:cut] for s,cut in zip(seqs,cuts)])
            dt=np.concatenate([np.array(s['dt'])[1:cut] for s,cut in zip(seqs,cuts)])
            for chain in range(4):
                loglam=np.log(samples['lam'][chain]);k=samples['kappa'][chain];logsigma=np.log(samples['sigma2'][chain])
                gamma=samples['gamma'][chain].mean(1)
                values=np.column_stack([loglam,k,logsigma]); corr=np.corrcoef(values.T)
                for name,other in [('log_lambda',loglam),('log_sigma2',logsigma),('gamma_mean',gamma)]:
                    rows.append(dict(fit=fit,chain=chain,statistic='correlation_with_kappa',parameter=name,value=float(np.corrcoef(k,other)[0,1])))
                for lag in (1,10,100):
                    rows.append(dict(fit=fit,chain=chain,statistic='kappa_autocorrelation',parameter=f'lag_{lag}',value=float(np.corrcoef(k[:-lag],k[lag:])[0,1])))
                for half,part in enumerate(np.array_split(k,2)):
                    rows.append(dict(fit=fit,chain=chain,statistic='kappa_half_mean',parameter=f'half_{half}',value=float(part.mean())))
                rows.append(dict(fit=fit,chain=chain,statistic='decay_correlation_condition_number',parameter='loglam_kappa_logsigma2',value=float(np.linalg.cond(corr))))
            # Use medians only to describe time-scale regimes; not a Fisher information estimate.
            lam=float(np.median(samples['lam']));k=float(np.median(samples['kappa']));a=lam*np.exp(-k*c)*dt
            w=(a*np.exp(-a))**2
            center=float(np.average(c,weights=w)); variance=float(np.average((c-center)**2,weights=w))
            design.append(dict(fit=fit,transitions=len(c),practice_min=float(c.min()),practice_max=float(c.max()),
                practice_distinct=len(np.unique(c)),rate_design_rank=int(np.linalg.matrix_rank(np.column_stack([np.ones(len(c)),c]))),
                median_lambda=lam,median_kappa=k,a_q05=float(np.quantile(a,.05)),a_q50=float(np.median(a)),a_q95=float(np.quantile(a,.95)),
                fraction_a_below_001=float(np.mean(a<.01)),fraction_a_above_10=float(np.mean(a>10)),
                phi_sensitivity_weighted_C=center,phi_sensitivity_weighted_C_variance=variance,
                interpretation='Local phi-Jacobian design proxy at marginal medians, not observed/full Fisher information or a structural-identification test.'))
            if kind=='nested_synthetic':
                truth=json.loads((source/f'{stem}-truth.json').read_text())['parameters']
                for scale in (.1,1.,4.):
                    oracle.append(dict(cohort=cohort,prior_shape=2,prior_scale=scale,truth=truth['tau2_gamma'],
                        prior_probability_below_truth=float(invgamma.cdf(truth['tau2_gamma'],a=2,scale=scale)),
                        realized_slope_variance=float(np.var(truth['gamma'],ddof=1)),
                        **hierarchy_oracle(truth['gamma'],truth['g0'],scale)))
    write_csv(out/'decay-chain-characterization.csv',rows);write_csv(out/'decay-design-characterization.csv',design)
    write_csv(out/'hierarchy-oracle.csv',oracle)


def curate(source, out, baseline):
    if out.exists():raise FileExistsError(out)
    if json.loads((source/'status.json').read_text())['state']!='complete':raise ValueError('Study not complete.')
    results=read(source/'results.csv'); rows=read(source/'diagnostics.csv');replay=read(source/'baseline-replay.csv')
    metadata=json.loads((source/'chain-metadata.json').read_text())
    if len(results)!=12 or len(metadata)!=48 or len(replay)!=64 or not all(r['identical']=='True' for r in replay):
        raise ValueError('Incomplete study or failed replay.')
    for r in results:
        group=[d for d in rows if d['fit']==r['fit']]
        if len(group)!=142 or len({d['parameter'] for d in group})!=142:raise ValueError('Monitor set changed.')
        passed=all(np.isfinite([float(d['rhat']),float(d['ess_bulk']),float(d['ess_tail'])]).all()
                   and float(d['rhat'])<1.01 and min(float(d['ess_bulk']),float(d['ess_tail']))>=400 for d in group)
        if passed!=(r['passed']=='True'):raise ValueError('Gate differs.')
    for m in metadata:
        s=m['sampling']
        if s['adaptation_draws']!=4000 or s['innovation_adaptations']!=4000 or s['coefficient_moves']!=8000:
            raise ValueError('Move/adaptation metadata differs.')
    out.mkdir(parents=True)
    atomic_json(out/'raw-artifact-manifest.json', {str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.rglob('*')) if p.is_file()})
    for name in ('protocol.json','plan.json','provenance.json','status.json','results.csv','diagnostics.csv','recovery.csv','chain-metadata.json','baseline-replay.csv'):
        shutil.copy2(source/name,out/name)
    write_csv(out/'failed-parameters.csv',[r for r in rows if r['passed']!='True'])
    comparisons=[];parameters=[];events=[]
    for r in results:
        ref=next(v for v in results if v['kind']==r['kind'] and v['cohort']==r['cohort'] and v['condition']==('recorded' if r['kind']=='order' else 'original'))
        if r['fit']==ref['fit']:continue
        old={d['parameter']:d for d in rows if d['fit']==ref['fit']}
        group=[d for d in rows if d['fit']==r['fit']]
        if set(old)!={d['parameter'] for d in group}:raise ValueError('Unequal monitors.')
        segments={m['execution_segment'] for m in metadata if m['fit'] in (r['fit'],ref['fit'])}
        if len(segments)!=1:raise ValueError('Unmatched CPU segments.')
        comparisons.append(dict(fit=r['fit'],reference=ref['fit'],analysis=r['kind'],both_fit_gates=r['passed']=='True' and ref['passed']=='True',
            delta_log_loss=float(r['log_loss'])-float(ref['log_loss']),delta_brier=float(r['brier'])-float(ref['brier']),
            cpu_ratio=float(r['cpu'])/float(ref['cpu']),min_ess_cpu_ratio=float(r['min_ess_per_cpu'])/float(ref['min_ess_per_cpu']),
            execution_segment=next(iter(segments)),timing_interpretation='Different order/target; descriptive computational comparison, not sampler efficiency improvement.'))
        for d in group:
            b=old[d['parameter']];diff=float(d['mean'])-float(b['mean']);se=np.hypot(float(d['mcse_mean']),float(b['mcse_mean']))
            parameters.append(dict(fit=r['fit'],reference=ref['fit'],parameter=d['parameter'],mean_difference=diff,combined_mcse=se,
                descriptive_mcse_units=abs(diff)/se if se else None,reference_sd_units=diff/float(b['sd']),
                both_parameter_gates=d['passed']=='True' and b['passed']=='True',both_fit_gates=r['passed']=='True' and ref['passed']=='True',
                target_note='Terminal review can change under permutation' if r['kind']=='order' and d['parameter'].startswith('psi_last') else 'Same parameter, changed data order or prior'))
        if r['kind']=='order':
            mapping=json.loads((source/f"real-{r['cohort']}-row-mapping.json").read_text())
            p=np.load(source/f"{r['fit']}-predictions.npz");base=np.load(source/f"{ref['fit']}-predictions.npz")
            probabilities=p['probability'].mean(0);refprob=base['probability'].mean(0);offset=0
            for m in mapping:
                cut=m['cut'];n=len(m['recorded_rows'])-cut
                order=m['longest_positions'] if r['condition']=='longest' else m['reverse_positions']
                for j,pos in enumerate(order[cut:]):
                    original=offset+pos-cut;current=offset+j
                    if p['outcome'][current]!=base['outcome'][original]:raise ValueError('Event alignment differs.')
                    events.append(dict(fit=r['fit'],dense_item=m['dense_item'],recorded_review_index=pos,outcome=int(p['outcome'][current]),
                        recorded_probability=float(refprob[original]),candidate_probability=float(probabilities[current]),
                        delta_probability=float(probabilities[current]-refprob[original])))
                offset+=n
    write_csv(out/'sensitivity-comparison.csv',comparisons);write_csv(out/'parameter-sensitivity.csv',parameters)
    write_csv(out/'aligned-order-predictions.csv',events)
    bottleneck(baseline,out)
    # Compare the two feasible candidates directly as well as recorded references.
    pair=[]
    for cohort in ('0','1'):
        a=f'real-{cohort}-longest-order-longest';b=f'real-{cohort}-reverse-order-reverse'
        old={d['parameter']:d for d in rows if d['fit']==a}
        for d in [d for d in rows if d['fit']==b]:
            v=old[d['parameter']];delta=float(d['mean'])-float(v['mean']);se=np.hypot(float(d['mcse_mean']),float(v['mcse_mean']))
            pair.append(dict(cohort=cohort,parameter=d['parameter'],reverse_minus_longest=delta,combined_mcse=se,
                descriptive_mcse_units=abs(delta)/se if se else None,
                conservative_mcse_bound=float(d['mcse_mean'])+float(v['mcse_mean']),
                longest_sd_units=delta/float(v['sd']),
                both_fit_gates=all(r['passed']=='True' for r in results if r['fit'] in (a,b))))
    write_csv(out/'feasible-pair-sensitivity.csv',pair)
    plot(out,results,rows)
    trace_plots(source,out,results)
    measured_report(out,results,rows,comparisons,pair)
    atomic_json(out/'curation.json',dict(raw_source=str(source.resolve()),baseline=str(baseline.resolve()),
        raw_results_sha256=hashlib.sha256((source/'results.csv').read_bytes()).hexdigest(),
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        raw_arrays_remain_local=True,baseline_arrays_identical=64,all_monitor_sets_preserved=True))


def trace_plots(source, out, results):
    import matplotlib.pyplot as plt
    for cohort in ('0','1'):
        group=[r for r in results if r['cohort']==cohort]
        fig,axes=plt.subplots(len(group),3,figsize=(13,2*len(group)),squeeze=False)
        for row,r in enumerate(group):
            with np.load(source/f"{r['fit']}-draws.npz") as samples:
                for col,key in enumerate(('lam','kappa','tau2_gamma')):
                    for chain in range(4):
                        axes[row,col].plot(np.arange(4000,8000),samples[key][chain],lw=.4,alpha=.65,label=f'chain {chain+1}')
                    axes[row,col].set_title(f"{r['kind']} / {r['condition']}: {key}",fontsize=8)
                    if key!='kappa':axes[row,col].set_yscale('log')
                    axes[row,col].tick_params(labelsize=7)
        axes[0,0].legend(fontsize=6,ncol=4)
        fig.suptitle(f'Cohort {int(cohort)+1}: all retained chains, no thinning; warmup excluded',fontsize=10)
        fig.tight_layout();fig.savefig(out/f'cohort-{cohort}-retained-traces.png',dpi=140);plt.close(fig)


def measured_report(out, results, rows, comparisons, pair):
    lines=['# Task 3 completion sensitivity: measured results','',
        'All 12 fits / 48 chains finished at the prespecified 8000/4000 budget with no thinning. '
        'Four unchanged references replay all 64 retained/available-warmup parameter arrays exactly. '
        'Each fit retains all 142 original scalar gates. Blue figure markers indicate whole-fit passes; red marks failures.','',
        '| Analysis / cohort | Condition | Max Rhat | Min bulk / tail ESS | Failures / 142 | CPU s | Log loss | Brier | Gate |',
        '|---|---|---:|---:|---:|---:|---:|---:|---|']
    for r in results:
        lines.append(f"| {r['kind']} / {int(r['cohort'])+1} | {r['condition']} | {float(r['max_rhat']):.6f} | {float(r['min_ess_bulk']):.2f} / {float(r['min_ess_tail']):.2f} | {r['failed_parameters']} | {float(r['cpu']):.1f} | {float(r['log_loss']):.6f} | {float(r['brier']):.6f} | {'PASS' if r['passed']=='True' else 'FAIL'} |")
    lines+=['','## Feasible-order posterior impact','',
        'Changed priors/orders define sensitivity targets, not sampler benefits. Recorded order is a reproducible reference, not authenticated truth. '
        'MCSE units are descriptive screens from quadrature of marginal mean MCSEs (shared seeds can correlate fits), not significance tests or multiplicity-adjusted evidence. '
        'Whole-fit failures limit every downstream interpretation. Terminal-state targets can refer to different final events.','',
        '| Cohort | Parameter | Reverse minus longest mean | Combined MCSE | Descriptive MCSE units | Both full gates |',
        '|---|---|---:|---:|---:|---|']
    for d in pair:
        if d['parameter'] in ('lam','kappa','sigma2','gamma_mean','tau2_gamma','population_level'):
            lines.append(f"| {int(d['cohort'])+1} | {d['parameter']} | {d['reverse_minus_longest']:+.5f} | {d['combined_mcse']:.5f} | {d['descriptive_mcse_units']:.2f} | {d['both_fit_gates']} |")
    lines+=['','Recorded-reference changes for all parameters are in `parameter-sensitivity.csv`; direct feasible-pair changes for all 142 monitors are in `feasible-pair-sensitivity.csv`. '
        'Event-aligned predictions retain every held-out outcome in `aligned-order-predictions.csv`. Two candidate orders are a lower bound on sensitivity, not exhaustive uncertainty.','',
        '## Hierarchy MODEL sensitivity and fixed-truth recovery','',
        'Only the shared hierarchy IG scale changes; shape remains 2, affecting all three variances. '
        'The original scale is 1, lower is 0.1, higher is 4. These alternatives were frozen before new sampling and are not selected to obtain coverage. '
        'The complete recovery file retains all dependent truths, all failures and intervals. This is not prior-drawn SBC.','',
        '| Cohort | Prior scale | Lag variance mean | 95% interval | Parameter Rhat | Bulk / tail ESS | Interval contains .04 | Whole fit |',
        '|---|---:|---:|---|---:|---:|---|---|']
    for r in results:
        if r['kind']!='MODEL':continue
        d=next(d for d in rows if d['fit']==r['fit'] and d['parameter']=='tau2_gamma')
        lines.append(f"| {int(r['cohort'])+1} | {r['prior_scale']} | {float(d['mean']):.5f} | [{float(d['low']):.5f}, {float(d['high']):.5f}] | {float(d['rhat']):.6f} | {float(d['ess_bulk']):.1f} / {float(d['ess_tail']):.1f} | {float(d['low'])<=.04<=float(d['high'])} | {r['passed']} |")
    recovered=read(out/'recovery.csv')
    lines+=['','| Synthetic fit | Truth-containing intervals / 142 | Parameter gates / 142 | Whole fit |','|---|---:|---:|---|']
    for r in results:
        if r['kind']!='MODEL':continue
        group=[d for d in recovered if d['fit']==r['fit']]
        lines.append(f"| {r['fit']} | {sum(d['covered']=='True' for d in group)} / {len(group)} | {sum(d['passed']=='True' for d in group)} / {len(group)} | {r['passed']} |")
    lines+=['','The oracle supplies exact realized slopes and exact g0 and is a different observation control, not a sampling result. '
        'Its closed-form conditional and prior mass are in `hierarchy-oracle.csv`. '
        'Few groups, near-zero suppression under IG(2,1), and sensitivity to the shared hierarchy scale limit population-variance recovery; '
        'convergence is not a guarantee of truth-containing intervals.','',
        '## Prediction and computation comparisons','',
        '| Fit | Delta log loss versus reference | Delta Brier | CPU ratio | Min ESS/CPU ratio | Both fit gates |','|---|---:|---:|---:|---:|---|']
    for d in comparisons:
        lines.append(f"| {d['fit']} | {d['delta_log_loss']:+.6f} | {d['delta_brier']:+.6f} | {d['cpu_ratio']:.3f} | {d['min_ess_cpu_ratio']:.3f} | {d['both_fit_gates']} |")
    lines+=['','These are descriptive comparisons in one execution segment using the inherited approximate prequential filter. '
        'They do not establish learner efficacy, component benefit or superiority of any prior/order. Timing is one realization per fit.','',
        '## Failed parameters','']
    for r in results:
        failed=[d['parameter'] for d in rows if d['fit']==r['fit'] and d['passed']!='True']
        lines.append(f"- {r['fit']}: "+(', '.join(failed) if failed else 'none')+'.')
    lines+=['','![Completion sensitivity](completion-sensitivity.png)','',
        'Raw source/private mappings remain under the ignored run directory; aggregate diagnostics retain all monitors and failures. '
        'See the final Task 3 report for the decay/practice characterization, identifiability derivation, earlier controlled ablations and scientific limits.','']
    (out/'report.md').write_text('\n'.join(lines))


def plot(out,results,rows):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for cohort in ('0','1'):
        for j,condition in enumerate(('original','lower_scale','higher_scale')):
            fit=f'nested_synthetic-{cohort}-MODEL-{condition}'
            d=next(r for r in rows if r['fit']==fit and r['parameter']=='tau2_gamma')
            mean=float(d['mean']);low=float(d['low']);high=float(d['high']);y=int(cohort)*4+j
            axes[0].errorbar(mean,y,xerr=np.array([[mean-low],[high-mean]]),fmt='o',color='C0' if next(r for r in results if r['fit']==fit)['passed']=='True' else 'C3')
            axes[0].text(high*1.05,y,f'cohort {int(cohort)+1} / {condition}',fontsize=8,va='center')
    axes[0].axvline(.04,color='black',ls='--',label='Fixed generating variance .04')
    axes[0].set_xscale('log');axes[0].set_yticks([]);axes[0].set_title('MODEL sensitivity: slope variance, 95% intervals');axes[0].legend(fontsize=8)
    real=[r for r in results if r['kind']=='order']
    axes[1].barh([r['fit'] for r in real],[float(r['min_ess_bulk']) for r in real],color=['C0' if r['passed']=='True' else 'C3' for r in real])
    axes[1].axvline(400,color='black',ls='--');axes[1].tick_params(axis='y',labelsize=7)
    axes[1].set_title('Order sensitivity minimum bulk ESS; pass blue');axes[1].set_xlabel('Bulk ESS')
    fig.tight_layout();fig.savefig(out/'completion-sensitivity.png',dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--baseline',type=Path,default=ROOT/'.task3-runs/task3-coefficient-investigation');args=p.parse_args()
    curate(args.source,args.out,args.baseline)


if __name__=='__main__':main()
