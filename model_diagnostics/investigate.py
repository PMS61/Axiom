"""Task 3 bounded dynamics ladder and matched nested-design recovery pilot.

python -m model_diagnostics.investigate --out .task3-runs/NEW
Uses the saved, outcome-independent Anki selections; never overwrites a run.
"""
import argparse
import ast
import csv
import shutil
import platform
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from .analysis import atomic_json, diagnostics, metrics, plots, scalar_draws, truth_values, write_csv
from .anki import ROOT, ANKI_BOUNDS
from .innovation import InnovationModel


def nested_simulate(template, seed):
    """Conditional-design simulation: actual nested IDs/gaps/lengths, NEW outcomes.

    Gaps are treated as exogenous fixed design. Real scheduling may be endogenous;
    this checks recovery conditional on design, not a simulator of Anki's scheduler.
    """
    rng=np.random.default_rng(seed)
    users=max(s['user'] for s in template)+1
    items=max(s['item'] for s in template)+1
    truth=dict(b0=.2,g0=.7,lam=.35,kappa=.6,sigma2=.3,tau2_alpha=.25,tau2_beta=.36,tau2_gamma=.04,
               alpha=rng.normal(0,.5,users),beta=rng.normal(0,.6,items),gamma=rng.normal(.7,.2,users))
    seqs,states=[],[]
    for s in template:
        r,psi,successes=[],[],0
        for t,dt in enumerate(s['dt']):
            mu=truth['b0']+truth['alpha'][s['user']]+truth['beta'][s['item']]+truth['gamma'][s['user']]*(r[-1] if r else 0)
            rate=truth['lam']*np.exp(-truth['kappa']*np.log1p(successes))
            phi=np.exp(-rate*dt)
            q=truth['sigma2']*(-np.expm1(-2*rate*dt))/(2*rate)
            state=rng.normal(mu,1) if t==0 else rng.normal(phi*psi[-1]+(1-phi)*mu,np.sqrt(q))
            outcome=int(rng.random()<.5*(1+np.tanh(state/2)))
            r.append(outcome);psi.append(state);successes+=outcome
        seqs.append(dict(user=s['user'],item=s['item'],dt=np.array(s['dt']).copy(),r=np.array(r),x=np.array([0]+r[:-1])))
        states.append(psi)
    return seqs,truth,states


def nested_summaries(draws,samples,seqs):
    # Identified likelihood combinations for nested cards; retain all original monitors too.
    for user in sorted({s['user'] for s in seqs}):
        cards=sorted({s['item'] for s in seqs if s['user']==user})
        mean=samples['beta'][:,:,cards].mean(axis=2)
        draws[f'user_level[{user}]']=samples['b0']+samples['alpha'][:,:,user]+mean
        for card in cards:
            draws[f'within_user_card[{card}]']=samples['beta'][:,:,card]-mean
    return draws


def job(train,test,control,interweave,seed,chain,iterations,warmup,fixed):
    variant='no_dynamics' if control=='static' else 'full'
    model=InnovationModel(train,variant=variant,sampler='joint_blocked',seed=seed,chain=chain,
                          dt_bounds=ANKI_BOUNDS,fixed=fixed,interweave=interweave)
    start=time.process_time(); model.run(iterations,warmup)
    cpu=time.process_time()-start
    start=time.process_time();p=np.concatenate(model.predict(test))
    return dict(samples=model.samples,warmup=model.warmup,prediction=p,cpu=cpu,
                prediction_cpu=time.process_time()-start,seed=seed,sampling=model.sampling_info)


def verified_json(path, data):
    """Retain a saved dataset byte-for-byte when deterministic regeneration agrees."""
    if not path.exists():
        atomic_json(path,data)
        return
    check=path.with_name(path.stem+'-regeneration-check.json')
    atomic_json(check,data)
    try:
        if path.read_bytes()!=check.read_bytes(): raise ValueError(f'Regenerated artifact differs: {path.name}')
    finally: check.unlink()


def read_checkpoint_csv(path):
    if not path.exists(): return []
    def parse(value):
        if value in ('True','False'): return value=='True'
        try: return json.loads(value)
        except (ValueError,TypeError): return value
    with path.open() as f: return [{k:parse(v) for k,v in row.items()} for row in csv.DictReader(f)]


def restore_checkpoint(source,out,config):
    if json.loads((source/'protocol.json').read_text())!=config: raise ValueError('Checkpoint protocol differs.')
    prior=json.loads((source/'provenance.json').read_text())
    for name in ('model_diagnostics/model.py','model_diagnostics/innovation.py','model_diagnostics/data.py','model_diagnostics/analysis.py','model_diagnostics/anki.py'):
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=prior['source_hashes'][name]:
            raise ValueError(f'Checkpoint sampler source changed: {name}')
    for package,version in prior['packages'].items():
        if importlib.metadata.version(package)!=version: raise ValueError(f'Checkpoint package changed: {package}')
    original=source/'source-snapshot/investigate.py'
    if not original.exists(): original=ROOT/'.task3-runs/task3-resume-source/investigate-before-resume.py'
    if hashlib.sha256(original.read_bytes()).hexdigest()!=prior['source_hashes']['model_diagnostics/investigate.py']:
        raise ValueError('Original orchestration source fingerprint differs.')
    old_functions={n.name:ast.dump(n) for n in ast.parse(original.read_text()).body if isinstance(n,ast.FunctionDef)}
    current_functions={n.name:ast.dump(n) for n in ast.parse(Path(__file__).read_text()).body if isinstance(n,ast.FunctionDef)}
    if any(old_functions[n]!=current_functions[n] for n in ('nested_simulate','nested_summaries','job')):
        raise ValueError('Numerical generation/job functions differ from checkpoint.')
    results=read_checkpoint_csv(source/'results.csv')
    rows=read_checkpoint_csv(source/'diagnostics.csv')
    recovery=read_checkpoint_csv(source/'recovery.csv')
    metadata=json.loads((source/'chain-metadata.json').read_text())
    for r in results:
        label=r['fit']
        if len([m for m in metadata if m['fit']==label])!=4 or not any(d['fit']==label for d in rows):
            raise ValueError(f'Incomplete checkpoint: {label}')
        with np.load(source/f'{label}-draws.npz') as arrays:
            if any(v.shape[:2]!=(4,config['iterations']-config['warmup']) for v in arrays.values()):
                raise ValueError(f'Invalid chain/draw axes: {label}')
    manifest={}
    for path in source.iterdir():
        if path.is_file() and path.name not in ('status.json','provenance.json'):
            shutil.copy2(path,out/path.name)
            manifest[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    shutil.copy2(source/'provenance.json',out/'previous-provenance.json')
    shutil.copy2(original,out/'investigate-before-resume.py')
    atomic_json(out/'resume-origin.json',dict(source=str(source.resolve()),completed_fits=[r['fit'] for r in results],
                prior_artifact_hashes=manifest,note='All completed NPZs are copied unchanged. Aggregate CSVs continue in the new directory. Original directory is untouched.'))
    for m in metadata:m['execution_segment']='before_interruption'
    return results,rows,recovery,metadata


def execute(out,iterations=8000,warmup=4000,resume_from=None):
    out.mkdir(parents=True,exist_ok=False)
    config=dict(iterations=iterations,warmup=warmup,chains=4,workers=4,seed=20261028,
                base_sampler='joint_blocked',mh_steps=5,cohorts=2,
                controls=['static','fixed_dynamics','free_diffusion','free_decay_diffusion','full'],
                interweave=[False,True],synthetic_seed=20261101,
                fixed_real=dict(lam=.05,sigma2=.5,kappa=0),
                note='CLI diagnostic pilot; fixed controls change the fitted model; no prior tuning or SBC.')
    results,rows,recovery,metadata=restore_checkpoint(resume_from,out,config) if resume_from else ([],[],[],[])
    atomic_json(out/'protocol.json',config)
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'model_diagnostics').glob('*.py')}
    snapshot=out/'source-snapshot';snapshot.mkdir(exist_ok=True)
    for name in hashes:(snapshot/Path(name).name).write_bytes((ROOT/name).read_bytes())
    atomic_json(out/'provenance.json',dict(revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_hashes=hashes,
                packages={p:importlib.metadata.version(p) for p in ('numpy','scipy','arviz','polyagamma','pyarrow')},
                execution_segment='after_interruption' if resume_from else 'initial',platform=platform.platform(),
                resume_from=str(resume_from.resolve()) if resume_from else None,
                source_run='task3-anki-joint_blocked',source_protocol_sha256=hashlib.sha256((ROOT/'experiments/task3-anki-joint_blocked.json').read_bytes()).hexdigest()))
    completed={r['fit'] for r in results}
    started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=4) as pool:
        for cohort in range(2):
            source=ROOT/'.task3-runs/task3-anki-joint_blocked'
            template=json.loads((source/f'anki-{cohort}-dataset.json').read_text())
            cuts=[m['training_events'] for m in json.loads((source/f'anki-{cohort}-selection.json').read_text())['mapping']]
            for kind in ('real','nested_synthetic'):
                seqs=[{k:np.array(v) if isinstance(v,list) else v for k,v in s.items()} for s in template]
                truth=None
                if kind=='nested_synthetic':
                    seqs,truth,states=nested_simulate(seqs,config['synthetic_seed']+cohort)
                    verified_json(out/f'{kind}-{cohort}-truth.json',dict(parameters=truth,latent_states=states))
                verified_json(out/f'{kind}-{cohort}-dataset.json',seqs)
                verified_json(out/f'{kind}-{cohort}-split.json',dict(cuts=cuts,dataset_sha256=hashlib.sha256((out/f'{kind}-{cohort}-dataset.json').read_bytes()).hexdigest()))
                train=[{k:v[:cut].copy() if isinstance(v,np.ndarray) else v for k,v in s.items()} for s,cut in zip(seqs,cuts)]
                test=[{k:v[cut:].copy() if isinstance(v,np.ndarray) else v for k,v in s.items()} for s,cut in zip(seqs,cuts)]
                y=np.concatenate([s['r'] for s in test])
                fixed_values=config['fixed_real'] if truth is None else {k:truth[k] for k in ('lam','sigma2','kappa')}
                for control in config['controls']:
                    fixed=({} if control in ('static','full') else dict(fixed_values) if control=='fixed_dynamics'
                           else {k:fixed_values[k] for k in ('lam','kappa')} if control=='free_diffusion' else {'kappa':fixed_values['kappa']})
                    # Static/fixed controls have no innovation parameter move; do not duplicate.
                    for interweave in ([False] if control in ('static','fixed_dynamics') else [False,True]):
                        label=f'{kind}-{cohort}-{control}-'+('interwoven' if interweave else 'centered')
                        if label in completed: continue
                        futures=[pool.submit(job,train,test,control,interweave,config['seed']+1009*cohort+31+7919*c,c,iterations,warmup,fixed) for c in range(4)]
                        chains=[f.result() for f in futures]
                        samples={k:np.stack([c['samples'][k] for c in chains]) for k in chains[0]['samples']}
                        warm={k:np.stack([c['warmup'][k] for c in chains]) for k in chains[0]['warmup']}
                        np.savez_compressed(out/f'{label}-draws.npz',**samples)
                        np.savez_compressed(out/f'{label}-warmup.npz',**warm)
                        variant='no_dynamics' if control=='static' else 'full'
                        draws=nested_summaries(scalar_draws(samples,variant),samples,train)
                        for k in fixed: draws.pop(k,None)
                        cpu=sum(c['cpu'] for c in chains)
                        diag=diagnostics(draws,cpu)
                        for r in diag: rows.append(dict(fit=label,**r))
                        pred=np.mean([c['prediction'] for c in chains],axis=0)
                        write_csv(out/f'{label}-predictions.csv',[dict(event=j,outcome=int(a),probability=float(b)) for j,(a,b) in enumerate(zip(y,pred))])
                        result=dict(fit=label,kind=kind,cohort=cohort,control=control,interweave=interweave,
                                    passed=all(r['passed'] for r in diag),max_rhat=max(r['rhat'] for r in diag),
                                    min_ess_bulk=min(r['ess_bulk'] for r in diag),min_ess_tail=min(r['ess_tail'] for r in diag),
                                    failed_parameters=sum(not r['passed'] for r in diag),cpu=cpu,
                                    min_ess_per_cpu=min(r['ess_per_cpu_second'] for r in diag),
                                    prediction_cpu=sum(c['prediction_cpu'] for c in chains),**metrics(y,pred))
                        results.append(result)
                        for c,ch in enumerate(chains): metadata.append(dict(fit=label,chain=c,seed=ch['seed'],cpu=ch['cpu'],sampling=ch['sampling'],execution_segment='after_interruption' if resume_from else 'initial'))
                        if truth is not None and control=='full':
                            expected=truth_values(truth)
                            expected.update({f'psi_last[{j}]': states[j][cut-1] for j,cut in enumerate(cuts)})
                            for user in sorted({s['user'] for s in train}):
                                cards=sorted({s['item'] for s in train if s['user']==user})
                                mean=np.mean(truth['beta'][cards])
                                expected[f'user_level[{user}]']=truth['b0']+truth['alpha'][user]+mean
                                for card in cards: expected[f'within_user_card[{card}]']=truth['beta'][card]-mean
                            for r in diag:
                                if r['parameter'] in expected:
                                    value=expected[r['parameter']]
                                    recovery.append(dict(fit=label,parameter=r['parameter'],truth=value,error=r['mean']-value,
                                                         covered=r['low']<=value<=r['high'],**r))
                        plots(out,label,draws,warm)
                        write_csv(out/'results.csv',results);write_csv(out/'diagnostics.csv',rows);write_csv(out/'recovery.csv',recovery)
                        atomic_json(out/'chain-metadata.json',metadata)
                        atomic_json(out/'status.json',dict(state='running',fits=len(results),total=32,elapsed_seconds=time.perf_counter()-started))
                        print(label,result['passed'],round(result['max_rhat'],4),round(result['min_ess_bulk'],1),flush=True)
    atomic_json(out/'status.json',dict(state='complete',fits=len(results),total=32,elapsed_seconds=time.perf_counter()-started))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--iterations',type=int,default=8000);p.add_argument('--warmup',type=int,default=4000)
    p.add_argument('--resume-from',type=Path,help='Copy a verified incomplete checkpoint into a NEW output directory.')
    args=p.parse_args()
    if args.out.exists(): p.error('Output exists; choose a new directory.')
    if not 50<=args.warmup<args.iterations or args.iterations-args.warmup<100: p.error('Invalid warmup/draw budget.')
    execute(args.out,args.iterations,args.warmup,args.resume_from)


if __name__=='__main__':main()
