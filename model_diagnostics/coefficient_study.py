"""Bounded full-model comparison; copies exact saved datasets, never overwrites.

Four matched datasets x three kernels x four chains, 8000/4000, no thinning.
All three kernels are rerun in one execution segment for comparable CPU costs.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from .analysis import atomic_json, diagnostics, metrics, plots, scalar_draws, truth_values, write_csv
from .anki import ROOT, ANKI_BOUNDS
from .coefficient_innovation import CoefficientInnovationModel
from .innovation import InnovationModel
from .investigate import nested_summaries, recovery_record


def job(train, test, kernel, seed, chain, iterations, warmup):
    cls = CoefficientInnovationModel if kernel == 'coefficient' else InnovationModel
    model = cls(train, variant='full', sampler='joint_blocked', seed=seed, chain=chain,
                dt_bounds=ANKI_BOUNDS, interweave=kernel != 'centered')
    start = time.process_time(); model.run(iterations, warmup)
    cpu = time.process_time()-start
    start = time.process_time(); prediction = np.concatenate(model.predict(test))
    return dict(samples=model.samples, warmup=model.warmup, prediction=prediction,
                cpu=cpu, prediction_cpu=time.process_time()-start, seed=seed, sampling=model.sampling_info)


def execute(out, source, iterations=8000, warmup=4000):
    if json.loads((source/'status.json').read_text())['state'] != 'complete':
        raise ValueError('Baseline source must be complete.')
    if (iterations, warmup) != (8000, 4000):
        raise ValueError('The prespecified matched budget is 8000/4000.')
    out.mkdir(parents=True, exist_ok=False)
    protocol = dict(iterations=iterations, warmup=warmup, chains=4, workers=4, seed=20261028,
                    kernels=['centered','interwoven','coefficient'], kinds=['real','nested_synthetic'], cohorts=2,
                    mh_steps=5, coefficient_moves_per_sweep=1, model='full', fixed={},
                    gate='Every scalar: rank folded split Rhat <1.01 AND bulk/tail ESS >=400; undefined fails.',
                    note='No adaptive stopping/extensions, prior changes, dropped monitors/chains, or reordered real data. Fixed-truth conditional-design recovery, not SBC.')
    atomic_json(out/'protocol.json', protocol)
    snapshot=out/'source-snapshot'; snapshot.mkdir()
    hashes={}
    for path in (ROOT/'model_diagnostics').glob('*.py'):
        hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        shutil.copy2(path, snapshot/path.name)
    artifacts={}
    for kind in protocol['kinds']:
        for cohort in range(2):
            stem=f'{kind}-{cohort}'
            for suffix in ['dataset.json','split.json']+(['truth.json'] if kind=='nested_synthetic' else []):
                path=source/f'{stem}-{suffix}'
                shutil.copy2(path, out/path.name)
                artifacts[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    atomic_json(out/'provenance.json',dict(revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                source_hashes=hashes, dataset_artifact_hashes=artifacts, baseline_source=str(source.resolve()),
                baseline_provenance_sha256=hashlib.sha256((source/'provenance.json').read_bytes()).hexdigest(),
                packages={p:importlib.metadata.version(p) for p in ('numpy','scipy','arviz','polyagamma','pyarrow')},
                platform=platform.platform(), execution_segment=out.name))
    results=[]; rows=[]; recovery=[]; metadata=[]; regressions=[]
    started=time.perf_counter()
    atomic_json(out/'status.json',dict(state='running',fits=0,total=12))
    with ProcessPoolExecutor(max_workers=4) as pool:
        for kind in protocol['kinds']:
            for cohort in range(2):
                stem=f'{kind}-{cohort}'
                seqs=[{k:np.array(v) if isinstance(v,list) else v for k,v in s.items()} for s in json.loads((out/f'{stem}-dataset.json').read_text())]
                split=json.loads((out/f'{stem}-split.json').read_text())
                if split['dataset_sha256'] != artifacts[f'{stem}-dataset.json']:
                    raise ValueError('Saved split/dataset fingerprint differs.')
                cuts=split['cuts']
                train=[{k:v[:cut].copy() if isinstance(v,np.ndarray) else v for k,v in s.items()} for s,cut in zip(seqs,cuts)]
                test=[{k:v[cut:].copy() if isinstance(v,np.ndarray) else v for k,v in s.items()} for s,cut in zip(seqs,cuts)]
                y=np.concatenate([s['r'] for s in test])
                expected={}
                if kind=='nested_synthetic':
                    truth=json.loads((out/f'{stem}-truth.json').read_text())
                    parameters=truth['parameters']; expected=truth_values(parameters)
                    expected.update({f'psi_last[{j}]':truth['latent_states'][j][cut-1] for j,cut in enumerate(cuts)})
                    for user in sorted({s['user'] for s in train}):
                        cards=sorted({s['item'] for s in train if s['user']==user})
                        mean=np.mean(np.asarray(parameters['beta'])[cards])
                        expected[f'user_level[{user}]']=parameters['b0']+parameters['alpha'][user]+mean
                        for card in cards: expected[f'within_user_card[{card}]']=parameters['beta'][card]-mean
                for kernel in protocol['kernels']:
                    label=f'{stem}-full-{kernel}'
                    futures=[pool.submit(job,train,test,kernel,protocol['seed']+1009*cohort+31+7919*c,c,iterations,warmup) for c in range(4)]
                    chains=[f.result() for f in futures]
                    samples={k:np.stack([c['samples'][k] for c in chains]) for k in chains[0]['samples']}
                    warm={k:np.stack([c['warmup'][k] for c in chains]) for k in chains[0]['warmup']}
                    np.savez_compressed(out/f'{label}-draws.npz',**samples)
                    np.savez_compressed(out/f'{label}-warmup.npz',**warm)
                    if kernel!='coefficient':
                        for suffix,arrays in [('draws',samples),('warmup',warm)]:
                            with np.load(source/f'{label}-{suffix}.npz') as saved:
                                if set(saved.files)!=set(arrays): raise ValueError('Baseline parameter set differs.')
                                for key in arrays:
                                    identical=bool(np.array_equal(arrays[key],saved[key]))
                                    regressions.append(dict(fit=label,phase=suffix,parameter=key,identical=identical))
                                    if not identical: raise ValueError(f'Baseline replay differs: {label}/{suffix}/{key}')
                    draws=nested_summaries(scalar_draws(samples,'full'),samples,train)
                    cpu=sum(c['cpu'] for c in chains); diag=diagnostics(draws,cpu)
                    rows.extend(dict(fit=label,**r) for r in diag)
                    pred=np.mean([c['prediction'] for c in chains],axis=0)
                    write_csv(out/f'{label}-predictions.csv',[dict(event=j,outcome=int(a),probability=float(b)) for j,(a,b) in enumerate(zip(y,pred))])
                    # Preserve chain predictions too for prediction-stability inspection.
                    np.savez_compressed(out/f'{label}-predictions.npz',probability=np.stack([c['prediction'] for c in chains]),outcome=y)
                    result=dict(fit=label,kind=kind,cohort=cohort,kernel=kernel,passed=all(r['passed'] for r in diag),
                                max_rhat=max(r['rhat'] for r in diag),min_ess_bulk=min(r['ess_bulk'] for r in diag),
                                min_ess_tail=min(r['ess_tail'] for r in diag),failed_parameters=sum(not r['passed'] for r in diag),
                                monitors=len(diag),cpu=cpu,min_ess_per_cpu=min(r['ess_per_cpu_second'] for r in diag),
                                prediction_cpu=sum(c['prediction_cpu'] for c in chains),**metrics(y,pred))
                    results.append(result)
                    for c,ch in enumerate(chains): metadata.append(dict(fit=label,chain=c,seed=ch['seed'],cpu=ch['cpu'],sampling=ch['sampling'],execution_segment=out.name))
                    recovery.extend(recovery_record(label,r,expected[r['parameter']]) for r in diag if r['parameter'] in expected)
                    write_csv(out/'results.csv',results); write_csv(out/'diagnostics.csv',rows); write_csv(out/'recovery.csv',recovery)
                    write_csv(out/'baseline-regression.csv',regressions); atomic_json(out/'chain-metadata.json',metadata)
                    atomic_json(out/'status.json',dict(state='running',fits=len(results),total=12,elapsed_seconds=time.perf_counter()-started))
                    print(label,result['passed'],result['max_rhat'],result['min_ess_bulk'],flush=True)
                    plots(out,label,draws,warm)
    atomic_json(out/'status.json',dict(state='complete',fits=len(results),total=12,elapsed_seconds=time.perf_counter()-started))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--source',type=Path,default=ROOT/'.task3-runs/task3-dynamics-completed')
    args=p.parse_args()
    if args.out.exists(): p.error('Output exists; choose a new directory.')
    try: execute(args.out,args.source)
    except Exception as error:
        if args.out.is_dir(): atomic_json(args.out/'status.json',dict(state='failed',message=str(error)))
        raise


if __name__=='__main__': main()
