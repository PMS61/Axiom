"""Prespecified Task 3 completion: order and MODEL prior sensitivity, CLI only.

No sampler extension or stopping rule. All fits use the tested coefficient kernel.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time
import numpy as np
from .analysis import atomic_json, diagnostics, metrics, scalar_draws, truth_values, write_csv
from .anki import ROOT, ANKI_BOUNDS
from .coefficient_innovation import CoefficientInnovationModel
from .investigate import nested_summaries, recovery_record
from .provenance_audit import timing_residual_range


def permutations(days, seconds):
    """Two bounded feasible candidates; never exchange partition membership."""
    a, b = [], []
    for day in np.unique(days):
        pos = np.flatnonzero(days == day)
        ordered = pos[np.argsort(-seconds[pos], kind='stable')]
        a.extend(ordered); b.extend(np.r_[ordered[:1], ordered[:0:-1]])
    return np.asarray(a, int), np.asarray(b, int)


def ordered_sequence(seq, rows, cut, order):
    """Keep each gap attached to its row and recompute all lag predictors."""
    if len(rows) != len(seq['r']) or sorted(order.tolist()) != list(range(len(rows))):
        raise ValueError('Permutation must preserve all source rows.')
    if set(order[:cut]) != set(range(cut)) or set(order[cut:]) != set(range(cut, len(rows))):
        raise ValueError('Permutation crosses the original split.')
    days = np.asarray([r['timestamp'] for r in rows])
    seconds = np.asarray([r['elapsed_seconds'] for r in rows], float)
    if timing_residual_range(days[order], seconds[order]) >= 1:
        raise ValueError('Candidate is not strictly feasible under the day-floor model.')
    r = np.asarray([rows[i]['y'] for i in order], int)
    dt = seconds[order] / 86400; dt[0] = 1.
    return dict(user=seq['user'], item=seq['item'], r=r, dt=dt, x=np.r_[0, r[:-1]])


def prepare_orders(source, cohort, out):
    import pyarrow.parquet as pq
    selection = ROOT/f'.task3-runs/task3-anki-joint_blocked/anki-{cohort}-selection.json'
    mapping = json.loads(selection.read_text())['mapping']
    seqs = json.loads((source/f'real-{cohort}-dataset.json').read_text())
    cuts = json.loads((source/f'real-{cohort}-split.json').read_text())['cuts']
    selected = {(m['user'], m['source_item']): [] for m in mapping}
    for partition, name in enumerate(('train_data.parquet', 'test_data.parquet')):
        table = pq.read_table(ROOT/'AnkiFirstRun'/name, columns=['user_id','item_id','timestamp','elapsed_seconds','y'])
        for index, row in enumerate(table.to_pylist()):
            key = row['user_id'], row['item_id']
            if key in selected:
                selected[key].append({**row, 'partition': partition, 'source_row': index})
    orders = {'longest': [], 'reverse': []}; records = []
    for s, m, cut in zip(seqs, mapping, cuts):
        rows = sorted(selected[m['user'], m['source_item']], key=lambda r: r['timestamp'])
        # Independently reconstruct and check the exact recorded dataset first.
        r = np.array([v['y'] for v in rows]); dt = np.array([v['elapsed_seconds'] for v in rows])/86400; dt[0] = 1.
        if (not np.array_equal(r, s['r']) or not np.array_equal(dt, s['dt'])
                or not np.array_equal(np.r_[0, r[:-1]], s['x'])
                or sum(v['partition'] == 0 for v in rows) != cut):
            raise ValueError('Raw-row reconstruction differs from recorded selection.')
        a, b = permutations(np.array([v['timestamp'] for v in rows]), np.array([v['elapsed_seconds'] for v in rows]))
        for key, order in zip(orders, (a, b)):
            orders[key].append(ordered_sequence(s, rows, cut, order))
        records.append(dict(dense_item=s['item'], dense_user=s['user'], mapping=m,
                            recorded_rows=rows, longest_positions=a, reverse_positions=b, cut=cut))
    atomic_json(out/f'real-{cohort}-row-mapping.json', records)
    shutil.copy2(selection, out/selection.name)
    return orders


def job(train, test, prior, seed, chain):
    model = CoefficientInnovationModel(train, variant='full', sampler='joint_blocked', seed=seed,
                                      chain=chain, dt_bounds=ANKI_BOUNDS, interweave=True)
    model.tau2_prior = tuple(prior)  # Explicit MODEL analysis: all three hierarchy variances.
    start = time.process_time(); model.run(8000, 4000); cpu = time.process_time()-start
    start = time.process_time(); prediction = np.concatenate(model.predict(test))
    return dict(samples=model.samples, warmup=model.warmup, prediction=prediction, cpu=cpu,
                prediction_cpu=time.process_time()-start, seed=seed, sampling=model.sampling_info)


def execute(out, source):
    if out.exists(): raise FileExistsError(out)
    if json.loads((source/'status.json').read_text())['state'] != 'complete':
        raise ValueError('Source must be complete.')
    out.mkdir(parents=True)
    protocol = dict(iterations=8000, warmup=4000, chains=4, workers=4, seed=20261028, fits=12,
        sampler='coefficient/path plus innovation, joint_blocked; unchanged algorithm',
        order_analysis=['recorded', 'longest', 'reverse'], cohorts=[0,1],
        hierarchy_MODEL_priors={'original':[2,1], 'lower_scale':[2,.1], 'higher_scale':[2,4]},
        rationale='IG shape 2 held fixed; hierarchy prior means 0.1, 1, 4 bracket scale uncertainty. Apply to ALL hierarchy variances, not selected gamma only. Chosen before new fits. Not SBC or a sampler improvement.',
        decay_analysis='Read-only chainwise effective-rate ridge and design-information characterization; no new remedy claimed.',
        gate='Every original 142 scalar monitors: finite rank folded split Rhat <1.01 and bulk/tail ESS >=400. No thinning, stopping, extensions or selection.',
        limits='Two feasible permutations are not authenticated chronology or exhaustive orders. Same data membership and split; recorded reference retained. Fixed-truth conditional-design sensitivity, not calibrated coverage.')
    atomic_json(out/'protocol.json', protocol)
    snapshot = out/'source-snapshot'; snapshot.mkdir(); hashes = {}
    for path in list((ROOT/'model_diagnostics').glob('*.py')) + [ROOT/'AnkiFirstRun/dynamic_retention_model_optimised.py']:
        relative = str(path.relative_to(ROOT)); hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        dest = snapshot/relative; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(path,dest)
    for path in source.glob('*-dataset.json'): shutil.copy2(path,out/path.name)
    for path in source.glob('*-split.json'): shutil.copy2(path,out/path.name)
    for path in source.glob('*-truth.json'): shutil.copy2(path,out/path.name)
    plans = []
    for cohort in range(2):
        orders = prepare_orders(source, cohort, out)
        for order in ('recorded','longest','reverse'):
            stem = f'real-{cohort}' if order == 'recorded' else f'real-{cohort}-{order}'
            if order != 'recorded':
                atomic_json(out/f'{stem}-dataset.json', orders[order])
                cuts = json.loads((out/f'real-{cohort}-split.json').read_text())['cuts']
                atomic_json(out/f'{stem}-split.json', dict(cuts=cuts, dataset_sha256=hashlib.sha256((out/f'{stem}-dataset.json').read_bytes()).hexdigest()))
            plans.append(dict(stem=stem,cohort=cohort,kind='order',condition=order,prior=[2,1]))
    for cohort in range(2):
        for condition, prior in protocol['hierarchy_MODEL_priors'].items():
            plans.append(dict(stem=f'nested_synthetic-{cohort}',cohort=cohort,kind='MODEL',condition=condition,prior=prior))
    atomic_json(out/'plan.json',plans)
    atomic_json(out/'provenance.json',dict(revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source=str(source.resolve()),source_hashes=hashes,
        artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.json')},
        parquet_hashes={n:hashlib.sha256((ROOT/'AnkiFirstRun'/n).read_bytes()).hexdigest() for n in ('train_data.parquet','test_data.parquet')},
        packages={p:importlib.metadata.version(p) for p in ('numpy','scipy','arviz','polyagamma','pyarrow')},
        platform=platform.platform(),execution_segment=out.name))
    results=[]; rows=[]; recovery=[]; metadata=[]; replays=[]; started=time.perf_counter()
    atomic_json(out/'status.json',dict(state='running',fits=0,total=12))
    with ProcessPoolExecutor(max_workers=4) as pool:
        for plan in plans:
            stem=plan['stem']; cohort=plan['cohort']; label=f"{stem}-{plan['kind']}-{plan['condition']}"
            seqs=[{k:np.array(v) if isinstance(v,list) else v for k,v in s.items()} for s in json.loads((out/f'{stem}-dataset.json').read_text())]
            split=json.loads((out/f'{stem}-split.json').read_text()); cuts=split['cuts']
            if split['dataset_sha256']!=hashlib.sha256((out/f'{stem}-dataset.json').read_bytes()).hexdigest(): raise ValueError('Split fingerprint differs.')
            train=[{k:v[:cut].copy() if isinstance(v,np.ndarray) else v for k,v in s.items()} for s,cut in zip(seqs,cuts)]
            test=[{k:v[cut:].copy() if isinstance(v,np.ndarray) else v for k,v in s.items()} for s,cut in zip(seqs,cuts)]
            futures=[pool.submit(job,train,test,plan['prior'],20261028+1009*cohort+31+7919*c,c) for c in range(4)]
            chains=[f.result() for f in futures]
            samples={k:np.stack([c['samples'][k] for c in chains]) for k in chains[0]['samples']}
            warm={k:np.stack([c['warmup'][k] for c in chains]) for k in chains[0]['warmup']}
            for phase,arrays in [('draws',samples),('warmup',warm)]:
                np.savez_compressed(out/f'{label}-{phase}.npz',**arrays)
                if plan['condition'] in ('recorded','original'):
                    with np.load(source/f'{stem}-full-coefficient-{phase}.npz') as old:
                        if set(old.files)!=set(arrays): raise ValueError('Replay keys differ.')
                        for k,v in arrays.items():
                            same=np.array_equal(v,old[k]); replays.append(dict(fit=label,phase=phase,parameter=k,identical=bool(same)))
                            if not same: raise ValueError('Replay arrays differ.')
            draws=nested_summaries(scalar_draws(samples,'full'),samples,train)
            if len(draws)!=142: raise ValueError('Original monitor set changed.')
            cpu=sum(c['cpu'] for c in chains); diag=diagnostics(draws,cpu)
            rows.extend(dict(fit=label,**r) for r in diag)
            pred=np.stack([c['prediction'] for c in chains]); y=np.concatenate([s['r'] for s in test])
            np.savez_compressed(out/f'{label}-predictions.npz',probability=pred,outcome=y)
            result=dict(fit=label,kind=plan['kind'],cohort=cohort,condition=plan['condition'],prior_scale=plan['prior'][1],
                passed=all(r['passed'] for r in diag),max_rhat=max(r['rhat'] for r in diag),min_ess_bulk=min(r['ess_bulk'] for r in diag),
                min_ess_tail=min(r['ess_tail'] for r in diag),failed_parameters=sum(not r['passed'] for r in diag),monitors=len(diag),cpu=cpu,
                min_ess_per_cpu=min(r['ess_per_cpu_second'] for r in diag),prediction_cpu=sum(c['prediction_cpu'] for c in chains),**metrics(y,pred.mean(0)))
            results.append(result)
            metadata.extend(dict(fit=label,chain=c,seed=ch['seed'],cpu=ch['cpu'],sampling=ch['sampling'],execution_segment=out.name) for c,ch in enumerate(chains))
            if plan['kind']=='MODEL':
                truth=json.loads((out/f'{stem}-truth.json').read_text()); p=truth['parameters']; expected=truth_values(p)
                expected.update({f'psi_last[{j}]':truth['latent_states'][j][cut-1] for j,cut in enumerate(cuts)})
                for user in sorted({s['user'] for s in train}):
                    cards=sorted(s['item'] for s in train if s['user']==user); mean=np.mean(np.asarray(p['beta'])[cards])
                    expected[f'user_level[{user}]']=p['b0']+p['alpha'][user]+mean
                    expected.update({f'within_user_card[{c}]':p['beta'][c]-mean for c in cards})
                recovery.extend(recovery_record(label,r,expected[r['parameter']]) for r in diag if r['parameter'] in expected)
            write_csv(out/'results.csv',results);write_csv(out/'diagnostics.csv',rows);write_csv(out/'recovery.csv',recovery)
            write_csv(out/'baseline-replay.csv',replays);atomic_json(out/'chain-metadata.json',metadata)
            atomic_json(out/'status.json',dict(state='running',fits=len(results),total=12,elapsed_seconds=time.perf_counter()-started))
            print(label,result['passed'],result['max_rhat'],result['min_ess_bulk'],flush=True)
    atomic_json(out/'status.json',dict(state='complete',fits=12,total=12,elapsed_seconds=time.perf_counter()-started))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--source',type=Path,default=ROOT/'.task3-runs/task3-coefficient-investigation');args=p.parse_args()
    if args.out.exists():p.error('Choose a new output directory.')
    try:execute(args.out,args.source)
    except Exception as error:
        if args.out.is_dir():atomic_json(args.out/'status.json',dict(state='failed',message=str(error)))
        raise


if __name__=='__main__':main()
