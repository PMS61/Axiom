"""Read-only audit of the local extract; no upstream identity is assumed."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from .analysis import atomic_json
from .anki import ROOT


def timing_residual_range(days, seconds):
    """Necessary day-floor consistency across the entire retained trajectory."""
    days=np.asarray(days,dtype=float);seconds=np.asarray(seconds,dtype=float)
    return float(np.ptp(days-(np.cumsum(seconds)-seconds[0])/86400))


def audit(root=ROOT):
    frames, files = [], {}
    for partition, name in enumerate(('train_data.parquet', 'test_data.parquet')):
        path = root/'AnkiFirstRun'/name
        table = pq.read_table(path)
        frame = table.to_pandas()
        frame['_partition'] = partition
        frame['_source_row'] = np.arange(len(frame))
        frames.append(frame)
        metadata = json.loads(table.schema.metadata.get(b'pandas', b'{}'))
        files[name] = dict(rows=len(frame), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           pandas_metadata=metadata,
                           git_blob=subprocess.check_output(['git','hash-object',str(path)], text=True).strip(),
                           committed_blob=subprocess.check_output(['git','rev-parse',f'HEAD:AnkiFirstRun/{name}'],text=True).strip())
    source = pd.concat(frames, ignore_index=True)
    keys = ['user_id','item_id']
    original_diff = source.groupby(keys, sort=False).timestamp.diff()
    ordered = source.sort_values(keys+['timestamp','_partition','_source_row'], kind='stable').copy()
    groups = ordered.groupby(keys, sort=False)
    day_diff = groups.timestamp.diff()
    gap = ordered.elapsed_seconds / 86400
    adjacent = day_diff.notna()
    mismatch = adjacent & ((day_diff-gap).abs() > 1.00001)
    # If exact gaps really span adjacent retained rows, their cumulative sum also
    # agrees with day offsets within <1 day over the entire observed history.
    cumulative = groups.elapsed_seconds.cumsum() - groups.elapsed_seconds.transform('first')
    calendar = ordered.timestamp-groups.timestamp.transform('first')
    cumulative_error = calendar-cumulative/86400
    ambiguous = ordered.loc[adjacent & (day_diff == 0), ['state']]
    per_user = source.groupby('user_id').timestamp.agg(['min','max'])
    ordered['_timing_residual']=cumulative_error
    ranges=ordered.groupby(keys)['_timing_residual'].agg(['min','max'])
    selected=[]
    for cohort in range(2):
        path=root/f'.task3-runs/task3-anki-joint_blocked/anki-{cohort}-selection.json'
        if path.exists():
            mapping=json.loads(path.read_text())['mapping']
            values=[float(ranges.loc[(m['user'],m['source_item']),'max']-ranges.loc[(m['user'],m['source_item']),'min']) for m in mapping]
            selected.append(dict(cohort=cohort,histories=len(values),infeasible_day_floor_paths=sum(v>1.00001 for v in values),maximum_residual_range_days=max(values)))
    alternative=source.sort_values(keys+['timestamp','elapsed_seconds'],ascending=[True,True,True,False],kind='stable').copy()
    alt_groups=alternative.groupby(keys,sort=False)
    alt_diff=alt_groups.timestamp.diff()
    alt_bad=alt_diff.notna() & ((alt_diff-alternative.elapsed_seconds/86400).abs()>1.00001)
    alternative['_residual']=alternative.timestamp-(alt_groups.elapsed_seconds.cumsum()-alt_groups.elapsed_seconds.transform('first'))/86400
    alt_ranges=alternative.groupby(keys)['_residual'].agg(['min','max'])
    alt_width=alt_ranges['max']-alt_ranges['min']
    alt_selected=[]
    for cohort in range(2):
        path=root/f'.task3-runs/task3-anki-joint_blocked/anki-{cohort}-selection.json'
        if path.exists():
            mapping=json.loads(path.read_text())['mapping']
            values=[float(alt_width.loc[m['user'],m['source_item']]) for m in mapping]
            alt_selected.append(dict(cohort=cohort,infeasible_day_floor_paths=sum(v>1.00001 for v in values),maximum_residual_range_days=max(values)))
    tie_sensitivity=dict(order='DIAGNOSTIC ONLY: longest elapsed gap first within each card/day',
                        adjacent_mismatches=int(alt_bad.sum()),mismatch_histories=int(alternative.loc[alt_bad].groupby(keys).ngroups),
                        infeasible_day_floor_histories=int((alt_width>1.00001).sum()),selected_cohorts=alt_selected,
                        limitation='Not authenticated chronology. Does not resolve the order of multiple short-gap outcomes, and is never applied to model datasets.')
    return dict(files=files, rows=len(source), histories=groups.ngroups,
                audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                tie_order_sensitivity=tie_sensitivity,
                selected_cohort_timing=selected,
                state_counts={str(k):int(v) for k,v in source.state.value_counts().items()},
                binary_rating_mismatches=int((source.y!=(source.rating>1).astype(int)).sum()),
                infeasible_day_floor_histories=int(((ranges['max']-ranges['min'])>1.00001).sum()),
                source_order_negative_card_day_steps=int((original_diff < 0).sum()),
                adjacent_intervals=int(adjacent.sum()), adjacent_day_ties=int((adjacent & (day_diff == 0)).sum()),
                adjacent_mismatches=int(mismatch.sum()),
                timestamp_exceeds_elapsed=int((mismatch & (day_diff > gap)).sum()),
                elapsed_exceeds_timestamp=int((mismatch & (day_diff < gap)).sum()),
                mismatch_histories=int(ordered.loc[mismatch].groupby(keys).ngroups),
                mismatch_by_state={str(k):int(v) for k,v in ordered.loc[mismatch].state.value_counts().items()},
                ties_by_state={str(k):int(v) for k,v in ambiguous.state.value_counts().items()},
                delta_matches_calendar=int((adjacent & (ordered.delta_t == day_diff)).sum()),
                delta_matches_elapsed_floor=int((ordered.delta_t == np.floor(gap)).sum()),
                delta_matches_elapsed_round=int((ordered.delta_t == np.rint(gap)).sum()),
                elapsed_nonpositive=int((source.elapsed_seconds <= 0).sum()),
                cumulative_discrepancy_over_day_histories=int(ordered.loc[cumulative_error.abs()>1.00001].groupby(keys).ngroups),
                first_observed_day_zero_users=int((per_user['min']==0).sum()), users=len(per_user),
                global_timestamp_monotone=bool((source.timestamp.diff().dropna()>=0).all()),
                partition_day_ranges=[[int(f.timestamp.min()),int(f.timestamp.max())] for f in frames],
                train_fraction=len(frames[0])/len(source),
                evidence_limit='No preparation script or upstream fingerprint found. Schema resemblance cannot authenticate day_offset or chronological ties. Timing mismatch is evidence of incompatible local fields/retained history, not proof of corrupt upstream data.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists(): parser.error('Choose a new output file.')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    result=audit(); atomic_json(args.out,result)
    print(json.dumps({k:v for k,v in result.items() if k!='files'},indent=2))


if __name__=='__main__': main()
