"""Read-only lower bound on feasible outcome-order ambiguity in saved cohorts.

Never changes sampler inputs. Two diagnostic permutations are not authenticated
chronology and are not an exhaustive enumeration of possible orders.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from .analysis import atomic_json
from .anki import ROOT
from .provenance_audit import timing_residual_range


def compare_orders(days, seconds, outcomes):
    days=np.asarray(days); seconds=np.asarray(seconds); outcomes=np.asarray(outcomes)
    # Stable longest-first ties, versus same first event with remaining ties reversed.
    longest=[]; alternative=[]
    for day in np.unique(days):
        positions=np.flatnonzero(days==day)
        ordered=positions[np.argsort(-seconds[positions],kind='stable')]
        longest.extend(ordered); alternative.extend(np.r_[ordered[:1],ordered[:0:-1]])
    a=np.asarray(longest,dtype=int); b=np.asarray(alternative,dtype=int)
    def predictors(order):
        r=outcomes[order]
        return np.r_[0,r[:-1]],np.r_[0,np.cumsum(r)[:-1]]
    ax,ac=predictors(a);bx,bc=predictors(b)
    return dict(events=len(days),same_day_groups=int(sum(np.sum(days==d)>1 for d in np.unique(days))),
                candidate_ambiguous_groups=int(sum(np.sum(days==d)>2 for d in np.unique(days))),
                recorded_residual_range=timing_residual_range(days,seconds),
                longest_first_residual_range=timing_residual_range(days[a],seconds[a]),
                alternative_residual_range=timing_residual_range(days[b],seconds[b]),
                both_strictly_feasible=bool(timing_residual_range(days[a],seconds[a])<1 and timing_residual_range(days[b],seconds[b])<1),
                different_row_positions=int(np.sum(a!=b)),different_outcome_positions=int(np.sum(outcomes[a]!=outcomes[b])),
                different_lag_positions=int(np.sum(ax!=bx)),different_practice_positions=int(np.sum(ac!=bc)))


def audit():
    import pandas as pd
    paths=[ROOT/'AnkiFirstRun'/n for n in ('train_data.parquet','test_data.parquet')]
    frame=pd.concat([pq.read_table(p,columns=['user_id','item_id','timestamp','elapsed_seconds','y']).to_pandas() for p in paths],ignore_index=True)
    cohorts=[]
    for cohort in range(2):
        selection=ROOT/f'.task3-runs/task3-anki-joint_blocked/anki-{cohort}-selection.json'
        mapping=json.loads(selection.read_text())['mapping']
        values=[]
        for m in mapping:
            rows=frame.loc[(frame.user_id==m['user'])&(frame.item_id==m['source_item'])].sort_values('timestamp',kind='stable')
            values.append(compare_orders(rows.timestamp.to_numpy(),rows.elapsed_seconds.to_numpy(),rows.y.to_numpy()))
        feasible=[v for v in values if v['both_strictly_feasible']]
        cohorts.append(dict(cohort=cohort,histories=len(values),events=sum(v['events'] for v in values),
                            both_orders_strictly_feasible=len(feasible),
                            histories_with_different_feasible_outcome_order=sum(v['different_outcome_positions']>0 for v in feasible),
                            histories_with_different_feasible_lag_predictors=sum(v['different_lag_positions']>0 for v in feasible),
                            histories_with_different_feasible_practice_predictors=sum(v['different_practice_positions']>0 for v in feasible),
                            different_feasible_outcome_positions=sum(v['different_outcome_positions'] for v in feasible),
                            different_feasible_lag_positions=sum(v['different_lag_positions'] for v in feasible),
                            different_feasible_practice_positions=sum(v['different_practice_positions'] for v in feasible),
                            same_day_groups=sum(v['same_day_groups'] for v in values),
                            candidate_ambiguous_groups=sum(v['candidate_ambiguous_groups'] for v in values),
                            max_longest_first_residual_range=max(v['longest_first_residual_range'] for v in values),
                            max_alternative_residual_range=max(v['alternative_residual_range'] for v in values),
                            selection_sha256=hashlib.sha256(selection.read_bytes()).hexdigest()))
    return dict(source_files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cohorts=cohorts,
                permutation='Stable longest-gap-first within card/day versus keeping that first event and reversing the remaining events.',
                interpretation='A lower bound from two feasible permutations, not exhaustive sensitivity and not recovered chronology. Outcome/predictor positions refer to review index; no model datasets are changed. No posterior/efficacy conclusion is drawn.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    if args.out.exists():p.error('Output exists; choose a new file.')
    args.out.parent.mkdir(parents=True,exist_ok=True);atomic_json(args.out,audit())


if __name__=='__main__':main()
