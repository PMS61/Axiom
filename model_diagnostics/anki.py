"""Deterministic local Anki cohorts with the provided temporal partition intact.

No outcome-based selection, silent time clipping, sequence truncation or resplit.
Parquet source row order breaks within-day timestamp ties deterministically.
"""
import hashlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ANKI_BOUNDS = (1 / 86400, 3650.)


def select_cohort(train_rows, test_rows, seed, cohort, users, cards, max_events):
    groups = {}
    exclusions = dict(short_history=0, long_history=0, partition_overlap=0,
                      invalid_elapsed=0, inconsistent_time=0)
    for partition, rows in enumerate((train_rows, test_rows)):
        for row in rows:
            key = int(row['user_id']), int(row['item_id'])
            groups.setdefault(key, [[], []])[partition].append(row)
    eligible = {}
    with_test = 0
    for (user, item), (train, test) in sorted(groups.items()):
        if not test: continue
        with_test += 1
        if len(train) < 6 or len(test) < 2:
            exclusions['short_history'] += 1
            continue
        if len(train) + len(test) > max_events:
            exclusions['long_history'] += 1
            continue
        # Stable sorting retains source order for ties, never sorts on outcome.
        train = sorted(train, key=lambda r: r['timestamp'])
        test = sorted(test, key=lambda r: r['timestamp'])
        if train[-1]['timestamp'] >= test[0]['timestamp']:
            exclusions['partition_overlap'] += 1
            continue
        rows = train + test
        seconds = np.array([r['elapsed_seconds'] for r in rows], float)
        gaps = seconds / 86400
        if not np.isfinite(gaps).all() or np.any(gaps[1:] < ANKI_BOUNDS[0]) or np.any(gaps[1:] > ANKI_BOUNDS[1]):
            exclusions['invalid_elapsed'] += 1
            continue
        # Source timestamps are integer days; exact elapsed gaps may differ by <1 day.
        days = np.array([r['timestamp'] for r in rows], float)
        if np.any(np.abs(np.diff(days) - gaps[1:]) > 1.00001):
            exclusions['inconsistent_time'] += 1
            continue
        outcomes = np.array([r['y'] for r in rows])
        if not np.isin(outcomes, [0, 1]).all(): raise ValueError('Anki outcomes must be binary.')
        if any(int(r['y']) != int(r['rating'] > 1) for r in rows):
            raise ValueError('Anki y does not match rating > 1.')
        gaps[0] = 1.  # unused first gap; retain all subsequent exact gaps
        seq = dict(user=user, item=item, r=outcomes.astype(int), dt=gaps,
                   x=np.r_[0, outcomes[:-1]].astype(int))
        eligible.setdefault(user, []).append((item, len(train), seq))
    candidate_users = np.array(sorted(u for u, seqs in eligible.items() if len(seqs) >= cards))
    order = np.random.default_rng(seed).permutation(candidate_users)
    chosen = order[cohort*users:(cohort+1)*users]
    if len(chosen) != users:
        raise ValueError(f'Need {(cohort+1)*users} eligible users for disjoint cohorts; have {len(order)}.')
    # IDs are user-nested even when source item numbers coincide across users.
    selection = []
    for user in sorted(chosen):
        available = eligible[int(user)]
        rng = np.random.default_rng(np.random.SeedSequence([seed, int(user)]))
        for index in sorted(rng.choice(len(available), cards, replace=False)):
            selection.append(available[index])
    user_map = {int(user): j for j, user in enumerate(sorted(chosen))}
    train, test, full, mapping = [], [], [], []
    for dense_item, (item, cut, seq) in enumerate(selection):
        mapped = {**seq, 'user': user_map[seq['user']], 'item': dense_item}
        full.append(mapped)
        train.append({k: v[:cut].copy() if isinstance(v, np.ndarray) else v for k, v in mapped.items()})
        test.append({k: v[cut:].copy() if isinstance(v, np.ndarray) else v for k, v in mapped.items()})
        mapping.append(dict(user=seq['user'], source_item=item, dense_user=mapped['user'], dense_item=dense_item, training_events=cut))
    all_gaps = np.concatenate([s['dt'][1:] for s in full])
    metadata = dict(source='local Anki Parquet files', cohort=cohort, selection_seed=seed,
                    provided_partition_preserved=True, time_unit='days from elapsed_seconds / 86400',
                    timestamp_ties='stable source row order; original intra-day ordering not independently verified',
                    candidates_with_test=with_test, eligible_users=len(candidate_users),
                    eligible_sequences=sum(len(v) for v in eligible.values()), exclusions=exclusions,
                    users=len(chosen), sequences=len(full), training_events=sum(len(s['r']) for s in train),
                    test_events=sum(len(s['r']) for s in test), minimum_gap_days=float(all_gaps.min()),
                    maximum_gap_days=float(all_gaps.max()), gaps_below_old_minimum=int(sum(all_gaps < .01)),
                    gaps_above_old_maximum=int(sum(all_gaps > 365)), mapping=mapping)
    return train, test, full, metadata


def load_cohort(seed, cohort, users, cards, max_events, root=ROOT):
    import pyarrow.parquet as pq
    columns = ['user_id', 'item_id', 'timestamp', 'elapsed_seconds', 'y', 'rating']
    paths = [root/'AnkiFirstRun'/name for name in ('train_data.parquet', 'test_data.parquet')]
    rows = [pq.read_table(path, columns=columns).to_pylist() for path in paths]
    train, test, full, metadata = select_cohort(*rows, seed, cohort, users, cards, max_events)
    metadata['files'] = {str(p.relative_to(root)): dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(), rows=len(r)) for p, r in zip(paths, rows)}
    return train, test, full, metadata
