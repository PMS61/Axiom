"""
common.py -- Shared data-loading entry point for analysis_suite.py and run_experiments.py
"""
import pickle
import numpy as np
import pandas as pd


def add_data_args(ap):
    ap.add_argument('--data', choices=['synthetic', 'duolingo', 'anki', 'parquet_benchmark'], required=True)
    ap.add_argument('--pickle', default='processed_sequences.pkl')
    ap.add_argument('--anki_root', default=None)
    ap.add_argument('--train_parquet', default='train_data.parquet')
    ap.add_argument('--test_parquet', default='test_data.parquet')
    ap.add_argument('--n_users', type=int, default=100)
    ap.add_argument('--max_cards', type=int, default=300, help='anki loader: max cards sampled per user')
    ap.add_argument('--min_events', type=int, default=300, help='anki loader: skip users with fewer rows')
    ap.add_argument('--user_frac', type=float, default=1.0,
                    help='Duolingo only: fraction of users sampled per seed (e.g. 0.8)')


def load_parquet_splits_correctly(train_path, test_path, seed=0, test_frac=0.2, max_seqs=20000):
    """
    Combines train and test parquet files to reconstruct continuous review sequences,
    splits each sequence chronologically, and caps sequence counts to fit safely in RAM.
    """
    print(f"Loading local parquet benchmark files:\n  Train: {train_path}\n  Test:  {test_path}")
    
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    
    # Combine to reconstruct full contiguous trajectories per card
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    
    # Filter out learning steps if present (delta_t < 0)
    full_df = full_df[full_df['delta_t'] >= 0].copy()
    
    train_seqs = []
    test_seqs = []
    
    grouped = full_df.groupby(['user_id', 'item_id'])
    
    for (uid, item_id), group in grouped:
        group_sorted = group.sort_values(by='timestamp')
        
        dts = group_sorted['delta_t'].values.astype(float)
        ys = group_sorted['y'].values.astype(int)
        
        N_total = len(ys)
        # Skip micro-sequences that don't have enough history to evaluate
        if N_total < 3:
            continue
            
        x_lag = np.concatenate([[0], ys[:-1]]).astype(int)
        
        # Split index per sequence (Suffix Split)
        split_idx = max(1, int(N_total * (1.0 - test_frac)))
        
        # Train sequence (History up to split_idx)
        train_seqs.append({
            'user': int(uid),
            'item': int(item_id),
            'dt': dts[:split_idx],
            'y': ys[:split_idx],
            'r': ys[:split_idx],
            'x': x_lag[:split_idx]
        })
        
        # Test sequence (Future evaluation hold-out)
        test_seqs.append({
            'user': int(uid),
            'item': int(item_id),
            'dt': dts[split_idx:],
            'y': ys[split_idx:],
            'r': ys[split_idx:],
            'x': x_lag[split_idx:]
        })

    # Subsample sequences using the passed seed to prevent Linux OOM crashes
    if len(train_seqs) > max_seqs:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(train_seqs), max_seqs, replace=False)
        train_seqs = [train_seqs[i] for i in idx]
        test_seqs = [test_seqs[i] for i in idx]
        
    print(f"Successfully constructed {len(train_seqs)} continuous sequence trajectories.")
    return train_seqs, test_seqs


def load(args, seed):
    if args.data == 'parquet_benchmark' or (args.data == 'anki' and args.anki_root is None):
        return load_parquet_splits_correctly(args.train_parquet, args.test_parquet, seed=seed)

    if args.data == 'synthetic':
        from synthetic_check_dt import simulate, split_suffix
        seqs, _ = simulate(np.random.default_rng(seed))
        return split_suffix(seqs)

    if args.data == 'duolingo':
        from data_pipeline import train_test_split_sequences
        with open(args.pickle, 'rb') as fh:
            seqs = pickle.load(fh)['sequences']
        if args.user_frac < 1.0:
            rng = np.random.default_rng(seed)
            users = np.unique([s['user'] for s in seqs])
            keep = set(rng.choice(users, int(args.user_frac * len(users)), replace=False).tolist())
            seqs = [s for s in seqs if s['user'] in keep]
            um = {u: k for k, u in enumerate(sorted(keep))}
            im = {i: k for k, i in enumerate(sorted({s['item'] for s in seqs}))}
            seqs = [{**s, 'user': um[s['user']], 'item': im[s['item']]} for s in seqs]
        return train_test_split_sequences(seqs, test_frac_per_seq=0.25, min_train_len=2, random_state=seed)

    if args.data == 'anki':
        from anki_data import load_anki
        return load_anki(args.anki_root, n_users=args.n_users, seed=seed,
                         min_events=args.min_events, max_cards=args.max_cards)

    raise ValueError(args.data)