"""
Loads and preprocesses the Duolingo half-life-regression dataset
(Settles & Meeder 2016) into the (user, item, r, x, dt) sequence format
that DynamicRetentionModel / StaticRetentionModel expect.

Raw dataset columns (confirmed from duolingo/halflife-regression README):
    p_recall          - proportion of exercises correctly recalled in this
                         practice event (we binarize: >=0.5 -> 1 else 0,
                         since our model's observation is Bernoulli per
                         review EVENT, not a per-lesson recall rate)
    timestamp         - UNIX timestamp of this practice event
    delta             - seconds since the last practice of this lexeme
    user_id           - anonymized student ID
    lexeme_string     - the item identifier (word/lexeme + grammatical tag)
    history_seen      - total times user has seen this lexeme before
    history_correct   - total times user got this lexeme correct before
    session_seen      - times seen in this session
    session_correct   - times correct in this session

We do NOT use history_seen/history_correct as model inputs (that would be
leaking the model's own job -- estimating retention from history -- into
a feature), but they ARE useful for a sanity-check baseline and for
choosing a well-observed subsample.
"""
import numpy as np
import pandas as pd


def load_duolingo_subsample(path, n_users=500, min_reviews_per_pair=3,
                             max_reviews_per_pair=30, language_filter=None,
                             random_state=0):
    """
    Loads a manageable subsample of the (large, ~13M-row) Duolingo CSV.

    Parameters
    ----------
    path : str
        Path to the (possibly gzipped) CSV, e.g.
        '/content/drive/MyDrive/axiom_data/learning_traces.13m.csv.gz'
    n_users : int
        Number of distinct users to sample (this bounds memory usage --
        13M rows is too large to hold all trajectories in Python objects
        on a Colab free-tier instance without subsampling first).
    min_reviews_per_pair, max_reviews_per_pair : int
        Filters on trajectory length. Very short sequences carry little
        information about dynamics (the whole point of the paper); very
        long ones are truncated to keep runtime bounded.
    language_filter : str or None
        If given (e.g. 'en' for English learners), restricts to that
        learning_language value if the column is present.
    random_state : int

    Returns
    -------
    sequences : list of dicts (see dynamic_retention_model.py docstring
        for the expected schema), plus integer-encoded user/item ids.
    user_id_map, item_id_map : dict mapping original ids -> 0-indexed ints
    """
    # NOTE: for a 13M-row file, pd.read_csv on the whole thing at once on
    # Colab free tier risks OOM. Read in chunks and stop once we have
    # enough distinct users. Columns used: user_id, lexeme_string,
    # timestamp, delta, p_recall.
    rng = np.random.default_rng(random_state)
    wanted_cols = ['p_recall', 'timestamp', 'delta', 'user_id', 'lexeme_string']

    chunks = []
    seen_users = set()
    chunk_iter = pd.read_csv(path, usecols=lambda c: c in wanted_cols or c == 'learning_language',
                              chunksize=500_000)
    for chunk in chunk_iter:
        if language_filter is not None and 'learning_language' in chunk.columns:
            chunk = chunk[chunk['learning_language'] == language_filter]
        chunks.append(chunk)
        seen_users.update(chunk['user_id'].unique().tolist())
        if len(seen_users) >= n_users * 3:  # oversample users, then subsample below
            break

    df = pd.concat(chunks, ignore_index=True)

    # Subsample down to exactly n_users distinct users for a bounded run
    all_users = df['user_id'].unique()
    if len(all_users) > n_users:
        chosen_users = rng.choice(all_users, size=n_users, replace=False)
        df = df[df['user_id'].isin(chosen_users)]

    df = df.sort_values(['user_id', 'lexeme_string', 'timestamp'])

    # Binarize p_recall into a single Bernoulli outcome per review EVENT
    df['r'] = (df['p_recall'] >= 0.5).astype(int)

    sequences = []
    user_id_map, item_id_map = {}, {}

    for (uid, lex), group in df.groupby(['user_id', 'lexeme_string']):
        T = len(group)
        if T < min_reviews_per_pair:
            continue
        group = group.iloc[:max_reviews_per_pair]
        T = len(group)

        if uid not in user_id_map:
            user_id_map[uid] = len(user_id_map)
        if lex not in item_id_map:
            item_id_map[lex] = len(item_id_map)

        r = group['r'].values.astype(int)
        dt_days = (group['delta'].values.astype(float) / 86400.0)  # seconds -> days
        dt_days = np.clip(dt_days, 1e-3, None)  # avoid zero/negative gaps

        # x_t = previous review's success indicator (t=0 has no previous
        # review for this pair, so x[0] = 0 -- this is the indexing fix
        # discussed earlier: the drift at time t must depend on t-1's
        # outcome, not t's own outcome, to avoid same-timestep circularity)
        x = np.concatenate(([0], r[:-1]))

        sequences.append({
            'user': user_id_map[uid],
            'item': item_id_map[lex],
            'r': r,
            'x': x,
            'dt': dt_days,
        })

    return sequences, user_id_map, item_id_map


def train_test_split_sequences(sequences, test_frac_per_seq=0.25, min_train_len=2,
                                random_state=0):
    """
    Splits EACH sequence into a train prefix and a held-out suffix (rather
    than holding out whole sequences), so we can compute the report's
    "late-sequence AUC" metric (Section 5.1): can the model use early
    reviews to predict later ones better than a static model can?

    Sequences too short to leave both a valid train prefix and a non-empty
    test suffix are dropped.
    """
    train_seqs, test_seqs = [], []
    for s in sequences:
        T = len(s['r'])
        n_test = max(1, int(round(T * test_frac_per_seq)))
        n_train = T - n_test
        if n_train < min_train_len:
            continue

        train_seqs.append({
            'user': s['user'], 'item': s['item'],
            'r': s['r'][:n_train], 'x': s['x'][:n_train], 'dt': s['dt'][:n_train],
        })
        test_seqs.append({
            'user': s['user'], 'item': s['item'],
            'r': s['r'][n_train:], 'x': s['x'][n_train:], 'dt': s['dt'][n_train:],
            'n_train': n_train,   # kept so we know how many reviews preceded this suffix
        })
    return train_seqs, test_seqs
