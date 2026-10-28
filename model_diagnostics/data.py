"""Synthetic known-truth experiments and strict JSON review-sequence validation."""
import numpy as np

SCENARIOS = ('full', 'no_practice', 'no_hierarchy', 'no_dynamics', 'short_gaps')


def validate_sequences(raw):
    if not isinstance(raw, list) or not 2 <= len(raw) <= 120:
        raise ValueError('Supply 2–120 sequences in a JSON array.')
    seen, users, items, result = set(), set(), set(), []
    for s in raw:
        if not isinstance(s, dict) or not all(k in s for k in ('user', 'item', 'r', 'dt')):
            raise ValueError('Each sequence requires user, item, r and dt.')
        u, i, r, dt = s['user'], s['item'], s['r'], s['dt']
        if any(type(v) is not int or v < 0 for v in (u, i)):
            raise ValueError('User and item IDs must be nonnegative integers.')
        if (u, i) in seen:
            raise ValueError('Duplicate user/item sequence; combine and sort events first.')
        if not isinstance(r, list) or not 8 <= len(r) <= 100:
            raise ValueError('Each sequence needs 8–100 ordered binary outcomes.')
        if any(type(v) is not int or v not in (0, 1) for v in r):
            raise ValueError('Outcomes must be integer 0 or 1.')
        if not isinstance(dt, list) or len(dt) != len(r):
            raise ValueError('dt and r must have equal length.')
        if any(type(v) not in (int, float) for v in dt):
            raise ValueError('dt must contain numeric days, not strings or booleans.')
        try:
            gaps = np.asarray(dt, dtype=float)
        except (ValueError, TypeError):
            raise ValueError('dt must be numeric days.') from None
        if not np.all(np.isfinite(gaps)) or np.any(gaps < .01) or np.any(gaps > 365):
            raise ValueError('dt must contain finite days in [0.01, 365]. First gap is ignored.')
        seen.add((u, i)); users.add(u); items.add(i)
        result.append(dict(user=u, item=i, r=np.array(r), dt=gaps,
                           x=np.array([0] + r[:-1])))
    # Remap arbitrary external identifiers without allocating max(id) arrays.
    um = {u: j for j, u in enumerate(sorted(users))}
    im = {i: j for j, i in enumerate(sorted(items))}
    for s in result:
        s['user'], s['item'] = um[s['user']], im[s['item']]
    return result, dict(users=um, items=im)


def simulate(seed=20261027, scenario='full', users=4, items=5, events=32):
    if scenario not in SCENARIOS:
        raise ValueError('Unknown simulation scenario')
    rng = np.random.default_rng(seed)
    alpha = rng.normal(0, .5, users); alpha -= alpha.mean()
    beta = rng.normal(0, .6, items); beta -= beta.mean()
    gamma = rng.normal(.7, .2, users)
    if scenario == 'no_hierarchy':
        alpha[:] = 0; beta[:] = 0; gamma[:] = .7
    truth = dict(b0=.2, lam=.35, kappa=.6 if scenario not in ('no_practice', 'no_dynamics') else 0.,
                 sigma2=.3, alpha=alpha, beta=beta, gamma=gamma, g0=.7)
    sequences, states = [], []
    for u in range(users):
        for i in range(items):
            dt = np.exp(rng.uniform(np.log(.05), np.log(20), events))
            if scenario == 'short_gaps':
                dt = np.exp(rng.uniform(np.log(.01), np.log(.08), events))
            r, psi, successes = [], [], 0
            for t in range(events):
                x = r[-1] if r else 0
                mu = truth['b0'] + alpha[u] + beta[i] + gamma[u] * x
                rate = truth['lam'] * np.exp(-truth['kappa'] * np.log1p(successes))
                phi = np.exp(-rate * dt[t])
                q = truth['sigma2'] * -np.expm1(-2 * rate * dt[t]) / (2 * rate)
                state = (rng.normal(mu, 1) if t == 0 else
                         rng.normal(phi * psi[-1] + (1 - phi) * mu, np.sqrt(q)))
                if scenario == 'no_dynamics':
                    state = mu
                outcome = int(rng.random() < 1 / (1 + np.exp(-state)))
                r.append(outcome); psi.append(state); successes += outcome
            sequences.append(dict(user=u, item=i, dt=dt, r=np.array(r), x=np.array([0] + r[:-1])))
            states.append(np.array(psi))
    if scenario == 'no_dynamics':
        # These do not exist in the generating model; never claim their recovery.
        for key in ('lam', 'sigma2', 'kappa'):
            truth.pop(key)
    return sequences, truth, states


def split_suffix(sequences, fraction=.25):
    train, test = [], []
    for s in sequences:
        split = max(4, min(len(s['r']) - 1, int(len(s['r']) * (1 - fraction))))
        train.append({k: v[:split].copy() if isinstance(v, np.ndarray) else v for k, v in s.items()})
        test.append({k: v[split:].copy() if isinstance(v, np.ndarray) else v for k, v in s.items()})
    return train, test
