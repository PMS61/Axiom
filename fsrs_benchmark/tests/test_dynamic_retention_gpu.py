"""Tests for fsrs_benchmark.models.dynamic_retention_gpu (torch port of DynamicRetentionModelDT).

The port must be the same model. Deterministic pieces are compared with the numpy original to
1e-9 (given identical state / identical noise); the stochastic parts are compared in
distribution (PG mean, and a short fit on small synthetic data against the original).
"""
import time

import numpy as np
import pytest
import torch

from fsrs_benchmark.models.dynamic_retention_gpu import DynamicRetentionModelGPU, sample_pg1_torch
from dynamic_retention_dt import DynamicRetentionModelDT, ess  # sys.path set by the module above

DEV = "cpu"


def make_data(n_users=15, items_per_user=6, max_len=30, seed=0, lam=0.1):
    """Synthetic cards: AR(1)-on-logit retention with irregular gaps. Returns (train, test)."""
    rng = np.random.default_rng(seed)
    train, test = [], []
    item = 0
    ue = rng.normal(0, 0.5, n_users)
    for u in range(n_users):
        for _ in range(items_per_user):
            T = int(rng.integers(1, max_len + 1))
            dt = rng.exponential(2.0, T) + 0.05
            psi = 0.8 + ue[u] + rng.normal(0, 0.5)
            r = np.zeros(T)
            for t in range(T):
                if t:
                    ph = np.exp(-lam * dt[t])
                    psi = ph * psi + (1 - ph) * (0.8 + ue[u]) + rng.normal(0, np.sqrt(0.5 * (1 - ph * ph) / (2 * lam)) * 0.5)
                r[t] = rng.random() < 1 / (1 + np.exp(-psi))
            x = np.concatenate([[0.0], r[:-1]])
            cut = max(1, int(T * 0.7)) if T > 1 else 1
            train.append(dict(user=u, item=item, r=r[:cut], x=x[:cut], dt=dt[:cut]))
            if T > cut:
                test.append(dict(user=u, item=item, r=r[cut:], x=x[cut:], dt=dt[cut:]))
            item += 1
    # cold cases: unseen card of a known user, unseen user
    test.append(dict(user=0, item=item, r=np.array([1.0, 0.0, 1.0]), x=np.zeros(3), dt=np.array([1.0, 3.0, 0.5])))
    test.append(dict(user=n_users, item=item + 1, r=np.array([0.0, 1.0]), x=np.zeros(2), dt=np.array([2.0, 2.0])))
    return train, test


@pytest.fixture(scope="module")
def data():
    return make_data()


def _same_state(orig, gpu, seed=3, kappa=0.3):
    rng = np.random.default_rng(seed)
    st = dict(b0=0.4, g0=0.1, lam=0.13, kappa=kappa, sigma2=0.7,
              alpha=rng.normal(0, .5, orig.n_users), beta=rng.normal(0, .5, orig.n_items),
              gamma=rng.normal(0, .3, orig.n_users))
    for k in ('b0', 'g0', 'lam', 'kappa', 'sigma2'):
        setattr(orig, k, st[k]); setattr(gpu, k, st[k])
    for k in ('alpha', 'beta', 'gamma'):
        setattr(orig, k, st[k].copy())
        setattr(gpu, k, torch.as_tensor(st[k], dtype=torch.float64, device=DEV))


def _np(t):
    return t.detach().cpu().numpy()


def _numpy_forward(R, omega, phi, q, mu):
    N, Tm = R.shape
    z, ov = (R - 0.5) / omega, 1.0 / omega
    m, P, a, Rp = (np.zeros((N, Tm)) for _ in range(4))
    mp = np.zeros(N); Pp = np.zeros(N)
    for t in range(Tm):
        a[:, t] = phi[:, t] * mp + (1 - phi[:, t]) * mu[:, t]
        Rp[:, t] = phi[:, t] ** 2 * Pp + q[:, t]
        K = Rp[:, t] / (Rp[:, t] + ov[:, t])
        m[:, t] = a[:, t] + K * (z[:, t] - a[:, t])
        P[:, t] = (1 - K) * Rp[:, t]
        mp, Pp = m[:, t], P[:, t]
    return m, P, a, Rp


def _numpy_backward(m, P, a, Rp, phi, eps, lens):
    """Direct transcription of the numba backward loop (per sequence, per t)."""
    N, Tm = m.shape
    last = lens - 1
    psi = np.zeros((N, Tm))
    for t in range(Tm - 1, -1, -1):
        for n in range(N):
            var_end = m[n, t] + np.sqrt(max(P[n, t], 1e-12)) * eps[n, t]
            if t == Tm - 1:
                val = var_end
            else:
                J = P[n, t] * phi[n, t + 1] / Rp[n, t + 1]
                h = m[n, t] + J * (psi[n, t + 1] - a[n, t + 1])
                V = P[n, t] - J * phi[n, t + 1] * P[n, t]
                cond = h + np.sqrt(max(V, 1e-12)) * eps[n, t]
                val = var_end if last[n] == t else cond
            psi[n, t] = val if t <= last[n] else 0.0
    return psi


def _padded(flat, orig):
    out = np.zeros(orig.M.shape)
    out[orig.M] = flat
    return out


def test_dyn_mu_parity(data):
    train, _ = data
    orig = DynamicRetentionModelDT(train, rng=np.random.default_rng(0), use_kappa=True)
    gpu = DynamicRetentionModelGPU(train, rng=np.random.default_rng(0), use_kappa=True, device=DEV)
    _same_state(orig, gpu)
    assert gpu.N == orig.N and gpu.Tm == orig.Tm and (gpu.lens == orig.lens).all()
    phi_o, q_o = orig._dyn()
    phi_g, q_g = gpu._dyn()
    assert np.abs(_np(phi_g) - phi_o[orig.M]).max() < 1e-9
    assert np.abs(_np(q_g) - q_o[orig.M]).max() < 1e-9
    assert np.abs(_np(gpu._mu()) - orig._mu()[orig.M]).max() < 1e-9
    assert np.abs(_np(gpu.C) - orig.C[orig.M]).max() < 1e-12


@pytest.mark.parametrize("bucket", [False, True])
def test_ffbs_parity_fixed_eps(data, bucket):
    """Forward filter and backward recursion match numpy, with and without length buckets."""
    train, _ = data
    orig = DynamicRetentionModelDT(train, rng=np.random.default_rng(0), use_kappa=True)
    gpu = DynamicRetentionModelGPU(train, rng=np.random.default_rng(0), use_kappa=True,
                                   device=DEV, bucket=bucket)
    assert (len(gpu.buckets) > 1) == bucket
    _same_state(orig, gpu)
    rng = np.random.default_rng(5)
    phi, q = orig._dyn(); mu = orig._mu()
    omega = rng.uniform(0.05, 0.6, orig.R.shape)
    eps = rng.standard_normal(orig.R.shape)
    m, P, a, Rp = _numpy_forward(orig.R, omega, phi, q, mu)
    psi_ref = _numpy_backward(m, P, a, Rp, phi, eps, orig.lens)
    # single-bucket public API on (N, Tm) arrays
    if not bucket:
        tt = lambda x: torch.as_tensor(x, dtype=torch.float64)  # noqa: E731
        mg, Pg, ag, Rg = gpu._ffbs_forward(tt(omega), tt(phi), tt(q), tt(mu))
        for got, ref in ((mg, m), (Pg, P), (ag, a), (Rg, Rp)):
            assert np.abs(_np(got) - ref)[orig.M].max() < 1e-9
        pg = gpu._ffbs_backward(mg, Pg, ag, Rg, tt(phi), tt(eps))
        assert np.abs(_np(pg) - psi_ref).max() < 1e-9
    # flat _ffbs over buckets with the same eps (sliced per bucket)
    f = lambda x: torch.as_tensor(x[orig.M], dtype=torch.float64)  # noqa: E731
    eps_list = [torch.as_tensor(eps[b['idx'], :b['T']]) for b in gpu.buckets]
    got = _np(gpu._ffbs(f(omega), f(phi), f(q), f(mu), eps_list=eps_list))
    assert np.abs(got - psi_ref[orig.M]).max() < 1e-9


def test_pg_mean():
    gen = torch.Generator().manual_seed(1)
    for c in (0.0, 0.5, 1.0, 2.0, 5.0, -3.0):
        d = sample_pg1_torch(torch.full((20000,), c, dtype=torch.float64), gen, chunk_elems=2_000_000)
        want = 0.25 if c == 0 else np.tanh(c / 2) / (2 * c)
        assert abs(d.mean().item() - want) < 0.01, c
        assert (d > 0).all()


@pytest.fixture(scope="module")
def fitted_pair(data):
    train, test = data
    orig = DynamicRetentionModelDT(train, rng=np.random.default_rng(1), use_kappa=True)
    orig.run(n_iter=60, burn_in=30, thin=2)
    return orig


@pytest.mark.parametrize("draws", [False, True])
def test_predict_parity_given_samples(data, fitted_pair, draws):
    train, test = data
    orig = fitted_pair
    gpu = DynamicRetentionModelGPU(train, rng=np.random.default_rng(0), use_kappa=True, device=DEV,
                                   chunk_elems=3000)           # tiny budget -> many chunks
    gpu.samples = {k: v.copy() for k, v in orig.samples.items()}
    assert gpu.key2idx == orig.key2idx
    po = orig.predict(test, max_draws=7, return_draws=draws)
    pg = gpu.predict(test, max_draws=7, return_draws=draws)
    if draws:
        po, do = po; pg, dg = pg
        for a, b in zip(do, dg):
            assert a.shape == b.shape and np.abs(a - b).max() < 1e-9
    assert len(po) == len(pg) == len(test)
    for a, b in zip(po, pg):
        assert a.shape == b.shape and np.abs(a - b).max() < 1e-9


def test_samples_keys_shapes_types(data, fitted_pair):
    train, _ = data
    orig = fitted_pair
    gpu = DynamicRetentionModelGPU(train, rng=np.random.default_rng(2), use_kappa=True, device=DEV)
    calls = []
    inner = gpu._update_lam_sigma2
    gpu._update_lam_sigma2 = lambda psi, it, adapt: (calls.append(it), inner(psi, it, adapt))
    out = gpu.fit(n_iter=60, burn_in=30, thin=2).samples
    assert calls == list(range(60))                    # wrapper hook called once per sweep
    assert set(out) == set(orig.samples)
    for k, v in orig.samples.items():
        assert isinstance(out[k], np.ndarray) and out[k].shape == v.shape and out[k].dtype == v.dtype, k
    for k in ('b0', 'g0', 'lam', 'kappa', 'sigma2'):
        assert isinstance(getattr(gpu, k), float)
    assert all(isinstance(v, float) for v in gpu.tau2.values())
    assert np.isfinite(gpu.acc_rate)
    s = gpu.summary()
    assert set(s) == set(orig.summary())
    assert all(np.isfinite(d['mean']) for d in s.values())
    gpu.print_summary()
    assert all(np.isfinite(v).all() for v in out.values())


def test_fixed_lam_and_all_length_one():
    seqs = [dict(user=i % 3, item=i, r=np.array([float(i % 2)]), x=np.zeros(1), dt=np.ones(1)) for i in range(9)]
    g = DynamicRetentionModelGPU(seqs, rng=np.random.default_rng(0), fixed_lam=0.2, device=DEV)
    g.run(n_iter=12, burn_in=6, thin=2)
    assert g.lam == 0.2 and len(g.samples['lam']) == 3 and np.isfinite(g.samples['sigma2']).all()
    assert len(g.predict(seqs[:2])) == 2


def test_statistical_parity_and_timing():
    """Several short chains of each implementation; the model mixes slowly, so single chains are
    compared through the spread of chain means (the original's own between-chain spread)."""
    train, test = make_data(n_users=10, items_per_user=6, max_len=15, seed=7)
    n_chain, n_it, burn = 6, 400, 150
    y = np.concatenate([s['r'] for s in test])
    ll = lambda p: -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))  # noqa: E731
    res = {}
    for name, make in (("orig", lambda s: DynamicRetentionModelDT(train, rng=np.random.default_rng(s))),
                       ("gpu", lambda s: DynamicRetentionModelGPU(train, rng=np.random.default_rng(s), device=DEV))):
        rows = []
        for c in range(n_chain):
            m = make(100 + c)
            m.run(n_iter=n_it, burn_in=burn, thin=1)
            p = np.clip(np.concatenate(m.predict(test)), 1e-6, 1 - 1e-6)
            rows.append([m.samples[k].mean() for k in ('b0', 'lam', 'sigma2')] + [p.mean(), ll(p)])
        res[name] = np.array(rows)
    for j, k in enumerate(('b0', 'lam', 'sigma2', 'mean_p')):
        a, b = res["orig"][:, j], res["gpu"][:, j]
        se = np.sqrt(a.var(ddof=1) / n_chain + b.var(ddof=1) / n_chain)
        assert abs(a.mean() - b.mean()) < 4 * se + 0.01 * abs(a.mean()), (k, a.mean(), b.mean(), se)
    assert abs(res["orig"][:, 4].mean() - res["gpu"][:, 4].mean()) < 0.01, res

    # rough timing, not an assertion: one sweep at N about 5000 sequences
    big, _ = make_data(n_users=500, items_per_user=10, max_len=30, seed=9)
    m = DynamicRetentionModelGPU(big, rng=np.random.default_rng(0), device=DEV)
    m.run(n_iter=1, burn_in=0, thin=1)
    t0 = time.perf_counter()
    m.run(n_iter=3, burn_in=3, thin=1)
    print(f"\n[timing] N={m.N} E={m.E} Tm={m.Tm} buckets={len(m.buckets)} "
          f"{(time.perf_counter() - t0) / 3:.3f} s/sweep on {DEV}")
