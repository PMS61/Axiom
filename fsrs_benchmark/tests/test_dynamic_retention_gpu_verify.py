"""Verifier tests for fsrs_benchmark.models.dynamic_retention_gpu (Task 5.3): gaps not covered by
test_dynamic_retention_gpu.py. Synthetic data only. Runtime kept small (CPU torch).
"""
import numpy as np
import pytest
import torch

from fsrs_benchmark.models.dynamic_retention_gpu import DynamicRetentionModelGPU, sample_pg1_torch
from fsrs_benchmark.models.ours import _record_lam_acceptance
from dynamic_retention_dt import DynamicRetentionModelDT  # sys.path set by the GPU module
from fsrs_benchmark.tests.test_dynamic_retention_gpu import (
    DEV, _numpy_backward, _numpy_forward, _padded, _same_state, make_data,
)


def _T(a):
    return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float64)


def _small(seed=1):
    return make_data(n_users=4, items_per_user=3, max_len=8, seed=seed)


# ---------------------------------------------------------------- deterministic parity

@pytest.mark.parametrize("use_kappa", [False, True])
def test_update_lam_sigma2_matches_original_exactly(use_kappa):
    """All scalar draws (MH proposals, accepts, sigma2 gamma) use the numpy rng in the same
    order in both models, so given identical psi and state the outputs must agree."""
    train, _ = _small()
    orig = DynamicRetentionModelDT(train, rng=np.random.default_rng(0), use_kappa=use_kappa)
    gpu = DynamicRetentionModelGPU(train, rng=np.random.default_rng(0), use_kappa=use_kappa, device=DEV)
    _same_state(orig, gpu)
    orig.rng = np.random.default_rng(42)
    gpu.rng = np.random.default_rng(42)
    psi_flat = np.random.default_rng(9).normal(0.5, 1.0, gpu.E)
    orig._update_lam_sigma2(_padded(psi_flat, orig), 0, adapt=True)
    gpu._update_lam_sigma2(_T(psi_flat), 0, adapt=True)
    assert abs(gpu.lam - orig.lam) < 1e-9
    assert abs(gpu.kappa - orig.kappa) < 1e-12
    assert abs(gpu.sigma2 - orig.sigma2) < 1e-9
    assert abs(gpu.step - orig.step) < 1e-9
    assert abs(gpu.step_k - orig.step_k) < 1e-9
    assert gpu.acc_rate == orig.acc_rate


def test_fixed_lam_is_kept_and_no_lam_mh():
    train, _ = _small()
    gpu = DynamicRetentionModelGPU(train, rng=np.random.default_rng(0), fixed_lam=0.2, device=DEV)
    gpu.run(n_iter=12, burn_in=4, thin=2)
    assert np.all(gpu.samples['lam'] == 0.2)
    assert np.isnan(gpu.acc_rate)


def test_ffbs_padding_does_not_leak_into_real_events():
    """Padded columns (after each sequence's last event) hold junk values; the real events'
    psi must not change, and must match the numpy transcription of the numba kernel."""
    train, _ = _small()
    gpu = DynamicRetentionModelGPU(train[:6], rng=np.random.default_rng(0), device=DEV, bucket=False)
    N, Tm, lens = gpu.N, gpu.Tm, gpu.lens
    rs = np.random.default_rng(3)
    omega = rs.uniform(0.1, 2.0, (N, Tm)); phi = rs.uniform(0.05, 0.95, (N, Tm))
    q = rs.uniform(0.05, 1.0, (N, Tm)); mu = rs.normal(0, 1, (N, Tm))
    R = rs.integers(0, 2, (N, Tm)).astype(float); eps = rs.normal(size=(N, Tm))
    pad = 7
    junk = lambda shape, lo, hi: rs.uniform(lo, hi, shape)  # noqa: E731
    big = lambda a, lo, hi: np.hstack([a, junk((N, pad), lo, hi)])  # noqa: E731
    Ob, Pb, Qb, Mb = big(omega, 0.1, 2.0), big(phi, 0.05, 0.95), big(q, 0.05, 1.0), big(mu, -2, 2)
    Rb, Eb = big(R, 0, 2), big(eps, -3, 3)
    Ob[:, Tm:] = 1.0

    def run_gpu(o, p, qq, m_, r_, e_):
        t = lambda x: _T(x)  # noqa: E731
        m, P, a, Rp = gpu._ffbs_forward(t(o), t(p), t(qq), t(m_), t(r_))
        return gpu._ffbs_backward(m, P, a, Rp, t(p), t(e_), lens=lens).numpy()

    small = run_gpu(omega, phi, q, mu, R, eps)
    padded = run_gpu(Ob, Pb, Qb, Mb, Rb, Eb)
    for n in range(N):
        assert np.abs(small[n, :lens[n]] - padded[n, :lens[n]]).max() < 1e-12, n
    m, P, a, Rp = _numpy_forward(R, omega, phi, q, mu)
    ref = _numpy_backward(m, P, a, Rp, phi, eps, lens)
    for n in range(N):
        assert np.abs(small[n, :lens[n]] - ref[n, :lens[n]]).max() < 1e-9, n


def test_bucketed_ffbs_equals_single_bucket_with_same_eps():
    train, _ = make_data(n_users=5, items_per_user=4, max_len=20, seed=4)
    g1 = DynamicRetentionModelGPU(train, rng=np.random.default_rng(0), device=DEV, bucket=False)
    g2 = DynamicRetentionModelGPU(train, rng=np.random.default_rng(0), device=DEV, bucket=True)
    assert len(g2.buckets) > 2
    rs = np.random.default_rng(5)
    E = g1.E
    omega, phi = rs.uniform(0.1, 2.0, E), rs.uniform(0.05, 0.95, E)
    q, mu = rs.uniform(0.05, 1.0, E), rs.normal(0, 1, E)
    eps = rs.normal(size=(g1.N, g1.Tm))
    single = g1._ffbs(_T(omega), _T(phi), _T(q), _T(mu), eps_list=[_T(eps)]).numpy()
    eps_b = [_T(eps[b['idx']][:, :b['T']]) for b in g2.buckets]
    bucketed = g2._ffbs(_T(omega), _T(phi), _T(q), _T(mu), eps_list=eps_b).numpy()
    assert np.abs(single - bucketed).max() < 1e-12


def test_ffbs_kernel_matches_numpy_reference_on_bucketed_model():
    train, _ = make_data(n_users=5, items_per_user=4, max_len=20, seed=4)
    g2 = DynamicRetentionModelGPU(train, rng=np.random.default_rng(0), device=DEV, bucket=True)
    rs = np.random.default_rng(8)
    E = g2.E
    omega, phi = rs.uniform(0.1, 2.0, E), rs.uniform(0.05, 0.95, E)
    q, mu = rs.uniform(0.05, 1.0, E), rs.normal(0, 1, E)
    eps = rs.normal(size=(g2.N, g2.Tm))
    eps_b = [_T(eps[b['idx']][:, :b['T']]) for b in g2.buckets]
    got = g2._ffbs(_T(omega), _T(phi), _T(q), _T(mu), eps_list=eps_b).numpy()
    # dense reference over the full (N, Tm) layout
    R = _dense(g2.R.numpy(), g2)
    m, P, a, Rp = _numpy_forward(R, _dense(omega, g2, 1.0), _dense(phi, g2), _dense(q, g2, 1.0),
                                 _dense(mu, g2))
    ref = _numpy_backward(m, P, a, Rp, _dense(phi, g2), eps, g2.lens)
    for n in range(g2.N):
        L = g2.lens[n]
        seg = got[g2.offsets[n]:g2.offsets[n] + L]
        assert np.abs(seg - ref[n, :L]).max() < 1e-9, n


def _dense(flat, g, fill=0.0):
    """Flat per-event vector -> (N, Tm) matrix, sequences in input order."""
    out = np.full((g.N, g.Tm), fill)
    for n in range(g.N):
        out[n, :g.lens[n]] = flat[g.offsets[n]:g.offsets[n] + g.lens[n]]
    return out


def test_dyn_with_kappa_nonzero_matches_original():
    train, _ = _small()
    orig = DynamicRetentionModelDT(train, rng=np.random.default_rng(0), use_kappa=True)
    gpu = DynamicRetentionModelGPU(train, rng=np.random.default_rng(0), use_kappa=True, device=DEV)
    _same_state(orig, gpu, kappa=-0.7)
    phi_o, q_o = orig._dyn()
    phi_g, q_g = gpu._dyn()
    assert np.abs(phi_g.numpy() - phi_o[orig.M]).max() < 1e-9
    assert np.abs(q_g.numpy() - q_o[orig.M]).max() < 1e-9


# ---------------------------------------------------------------- lookup / cold paths

def test_lookup_cold_pair_with_item_past_stride_is_not_warm():
    """BUG candidate: _lookup encodes keys as user*stride + item with stride = n_items + 1.
    A test item >= stride spills into the next user's key range, so an unseen (user, item) can
    match another user's training card. The dict-based original returns cold (-1)."""
    train = [dict(user=0, item=0, r=np.array([1.0]), x=np.zeros(1), dt=np.ones(1)),
             dict(user=1, item=0, r=np.array([0.0]), x=np.zeros(1), dt=np.ones(1))]
    gpu = DynamicRetentionModelGPU(train, rng=np.random.default_rng(0), device=DEV)
    orig = DynamicRetentionModelDT(train, rng=np.random.default_rng(0))
    q_item = gpu.n_items + 1
    assert orig.key2idx.get((0, q_item)) is None
    assert gpu._lookup(np.array([0]), np.array([q_item]))[0] == -1


def test_lookup_matches_dict_on_in_range_pairs():
    train, test = _small()
    gpu = DynamicRetentionModelGPU(train, rng=np.random.default_rng(0), device=DEV)
    pairs = [(s['user'], s['item']) for s in test]
    pairs += [(u, i) for u in range(gpu.n_users + 2) for i in range(gpu.n_items)]
    got = gpu._lookup(np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs]))
    want = [gpu.key2idx.get(p, -1) for p in pairs]
    assert list(got) == want


def test_duplicate_training_pairs_resolve_like_original():
    """Original dict keeps the LAST training sequence for a repeated (user, item); _lookup
    picks the first. Only matters when a (user, item) appears twice in training."""
    train, _ = _small()
    train = train + [dict(train[0])]
    orig = DynamicRetentionModelDT(train, rng=np.random.default_rng(0))
    gpu = DynamicRetentionModelGPU(train, rng=np.random.default_rng(0), device=DEV)
    u, i = train[0]['user'], train[0]['item']
    assert gpu._lookup(np.array([u]), np.array([i]))[0] == orig.key2idx[(u, i)]


def test_predict_cold_item_past_stride_matches_original():
    """Same collision as above, end to end: user 0 / item past stride must predict as cold."""
    train = [dict(user=0, item=0, r=np.array([1.0, 1.0, 0.0, 1.0]), x=np.zeros(4), dt=np.ones(4)),
             dict(user=1, item=0, r=np.array([0.0, 0.0, 0.0, 0.0]), x=np.zeros(4), dt=np.ones(4) * 3)]
    orig = DynamicRetentionModelDT(train, rng=np.random.default_rng(0))
    orig.run(n_iter=40, burn_in=10, thin=2)
    gpu = DynamicRetentionModelGPU(train, rng=np.random.default_rng(0), device=DEV)
    gpu.samples = {k: np.array(v).copy() for k, v in orig.samples.items()}
    probe = [dict(user=0, item=gpu.n_items + 1, r=np.array([1.0, 0.0]), x=np.zeros(2),
                  dt=np.array([1.0, 2.0]))]
    po = orig.predict(probe, max_draws=5)
    pg = gpu.predict(probe, max_draws=5)
    assert np.abs(po[0] - pg[0]).max() < 1e-9


def test_predict_chunking_does_not_change_output():
    train, test = _small()
    orig = DynamicRetentionModelDT(train, rng=np.random.default_rng(0), use_kappa=True)
    orig.run(n_iter=30, burn_in=10, thin=2)
    preds = []
    for budget in (3000, 10 ** 9):
        g = DynamicRetentionModelGPU(train, rng=np.random.default_rng(0), use_kappa=True,
                                     device=DEV, chunk_elems=budget)
        g.samples = {k: np.array(v).copy() for k, v in orig.samples.items()}
        preds.append(g.predict(test, max_draws=6))
    for a, b in zip(*preds):
        assert np.abs(a - b).max() < 1e-12


# ---------------------------------------------------------------- PG, wrapper, misc

@pytest.mark.parametrize("chunk", [10 ** 8, 1000])
def test_pg_mean_with_chunking(chunk):
    gen = torch.Generator().manual_seed(1)
    for c in (0.0, 0.5, 1.0, 2.0, 5.0, -3.0):
        draws = sample_pg1_torch(torch.full((20000,), c, dtype=torch.float64), gen, 200, chunk)
        theo = 0.25 if c == 0 else np.tanh(c / 2) / (2 * c)
        assert abs(draws.mean().item() - theo) < 0.01, (c, chunk)


def test_record_lam_acceptance_wrapper_called_once_per_sweep():
    train, _ = _small()
    gpu = DynamicRetentionModelGPU(train, rng=np.random.default_rng(0), device=DEV)
    recs = _record_lam_acceptance(gpu, total=6)
    gpu.run(n_iter=6, burn_in=2, thin=1)
    assert [r[0] for r in recs] == list(range(6))
    assert all(np.isfinite(r[1]) for r in recs)


def test_samples_are_numpy_and_scalars_are_floats():
    train, _ = _small()
    gpu = DynamicRetentionModelGPU(train, rng=np.random.default_rng(0), device=DEV)
    s = gpu.run(n_iter=8, burn_in=2, thin=2)
    for k in ('alpha', 'beta', 'gamma', 'psi_last'):
        assert isinstance(s[k], np.ndarray)
    assert s['alpha'].shape == (3, gpu.n_users)
    assert s['psi_last'].shape == (3, gpu.N)
    assert all(isinstance(gpu.__dict__[k], float) for k in ('lam', 'kappa', 'sigma2', 'b0', 'g0'))
    gpu.summary()
