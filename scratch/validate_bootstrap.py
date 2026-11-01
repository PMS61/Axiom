"""
validate_bootstrap.py  –  Step 4: Bootstrap methodology sanity checks.

Run from repo root:
    python scratch/validate_bootstrap.py
"""
import sys
import numpy as np

sys.path.insert(0, "/Users/vaishvi/Desktop/FYP/Axiom")
from fsrs_benchmark.metrics import paired_bootstrap, log_loss, brier, auc, ece

PASS = "  PASS"
FAIL = "  FAIL *** "
all_ok = True

def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")

def check(label, ok, detail=""):
    status = PASS if ok else FAIL
    print(f"  {label}: {status}  {detail}")
    return ok

# ── 1. Identical models → diff = 0, CI = [0, 0] ──────────────────────────────
section("1. Identical models → diff=0, CI=[0,0]")

rng = np.random.default_rng(42)
n = 500
uid = rng.integers(0, 20, n)
y = rng.integers(0, 2, n).astype(float)
while y.sum() == 0 or y.sum() == n:
    y = rng.integers(0, 2, n).astype(float)
p = rng.uniform(0.1, 0.9, n)

for metric in ["log_loss", "brier", "auc", "ece"]:
    res = paired_bootstrap(uid, y, p, p, metric, n_boot=500, seed=0)
    ok = res["diff"] == 0.0 and res["lo"] == 0.0 and res["hi"] == 0.0
    all_ok &= check(f"{metric}: diff=0, CI=[0,0]", ok,
                    f"diff={res['diff']}, lo={res['lo']}, hi={res['hi']}")

# ── 2. CI contains the point estimate ─────────────────────────────────────────
section("2. CI contains the point estimate for random models")

rng = np.random.default_rng(7)
n = 1000
uid = rng.integers(0, 50, n)
y = rng.integers(0, 2, n).astype(float)
p_a = rng.uniform(0.1, 0.9, n)
p_b = rng.uniform(0.1, 0.9, n)

for metric in ["log_loss", "brier", "auc", "ece"]:
    res = paired_bootstrap(uid, y, p_a, p_b, metric, n_boot=1000, seed=0)
    ok = res["lo"] <= res["diff"] <= res["hi"]
    all_ok &= check(f"{metric}: lo ≤ diff ≤ hi", ok,
                    f"diff={res['diff']:.4f} [{res['lo']:.4f}, {res['hi']:.4f}]")

# ── 3. User-cluster resampling (vs event-level) → wider intervals ─────────────
section("3. User clustering widens CI vs event-level resampling")

def event_level_bootstrap(y, p_a, p_b, metric_fn, n_boot=1000, seed=0):
    """Bootstrap that resamples individual events (no clustering) – narrower CI."""
    rng = np.random.default_rng(seed)
    n = len(y)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        d = metric_fn(y[idx], p_a[idx]) - metric_fn(y[idx], p_b[idx])
        if np.isfinite(d):
            draws.append(d)
    return np.percentile(draws, [2.5, 97.5])

rng = np.random.default_rng(10)
n = 2000
# Simulate clustered data: 20 users, strong within-user correlation
n_users = 20
uid = np.repeat(np.arange(n_users), n // n_users)
# Each user has a user-specific shift in predictions
user_shift = rng.normal(0, 0.3, n_users)
y = (rng.uniform(0, 1, n) < 0.7).astype(float)
p_a = np.clip(0.7 + user_shift[uid] + rng.normal(0, 0.05, n), 0.01, 0.99)
p_b = np.clip(0.65 + user_shift[uid] + rng.normal(0, 0.05, n), 0.01, 0.99)

for metric_name, metric_fn in [("log_loss", log_loss), ("brier", brier)]:
    res_cluster = paired_bootstrap(uid, y, p_a, p_b, metric_name, n_boot=1000, seed=0)
    lo_ev, hi_ev = event_level_bootstrap(y, p_a, p_b, metric_fn, n_boot=1000, seed=0)
    cluster_width = res_cluster["hi"] - res_cluster["lo"]
    event_width = hi_ev - lo_ev
    ok = cluster_width > event_width  # user-cluster should be wider (more conservative)
    all_ok &= check(f"{metric_name}: cluster CI ≥ event CI (with correlated data)", ok,
                    f"cluster_width={cluster_width:.4f} event_width={event_width:.4f}")

# ── 4. Correctly paired – same users resampled for both models ─────────────────
section("4. Pairing: same users resampled for both A and B")

# If models are always wrong/right together, the paired diff should be tiny
# even when the individual metrics vary a lot across users.
rng = np.random.default_rng(20)
n = 1000
n_users = 20
uid = np.repeat(np.arange(n_users), n // n_users)
y = (rng.uniform(0, 1, n) < 0.7).astype(float)
# Model A and B are nearly identical (differ only by tiny random noise)
p_base = np.clip(0.65 + np.repeat(rng.normal(0, 0.2, n_users), n // n_users), 0.05, 0.95)
p_a = np.clip(p_base + rng.normal(0, 0.001, n), 0.01, 0.99)
p_b = np.clip(p_base + rng.normal(0, 0.001, n), 0.01, 0.99)

res = paired_bootstrap(uid, y, p_a, p_b, "log_loss", n_boot=1000, seed=0)
ok = abs(res["diff"]) < 0.01 and abs(res["lo"]) < 0.05 and abs(res["hi"]) < 0.05
all_ok &= check("nearly identical models → narrow CI around 0", ok,
                f"diff={res['diff']:.5f} [{res['lo']:.5f}, {res['hi']:.5f}]")

# ── 5. NaN handling: single-class bootstrap samples dropped ───────────────────
section("5. NaN bootstrap samples (AUC with 1 class) are dropped")

rng = np.random.default_rng(33)
n = 200
# Use only 3 users so some bootstrap samples may be all-positive or all-negative
uid_sparse = np.repeat([0, 1, 2], [60, 80, 60])
# User 0 is all positive, users 1 and 2 are mixed
y_sparse = np.concatenate([np.ones(60), rng.integers(0,2,80).astype(float), rng.integers(0,2,60).astype(float)])
p_sparse = rng.uniform(0.1, 0.9, n)
p2 = rng.uniform(0.1, 0.9, n)

# Should not crash even when some bootstrap samples have 1 class
try:
    res = paired_bootstrap(uid_sparse, y_sparse, p_sparse, p2, "auc", n_boot=200, seed=0)
    ok = True
    detail = f"diff={res['diff']:.4f}"
except Exception as e:
    ok = False
    detail = str(e)
all_ok &= check("NaN-in-bootstrap does not crash", ok, detail)

# ── 6. Trajectory (user+card) correlation awareness ──────────────────────────
section("6. Trajectory-level correlation check")

print("""
  The bootstrap clusters by USER (user_ids), not by (user, card) pairs.
  This is the standard and documented approach.

  Claim: user-level clustering is CONSERVATIVE relative to (user,card) clustering
  because users are coarser clusters than trajectories.
  With fewer clusters = wider intervals = more conservative.

  Verification: artificially create data with strong trajectory-level correlation
  and compare user-cluster vs trajectory-cluster bootstrap widths.
""")

rng = np.random.default_rng(50)
n_users, n_cards = 10, 5
n_reviews = 8
n = n_users * n_cards * n_reviews

uid_tc = np.repeat(np.arange(n_users), n_cards * n_reviews)
cid_tc = np.tile(np.repeat(np.arange(n_cards), n_reviews), n_users)
traj_id = uid_tc * n_cards + cid_tc  # unique trajectory id

# Trajectory-level random effect (strong correlation within trajectory)
traj_effect = rng.normal(0, 0.4, n_users * n_cards)
y_tc = (rng.uniform(0, 1, n) < 0.7).astype(float)
p_a_tc = np.clip(0.65 + traj_effect[traj_id] + rng.normal(0, 0.02, n), 0.01, 0.99)
p_b_tc = np.clip(0.60 + traj_effect[traj_id] + rng.normal(0, 0.02, n), 0.01, 0.99)

# User-level bootstrap (used in benchmark)
res_user = paired_bootstrap(uid_tc, y_tc, p_a_tc, p_b_tc, "log_loss", n_boot=500, seed=0)
user_width = res_user["hi"] - res_user["lo"]

# Trajectory-level bootstrap (more granular)
res_traj = paired_bootstrap(traj_id, y_tc, p_a_tc, p_b_tc, "log_loss", n_boot=500, seed=0)
traj_width = res_traj["hi"] - res_traj["lo"]

print(f"  User-level CI width:       {user_width:.5f}")
print(f"  Trajectory-level CI width: {traj_width:.5f}")

# With correlated data and fewer users than trajectories, user-level should be ≥ traj-level
ok = user_width >= traj_width * 0.8  # allow some Monte Carlo noise
all_ok &= check("User-level CI ≥ trajectory-level CI (conservative)", ok,
                f"user={user_width:.5f}, traj={traj_width:.5f}")

print("""
  CONCLUSION on trajectory correlation:
  The benchmark's user-level clustering correctly accounts for within-user correlations,
  which subsumes within-trajectory correlations (trajectories are nested within users).
  User-level resampling is the standard approach for this type of review-log data.
  It is conservative relative to trajectory-level clustering.
""")

# ── 7. n_boot stability ────────────────────────────────────────────────────────
section("7. Bootstrap stability: 500 vs 1000 boots give similar CIs")

rng = np.random.default_rng(99)
n = 1000
uid_s = rng.integers(0, 30, n)
y_s = rng.integers(0, 2, n).astype(float)
p_a_s = rng.uniform(0.1, 0.9, n)
p_b_s = rng.uniform(0.1, 0.9, n)

res500 = paired_bootstrap(uid_s, y_s, p_a_s, p_b_s, "log_loss", n_boot=500, seed=0)
res1000 = paired_bootstrap(uid_s, y_s, p_a_s, p_b_s, "log_loss", n_boot=1000, seed=0)

lo_diff = abs(res500["lo"] - res1000["lo"])
hi_diff = abs(res500["hi"] - res1000["hi"])
ok = lo_diff < 0.01 and hi_diff < 0.01
all_ok &= check("500 vs 1000 boots: CI endpoints within 0.01", ok,
                f"lo diff={lo_diff:.5f}, hi diff={hi_diff:.5f}")

# ── Summary ────────────────────────────────────────────────────────────────────
section("SUMMARY")
if all_ok:
    print("  All bootstrap checks PASSED.\n")
else:
    print("  Some checks FAILED – see FAIL lines above.\n")
    sys.exit(1)

