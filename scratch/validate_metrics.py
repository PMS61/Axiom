"""
validate_metrics.py  –  Step 2: Independent cross-check of fsrs_benchmark.metrics
against sklearn and brute-force reference implementations.

Run from repo root:
    python scratch/validate_metrics.py
"""
import sys
import numpy as np

sys.path.insert(0, "/Users/vaishvi/Desktop/FYP/Axiom")
from fsrs_benchmark.metrics import log_loss, brier, auc, ece, reliability_curve, paired_bootstrap

try:
    from sklearn.metrics import (
        log_loss as sk_log_loss,
        brier_score_loss as sk_brier,
        roc_auc_score as sk_auc,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("WARNING: sklearn not available – sklearn cross-checks skipped")

PASS = "  PASS"
FAIL = "  FAIL *** "

def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")

def check(label, ok, detail=""):
    status = PASS if ok else FAIL
    print(f"  {label}: {status}  {detail}")
    return ok

all_ok = True

# ── 1. Log loss ────────────────────────────────────────────────────────────────
section("1. Log loss")

# 1a. Hand-computed known case
y_hand = np.array([1.0, 0.0, 1.0])
p_hand = np.array([0.9, 0.2, 0.7])
eps = 1e-6
expected_ll = -np.mean([np.log(0.9), np.log(0.8), np.log(0.7)])
got_ll = log_loss(y_hand, p_hand)
ok = abs(got_ll - expected_ll) < 1e-12
all_ok &= check("hand-computed", ok, f"got={got_ll:.8f} expected={expected_ll:.8f}")

# 1b. Clipping: p=0 and p=1 should not crash and should use eps
y_clip = np.array([1.0, 0.0])
p_clip = np.array([0.0, 1.0])
got_clipped = log_loss(y_clip, p_clip)
expected_clipped = -np.mean([np.log(eps), np.log(eps)])
ok = abs(got_clipped - expected_clipped) < 1e-10
all_ok &= check("clipping p=0,1 uses eps=1e-6", ok, f"got={got_clipped:.6f} expected={expected_clipped:.6f}")

# 1c. sklearn cross-check
if HAS_SKLEARN:
    rng = np.random.default_rng(42)
    y_r = rng.integers(0, 2, 500).astype(float)
    p_r = rng.uniform(0.01, 0.99, 500)
    ours = log_loss(y_r, p_r)
    # sklearn clips at eps=1e-15 by default; re-clip at 1e-6 to match ours
    p_clipped = np.clip(p_r, 1e-6, 1 - 1e-6)
    sk = sk_log_loss(y_r, p_clipped)
    ok = abs(ours - sk) < 1e-10
    all_ok &= check("matches sklearn (p already in (1e-6, 1-1e-6))", ok, f"ours={ours:.8f} sklearn={sk:.8f}")

# ── 2. Brier score ─────────────────────────────────────────────────────────────
section("2. Brier score")

y_b = np.array([1.0, 0.0, 1.0, 0.0])
p_b = np.array([0.8, 0.3, 0.6, 0.4])
expected_brier = np.mean([(0.8-1)**2, (0.3-0)**2, (0.6-1)**2, (0.4-0)**2])
got_brier = brier(y_b, p_b)
ok = abs(got_brier - expected_brier) < 1e-14
all_ok &= check("hand-computed", ok, f"got={got_brier:.8f} expected={expected_brier:.8f}")

if HAS_SKLEARN:
    rng = np.random.default_rng(7)
    y_r = rng.integers(0, 2, 1000).astype(float)
    p_r = rng.uniform(0.0, 1.0, 1000)
    ours = brier(y_r, p_r)
    sk = sk_brier(y_r, p_r)   # sklearn uses mean((p - y)^2) for binary
    ok = abs(ours - sk) < 1e-13
    all_ok &= check("matches sklearn", ok, f"ours={ours:.10f} sklearn={sk:.10f}")

# ── 3. AUC ────────────────────────────────────────────────────────────────────
section("3. AUC (Mann-Whitney rank-sum)")

# 3a. Textbook case: AUC = 0.75
y_auc = np.array([0.0, 0.0, 1.0, 1.0])
p_auc = np.array([0.1, 0.4, 0.35, 0.8])
got = auc(y_auc, p_auc)
ok = abs(got - 0.75) < 1e-12
all_ok &= check("textbook AUC=0.75", ok, f"got={got:.6f}")

# 3b. Perfect predictor → AUC = 1.0
y_perf = np.array([0.0, 0.0, 1.0, 1.0])
p_perf = np.array([0.1, 0.2, 0.8, 0.9])
ok = abs(auc(y_perf, p_perf) - 1.0) < 1e-12
all_ok &= check("perfect predictor → AUC=1.0", ok)

# 3c. Worst predictor → AUC = 0.0
p_worst = np.array([0.9, 0.8, 0.2, 0.1])
ok = abs(auc(y_perf, p_worst) - 0.0) < 1e-12
all_ok &= check("worst predictor → AUC=0.0", ok)

# 3d. Ties: constant predictor → AUC = 0.5
y_tie = np.array([0.0, 0.0, 1.0, 1.0])
p_const = np.array([0.6, 0.6, 0.6, 0.6])
ok = abs(auc(y_tie, p_const) - 0.5) < 1e-12
all_ok &= check("constant predictor (ties) → AUC=0.5", ok, f"got={auc(y_tie, p_const):.6f}")

# 3e. Single class → NaN
y_single = np.array([1.0, 1.0, 1.0])
p_single = np.array([0.7, 0.8, 0.9])
ok = np.isnan(auc(y_single, p_single))
all_ok &= check("single class → NaN", ok)

# 3f. Brute-force pairwise comparison vs rank-sum
rng = np.random.default_rng(99)
for trial in range(5):
    n = 200
    y_r = rng.integers(0, 2, n).astype(float)
    while y_r.sum() == 0 or y_r.sum() == n:
        y_r = rng.integers(0, 2, n).astype(float)
    p_r = rng.uniform(0.0, 1.0, n)
    got = auc(y_r, p_r)
    # Brute-force: count concordant pairs (with ties as 0.5)
    pos_idx = np.where(y_r == 1)[0]
    neg_idx = np.where(y_r == 0)[0]
    total = 0.0
    for pi in pos_idx:
        for ni in neg_idx:
            if p_r[pi] > p_r[ni]:
                total += 1.0
            elif p_r[pi] == p_r[ni]:
                total += 0.5
    brute = total / (len(pos_idx) * len(neg_idx))
    ok = abs(got - brute) < 1e-12
    all_ok &= check(f"brute-force trial {trial+1} (n={n})", ok,
                    f"rank-sum={got:.8f} brute={brute:.8f}")

if HAS_SKLEARN:
    rng = np.random.default_rng(55)
    y_r = rng.integers(0, 2, 500).astype(float)
    p_r = rng.uniform(0.0, 1.0, 500)
    ours = auc(y_r, p_r)
    sk = sk_auc(y_r, p_r)
    ok = abs(ours - sk) < 1e-10
    all_ok &= check("matches sklearn roc_auc_score", ok, f"ours={ours:.8f} sklearn={sk:.8f}")

# ── 4. ECE ─────────────────────────────────────────────────────────────────────
section("4. ECE (10 equal-width bins)")

# 4a. Perfect calibration: mean_p = mean_y in every bin → ECE = 0
y_ec = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0])
p_ec = np.array([0.1, 0.2, 0.7, 0.8, 0.15, 0.75])
# mean_p ≈ 0.175 in [0,0.3) bin; actual outcomes: two 0s, zero 1s → not perfect calibration
# Build perfectly calibrated data: for each bin, set p = fraction of 1s
rng = np.random.default_rng(0)
n_per_bin = 100
y_perf_list, p_perf_list = [], []
for b in range(10):
    mid = (b + 0.5) / 10.0
    # Assign exactly round(mid*n) outcomes = 1
    n_ones = round(mid * n_per_bin)
    y_perf_list.extend([1.0] * n_ones + [0.0] * (n_per_bin - n_ones))
    p_perf_list.extend([mid] * n_per_bin)
y_perf_ec = np.array(y_perf_list)
p_perf_ec = np.array(p_perf_list)
got_ece_perf = ece(y_perf_ec, p_perf_ec)
ok = got_ece_perf < 0.05  # should be very small (near-perfect calibration)
all_ok &= check("near-perfect calibration → small ECE", ok, f"ECE={got_ece_perf:.6f}")

# 4b. Hand-computed: all predictions in bin 0 ([0, 0.1)), all outcomes 1 → gap = |0.05 - 1.0|
y_hand_ece = np.array([1.0, 1.0, 1.0, 1.0])
p_hand_ece = np.array([0.05, 0.05, 0.05, 0.05])
expected_ece_hand = abs(0.05 - 1.0)  # = 0.95, all 4 in one bin, weight = 4/4 = 1
got_hand = ece(y_hand_ece, p_hand_ece)
ok = abs(got_hand - expected_ece_hand) < 1e-12
all_ok &= check("hand-computed single bin", ok, f"got={got_hand:.6f} expected={expected_ece_hand:.6f}")

# 4c. Independent loop implementation
def ece_ref(y, p, n_bins=10):
    y, p = np.asarray(y), np.asarray(p)
    total = len(y)
    result = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        if b == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        n_b = mask.sum()
        if n_b == 0:
            continue
        gap = abs(p[mask].mean() - y[mask].mean())
        result += (n_b / total) * gap
    return result

rng = np.random.default_rng(1234)
for trial in range(5):
    y_r = rng.integers(0, 2, 300).astype(float)
    p_r = rng.uniform(0.0, 1.0, 300)
    ours = ece(y_r, p_r)
    ref = ece_ref(y_r, p_r)
    ok = abs(ours - ref) < 1e-12
    all_ok &= check(f"independent loop trial {trial+1}", ok,
                    f"ours={ours:.8f} ref={ref:.8f}")

# 4d. ECE with high recall rate (like the Anki dataset, ~83% recall)
rng = np.random.default_rng(77)
n = 10000
y_anki = (rng.uniform(0, 1, n) < 0.83).astype(float)
p_anki = rng.beta(8, 2, n)  # skewed toward high p
ece_anki = ece(y_anki, p_anki)
rc = reliability_curve(y_anki, p_anki)
n_empty_low = int((rc["n"][:5] == 0).sum())
all_ok &= check("high-recall scenario: ECE computed without crash", True,
                f"ECE={ece_anki:.4f}, empty low bins (0-0.5)={n_empty_low}/5")

# Report how many low-p bins are empty (calibration concern noted in plan)
print(f"\n    [INFO] Reliability bins for high-recall data (83%):")
print(f"    {'Bin':<12}{'n':>6}{'mean_p':>10}{'mean_y':>10}")
for _, row in rc.iterrows():
    print(f"    [{row.bin_lo:.1f}-{row.bin_hi:.1f})  {row.n:>6}  {row.mean_p:>10.4f}  {row.mean_y:>10.4f}")

# ── 5. Reliability curve ───────────────────────────────────────────────────────
section("5. Reliability curve")

rc_test = reliability_curve(np.array([1,0,1,1,0,0]), np.array([0.75,0.25,0.85,0.65,0.15,0.35]))
ok = (rc_test["n"].sum() == 6)
all_ok &= check("all events counted", ok)
ok = (rc_test.columns.tolist() == ["bin_lo","bin_hi","n","mean_p","mean_y"])
all_ok &= check("correct columns", ok)
ok = all(rc_test.loc[rc_test["n"]==0, "mean_p"].isna())
all_ok &= check("empty bins have NaN mean", ok)

# ── 6. Summary ─────────────────────────────────────────────────────────────────
section("SUMMARY")
if all_ok:
    print("  All checks PASSED – metrics implementations are correct.\n")
else:
    print("  Some checks FAILED – see FAIL lines above.\n")
    sys.exit(1)

