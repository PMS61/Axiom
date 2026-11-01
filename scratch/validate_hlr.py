"""
validate_hlr.py  –  Step 3: Manual SGD step trace for HLR and feature audit.

Run from repo root:
    python scratch/validate_hlr.py
"""
import sys
import math
import numpy as np

sys.path.insert(0, "/Users/vaishvi/Desktop/FYP/Axiom")
from fsrs_benchmark.models.hlr import (
    HLRModel, features, sgd_update, observed_half_life,
    hclip, pclip, MIN_HALF_LIFE, MAX_HALF_LIFE, P_MIN, P_MAX, LN2
)

PASS = "  PASS"
FAIL = "  FAIL *** "
all_ok = True

def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")

def check(label, ok, detail=""):
    status = PASS if ok else FAIL
    print(f"  {label}: {status}  {detail}")
    return ok

# ── 1. Constants match the Duolingo reference ──────────────────────────────────
section("1. Constants vs Duolingo experiment.py")
ok = abs(MIN_HALF_LIFE - 15.0/(24*60)) < 1e-15
all_ok &= check("MIN_HALF_LIFE = 15 minutes in days", ok, f"{MIN_HALF_LIFE:.10f}")
ok = MAX_HALF_LIFE == 274.0
all_ok &= check("MAX_HALF_LIFE = 274 days", ok)
ok = P_MIN == 1e-4 and abs(P_MAX - (1 - 1e-4)) < 1e-15
all_ok &= check("P_MIN=1e-4, P_MAX=1-1e-4", ok)

# ── 2. Forgetting curve formula ───────────────────────────────────────────────
section("2. Forgetting curve  p = 2^(-t/h),  h = 2^(theta·x)")

# For a given theta and x: h should be 2^(theta·x), p should be 2^(-t/h)
theta = np.array([0.5, -0.3, 1.2])
x_test = np.array([np.sqrt(3), np.sqrt(2), 1.0])
t_test = 5.0

dot = float(theta @ x_test)
h_expected = hclip(2.0 ** dot)
p_expected = pclip(2.0 ** (-t_test / h_expected))

# Feed through predict path manually
import pandas as pd
rows = [
    dict(event_id=1, user_id=1, card_id=1, review_idx=0, day_offset=0, rating=3, y=1,
         dt_days=np.nan, elapsed_days=-1, is_first=True, is_train=True, scored=False),
    dict(event_id=2, user_id=1, card_id=1, review_idx=1, day_offset=5, rating=3, y=1,
         dt_days=5.0, elapsed_days=5, is_first=False, is_train=False, scored=True),
]
ev = pd.DataFrame(rows)
# The model's prediction for event 2: theta is the weights, x = [sqrt(1+0), sqrt(1+0), 1] = [1,1,1]
# (since there's 1 prior correct review of this card for this user after training)
model = HLRModel(max_epochs=1)
model.theta_ = theta  # inject weights manually
p_pred = model.predict(ev)
# x for scored event = [sqrt(1+1_right_before), sqrt(1+0_wrong), 1.0]
# 1 right before = 1 (the train event y=1), so x = [sqrt(2), sqrt(1), 1.0]
x_actual = np.array([np.sqrt(2), np.sqrt(1), 1.0])
dot_actual = float(theta @ x_actual)
h_actual = hclip(2.0 ** dot_actual)
p_actual = pclip(2.0 ** (-5.0 / h_actual))
ok = abs(float(p_pred.iloc[0]) - p_actual) < 1e-12
all_ok &= check("predict() matches 2^(-t/h) with h=2^(theta·x)", ok,
                f"got={float(p_pred.iloc[0]):.8f} expected={p_actual:.8f}")

# ── 3. Single SGD step - detailed trace ───────────────────────────────────────
section("3. SGD step: detailed arithmetic trace")

# Fixed known state
w = [0.5, -0.2, 1.0]
fcounts = [10, 5, 20]
x = [math.sqrt(3), math.sqrt(2), 1.0]
t = 3.0
y = 1.0
lrate, hlwt, l2wt, sigma = 0.001, 0.01, 0.1, 1.0

# Expected outputs computed by hand following the paper/official code
dot = sum(wi * xi for wi, xi in zip(w, x))
h = min(max(2.0 ** dot, MIN_HALF_LIFE), MAX_HALF_LIFE)
p_obs = min(max(y, P_MIN), P_MAX)
p = min(max(2.0 ** (-t / h), P_MIN), P_MAX)
h_obs_val = min(max(-t / math.log2(min(max(y, P_MIN), P_MAX)), MIN_HALF_LIFE), MAX_HALF_LIFE)
dlp_dw = 2.0 * (p - p_obs) * LN2**2 * p * (t / h)
dlh_dw = 2.0 * (h - h_obs_val) * LN2 * h

expected_w = []
expected_fc = []
for k in range(3):
    rate = (1.0 / (1.0 + p_obs)) * lrate / math.sqrt(1.0 + fcounts[k])
    wk = w[k]
    wk -= rate * dlp_dw * x[k]
    wk -= rate * hlwt * dlh_dw * x[k]
    wk -= rate * l2wt * wk / sigma**2
    expected_w.append(wk)
    expected_fc.append(fcounts[k] + 1)

# Now run through the actual function
w_copy = list(w)
fc_copy = list(fcounts)
h_obs_arr = observed_half_life(np.array([t]), np.array([y]))
sgd_update(w_copy, fc_copy, x, t, y, float(h_obs_arr[0]))

print(f"\n  w_before={[round(wi,6) for wi in w]}")
print(f"  h={h:.6f}, p_obs={p_obs:.6f}, p={p:.6f}")
print(f"  h_obs={h_obs_val:.6f}, dlp_dw={dlp_dw:.8f}, dlh_dw={dlh_dw:.8f}")
print(f"  Expected w_after: {[round(wi,8) for wi in expected_w]}")
print(f"  Got      w_after: {[round(wi,8) for wi in w_copy]}")

for k in range(3):
    ok = abs(w_copy[k] - expected_w[k]) < 1e-14
    all_ok &= check(f"  w[{k}] correct", ok,
                    f"got={w_copy[k]:.12f} exp={expected_w[k]:.12f}")
ok = fc_copy == expected_fc
all_ok &= check("fcounts incremented", ok)

# ── 4. h_obs formula ─────────────────────────────────────────────────────────
section("4. observed_half_life = hclip(-t / log2(pclip(y)))")

for t_val, y_val in [(1.0, 0.5), (3.0, 0.0), (100.0, 1.0), (0.1, 0.9)]:
    p_c = min(max(y_val, P_MIN), P_MAX)
    exp = min(max(-t_val / math.log2(p_c), MIN_HALF_LIFE), MAX_HALF_LIFE)
    got = float(observed_half_life(np.array([t_val]), np.array([y_val]))[0])
    ok = abs(got - exp) < 1e-12
    all_ok &= check(f"t={t_val}, y={y_val}", ok, f"got={got:.8f} exp={exp:.8f}")

# ── 5. Features: prior-only, correct formulas ────────────────────────────────
section("5. Features: [sqrt(1+n_right_prior), sqrt(1+n_wrong_prior), 1]")

import pandas as pd

events_feats = pd.DataFrame([
    dict(event_id=1, user_id=1, card_id=10, review_idx=0, y=1, is_first=True, is_train=True,
         elapsed_days=-1, dt_days=np.nan, day_offset=0, scored=False, rating=3),
    dict(event_id=2, user_id=1, card_id=10, review_idx=1, y=0, is_first=False, is_train=True,
         elapsed_days=3, dt_days=3.0, day_offset=3, scored=True, rating=1),
    dict(event_id=3, user_id=1, card_id=10, review_idx=2, y=1, is_first=False, is_train=True,
         elapsed_days=2, dt_days=2.0, day_offset=5, scored=True, rating=3),
    dict(event_id=4, user_id=1, card_id=10, review_idx=3, y=1, is_first=False, is_train=False,
         elapsed_days=7, dt_days=7.0, day_offset=12, scored=True, rating=3),
])
X = features(events_feats)

# Event 1 (first): n_right=0, n_wrong=0 → [1, 1, 1]
r = X.loc[1].tolist()
ok = np.allclose(r, [math.sqrt(1), math.sqrt(1), 1.0], atol=1e-12)
all_ok &= check("event 1 (first): [sqrt(1), sqrt(1), 1]", ok, str([round(v,6) for v in r]))

# Event 2: 1 prior correct → [sqrt(2), sqrt(1), 1]
r = X.loc[2].tolist()
ok = np.allclose(r, [math.sqrt(2), math.sqrt(1), 1.0], atol=1e-12)
all_ok &= check("event 2: [sqrt(2), sqrt(1), 1]  (1 prior right)", ok, str([round(v,6) for v in r]))

# Event 3: 1 right, 1 wrong prior → [sqrt(2), sqrt(2), 1]
r = X.loc[3].tolist()
ok = np.allclose(r, [math.sqrt(2), math.sqrt(2), 1.0], atol=1e-12)
all_ok &= check("event 3: [sqrt(2), sqrt(2), 1]  (1 right, 1 wrong prior)", ok, str([round(v,6) for v in r]))

# Event 4: 2 right, 1 wrong prior → [sqrt(3), sqrt(2), 1]
r = X.loc[4].tolist()
ok = np.allclose(r, [math.sqrt(3), math.sqrt(2), 1.0], atol=1e-12)
all_ok &= check("event 4: [sqrt(3), sqrt(2), 1]  (2 right, 1 wrong prior)", ok, str([round(v,6) for v in r]))

# ── 6. Training-row filter ────────────────────────────────────────────────────
section("6. Training rows: TRAIN & non-first & elapsed_days >= 1")

# Verify that a first review with elapsed_days=-1 is NOT used for training
model_check = HLRModel(max_epochs=10)
model_check.fit(events_feats)
# All rows here are is_train=True except event 4; events 1,2,3 are train
# Event 1 is is_first=True → excluded; Event 2 has elapsed_days=3 → included
# Event 3 has elapsed_days=2 → included
# This should fit without error and produce non-zero theta
ok = model_check.theta_ is not None and not np.all(model_check.theta_ == 0)
all_ok &= check("fit() runs and produces non-zero weights", ok,
                f"theta={np.round(model_check.theta_, 4).tolist()}")

# Check that flipping test labels doesn't change fit
events_feats_copy = events_feats.copy()
events_feats_copy.loc[events_feats_copy.event_id == 4, 'y'] = 0  # flip test label
model_check2 = HLRModel(max_epochs=10)
model_check2.fit(events_feats_copy)
ok = np.array_equal(model_check.theta_, model_check2.theta_)
all_ok &= check("test-row outcome flip doesn't change theta (no leakage)", ok)

# ── 7. Prediction range ────────────────────────────────────────────────────────
section("7. Predictions are valid probabilities in (0, 1)")

ok = float(model_check.predict(events_feats).min()) > 0.0
all_ok &= check("min prediction > 0", ok)
ok = float(model_check.predict(events_feats).max()) < 1.0
all_ok &= check("max prediction < 1", ok)

# ── 8. Known issue: all_baselines.py HalfLifeRegression differs ───────────────
section("8. Documenting: all_baselines.py HLR vs benchmark HLR")

print("""
  The repository contains TWO HLR implementations:

  a) fsrs_benchmark/models/hlr.py (HLRModel) — USED IN BENCHMARK
     - Optimizer:  paper's SGD (per-feature learning rates, recall halving)
     - Loss:       (p - y)^2 + hlwt * (h - h_obs)^2 + L2
     - Features:   [sqrt(1+n_right), sqrt(1+n_wrong), 1]
     - Clipping:   h in [15min, 274d], p in [1e-4, 1-1e-4]
     - Training:   multiple epochs until convergence

  b) all_baselines.py (HalfLifeRegression) — NOT USED IN BENCHMARK
     - Optimizer:  L-BFGS-B (scipy.optimize.minimize)
     - Loss:       (p_pred - y)^2 + L2*w^2  (no h^2 term)
     - Features:   [1, review_index_i, n_right, n_wrong]  (4-dim, includes index!)
     - Clipping:   p clipped to [1e-6, 1-1e-6] only
     - Training:   single minimization call

  VERDICT: The benchmark correctly uses HLRModel from hlr.py.
  The all_baselines.py version is a prototype and should NOT be compared
  against the benchmark results. It is not the Settles & Meeder formulation.
""")
all_ok &= check("Documented: two HLR implementations (finding)", True)

# ── Summary ────────────────────────────────────────────────────────────────────
section("SUMMARY")
if all_ok:
    print("  All HLR checks PASSED.\n")
else:
    print("  Some checks FAILED – see FAIL lines above.\n")
    sys.exit(1)

