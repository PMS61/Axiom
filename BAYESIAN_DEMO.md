# Bayesian scheduler prototype

The `BayesianScheduler` branch now has a public local workbench at
[http://localhost:3000/bayesian-demo/recall](http://localhost:3000/bayesian-demo/recall).
No account, database, Gemini key, downloaded dataset or training job is needed.

## Start

From the repository root (Python 3.10+ and Node 22.18+):

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements-demo.txt
cd client
npm install
AXIOM_PYTHON="$PWD/../.venv/bin/python" npm run dev
```

If your `python3` already has NumPy installed, simply use `npm run dev` from
`client/`. The Python process runs locally on the Next.js server. Keep the full
repository present; this adapter is not packaged for serverless deployment.

## Three-minute walkthrough

1. Open `/bayesian-demo/recall`. Six synthetic topic histories fit immediately. Dots
   in each history mean remembered; crosses mean forgot. Hover for each outcome.
2. Point out the dynamic prediction versus the static-model diamond. The wide
   bands show a 90% interval over next-event probabilities, including process
   uncertainty. They are not confidence intervals for a measured learning gain.
3. Inspect the risk-first queue starting at 09:00. Change the study window from
   60 to 30 or 90 minutes. Whole review sessions always fit within the window;
   any leftover time is shown. The demo assumes continuous availability.
4. Expand the fixed-order comparison: the same topics and durations are packed
   in original topic order without recall predictions.
5. Select Bayes’ theorem, try its question and reveal the answer. Click
   **I remembered** or **I forgot**. The actual Python samplers refit, the review
   count increases, and the queue is recomputed. The status shows the prediction
   before and after. Hierarchical sharing can also change other topics.
6. Reset to reproduce the starting predictions. State is page-session only;
   reload also resets. Feedback is capped at 40 reviews per topic.

## What is connected

- `dynamic_retention_model.py`: existing AR(1) logit retention model with
  Pólya–Gamma Gibbs sampling and FFBS. Now retains terminal-state draws alongside
  the corresponding parameter draws for posterior predictive forecasts. Existing
  posterior-mean trajectories and evaluation interfaces are preserved.
- `bayesian_demo.py`: deterministic fixture adapter, seed 42, 480 iterations,
  160 burn-in, thinning 2 (160 retained draws). Uses 30 process-noise replicates
  per posterior draw and the real `StaticRetentionModel` reference.
- `/api/bayesian-demo/recall`: validated, bounded binary histories; invokes Python
  without a shell and times out after 30 seconds. Returns an actionable error
  if Python/NumPy is unavailable. No history is saved on the server.
- `client/src/lib/bayesian/demo.ts`: explicitly heuristic, ascending recall
  order, budget-constrained whole-session placement. Fixed-order comparison
  uses the same packing rule. These are separate from the old app scheduler.

## Research boundaries

This demonstrates a retention-informed scheduling loop, not the full M1–M4
cognitive-load research system. It uses labelled synthetic data and self-reported
feedback. It does not initialize or mutate the adaptive profile. Durations and
questions are fixtures, not M4 estimates. There are no availability windows,
focus calibration, deadline optimization or measured scheduling outcomes.

The existing dynamic model does **not** use `dt`: predictions are per review
event, not a forgetting curve over hours/days. The existing Pólya–Gamma sampler
uses a truncated series; persistence draws are clipped. A short single chain is
suitable for demonstrating behavior, not establishing posterior convergence or
research accuracy. The static model shown is the branch's existing implementation.
Risk-first scheduling is a new demo heuristic, not an optimal scheduling result.
The fixed-order illustration is not a frozen evaluation baseline or evidence of
improved learning.

## Checks

From `client/`:

```sh
npm run test:bayesian
npm run lint
npm run build
```

`test:bayesian` uses Node's built-in test runner (no added test framework) for
budget, ordering and baseline invariants, then Python unittest for reproducibility,
feedback sensitivity, forecast bounds, terminal-state draw alignment and invalid
histories. Python tests use `python3`; activate the virtual environment first
when needed. The existing repository-wide lint issues are outside this demo.
