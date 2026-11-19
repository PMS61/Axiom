# Axiom presentation code

Open an experiment's `run.py` to show its implementation. From the repository root, run the same file in the [diagnostics Python environment](../RESEARCH_WORKSPACE.md).

```bash
.venv-diagnostics/bin/python presentation/experiments/srs_final/run.py
```

The default command runs the checked-in model and online update on synthetic reviews and prints freshly computed metrics. For full public benchmark reproduction, pass `--live --out /tmp/axiom-test.jsonl` and provide the original Anki data, side tables and external srs-benchmark checkout described in the [benchmark protocol](../paper/docs/SRSB_BEAT.md). Synthetic results demonstrate the implementation and do not estimate public benchmark performance.

## Experiment files

| Experiment | Source and entry point | Run mode or inputs |
|---|---|---|
| Frozen GBDT + online adaptation | [run.py](experiments/srs_final/run.py) | Synthetic inference with the checked-in model. Use `--saved` for the historical table. |
| Pooled GBDT | [run.py](experiments/srs_first/run.py) | Synthetic training with separate fit, early-stopping and test users. |
| Online adaptation | [run.py](experiments/srs_online_adaptation/run.py) | Review-by-review synthetic trace using the actual update kernel. |
| Public FSRS ablation | [run.py](experiments/srs_fsrs_ablation/run.py) | Retrains both GBDTs on synthetic data. Use `--live` for the benchmark cache. |
| Phase 7 feature ablation | [run.py](experiments/gbdt_ablation/run.py) | Synthetic training by default. Use `--dataset anki --models all no_fsrs` for the real validation cache. |
| Static vs dynamic Bayesian recall | [run.py](experiments/bayesian_recall/run.py) | Fits both models to six synthetic card histories. |
| Model stacking | [run.py](experiments/model_stacking/run.py) | Fits the stack and blend out of fold on synthetic probabilities. Use `--live` for cached predictions. |
| Anki | [run.py](experiments/anki/run.py) | Use `--demo` for synthetic revlogs. The default uses gated Anki data. |
| Duolingo | [run.py](experiments/duolingo/run.py) | Fits FSRS and GBDT on a small sample from the local Duolingo archive. |
| GBDT feature and scale sweep | [run.py](experiments/srs_feature_sweep/run.py) | Requires revlogs, training and validation user IDs, and an output path. |
| GBDT-Curve | [run.py](experiments/srs_curve/run.py) | Requires benchmark inputs and cohort arguments. |
| Residual GRU | [run.py](experiments/srs_residual_gru/run.py) | Requires PyTorch, benchmark inputs and model artifacts. |
| Per-user boosting | [run.py](experiments/srs_user_boosting/run.py) | Requires benchmark inputs and model artifacts. |
| Kalman drift | [run.py](experiments/srs_kalman/run.py) | Requires benchmark inputs and model artifacts. |
| Time-aware attention | [run.py](experiments/srs_attention/run.py) | Requires PyTorch and benchmark inputs. |
| User-type experts | [run.py](experiments/srs_user_experts/run.py) | Requires benchmark feature caches. |
| Mixture-of-experts gate | [run.py](experiments/srs_expert_gate/run.py) | Requires prediction parquet files. |
| Form-by-gap experts | [run.py](experiments/srs_grid_experts/run.py) | Requires benchmark feature caches. |
| Elo-band experts | [run.py](experiments/srs_elo_experts/run.py) | Requires benchmark feature caches. Pass a fresh `--out` directory. |
| Elo experts without FSRS | [run.py](experiments/srs_elo_without_fsrs/run.py) | Same trainer, with FSRS inputs masked. Pass `--mode none` or `mixed` and a fresh `--out` directory. |
| Inference speed | [run.py](experiments/srs_speed/run.py) | Requires benchmark revlogs and model checkpoints. |
| Bayesian v3 variants | [run.py](experiments/bayesian_variants/run.py) | Requires PyTorch, a validation dataset and the Phase 6 reference prediction cache. |
| Task 3 sampler diagnostics | [run.py](experiments/sampler_diagnostics/run.py) | Requires the coefficient study output and a fresh `--out` directory. |

Most data-dependent runners list their CLI arguments with `--help`. Runners that use the original fixed paths expect to be launched from the repository root. If a feature-cache path is empty, rebuild it with the feature sweep runner. The learned-FSRS experiment is [documented separately](experiments/learned_fsrs/README.md) because its source branch is unavailable.

## Interfaces

Start the app from `client/` with the commands below, then open the route. The Webpack dev server serves the nested guest dashboard routes in this checkout.

```bash
cd client
npm exec -- next dev --webpack -p 3000
```

Each presentation interface has its own Next.js route folder. Shared components and model code are imported from their existing locations.

| Interface | Browser route | Source folder |
|---|---|---|
| Research workspace | [/bayesian-demo](http://localhost:3000/bayesian-demo) | [bayesian-demo](../client/src/app/bayesian-demo/) |
| Bayesian recall | [/bayesian-demo/recall](http://localhost:3000/bayesian-demo/recall) | [recall](../client/src/app/bayesian-demo/recall/) |
| Benchmark comparison | [/bayesian-demo/benchmarks](http://localhost:3000/bayesian-demo/benchmarks) | [benchmarks](../client/src/app/bayesian-demo/benchmarks/) |
| Research audit | [/bayesian-demo/audit](http://localhost:3000/bayesian-demo/audit) | [audit](../client/src/app/bayesian-demo/audit/) |
| Review demo | [/review-demo](http://localhost:3000/review-demo) | [review-demo](../client/src/app/review-demo/) |
| Learning library | [/learn](http://localhost:3000/learn) | [learn](../client/src/app/learn/) |
| Trivia | [/learn/trivia](http://localhost:3000/learn/trivia) | [trivia](../client/src/app/learn/trivia/) |
| Guest dashboard | [/dashboard/demo](http://localhost:3000/dashboard/demo) | [dashboard/demo](../client/src/app/dashboard/demo/) |
| Guest roadmap | [/dashboard/demo/roadmap](http://localhost:3000/dashboard/demo/roadmap) | [roadmap](../client/src/app/dashboard/demo/roadmap/) |
| Guest discovery | [/dashboard/demo/discover](http://localhost:3000/dashboard/demo/discover) | [discover](../client/src/app/dashboard/demo/discover/) |
| Guest study | [/dashboard/demo/study](http://localhost:3000/dashboard/demo/study) | [study](../client/src/app/dashboard/demo/study/) |
