# Plan 1 — Research Paper: The Scheduler (When Pillar)

## Branch strategy

Build this plan on its own branch (e.g. `plan/01-research-scheduler`), off current `master`. Fully independent of Plans 2, 3, and 5 — the `research/` harness this plan builds touches no product code path, so it can be developed and merged in any order relative to the other three remaining plans. Plan 4 (personalization) is already merged and has no bearing on this plan either way.

## Context

Two documents already exist that both talk about the scheduler and disagree with each other:

1. **`codewiser (1).pdf`** — a drafted paper ("Axiom: A Deterministic Cognitive Scheduling System for Structured Learning", authored with Ghruank Kothare and Harshal Kamble) claiming the pipeline includes a 0/1 Knapsack DP optimizer (`/lib/knapsack.ts`), Lagrangian relaxation (`/lib/lagrangian.ts`), a Markov transition model (`/lib/markov.ts`), and DFT-based energy-rhythm analysis (`/lib/fourier.ts`), evaluated on 100 simulated learners with results (84% completion, 81% adherence, 9% overload, 7% deadline miss) that beat an "AI Scheduler" and "Static Planner" baseline.
2. **The actual codebase** (`client/src/lib/engine.ts`, `schedule.ts`, `energyModel.ts`, `scoring.ts`, `constraints.ts`) — none of the four files above exist. The real scheduler is a **single-pass, order-dependent greedy heuristic**: sort tasks once, place each into the best-fitness free slot, never revisit. There is no DP table, no Lagrangian objective, no Markov chain, no Fourier transform anywhere in the repo. The energy model, urgency function, and composite score formulas in the PDF (§5.1–5.4) are real and match the code — everything past §5.5 (chunking, Knapsack, Lagrangian, Markov, Fourier) is aspirational text with no implementation and no real simulation behind the numbers.
3. **The project's original research brief** — a more recent, honest document that already admits this: it calls the greedy loop "a reasonable product heuristic, but an unverified one," names that gap as the thesis, and lays out a fully computational (no human subjects) research design: synthetic user/task generators, an algorithm bank (FCFS, EDF, current greedy, proposed batch optimizer, optional RL), a metrics engine, and paired statistical comparison. (This document was superseded by the `plans/` directory and removed from the repo — referenced here only for the research design it originated, which the rest of this plan carries forward.)

**The paper cannot be finished by writing prose around the PDF's claims — the claims aren't real.** The only honest path is to build the real thing the PDF describes (or an honestly-scoped subset of it), run it, and write the paper around what actually happened. This plan is that build-then-write path, using the brief's research design as the methodology and the PDF's structure/formatting as the paper skeleton.

## Scope decision: what to actually build

Not all four "missing" techniques are equally worth building. Triaged by effort vs. payoff:

| Technique | Verdict | Why |
|---|---|---|
| **0/1 Knapsack DP optimizer** | **Build — core deliverable** | Directly replaces the greedy per-section placement (`schedule.ts`'s `tryAllocate` loop) with a real optimal-subset solver over each section's axiom budget. Small, deterministic, provably correct, and is the single biggest gap between the PDF's claim and reality. This is what makes the "batch optimizer" comparison in the original brief §05 real instead of theoretical. |
| **Global batch reformulation (ILP/CP-SAT or GA fallback)** | **Build — core deliverable** | Answers the brief's actual research question: how much schedule quality does the order-dependent greedy loop leave on the table vs. a global optimizer over the full task pool and horizon. Knapsack-per-section is a stepping stone; this is the real experiment. |
| **Lagrangian relaxation** | **Build — secondary** | Cheap to add on top of the Knapsack/ILP result: soft penalties for excess energy, load-balance deviation, and habit disruption (the exact three terms the PDF already specifies in §5.11). Turns the paper's optimality story into a *multi-objective* one, which lines up with the brief's Pareto-front framing (§05, §09). |
| **Markov transition model** | **Drop from core claims, keep as future work** | The PDF's states (Focused/Distracted/Break) have no corresponding telemetry anywhere in the app — there is no session-state logging to estimate `P_ij` from. Building fake transition data to feed a Markov model would repeat the exact sin being fixed. State honestly in Limitations as unimplemented, propose it as future work once session telemetry exists. |
| **DFT energy-rhythm analysis** | **Build — small, real, and cheap** | Unlike the Markov model, this has a real input: `generateDefaultBandwidthCurve()` in `engine.ts` is already a periodic 96-point signal. Running a DFT over historical per-user energy logs (once they exist) or even the synthetic bandwidth curves in the research harness is a few hours of work and makes one PDF section true instead of fictional. Cheap win, include it. |
| **Task chunking** | **Drop — out of scope** | `schedule.ts`'s own header states "No chunking — tasks are scheduled as atomic units" and `extractScheduleChunks` is a no-op passthrough in the current model. Reintroducing chunking is a scheduler redesign, not a research add-on. Remove references to `/lib/chunking.ts` from the paper; note the atomic-task model explicitly as a design choice, not an oversight. |

Net effect: the rewritten paper keeps §5.1–5.4 and §5.6–5.9 as-is (they're already true), replaces §5.5 (chunking) with an honest note on atomic scheduling, replaces §5.10 (Knapsack) with the *real* implementation, keeps §5.11 (Lagrangian) but backed by real code, replaces §5.12 (Markov) with a Limitations entry, and replaces §5.13 (Fourier) with a real DFT run over synthetic energy logs. Section 6 (Evaluation) is entirely rebuilt on real simulation output — the 84%/81%/9%/7% table is deleted and replaced with real numbers, whatever they turn out to be.

## Framing: the constants are priors, not fitted parameters

Every formula in the model has at least one hand-picked constant: `E_task = 0.6·d² + 0.8·p`, the 0.40/0.35/0.25 morning/afternoon/evening split, the feedback learning rate α=0.2, the `Score = (2p + d + 3U)/E_task` weighting, the Lagrangian multipliers λ₁/λ₂/λ₃ once added. None of these were fit to real usage data — there is no deployed cohort to fit them against, and no comparable prior system's constants to borrow (the PDF's own baselines are a synthetic "AI Scheduler" and a synthetic "Static Planner," not real products with known-good parameters).

**Do not claim these specific numbers are optimal.** That claim isn't supportable and would be the same overreach being fixed elsewhere in this plan (fabricated Knapsack/Markov/Fourier results). Instead, frame explicitly:

- The contribution is the **formulation** — a constrained-optimization structure for cognitive-load scheduling, with a provably feasible optimizer and measurable multi-objective trade-offs — not a claim that these particular coefficients are correct.
- The constants are **hand-tuned priors**, stated as such wherever they appear (§5.1–5.4 text, not just a caveat buried in Limitations).
- Add a **sensitivity analysis** to the evaluation (§6): sweep each constant (e.g. the 0.6/0.8 split in `E_task`, the α feedback rate, the composite-score weighting) across a plausible range and show how feasibility rate / deadline-miss rate / burnout-incident rate respond. This replaces false precision with an honest map of "how much does this constant matter, and in which direction" — which is itself a useful, checkable result.
- Frame the paper's numeric baseline explicitly as **a starting point for calibration against real usage**, not a finished tuning. This is consistent with the original brief's own Bayesian-calibration proposal (§05) and gives the paper a legitimate future-work hook instead of an unfalsifiable claim.

This framing is a strict improvement, not a hedge: it turns "we don't know if these numbers are right" from a weakness into the honest scope of a first paper on a system with no prior deployment to calibrate against — real usage data becomes the natural follow-up study, and the sensitivity analysis is itself a real, checkable contribution the fabricated draft never had.

## Deliverables

1. **`research/` harness directory** (new, sibling to `client/`, doesn't touch production data or `client/src/lib` behavior at runtime):
   - `generators/syntheticUser.ts` — parameterised wake/sleep, peak-focus windows, section weights, energy multiplier sampling, grounded in `generateDefaultBandwidthCurve()`'s existing shape.
   - `generators/syntheticTasks.ts` — Poisson task arrivals; difficulty/duration/deadline/type/priority drawn from ranges matching `TaskType`/`TaskPriority` in `client/src/lib/types.ts`.
   - `algorithms/fcfs.ts`, `algorithms/edf.ts` — naive baselines.
   - `algorithms/greedyCurrent.ts` — thin wrapper around the real `runSchedulingAlgorithm`/`runScheduler` so the production baseline is the actual shipped code, not a reimplementation.
   - `algorithms/knapsackSection.ts` — real 0/1 Knapsack DP per section-budget allocation (replaces `tryAllocate`'s greedy fill for the "proposed" arm).
   - `algorithms/batchOptimizer.ts` — global reformulation over the full task pool and horizon; OR-Tools CP-SAT via a Python subprocess or a JS ILP library, with a GA/simulated-annealing fallback path if solve time blows up per the original brief §13's risk mitigation.
   - `algorithms/lagrangian.ts` — soft-penalty layer on top of the batch optimizer's objective (excess energy, load deviation from target, habit disruption).
   - `analysis/fourierRhythm.ts` — DFT over synthetic/logged energy time series, feeding calibrated section weights back into `buildSections`.
   - `metrics/engine.ts` — computes every metric in the original brief §09 (feasibility rate, deadline-miss rate, CL variance, context-switches/day, burnout-incident rate, axiom utilisation, solve time, explanation fidelity) per run, logged automatically.
   - `analysis/statisticalCompare.ts` — paired t-test / Wilcoxon signed-rank across identical seeds, Pareto-front plot data (deadline-risk vs. burnout-risk) per algorithm.
   - `analysis/sensitivitySweep.ts` — reruns the algorithm bank across a grid of constant values (E_task's 0.6/0.8, feedback α, composite-score weights) holding seeds fixed, so results are attributable to the constant change and not seed noise. Backs the "priors, not fitted parameters" framing above with an actual sensitivity curve instead of an assertion.
   - `run.ts` — CLI entry point: N seeds × {7, 14, 30}-day horizons × algorithm bank, writes raw results to `research/results/`.
2. **Rewritten paper** at `research/paper/axiom-scheduler.md` (or `.tex` if the target venue needs LaTeX — check with the co-authors before choosing format): same author list and section structure as the PDF, but:
   - §5.5, §5.10–§5.13 rewritten to match what's actually built (per the triage table above).
   - §6 entirely replaced with real simulation output from the harness.
   - New §7 Limitations honestly scoped: synthetic-only evaluation (already true, keep), Markov model explicitly deferred pending session telemetry, and all model constants stated as hand-tuned priors backed by the sensitivity sweep — not claimed as fitted-optimal, with real-usage calibration named as the natural follow-up.
   - Every formula gets a real `File:` pointer, and every pointer must resolve to an actual file — no more references to nonexistent modules.
3. **A short reconciliation note** (can live at the top of the new paper or as a separate `research/NOTE.md`) explicitly stating the earlier draft's Table 1/Table 2 numbers were placeholders never backed by an implementation, and are superseded by this version. This is the honest thing to do given the co-authors' names are on the original draft — flag it before anyone submits or shares the old PDF further.

## Suggested sequencing (condensed from the brief's 16-week plan)

1. Formalize the model + literature grounding (Cognitive Load Theory, RCPSP/bin-packing, classical EDF/priority scheduling, GA/SA metaheuristics, XAI ablation fidelity — brief §14).
2. Build the harness: generators, baselines (FCFS, EDF, current greedy wrapper), metrics engine.
3. Build the Knapsack-per-section optimizer and validate it against the greedy baseline on the same synthetic scenarios.
4. Build the global batch optimizer (ILP/CP-SAT, GA fallback) and the Lagrangian layer.
5. Run the full experiment matrix across seeds/horizons, collect metrics, run statistical tests, plot Pareto fronts.
6. Run the DFT rhythm analysis as a smaller side-experiment; feed calibrated weights back into a variant run to see if it changes outcomes.
7. Write up: rebuild §6 with real numbers, rewrite the flagged sections, add the reconciliation note, circulate to co-authors before any external submission.

## Open question for the user

Target venue/format for the final paper (student conference, arXiv preprint, internal B.Tech submission) isn't specified — this determines LaTeX vs. Markdown and citation style. Confirm before the write-up phase; doesn't block the harness build.
