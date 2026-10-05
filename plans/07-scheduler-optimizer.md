# Plan 07 — Scheduling Method Based on the Cognitive Load Formula

**Owner: M2. Evaluation: jointly owned with M1. Status: planned; previous solver choices are superseded.**

Follow [the final brief](../AXIOM_FINAL_BRIEF.md). M2 develops an optimal scheduling method using **M1's formula** and **M4's task-variable estimates**, constrained by **M3's versioned profile**. M2 writes schedules, not the adaptive profile. Its first implementation is a stub so integration can start early.

## Inputs and outputs

Consume task metadata/prerequisites/deadlines, M4's estimate artifacts, M1's formula/variable registry, and M3's scheduling projection from one profile version. Record every input version. The current interface stub is `client/src/lib/research/m2-fixed-interval.ts`; extend the research contracts rather than adding unrelated product features.

Return placements with start/end times, formula-derived load, structured unscheduled reasons, objective value, solver status, and reproducibility metadata. Candidate-time/context variables must be evaluated as defined by M1. If load is not separable across tasks or depends on order, model those dependencies explicitly; the previous additive-load assumption is no longer locked.

Freeze a shared research contract early under [Plan 05](05-what-how-when-loop-integration.md); do not force new evidence into fields that cannot represent it.

## Feasibility and objective

M2 specifies the optimization problem with M1 before selecting an algorithm. At minimum account for:

- No overlapping tasks, and all occupied time inside known available windows.
- Explicit non-availability, timezone, date-specific/recurring windows, and midnight boundaries.
- Prerequisite completion and agreed deadline semantics.
- M1-defined cognitive capacity limits, with units consistent with estimated load.
- Task/session-size policy shared with the baseline; no silent task splitting or dropping.

Peak/minimum focus windows affect suitability according to the formula/objective. Minimum focus does not itself imply unavailability. If no feasible placement exists, retain the task with a structured reason. Never force an oversized task through a hard capacity constraint or fall back to a constraint-violating legacy schedule.

Propose and freeze the primary objective, such as completion under constraints followed by load/focus suitability, before evaluation. Priority, deadline urgency, focus suitability, and cognitive load must remain distinct quantities unless M1 explicitly defines otherwise. M4 estimates inputs; it does not choose optimizer weights or placements.

## Method selection and optimality

Choose an exact optimization formulation if it fits M1's model and the measured problem size. Compare candidate implementations using representative learning-task instances. The previous mandated Knapsack, Lagrangian, browser-only LNS, WASM oracle, 30-day horizon, fixed budget, and mandatory additive formula are no longer final decisions.

Define the research horizon, runtime budget, tie-breaking, and objective with the team. A heuristic may provide an incumbent or an early integration stub. Report it as a heuristic unless the solver certifies an optimum. A bounded run reports its incumbent, bound/gap if available, and termination reason. Do not call a method globally optimal because it improves a greedy schedule or is exact only for a small repair subset.

Recorded estimates and normalized inputs allow scheduler replay. LLM estimation is upstream; determinism of the solver does not establish determinism of fresh LLM calls.

## Fixed-interval baseline and evaluation

M1 and M2 jointly freeze the baseline in [Plan 01](01-research-paper-scheduler.md): static session/break cadence, start rule, task ordering, task-sizing policy, and failure handling. It uses the same eligible tasks and externally known availability restrictions. It does not adapt to learned focus windows, skill, or task-history calibration.

Implement the baseline through the same research interface as the proposed scheduler. A 15-minute internal grid is not automatically a fixed-interval scheduling baseline.

For small instances, verify the optimum against exhaustive enumeration or an independently checked exact formulation. For realistic instances, report feasibility, objective, runtime, and any certificate/gap alongside adherence, completion, deadlines, and reported difficulty. Separate synthetic correctness from observed learning-domain outcomes.

## Integration priorities

- Keep predicted duration/difficulty separate from actual learner observations.
- Normalize non-availability windows without double-counting; treat minimum-focus periods as preferences.
- Validate prerequisite graphs; an impossible task is reported rather than forced into the schedule.
- M2 consumes profile snapshots and never mutates them.

## Implementation sequence and acceptance

1. Shared-contract scheduling stub and baseline stub with deterministic fixtures.
2. Availability normalization, version checks, formula integration, and invariant checks.
3. Formal objective/constraints and method selection using measured instances.
4. Solver implementation with structured infeasibility and honest optimality status.
5. Research harness, small-instance cross-checks, fixed-interval comparison, and weekly report integration.
6. App integration using saved input references and explicit future-task rescheduling policy.

Verify no overlap/unavailable placement, model-capacity and prerequisite compliance, complete task accounting, replay of saved inputs, and absence of profile mutations. The complete loop must run before pilot evaluation; content/trend workstreams are not prerequisites.
