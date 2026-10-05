# Plan 05 — Four-Member Loop and Early Integration

**Status: active integration plan. Supersedes the What/How/When bridges and content/trend merge dependencies.**

Follow [the final brief](../AXIOM_FINAL_BRIEF.md). The loop is **M1 defines → M4 estimates → M2 schedules → M3 updates → M4 estimates again**. Integration starts with early stubs from every member; it does not wait for completed modules.

## Shared contracts

| Boundary | Producer | Consumer | Required artifact |
|---|---|---|---|
| Formula | M1 | M4, M2 | Versioned variable registry, units/ranges/source rules, evaluator |
| Adaptive profile | M3 | M4, M2, reports | Immutable snapshot with version/cutoff; skill/history and calendar projections |
| Task estimates | M4 | M2 | Task revision, variable estimates, duration, uncertainty, provenance, versions, readiness status |
| Schedule | M2 | Execution UI, reports | Placements, unscheduled reasons, objective/solver status, estimate/profile/formula references |
| Learning observations | Execution/assessment/settings handlers | M3 | Idempotent actual-duration/difficulty events and user corrections |
| Weekly evidence | M3, M4, M2 | M1 + M2 evaluation/reporting | Profile deltas, prediction errors, and adaptive/baseline results with evidence labels |

Keep common contracts in proposed `client/src/lib/research-contracts/`. Runtime validation is required at persisted/LLM boundaries. Document fixture values as provisional, not formula decisions or validated measurements.

## Stub milestone: all four members

1. **M1:** registry and evaluator stub, with candidate/context-dependent variables distinguished from task estimates.
2. **M3:** profile version 1 fixture, a relevant-history read interface, and an event receiver that produces version 2 in fixture mode.
3. **M4:** deterministic valid estimates for those fixtures; no LLM/network dependency; explicit `stub` provenance.
4. **M2:** scheduling adapter that accepts M4's artifact and M3's calendar view, plus a fixed-interval baseline stub under the same experiment interface.

Run a shared learning-task fixture through all four stubs before replacing any with full implementations. Stubs retain the production contract; swaps do not require M2 to learn a different response shape. Retain the fixture suite as a compatibility check.

## Runtime sequence

1. Read M1's active formula contract and a single M3 profile snapshot.
2. M4 estimates each eligible task revision using the profile's observation cutoff. Persist or return an immutable estimation artifact outside the profile.
3. M2 validates matching versions, obtains calendar/focus context from the same snapshot, evaluates M1's formula, and schedules. Candidate-time variables are resolved for each candidate by their registered source, not guessed once by the LLM.
4. Persist the schedule and all input references before execution.
5. Learner execution/assessment handlers submit actual observations to M3. Preserve the original estimate even if the task is later edited.
6. M3 alone updates task memory, skill, and windows, then publishes a new profile version.
7. M4 re-estimates eligible future tasks on the next agreed planning trigger. M2 reschedules future work under its policy; preserve completed/in-progress history.
8. Produce a weekly report showing initial-profile changes and the agreed fixed-interval comparison. M1 + M2 own interpretation.

A fresh profile version is an input change, not permission to rewrite past predictions. Re-estimation triggers include task revision, profile change, formula change, and explicit planning requests. Avoid a write/estimate/reschedule feedback loop with no new observation.

## Failure and consistency rules

- Missing required variables: return a blocked estimate with reasons; do not substitute zero effort.
- LLM outage: M4 returns a validated, labelled fallback if policy permits, or a blocked result. Stub outputs cannot enter measured study runs.
- Unschedulable task: M2 returns a reason; no force-placement into non-availability or capacity violations.
- Stale formula/task/profile versions: refresh the relevant artifact before placement; never mix history and windows from different profile snapshots.
- Duplicate feedback: M3 deduplicates by event ID. Schema migration is separate from snapshot reads.

## Shared acceptance scenario

Use a fixture learner and two comparable learning tasks:

1. Profile version 1 has no matching history. M4 provides a labelled cold-start estimate; M2 produces an adaptive schedule and the baseline.
2. The learner completes task A; observed active duration and difficulty differ from its prediction. The handler sends one event to M3.
3. M3 records task A and publishes version 2, with supported skill/window changes where justified.
4. M4 reads version 2 and adjusts task B's estimates using task A's evidence. It performs no profile write.
5. M2 schedules B using the updated artifact; version 1's estimate/schedule remain reproducible from saved inputs.
6. The weekly report shows the profile change, estimate error, and observed versus replay comparison labels.

Follow with sparse/missing-feedback, duplicate-event, unavailable-window, and incompatible-formula fixtures. These are integration checks; learning-domain evaluation is specified in [Plan 01](01-research-paper-scheduler.md).
