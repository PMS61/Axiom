# Plan 01 — Formula and Research Evaluation

**Owners: M1 (formula) and M1 + M2 (joint evaluation). Status: planned under the final direction of 5 October 2026.**

Follow [the final brief](../AXIOM_FINAL_BRIEF.md). This replaces the previous synthetic-only paper plan, prescribed Knapsack/Lagrangian/Fourier pipeline, and comparisons against invented AI schedulers. Algorithm choices and numerical results must follow actual implementation and experiments.

## M1: formula contract

M1 defines the cognitive load formula's structure and variables, not their per-task values. Publish:

- A formula version and a pure evaluator accepting validated variables.
- A variable registry: stable key, meaning, unit, valid range, required/optional status, and whether it comes from task metadata, M3's profile, M4's estimation, or M2's candidate scheduling context.
- Definitions of load and capacity that M2 can use consistently, including how duration enters the model and whether load depends on candidate time or task order.
- Explicit treatment of unknown variables and provisional assumptions. A stub evaluator must be labelled provisional.

The current formula evaluator is a labelled stub, not the approved final research model. M4 does not choose a formula to accommodate its estimates; M1 owns such changes.

## M1 + M2: evaluation protocol

Freeze the protocol before the measured study. M1 owns interpretation of cognitive load and measurement scales; M2 owns scheduler/baseline implementations and feasibility instrumentation. Both review the comparison and research claims.

The required comparator is a **fixed-interval scheduler**. Specify its study-session and break lengths, start-time rule, task ordering, handling of tasks longer than a session, deadline policy, and handling of unschedulable tasks. Do not silently change these settings after seeing adaptive outcomes. Fixed interval means a fixed study cadence, not merely a fixed internal slot resolution.

Both arms receive the same task set, prerequisites, deadlines, study horizon, and externally known non-availability restrictions. At each decision point, shared restrictions are aligned; the adaptive arm may additionally use M3's learned focus/skill/history signals. The baseline's own policy stays static. Record the task-sizing policy so splitting or excluding large tasks cannot favour one arm.

Run two comparisons to separate effects:

1. **Scheduling comparison:** give both methods the same frozen duration estimates and task metadata; compare placement and feasibility without changing estimates mid-run.
2. **Complete-system comparison:** compare the adaptive loop with the fixed-interval policy and its frozen estimation policy; report that differences combine estimation, scheduling, and adaptation.

An LLM-only estimator and a profile-frozen adaptive scheduler may be supplementary ablations. They do not replace the fixed-interval baseline.

## Evidence collection

Start with synthetic fixtures to check contracts, feasibility, reproducibility of recorded inputs, and small-instance optimality. Then run a learning-domain pilot with real task descriptions and recorded learner feedback. Simulation alone cannot establish learner benefit.

Agree the participant count, pilot length, task set, feedback scales, and scheduling cadence before collection. For an observed comparison, agree a randomized/counterbalanced within-learner design where practical, documenting learning and order effects. If only one schedule is followed, report the other as a replay or simulation and do not assign it fabricated adherence or difficulty outcomes.

Keep each estimate's creation time, formula/profile/task versions, actual active minutes, reported difficulty, interruption/missing-data flags, completion, planned and actual start times, and deadline status. For estimator evaluation, use chronological splits: only observations available before a prediction may inform it. Separate tuning data from held-out evaluation data.

## Metrics and weekly report

| Measure | Definition to freeze before the study |
|---|---|
| Completion | Completed eligible tasks / eligible tasks; include unscheduled tasks in the denominator |
| Adherence | Planned sessions followed within an agreed time tolerance / eligible planned sessions |
| Deadline misses | Missed due tasks / tasks due in the measurement window |
| Reported difficulty/overload | Learner ratings on an agreed scale; formula-derived load is a separate model output |
| Duration error | Absolute predicted versus actual active-minute error on comparable completed tasks |
| Difficulty error | Absolute predicted versus reported-rating error on the same agreed scale |
| Feasibility | Overlap, non-availability, prerequisite, deadline, and model-capacity violations |
| Solver quality | Objective value, runtime, and verified optimum or bound/gap where available |

Weekly reporting has two parts: change from the initial **profile baseline**, and comparison with the **fixed-interval scheduling baseline**. M3 supplies profile snapshots/change reasons; M4 supplies prediction artifacts and observed error; M1 + M2 own definitions and interpretation. Show denominators, sample sizes, uncertainty, missing values, and observed versus simulated labels. Sparse data produces an insufficient-evidence label, not a fabricated improvement.

## Deliverables and sequence

1. M1's formula/registry stub and M2's baseline stub; share fixtures immediately under [Plan 05](05-what-how-when-loop-integration.md).
2. A frozen protocol and baseline configuration, including metric definitions and consent/data-handling arrangements for the pilot.
3. A research harness using the same formula, estimate artifacts, and scheduling interfaces as the app.
4. Synthetic correctness results followed by the learning-domain pilot.
5. Weekly comparison output, held-out estimator analysis, and an evidence-based paper with limitations.

Research harness location: proposed `research/` at repository root. No harness, fitted coefficients, participant results, or new optimal solver is claimed implemented by this plan.
