# Research implementation plans

The authoritative project scope is [AXIOM_FINAL_BRIEF.md](../AXIOM_FINAL_BRIEF.md): a cognitive load based adaptive scheduler, built by four members and evaluated in the learning domain.

| Document | Active responsibility |
|---|---|
| [01 — Research and evaluation](01-research-paper-scheduler.md) | M1 + M2: formula contract, fixed-interval baseline, protocol, metrics, paper |
| [04 — Adaptive profile](04-personalization-unification-plan.md) | M3: sole profile writer, windows, skill, task memory |
| [05 — Loop integration](05-what-how-when-loop-integration.md) | Shared interfaces and early stubs; M1 → M4 → M2 → M3 → M4 |
| [06 — M4 implementation](06-task-variable-estimation.md) | M4: profile-informed task-variable estimation using an LLM |
| [07 — Scheduling method](07-scheduler-optimizer.md) | M2: objective, constraints, optimizer, baseline adapter |

Plan numbers are document identifiers, not member numbers. In particular, Plan 04 belongs to M3; Plan 06 is M4's work.

Only the scheduler research workstreams are retained in this repository. The old paper PDF and slide decks were removed at the project owner's request and are not research evidence.
