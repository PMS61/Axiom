# Plan 04 — M3 Adaptive Profile

**Owner: M3.** M3 is the only member allowed to write learner-profile state. The first implementation stores the profile in browser localStorage so the research workflow has no account, authentication, or database dependency.

## Profile fields

- Non-availability windows (hard scheduling constraints).
- Peak-focus and minimum-focus windows (preferences, not availability constraints).
- Topic skill with its evidence and uncertainty.
- A durable-in-profile history of each performed task: task/topic/type, predicted and actual duration, reported difficulty, and timestamp.
- A frozen initial profile snapshot and version for the weekly behaviour report.

## Current interface and ownership

`client/src/lib/research/m3-profile.ts` owns browser profile reads, initialisation, edits, and observation writes. UI controls and feedback submit changes through this module. M4 receives a profile snapshot as input and only selects relevant evidence for its prompt; M2 receives scheduling windows. Neither can modify the profile. M4 estimation artifacts are not profile state.

## Iteration sequence

1. Keep the local profile schema explicit and versioned; preserve unknown skill and missing observations as unknown.
2. Record actual duration and reported difficulty for each completed learning task, including the matching prediction.
3. Improve focus-window and topic-skill updates only when observations support the change; retain evidence and confidence.
4. Show week-over-week profile changes and task outcomes against the frozen initial profile. M1/M2 own the separate fixed-interval scheduler comparison.
5. Replace localStorage only if the study requires multi-device persistence; keep the same M3-only write boundary.

## Acceptance

The profile is visible to the learner, survives reload in the same browser, and exposes only relevant task history to M4. An estimate can never write it. M3 is the only module that changes the profile version. The weekly report labels observed, missing, and simulated values and does not claim improvement without collected comparative evidence.
