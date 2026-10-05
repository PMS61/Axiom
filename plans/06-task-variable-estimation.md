# Plan 06 — M4 Task-Variable Estimation

**Owner: M4.** The current root page accepts a natural-language learning goal and returns an ordered list of study tasks with metrics. This is a stub-backed research implementation, not a validated cognitive-load estimator.

## Boundary

M4 reads one M3 profile snapshot and combines relevant learner history/skill evidence with general task knowledge from an LLM. It returns per-task values and provenance to M2. It does not define M1's formula, schedule tasks, or update the profile. M3 remains the sole writer. No login, account, database, or unrelated learning-product feature is required.

## Current implementation

- `client/src/app/page.tsx` and `TaskEstimatorClient.tsx`: natural-language input, metric cards, profile-memory disclosure, outcome feedback, and weekly report.
- `client/src/app/actions/task-estimation.ts`: validates input, reads the supplied profile snapshot, asks Gemini for a structured task list, validates its response, and falls back to a labelled stub.
- `client/src/lib/research/contracts.ts`: provisional shared task/profile contracts.
- `client/src/lib/research/formula.ts`: M1-owned provisional formula interface and load proxy.
- `client/src/lib/research/m3-profile.ts`: M3-only profile mutations and read snapshot.
- `client/src/lib/research/m2-fixed-interval.ts`: M2 fixed-interval baseline stub.

The page displays the exact bounded task history included in M4's model context and separately labels focus/availability windows retained for M2. The LLM result is not a measured outcome; stub results are explicitly labelled.

## Build-out order

1. Agree M1's versioned list of variables, meaning, units, ranges, and required inputs. Implement the registry as a stub before tuning estimates.
2. Agree M3's profile snapshot schema and provide representative cold-start, sparse-history, and experienced-learner fixtures.
3. Normalize the natural-language goal into scoped tasks, prerequisites, and quantities. Keep user-provided deadlines/availability separate from estimated values.
4. Select similar past tasks using topic, activity, scope, and skill context. Report evidence count and uncertainty; never treat missing history as zero skill.
5. Use the LLM for task decomposition and general-knowledge priors. Validate structured output against M1's registry; attach source, range/uncertainty, prompt/model version, and warnings to each estimate.
6. Return unresolved required values explicitly. Use a labelled fallback only when its assumptions are visible; do not invent precise workload from an underspecified goal.
7. Hand the estimate and profile version to M2. M2 applies the formula at candidate schedule times if any variables are time-dependent.
8. Compare predictions with M3-recorded actual duration/difficulty on held-out learning tasks. M1 and M2 own the fixed-interval evaluation and claims.

## Acceptance

Natural-language goals return an ordered, inspectable task list with all available metrics, prerequisites, rationale, and uncertainty. The page shows the profile memory sent to M4. Estimates differ when relevant learner evidence differs, while the profile remains unchanged until the user records an outcome through M3. Gemini failures produce a usable labelled stub or a clear error, never an unlabelled live-looking estimate.
