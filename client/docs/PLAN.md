# Axiom demo rebuild — plan

## Intent

The FYP presentation (`/home/amal/Downloads/FYP-ppt.pdf`, 30 slides) ends with a
live demo of Axiom (slide 29): **Trend view → Roadmap → Scheduler (with reasoning
chain) → Recall prototype**. The current demo is slow and weak: `/` runs a
Python MCMC refit on every click, and the dashboard needs login, Postgres and
Gemini. We are building a new **guided demo at `/demo`** that is:

- **short** (about 4 minutes, 5 steps, keyboard arrows to move between them),
- **instant** (all data seeded, no network, no Python, no DB, no API keys),
- **impactful**: it shows the *real* improvement the deck claims, using the real
  per-review predictions in `results/full_u100_c1000_final/` (100 Anki users,
  95,400 scored test reviews) and the real srs-benchmark final-test numbers.

Numbers that come from real results must be exported from the result files, not
typed in by hand, except the srs-benchmark final-test numbers below (from the
deck, slide 23/24). Anything illustrative must carry a small caption
"illustrative" so the demo stays honest.

## Fixed decisions (orchestrator; do not change without asking)

- App: Next.js 16 / React 19 / TypeScript in `/home/amal/fyp/Axiom/client`.
  Run: `cd client && npm run dev` (port 3000). Lint: `npm run lint` (biome).
  Types: `npm run typecheck`.
- New route `client/src/app/demo/` (page + components under
  `client/src/components/demo/`). The old course sandbox currently at
  `src/app/demo/page.tsx` + `actions.ts` moves to `src/app/demo/course/`.
  `src/app/page.tsx` (the `/` route) must render the new demo.
  Do not touch `/dashboard`, `/bayesian-demo` or any Python model code.
- Demo route is **client-only and static**: no fetch, no server actions, no
  Gemini, no Postgres. Seed data lives in `client/src/lib/demo/`.
- Charts are hand-written inline SVG in React (no new chart library). No new npm
  dependencies at all.
- Style: reuse the app design tokens in `src/app/globals.css` (`--bg`, `--ink`,
  `--rule`, `--vermillion`, `--muted`, `--card-bg`, `--safe`, `--watch`) and
  fonts `--font-playfair` (headings) / `--font-ibm-plex-mono` (body/meta). CSS
  modules (`*.module.css`). Must look right in light and dark themes. Consistent
  colours across all charts: **FSRS-6 = muted grey `--muted`**, **Ours (GBDT +
  online) = `--vermillion`**, **RWKV / other = `--rule`/ink outline**, actual
  outcome dots = `--ink`.
- Projector-friendly: base font ≥ 16px, chart labels ≥ 13px, works at 1280×720
  without scrolling inside a step where possible.
- Tests: TS logic in `client/src/lib/demo/*.test.mjs` run with
  `node --experimental-strip-types --test` (same style as
  `src/lib/bayesian/demo.test.mjs`); add npm script `test:demo`. Python export
  tests run with `uv run --with pandas --with pyarrow --with pytest pytest`.
- Do not commit. Do not run more than one heavy job at once (PC has 19 GB RAM).

## Demo storyline (5 steps)

A learner, "Riya", is preparing for an ML interview.

1. **What to learn — Trends.** 6 seeded rising topics, each with score
   `100·(0.4F + 0.4E + 0.2R)` broken down as three small bars (F frequency,
   E engagement, R recency) and a tiny sparkline. One topic ("Transformers &
   attention") is highlighted/selected.
2. **How — Roadmap.** A DAG of 6 nodes for that topic (prereqs → advanced),
   drawn in SVG, plus a row showing the 7 content agents (roadmap, course,
   assessment, cheat-sheet, flashcard, short-bit, story) and what each produced
   for the selected node (seeded titles).
3. **When — Scheduler.** Today's timeline 07:30–23:30 with the bandwidth
   (energy) curve, peak/low windows, fixed commitments, and placed tasks.
   Clicking a task shows its reasoning chain (3–5 seeded steps). One block is a
   "Review: 6 flashcards" slot fed by step 4.
4. **Memory engine — Recall.** Two parts:
   a. *Real card replay*: choose one of 4 real Anki cards; chart its review
      history over time (actual outcome ✓/✗), and at each scored test review
      the predicted P(recall) from FSRS-6 (grey) and Ours GBDT + online layer (red), real srs-benchmark test predictions from Task 1.1b. Running
      log-loss tally for both on that card.
   b. *Live queue + online layer*: Riya's 6 flashcards (Transformers topic)
      with predicted recall and a Bayesian 90% band (illustrative), sorted
      riskiest first. Presenter clicks "Remembered"/"Forgot" on a card; the
      per-user online layer `logit(p') = logit(p) + w·x`,
      `w ← w + η·(y − p')·x` updates instantly, every card's prediction moves,
      and the queue reorders with animation. Show the 6 weights as small bars.
5. **Evidence — Why it matters.** (a) srs-benchmark final test, 1,000 unseen
   users: log-loss bars FSRS-6 0.338, Ours 0.2999, RWKV-Instant 0.2828 ("2nd of
   all published models"). (b) Real reliability diagram on our Anki test split (secondary panel, labelled "GBDT stage 1"),
   FSRS-6 opt vs GBDT (10 bins). (c) Waffle of 100 Anki users: 82 where ours
   beats FSRS-6. (d) Cost tiles: 0.09 ms vs 96 ms per prediction, 196k vs 2.76M
   stored numbers. (e) Log loss by review gap (5 bins, real).

## "Ours" = GBDT + online layer (user decision, 2026-11-19)

The headline model everywhere in the demo is the **frozen final model: GBDT +
per-user online layer** (variant B, lr 0.003) from
`results/srsb_beat/final/` (srs-benchmark test users 1–1000, scored once). The
Anki-100 GBDT numbers from Task 1.1 stay only as a secondary "our own Anki
split" panel, labelled "GBDT (stage 1)". Card replay (step 4a) uses real
per-review GBDT+online predictions from srs-benchmark test users (Task 1.1b).

## Phases

### Phase 1 — Seed data + demo shell + steps 1–3

**Task 1.1 — Export real results to seed JSON.**
Write `client/scripts/export_demo_data.py` that reads
`results/full_u100_c1000_final/{predictions.parquet,events.parquet,metrics.csv,user_wins.csv}`
and writes `client/src/lib/demo/seed/results.json` containing:
- `overall`: for models fsrs_default, fsrs_opt, hlr, ours, ours_v3, gbdt →
  log_loss, auc, ece, brier (from metrics.csv `scope=overall`).
- `gapBins`: for fsrs_opt and gbdt, log_loss per gap bin (metrics.csv `scope=gap`).
- `reliability`: for fsrs_opt and gbdt, 10 equal-width bins of predicted p over
  predictions.parquet: `{binLow, binHigh, meanPred, meanObs, n}` (skip empty bins).
- `userWins`: number of users (out of 100) where gbdt per-user log loss <
  fsrs_opt per-user log loss, computed from predictions.parquet, plus the
  list of per-user differences (gbdt − fsrs_opt), sorted.
- `cards`: 4 real cards chosen deterministically: cards with ≥ 6 scored test
  reviews and both outcomes present in the test period, ranked by
  (fsrs_opt card log loss − gbdt card log loss) descending; take the top 3 plus
  the single card where fsrs_opt beats gbdt most (honesty). For each card: its
  full review list from events.parquet (`day` = day_offset, `rating`, `y`,
  `scored`) and for scored rows `pFsrs` (fsrs_opt) and `pOurs` (gbdt). Anonymise:
  replace user/card ids with "Card A".."Card D".
Done when: script runs in < 60 s, JSON < 300 KB, a pytest test in
`client/scripts/test_export_demo_data.py` checks keys, that gbdt overall log loss
≈ 0.3584 and fsrs_opt ≈ 0.3794 (±1e-3), userWins == 82, reliability bins sum
to 95,400, every card has ≥ 6 scored reviews and probabilities in (0,1).

**Task 1.1b — Export the GBDT + online model (srs-benchmark final test).**
Write `client/scripts/export_online_demo.py`, run with
`cd /home/amal/fyp/Axiom && .venv/bin/python client/scripts/export_online_demo.py`
(that venv has lightgbm, numba, pandas). It writes
`client/src/lib/demo/seed/online.json` with:
- `final`: the `table` and `bootstrap` blocks from
  `results/srsb_beat/final/online_test1000.jsonl.comparison.json` (log loss
  weighted/unweighted, AUC, CIs; models: Online layer (ours, final), GBDT
  (stage 1), RWKV-Instant, RWKV-Curve, FSRS-6).
- `perUser`: for each of the 1,000 users, `{size, ours, gbdt, fsrs6}` log loss,
  ours/gbdt from `final/online_test1000.jsonl` and `final/gbdt_test1000.jsonl`,
  fsrs6 from `external/srs-benchmark/result/FSRS-6-short.jsonl` (match on
  `user`); plus `winsVsFsrs6` = count of users with ours < fsrs6, and
  `winsVsGbdt` = count with ours < gbdt.
- `cards`: real per-review predictions for 4 cards. Pick 3 small test users
  (≈ 4,000 test rows each, e.g. users 994, 574, 203 from
  `online_test1000.jsonl`). For each, reproduce the frozen final prediction
  exactly as `final_user()` in `fsrs_benchmark/srsb/online_adapt.py` does
  (`featurized_arrays` → `design` → `online_kernel` with variant "B", lr 0.003,
  `LAMBDA`; model/run_info/gbdt paths = the FINAL_* / defaults used by
  `main_final`). Keep, per test row: card id, review time (day), rating, y,
  p_gbdt, p_online, and the FSRS-6 default-weight R (`f_R` in the context
  columns, see `dump_expert_preds`). Pick 4 cards with ≥ 8 test reviews and both
  outcomes: top 3 by (FSRS log loss − online log loss) plus the 1 card where
  FSRS beats online most. Anonymise as "Card A".."Card D". Also include the
  card's earlier (non-test) review history (day, rating, y) if it is easy to get
  from the loaded user data; otherwise test rows only.
- Do NOT modify anything in `fsrs_benchmark/`; import from it.
Done when: script runs in < 10 min with peak RAM < 6 GB, JSON < 400 KB; a pytest
`client/scripts/test_export_online_demo.py` checks: final table weighted log
loss ours == 0.29991 and FSRS-6 == 0.33832; perUser has 1,000 users;
**for each of the 3 replayed users, the log loss of the reproduced p_online on
all its test rows matches `online_test1000.jsonl` within 1e-4** (proves the
replay is the real model); card probabilities in (0,1).

**Task 1.2 — Demo shell.**
Route `/demo` (and `/` re-exporting it), old sandbox moved to `/demo/course`.
`DemoShell` component: top bar (Axiom logo, step title, 5 step pills
"1 Trends · 2 Roadmap · 3 Scheduler · 4 Memory engine · 5 Evidence"),
←/→ keyboard and buttons, current step in the URL hash (`#3`) so reload keeps
the step, smooth fade between steps. Each step is its own component file in
`src/components/demo/` (placeholders for 4 and 5 for now). Theme toggle
(light/dark) in the top bar.
Done when: `npm run typecheck` and `npm run lint` pass for new files, `/demo`
and `/` load in the browser with no console errors, arrows switch steps, hash
persists across reload, `/demo/course` still renders.

**Task 1.3 — Steps 1–3 (Trends, Roadmap, Scheduler) with seeded data.**
Seed data in `src/lib/demo/story.ts` (topics, roadmap nodes/edges, agent
outputs, profile windows, tasks with reasoning chains). Implement the three
step components as described in the storyline, with SVG illustrations and a
short one-line caption per step that the presenter can read.
Done when: each step renders instantly, selected trend drives roadmap title,
clicking a roadmap node updates the agent outputs panel, clicking a task shows
its reasoning chain, the timeline tasks do not overlap fixed commitments
(unit test in `story.test.mjs` checks seeded tasks vs commitments/exclusions),
works in both themes, no console errors.

### Phase 2 — Memory engine + evidence + polish

**Task 2.1 — Online layer logic (TS) with tests.**
`src/lib/demo/online.ts`: `predict(baseP, w, x)`, `update(w, x, y, p, eta)`
(η default 0.003 scaled up for the demo: use η = 0.35 and say "sped up for the
demo" in the UI caption), Riya's 6 seeded cards with base GBDT p, features x
(6 signals: bias, GBDT margin, user recall-rate today, reviews today, recency,
lapse count; scaled to ~[−1,1]), illustrative Bayesian band, and `riskQueue`.
Tests: lr 0 → output equals base; a "forgot" lowers that card and (via shared
bias) others; a "remembered" raises; queue sorted ascending by p; p stays in
(0,1).

**Task 2.2 — Step 4 Memory engine UI** (card replay chart + live queue) as in
the storyline, using `results.json` and `online.ts`. Animate queue reorder.

**Task 2.3 — Step 5 Evidence UI** (5 visual blocks) from `results.json` and
the deck's srs-benchmark numbers.

**Task 2.4 — Speed + runbook.** `npm run demo` script (`next build && next
start -p 3000`), confirm production build passes, measure first load of `/demo`
(< 1.5 s on localhost prod), presenter notes `client/docs/DEMO_SCRIPT.md`
(≈ 4 minutes, what to click and say per step, tied to slide 29).

## Status

- [x] Phase 1 (data exports, shell, steps 1–3)
- [x] Phase 2 (memory engine, evidence, prod build, DEMO_SCRIPT.md); finished by the orchestrator directly at the user's request, without the per-task verify loop
