# Axiom

Axiom is a cognitive-load-based adaptive scheduler research project for the learning domain. The final direction and member ownership are in [AXIOM_FINAL_BRIEF.md](AXIOM_FINAL_BRIEF.md); active plans are indexed in [plans/README.md](plans/README.md).

The loop is **M1 defines the formula → M4 estimates task variables → M2 schedules → M3 updates the adaptive profile → M4 estimates again**. M3 is the sole profile writer. M1 and M2 jointly own evaluation against a fixed-interval baseline. Each member supplies a stub early so M2 can integrate against working interfaces.

The research workbench is in `client/`. It needs no account or database; the temporary M3 profile is stored in the current browser. Gemini task estimates require `GEMINI_API_KEY`; without one, the UI labels its deterministic task list as an illustrative stub.

## Run locally

```bash
cd client
npm install
npm run dev
```

Open `http://localhost:3000`. The root page accepts a natural-language learning goal and displays an ordered task list with its estimate context. Use the M3 profile editor to record focus/availability windows and topic skill. Record each task's actual duration and reported difficulty to add task memory for later M4 estimates.

## Checks

```bash
cd client
npm run lint
npm run typecheck
npm run build
```

The M1 cognitive-load formula, M2 adaptive optimizer, and measured baseline evaluation are stubs or planned work. The UI labels provisional metrics and reports accordingly. See [M4's implementation plan](plans/06-task-variable-estimation.md).
