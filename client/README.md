# Axiom research workbench

This Next.js app is a focused workbench for the Axiom cognitive-load-based adaptive scheduler research project, evaluated in the learning domain. See [the final brief](../AXIOM_FINAL_BRIEF.md) and [active plans](../plans/README.md).

## Run locally

```bash
npm install
npm run dev
```

Open `http://localhost:3000`. There is no login, authentication, or database requirement. The M3 profile lives in the current browser's local storage. Set `GEMINI_API_KEY` in `.env.local` for live M4 estimates; otherwise, the UI displays a labeled fallback stub.

## Research boundaries

- M1 defines the cognitive-load formula and variables. The current displayed load metric is provisional.
- M4 turns natural-language learning goals into task estimates using general task knowledge and the selected profile context.
- M2's fixed-interval baseline is a stub; the adaptive optimizer is planned.
- M3 alone writes browser-local availability/focus preferences, skill values, and task history.
- The weekly report describes local observations. The evaluation comparison is pending M1 and M2.

`src/app/page.tsx` is the workbench. Estimation lives in `src/app/actions/task-estimation.ts`, the M3 profile writer lives in `src/lib/research/m3-profile.ts`, and the formula/scheduler stubs live under `src/lib/research/`.

## Commands

```bash
npm run lint
npm run typecheck
npm run build
npm run start
```
