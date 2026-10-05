# Repository Guidelines

## Research Direction and Ownership

The authoritative scope is `AXIOM_FINAL_BRIEF.md`: a cognitive load based adaptive scheduler, developed by four members and evaluated in the learning domain. Active implementation documents are indexed in `plans/README.md`.

- M1 defines the formula structure and variables, not per-task values.
- M2 owns the optimal scheduling method based on that formula and M4's estimates.
- M3 owns adaptive behaviour, skill, non-availability windows, peak/minimum focus windows, and task memory with actual duration and reported difficulty. **Only M3's module writes to the adaptive profile**, including initialization and user corrections.
- M4 estimates task variables from the adaptive profile and an LLM's general task knowledge. It reads the profile only; estimate artifacts live outside profile state.
- The loop is M1 defines → M4 estimates → M2 schedules → M3 updates → M4. Every member ships a stub early so M2 can integrate against real interfaces.
- M1 and M2 jointly own evaluation against a fixed-interval scheduler. Weekly reports distinguish changes from the initial profile baseline from scheduling-baseline comparisons.

Keep the app limited to the scheduler research workflow. There is no authentication, account, or database requirement. M4's active implementation plan is `plans/06-task-variable-estimation.md`.

## Project Structure & Module Organization

The Next.js app is in `client/`. Work from `client/` for normal development. Source lives in `client/src`: routes and server actions are under `src/app`, reusable UI in `src/components`, and research contracts in `src/lib/research`.

## Build, Test, and Development Commands

Run commands from `client/`:

- `npm install`: install dependencies from `package-lock.json`.
- `npm run dev`: start the Next.js dev server on port 3000 with polling enabled.
- `npm run build`: create a production build with `next build`.
- `npm run start`: serve the production build.
- `npm run lint`: run `biome check`.
- `npm run format`: format files with Biome.

## Coding Style & Naming Conventions

Use TypeScript for new React/Next.js code unless a nearby file is already JavaScript. Biome is the source of truth: 2-space indentation, organized imports, and recommended React/Next rules. Use `PascalCase` for React components, `camelCase` for functions and variables, and route folders that match URL segments, for example `src/app/short-bit/demo/page.tsx`. Keep shared UI primitives in `src/components/ui` and domain-specific logic in `src/lib`.

## Testing Guidelines

No automated test script is currently configured. Before opening a PR, run `npm run lint` and `npm run build`, then manually exercise the affected route in `npm run dev`. If you add a test framework, wire it into `package.json` and document the command here. Prefer colocated tests with clear names such as `schedule.test.ts` or `TaskBlock.test.tsx`.

## Commit & Pull Request Guidelines

Recent history uses short imperative messages, sometimes with Conventional Commit prefixes such as `fix:` and `feat:`. Prefer `type: concise summary` for new work, for example `fix: update task generation preferences`. Pull requests should include a short description, affected routes or modules, verification steps, linked issues when applicable, and screenshots or screen recordings for UI changes.

## Agent-Specific Notes

This project uses a newer Next.js version with breaking changes. Before changing framework behavior, consult the local Next.js docs in `client/node_modules/next/dist/docs/` when available, and follow `client/AGENTS.md` if that file is present.
