# Repository Guidelines

## Project Structure & Module Organization

This repository is a thin wrapper around the Next.js app in `client/`. Work from `client/` for normal development. Source lives in `client/src`: route segments and pages are under `src/app`, reusable UI is in `src/components`, shared logic is in `src/lib`, and ambient TypeScript declarations are in `src/types`. Static browser assets, workers, ONNX models, and WASM files are in `client/public`. Demo and seed data live in `client/demoData.json` and `client/scripts/seed_demo_user.mjs`.

## Build, Test, and Development Commands

Run commands from `client/`:

- `npm install`: install dependencies from `package-lock.json`.
- `npm run dev`: start the Next.js dev server on port 3000 with polling enabled.
- `npm run dev:fresh`: remove `.next` and start a clean dev server.
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

`client/AGENTS.md` warns that this project uses a newer Next.js version with breaking changes. Before changing framework behavior, consult the local Next.js docs in `client/node_modules/next/dist/docs/` when available.
