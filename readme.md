# Axiom

Axiom is a cognitive task scheduler and learning planner built with Next.js. It turns goals, coursework, and everyday tasks into schedules that account for cognitive load, energy windows, deadlines, and user feedback. The app also includes roadmap generation, trend discovery, report tracking, and syllabus-driven study workflows.

## Features

- Cognitive-load scheduling with day sections, deadline pressure, burnout risk, and reasoning output.
- Dashboard views for tasks, weekly plans, matrix planning, profile settings, feedback, and reports.
- Roadmap, story, cheatsheet, flashcard, and short-bit learning experiences.
- Syllabus-to-task generation through the roadmap agent.
- Trend engine that can pull signals from Hacker News, Reddit, and NewsAPI.
- Auth and persistence through server actions, JWT cookies, and Vercel Postgres.

## Tech Stack

- Next.js 16 App Router
- React 19 and TypeScript
- Tailwind CSS 4
- Biome for linting and formatting
- Vercel Postgres, bcrypt, and JSON Web Tokens
- Gemini API integration
- 3D/visual libraries: Three.js, React Three Fiber, Drei

## Repository Structure

```text
.
├── AGENTS.md              # Contributor and agent guidelines
├── readme.md              # Project README
└── client/                # Next.js application
    ├── src/app/           # Routes, pages, layouts, and server actions
    ├── src/components/    # Shared UI and feature components
    ├── src/lib/           # Scheduler, AI, trends, and utility logic
    ├── public/            # Static assets
    └── scripts/           # Utility scripts, including demo user seeding
```

## Getting Started

### Prerequisites

- Node.js 20+
- npm
- A Postgres database compatible with `@vercel/postgres`
- Optional API keys for Gemini, NewsAPI, and Reddit enrichment

### Install

```bash
cd client
npm install
```

### Configure Environment

Create `client/.env.local`:

```env
JWT_SECRET=replace_with_a_long_random_secret

# Vercel Postgres / Neon-style connection values
POSTGRES_URL=...
POSTGRES_PRISMA_URL=...
POSTGRES_URL_NON_POOLING=...
POSTGRES_USER=...
POSTGRES_PASSWORD=...
POSTGRES_DATABASE=...
POSTGRES_HOST=...

# AI and trend integrations
GEMINI_API_KEY=...
NEWS_API_KEY=...
REDDIT_CLIENT_ID=...
REDDIT_SECRET=...

```

Only `JWT_SECRET` and database variables are required for authenticated persistence. AI and trend features degrade or skip sources when their keys are missing.

### Run Locally

```bash
npm run dev
```

Open `http://localhost:3000`.

## Scripts

Run these from `client/`:

```bash
npm run dev        # Start Next.js dev server on port 3000
npm run dev:fresh  # Remove .next and start a clean dev server
npm run build      # Build for production
npm run start      # Serve the production build
npm run lint       # Run Biome checks
npm run format     # Format with Biome
```

## Key Routes

- `/` - landing page
- `/login` - authentication
- `/onboarding` - profile setup
- `/dashboard` - main scheduler dashboard
- `/dashboard/tasks` - task planning
- `/dashboard/roadmap` - goal-to-roadmap workflow
- `/dashboard/trends` - trend discovery
- `/dashboard/report` - observation and report view
- `/short-bit`, `/flashcard`, `/demo` - learning demos and experiments

## Development Notes

- The main scheduling implementations live in `client/src/lib/schedule.ts` and `client/src/lib/engine.ts`.
- Server actions are in `client/src/app/actions`.
- Biome uses 2-space indentation and recommended React/Next rules.
- No automated test script is currently configured. Before submitting changes, run `npm run lint`, `npm run build`, and manually exercise affected routes.
- This project uses a newer Next.js version. For framework-specific changes, consult local docs under `client/node_modules/next/dist/docs/` when dependencies are installed.

## License

No license file is currently included.
