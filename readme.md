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

# Course narration (self-hosted, local dev only — see "Course Narration Setup" below)
PIPER_BINARY_PATH=piper                 # defaults to "piper" on PATH if unset
PIPER_MODEL_PATH=...                    # required — path to a Piper .onnx voice model, no default
PIPER_ESPEAK_DATA_PATH=...              # optional — Piper's bundled espeak-ng-data dir, needed for phonemization unless espeak-ng-data is already on the system search path
RHUBARB_BINARY_PATH=rhubarb             # optional — only read when LIPSYNC_ENGINE=rhubarb; defaults to "rhubarb" on PATH
LIPSYNC_ENGINE=amplitude                # optional — "amplitude" (default, fast, no Rhubarb needed) or "rhubarb"
RHUBARB_RECOGNIZER=phonetic             # optional — only used when LIPSYNC_ENGINE=rhubarb; "phonetic" (fast, default) or "pocketSphinx" (slower, word-aware)

```

Only `JWT_SECRET` and database variables are required for authenticated persistence. AI, trend, and narration features degrade or skip sources when their keys/binaries are missing.

### Course Narration Setup (Piper TTS)

The course player's AI Instructor avatar needs Piper installed locally — it's not an npm package, and (per the scope decision in `plans/02b-course-narration-video-chat.md`) is local-dev-only for now, not wired for Vercel's serverless runtime. Without it, narration silently degrades to no audio; nothing else breaks.

1. **Download Piper** (portable binary, no sudo needed):
   ```bash
   mkdir -p ~/.local/share/axiom-tools/piper && cd ~/.local/share/axiom-tools
   curl -sL -o piper_linux_x86_64.tar.gz \
     "$(curl -s https://api.github.com/repos/rhasspy/piper/releases/latest \
        | grep -o '"browser_download_url": *"[^"]*piper_linux_x86_64.tar.gz"' \
        | cut -d'"' -f4)"
   tar -xzf piper_linux_x86_64.tar.gz
   ```
   (For macOS/Windows, grab the matching asset from the [Piper releases page](https://github.com/rhasspy/piper/releases) instead.)

2. **Download a voice model** (English example — browse [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) on Hugging Face for other languages/voices):
   ```bash
   mkdir -p ~/.local/share/axiom-tools/piper-voices && cd ~/.local/share/axiom-tools/piper-voices
   curl -sL -o en_US-lessac-medium.onnx \
     "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
   curl -sL -o en_US-lessac-medium.onnx.json \
     "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
   ```

3. **Point `.env.local` at them**:
   ```env
   PIPER_BINARY_PATH=/home/you/.local/share/axiom-tools/piper/piper
   PIPER_MODEL_PATH=/home/you/.local/share/axiom-tools/piper-voices/en_US-lessac-medium.onnx
   PIPER_ESPEAK_DATA_PATH=/home/you/.local/share/axiom-tools/piper/espeak-ng-data
   ```

4. **Verify it works** standalone before trusting the app with it:
   ```bash
   echo "Hello from Piper." | ~/.local/share/axiom-tools/piper/piper \
     --model ~/.local/share/axiom-tools/piper-voices/en_US-lessac-medium.onnx \
     --output_file /tmp/piper-test.wav \
     --espeak_data ~/.local/share/axiom-tools/piper/espeak-ng-data
   ```
   A valid WAV at `/tmp/piper-test.wav` means it's working.

Lip sync is handled by `LIPSYNC_ENGINE=amplitude` (the default) with no extra install — it reads the Piper WAV's own loudness envelope directly, no separate binary. Only install Rhubarb if you specifically want its slower, word-aware visemes:

```bash
mkdir -p ~/.local/share/axiom-tools/rhubarb && cd ~/.local/share/axiom-tools/rhubarb
curl -sL -o rhubarb.zip \
  "$(curl -s https://api.github.com/repos/DanielSWolf/rhubarb-lip-sync/releases/latest \
     | grep -o '"browser_download_url": *"[^"]*Linux.zip"' \
     | cut -d'"' -f4)"
unzip -q rhubarb.zip && chmod +x Rhubarb-Lip-Sync-*/rhubarb
```
Then set `RHUBARB_BINARY_PATH` to the extracted `rhubarb` binary and `LIPSYNC_ENGINE=rhubarb` in `.env.local`. (Grab the macOS/Windows asset from the [Rhubarb releases page](https://github.com/DanielSWolf/rhubarb-lip-sync/releases) instead, if applicable.)

Restart `npm run dev` after changing any of these — Next only reads `.env.local` at server start.

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
