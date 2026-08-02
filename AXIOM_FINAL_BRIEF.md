# Axiom — Final Project Brief (Completed System)

**Prepared for: Pitch Deck**
**Status: describes what is built and working today, in `client/src/lib/` and `client/src/app/`, as of this brief. Future work lives separately in `plans/`.**

---

## 1. The Problem

Every learner today juggles three tools that don't talk to each other:

1. **What to learn** — trend-spotting is manual: scrolling Hacker News, Reddit, and tech newsletters, guessing what's worth the time investment.
2. **How to learn it** — course platforms are static and generic. The same course plays the same way for a beginner and someone halfway to mastery.
3. **When to learn it** — calendars and to-do lists have no concept of mental effort. They'll schedule three hard problem sets back-to-back with the same indifference as three errands.

None of these three systems share data. A trending topic never becomes a course. A course never adjusts to how well the learner is actually doing. A schedule never knows what it's scheduling is cognitively expensive versus trivial.

## 2. What Axiom Is

Axiom is a single closed loop across all three:

```
   WHAT                      HOW                       WHEN
Trend Engine  ──────►  Roadmap + Content  ──────►  Cognitive Scheduler
(discover)              Generators (teach)           (place in time)
```

It's a Next.js application, not a slideware concept — every piece named below is implemented, running code.

## 3. What's Built, Pillar by Pillar

### WHAT — Trend Discovery
- Aggregates live signals from **Hacker News, Reddit, and NewsAPI** in parallel.
- Maps raw article/post titles to ~60 canonical tech topics (AI, web dev, cloud, data, security, languages, mobile, blockchain, career/meta) via a keyword engine.
- Scores every topic on **frequency, engagement, and recency** (exponential recency decay, 24h half-life), ranks and surfaces the top movers on a live dashboard.
- Classifies each trend's **direction** (rising/stable/declining) and **momentum** (accelerating/steady/slowing) from real signal history — not a static label.
- An "Explore" mode lets a user type any topic and get an AI-generated market overview: relevance score, effort/ROI estimate, job demand, a full prerequisite-graph breakdown into subtopics, all grounded in the live trend data where available.
- Users declare interest domains and a profile type to personalize what surfaces.

### HOW — Adaptive Content Generation
- **Roadmap Agent** — turns a stated goal into a personalized Directed Acyclic Graph of learning nodes (courses + assessments), respecting prerequisites, sized to the learner's available daily time and experience level, fast-tracking topics they've already shown mastery in.
- **Course Agent** — generates full slide-deck courses from a topic: title, section, content, code, image, table, list, quote, and Mermaid-diagram slide types, composed automatically into a coherent crash course.
- **Assessment Agent** — generates mixed-format quizzes (MCQ, short answer, long answer, coding with test cases) per topic, and evaluates free-form answers with AI-graded feedback and rubric scoring.
- **Cheatsheet, Flashcard, Short-Bit, and Story Agents** — the same topic can be consumed as an exam cheat-sheet, a spaced-repetition flashcard deck, a bite-size micro-learning card, or a narrative web-novel chapter — four different learning styles from one content pipeline.
- Every generator is LLM-backed (Gemini) with structured-JSON contracts, retry logic, and rate-limit handling already production-hardened.

### WHEN — Cognitive-Load Scheduling
This is Axiom's core technical differentiator: a **fully deterministic, explainable scheduler** — the opposite of a black-box AI planner.

- **Axiom energy model** — every task's cognitive cost is computed from difficulty and priority (`E_task = 0.6·difficulty² + 0.8·priority`), against a daily 50-axiom budget split across morning/afternoon/evening.
- **Ultradian bandwidth curve** — models real attention rhythm across the day (peaks ~10am and ~3pm, troughs post-lunch and evening), personalized further by the user's own declared peak-focus and low-energy windows.
- **Deterministic placement engine** — sorts tasks by urgency, priority, and sequence, then places each into the highest-fitness available slot across a 7-day / 96-slot-per-day window, respecting sleep, fixed commitments, and hard exclusions.
- **Anti-starvation & diversity enforcement** — Shannon-entropy diversity scoring keeps a day from becoming monotonous; a rule guarantees a lower-priority task gets interleaved after three consecutive high-priority ones.
- **Burnout risk detection** — classifies upcoming days as safe/watch/warning/critical from rolling axiom-usage windows, before overload happens, not after.
- **Feedback-driven calibration** — section-time weights and per-task-type effort multipliers adjust automatically from the learner's actual completion history, not just their stated preferences.
- **Full reasoning chain** — every placement decision emits a human-readable explanation (why this slot, what was rejected and why), rendered live in the UI. Nothing is a black box.
- Conflict resolution UI lets a learner defer, sacrifice, or extend a deadline when a task can't be placed feasibly — the system never silently overloads a day.

### Product Surface
- Full dashboard: weekly matrix view, task list, roadmap DAG visualization, trend dashboard + explore mode, profile/onboarding flow, feedback/adherence reporting, report export.
- Auth and persistence: JWT-cookie sessions, Vercel Postgres, demo-mode seeding for instant trial.

## 4. Why This Is Different

| | Typical AI planner | Static calendar/to-do app | **Axiom** |
|---|---|---|---|
| Explainable | No — black box | Yes, but nothing to explain | **Yes — full reasoning chain per decision** |
| Reproducible (same input → same output) | No | Yes | **Yes — fully deterministic** |
| Models cognitive effort, not just time | No | No | **Yes — energy/axiom model** |
| Connects discovery → content → schedule | No | No | **Yes — one loop, not three tools** |
| Adapts content depth to demonstrated mastery | Rarely | No | Roadmap agent does today; full personalization is active development (see `plans/04`) |

The scheduler's determinism is a deliberate, defensible choice, not a limitation: every schedule Axiom produces can be audited, reproduced, and explained — a real gap in every commercial AI scheduling tool on the market today.

## 5. Tech Stack

- **Frontend/App**: Next.js 16 (App Router), React 19, TypeScript, Tailwind CSS 4
- **AI**: Gemini API, structured-JSON generation contracts across 6 content agents
- **Data**: Vercel Postgres, JWT-cookie auth
- **3D/Visual**: Three.js, React Three Fiber, Drei (foundation already in place for the in-progress course-narration work)

## 6. What's Next (kept out of this brief, tracked separately)

Five workstreams are actively planned and documented in `plans/`: a formal research paper on the scheduler (with a real optimizer implementation replacing the current heuristic for comparison), agentic course generation with 3D narrated video export, trend-prediction accuracy improvements, a unified personalization layer, and full closed-loop integration across all three pillars. None of that is claimed as done here — this brief is deliberately scoped to what a user can use today.
