# Axiom demo: presenter script (about 4 minutes, slide 29)

## Before you present

```sh
cd client
npm install        # first time only
npm run demo       # production build + server on http://localhost:3000
```

Open http://localhost:3000 (or /demo). No login, database, API keys or Python
needed: everything is seeded. Keys: ← → move between steps, 1–5 jump to a step.
"Light/Dark" sits top right. The URL hash (#1..#5) keeps your step on reload.

## 1 · Trends: what to learn (30 s)
"Axiom first finds what is rising. Each topic is scored 100·(0.4F + 0.4E + 0.2R)
from Hacker News, Reddit and NewsAPI signals. Riya picks Transformers."
Click a card to show the selection driving the next step.

## 2 · Roadmap: how (30 s)
"The roadmap agent builds a plan from prerequisites to advanced topics. For each
node, seven content agents produce the material." Click **Attention (QKV)**.

## 3 · Scheduler: when (40 s)
"The scheduler places work on Riya's energy curve, around her standup, lunch and
capstone meeting." Click task 3 to read its reasoning chain. Point to the red
**Review: 6 flashcards** block: "this block is fed by the memory engine. Next."

## 4 · Memory engine (80 s), the core
Left, **a real card**: "These are real reviews of an unseen srs-benchmark user.
Grey is FSRS-6, red is ours: GBDT plus the online layer." Press **Replay**. "When
she failed three times, FSRS kept guessing she'd forget; ours saw her recover.
Log loss on this card: 0.865 vs 0.426." Optional: **Card D** shows a card where
FSRS wins, for honesty.
Right, **Riya's queue**: "Riskiest card first." Click **✗ Forgot** on any card.
"Every prediction drops a few points and the queue re-sorts. That's the online
layer: six weights per learner, one small step per answer." Click **Reset**.

## 5 · Evidence (40 s)
"On 1,000 unseen users, frozen before scoring: log loss 0.300 vs FSRS-6 0.338,
2nd of all published models. Every dot is a user: ours is better for **995 of
1,000**. It runs 1,000× faster than RWKV-Instant with 14× fewer numbers, and its
probabilities are calibrated."

## Data provenance
- `src/lib/demo/seed/online.json`: `scripts/export_online_demo.py` (frozen final
  model in `results/srsb_beat/final/`; card replays are re-run by the exporter
  and match the recorded per-user log loss exactly).
- `src/lib/demo/seed/results.json`: `scripts/export_demo_data.py` (our 100-user
  Anki split, `results/full_u100_c1000_final/`).
- Steps 1–3, Riya's 6 cards and the 90% bands are seeded and illustrative.
- The card-replay FSRS line uses default weights; the headline 0.338 is FSRS-6
  fitted per user.
