---
title: "Predicting Memory: the GBDT Part of the FYP"
subtitle: "Presentation guide, from scratch: big picture, detailed picture, very detailed picture"
date: "19 November 2026"
geometry: margin=2.2cm
toc: true
toc-depth: 2
colorlinks: true
---

# How to read this guide

Every topic has three layers.

- **Big picture.** One or two sentences: what to say on a slide.
- **Detailed picture.** What you say when someone asks "how?".
- **Very detailed picture.** Formulas, numbers and worked examples, for hard questions.

Read the big pictures first (about 10 minutes), then go deeper where you feel unsure. Section 15 is a list of likely questions with short answers. Numbers marked *pending* come from runs still in progress when this was written.

# 0. The whole story on one page

1. **The problem.** A flashcard app (Anki) must decide *when* to show each card again. To do that it needs the **probability that the learner still remembers the card today**.
2. **The team's starting point.** A Bayesian memory model (v1, MCMC) and FSRS, the algorithm Anki actually uses, as the standard baseline.
3. **My part: a GBDT.** A gradient-boosted decision-tree model that reads a learner's past reviews and outputs the recall probability. On our own Anki and Duolingo tests it **beat both** FSRS and the Bayesian model on every metric.
4. **The public test: srs-benchmark.** It has 10,000 Anki users and about 727M reviews, and every published memory model is scored on it. Our first GBDT came **2nd**: it beat FSRS, GRU, LSTM and RWKV-Curve, but not **RWKV-Instant**, a 2.76M-parameter neural network.
5. **Improvements:**
   - more input features: exact time gaps, today's session, answer times, user and card "memory";
   - an **Elo-style online layer** that adjusts 6 numbers per user after every review.
6. **The final model** is GBDT + online layer, **frozen before testing**, then tested once on 1,000 users nobody had looked at.
7. **The result.** Clear 2nd place on the public benchmark:
   - ahead of every FSRS version, LSTM and RWKV-Curve;
   - behind RWKV-Instant;
   - about 14 times fewer stored numbers than RWKV;
   - CPU only.

# 1. The problem: spaced repetition

## Big picture

People forget at a predictable rate. Reviewing a card **just before** you would forget it is the most efficient way to learn. To schedule well, you need a good estimate of *P(recall)*: the chance the learner remembers this card right now.

## Detailed picture

- A **card** is one fact, e.g. "French: *chat*" → "cat".
- A **review** is one time the learner sees the card and grades themselves.
- A **scheduler** picks the next review date. A typical rule: schedule the next review for when predicted recall drops to 90%.
- Better predictions give fewer wasted reviews (seeing cards you already know) and fewer forgotten cards. **So the whole project is a prediction problem:** given everything that happened before, predict the outcome of the next review.

## Very detailed picture

- Memory decays roughly like a **forgetting curve**. Right after a review recall is near 100%, and it falls with time. Each successful review makes the decay slower; this is called "stability".
- Our job is *not* to schedule directly. We output a probability *p*. The scheduler (FSRS in Anki) can use any good *p*. This is why every model in the field is compared on **prediction quality**: log loss and AUC (Section 9).

# 2. The data: Anki review logs

## Big picture

The data is a list of reviews: which card, when, and how the learner graded themselves. We turn each grade into **remembered (1)** or **forgot (0)** and predict that.

## Detailed picture

The Anki dataset (`anki-revlogs-10k`) has 9,999 users and about 727 million reviews. Each review row has:

| Column | Meaning | Example |
|---|---|---|
| `card_id` | which card | 1234 |
| `day_offset` | which day (relative to the user's start) | 10 |
| `rating` | the learner's grade: 1 Again, 2 Hard, 3 Good, 4 Easy | 3 |
| `state` | card state: new, learning, review, relearning | review |
| `duration` | answer time in milliseconds | 4200 |
| `elapsed_days` | days since this card's previous review | 7 |
| `elapsed_seconds` | the same, in seconds (exact) | 604,800 |

Side tables also give each card's **note** (cards from the same note are "siblings") and **deck**.

**The label.** Grade 1 (Again) means *forgot*, so y = 0. Grades 2, 3 and 4 mean *remembered*, so y = 1.

The models do **not** predict Hard/Good/Easy. They predict one number, the probability that y = 1.

## Very detailed picture: one card's history (illustrative)

| Review | Day | Elapsed (days) | Grade | Label y | Scored? |
|---|---|---|---|---|---|
| 1 | 0 | – | Good (3) | – | no: first time seeing the card, nothing to predict |
| 2 | 3 | 3 | Good (3) | 1 | **yes** |
| 3 | 10 | 7 | Again (1) | 0 | **yes** |
| 4 | 10 | 0 | Good (3) | 1 | no: same-day re-study (elapsed = 0) |
| 5 | 12 | 2 | Easy (4) | 1 | **yes** |

**Benchmark rules:**

- A review is **scored** only if at least 1 day has passed since the card's last review (elapsed_days ≥ 1).
- Same-day reviews (like review 4) are **not scored, but they stay in the history**, so a model may use them as input.
- For review 5, the model may use reviews 1–4 (and all the user's other cards up to that moment), but **nothing from the future**. We call this **past-only**, and we have leak tests that check it.

# 3. FSRS: the baseline everyone compares to

## Big picture

FSRS is the algorithm inside Anki. It keeps two numbers per card, **stability S** (how slowly you forget) and **difficulty D**, and computes recall with a fixed forgetting-curve formula. It has about 21 learned weights per user.

## Detailed picture

- After each review FSRS updates S and D. A success raises S (raising it more if the review came late and the card is easy); a lapse cuts S.
- Recall after *t* days is read from a power-law curve built so that **R = 90% exactly when t = S**.
- "FSRS-6 optimised" means the 21 weights were fitted to each user's own past data. FSRS-7 is a newer version that also uses exact seconds.

## Very detailed picture

FSRS-6 forgetting curve:

$$R(t, S) = \left(1 + F \cdot \frac{t}{S}\right)^{-d}, \qquad F = 0.9^{-1/d} - 1$$

where *d* is the decay (a learned weight, about 0.15 by default).

**Worked example** (S = 10 days, d = 0.1542, so F ≈ 0.981):

| Days since review *t* | R |
|---|---|
| 1 | 0.986 |
| 10 (= S) | **0.900** |
| 30 | 0.809 |

**How we use FSRS.** Our GBDT does **not** sit "on top of" FSRS. We replay FSRS (default weights) over each card's history and give its outputs (S, D, R, logit R, t/S) to the GBDT as *some of its 78 inputs*. The ablation (Section 11) shows FSRS inputs help only a little.

# 4. The team's Bayesian model (context)

## Big picture

The team's model is a **hierarchical Bayesian** memory model. Each learner's parameters are drawn from a population distribution, so a new learner borrows strength from everyone else. v1 was fitted with MCMC; v2 and v3 are faster MAP (best-single-guess) versions.

## Detailed picture

On our own test splits (Section 12), the Bayesian v1 scored 0.4221 on Anki. That is worse than FSRS (0.3794), mainly because the MCMC had not converged. The MAP versions v3 and v2 were close to FSRS (0.378 and 0.379). **The GBDT (0.3584) beat all of them.** This is why the project moved to the GBDT.

## Very detailed picture

You don't need to defend the Bayesian model's internals: that's the team's part. Your line is: *"The Bayesian model gives uncertainty and interpretability. The GBDT gives better raw prediction, because it can use many more signals from the learner's history."*

# 5. The GBDT: what it is

## Big picture

A **gradient-boosted decision tree** model (LightGBM) is a sum of many small decision trees. Each tree asks yes/no questions about the inputs ("Has it been more than 30 days?", "Did the learner forget this card last time?") and adds a small correction. The sum is turned into a probability.

## Detailed picture

- **One decision tree:** a flowchart of questions ending in a *leaf* holding a number.
- **Boosting:** trees are built one after another. Each new tree is trained to fix the **errors of all previous trees**. After 1,047 trees, the sum is very accurate.
- **The output.** The trees' leaf values are added up to a **margin** (a "logit") *z*. Then

$$p = \sigma(z) = \frac{1}{1 + e^{-z}}.$$

- **Why a GBDT and not a neural network?** It works well on tables of numeric features, needs no GPU, trains in minutes, and handles missing values.

## Very detailed picture: a toy example with 3 trees

Inputs for one review: 40 days since the last review, the card was forgotten last time, the user's recent recall is 92%.

| Tree | Path taken | Leaf value |
|---|---|---|
| Tree 1 | "user recent recall > 85%?" yes → "gap > 30 days?" yes | +1.8 |
| Tree 2 | "card forgotten last time?" yes | −0.4 |
| Tree 3 | "answer time last time > 10 s?" no | +0.1 |

The margin is z = 1.8 − 0.4 + 0.1 = **1.5**, so p = σ(1.5) = **0.818**: an 82% chance of recall.

**The real model** (`sweep_a_l63_side`):

- 1,047 trees, each with up to 63 leaves, learning rate 0.05;
- each leaf needs at least 200 training reviews;
- 80% of features and rows are sampled per tree, with L2 regularisation 1.0;
- trained on 270 users (10.4M reviews), with 30 more users used for early stopping (stop adding trees when the holdout stops improving);
- about 47 minutes on 4 CPU cores;
- about 196,000 numbers that matter (the split thresholds and leaf values).

**Training objective.** Binary log loss, the same quantity we are scored on.

# 6. The GBDT's inputs: 78 past-only features

## Big picture

For every review we compute 78 numbers that describe the **card's past**, the **learner's past** and **today's session**. Everything is computed from data *before* the review.

## Detailed picture: the feature groups

| Group | What it captures | Example features |
|---|---|---|
| **Gap** | time since the card's last review | `el` (days), `log_el`, `el_frac`, `log_el_frac`, `same_day` |
| **Card history** | how this card went before | `card_n` (reviews so far), `card_lapses`, `card_rate`, last 3 grades `g1..g3`, outcomes `y1..y3`, gaps `gap1..gap3`, `card_max_gap`, `since_first` |
| **FSRS** | FSRS's view of the card | `f_S`, `f_D`, `f_R`, `f_logitR`, `f_el_over_S`, `f_R_secs` |
| **User** | the learner's overall form | `u_n`, `u_rate` (running recall), `u_recent`, `u_day_n`, `u_since_last`, `u_last5` |
| **User memory (EWMAs)** | recall averaged over the last ~10/50/200/1000 reviews | `ewm10`, `ewm50`, `ewm200`, `ewm1000` |
| **Today** | the current session | `n_today`, `n_first_today`, `rate_today`, `lapses_today` |
| **Answer time** | hesitation signals | `dur_card_last`, `dur_card_mean`, `dur_user_mean`, `dur_card_last_rel` |
| **Grade counts** | how often each grade was used | `rc1..rc4` |
| **State** | new / learning / review / relearning | `state`, `state_rate` |
| **Siblings and decks** | related cards | `sib_n`, `sib_rate`, `deck_rate`, `preset_rate`, `parent_rate` |
| **Model-error memory** | where FSRS has been wrong for this user | `resid_mean`, `resid_ewm100`, `resid_gap_mean` |
| **Bucket rates** | the user's recall in similar situations | `gap_bkt_rate` (same gap range), `cnt_bkt_rate` (same card age), `g1_rate` (same last grade) |

## Very detailed picture: what matters most

By the GBDT's own **gain** measure (how much each feature reduced the error):

| Feature | Share of gain |
|---|---|
| `ewm50` (the user's recall over roughly their last 50 reviews) | 24% |
| `f_el_over_S` (time since review divided by FSRS stability) | 14% |
| the FSRS recall features together | about 23% |
| user recency features | about 15% |

**In one line: the model is "the learner's current form × where the card sits on its forgetting curve".**

**Worked example** (illustrative). Review 5 of the card in Section 2:

- el = 2 days; card_n = 4; card_lapses = 1; g1 = Good (the same-day re-study), y1 = 1, y2 = 0;
- f_S ≈ 3 days (cut by the lapse), so f_R ≈ 0.96;
- the user's ewm50 = 0.88; rate_today = 0.9; n_today = 25.

The GBDT reads all 78 such numbers and outputs, say, z = 2.4, so p = 0.917.

# 7. The online layer ("Elo-style" per-user adjustment)

## Big picture

The GBDT is the same for everyone. The **online layer** gives each learner **6 personal numbers** that start neutral and are nudged after every review. If the model keeps over-predicting one learner, their numbers move to correct it. It's like an Elo rating that keeps learning.

## Detailed picture

- Inputs to the layer, for each review: the GBDT's margin *z* and 4 context signals: the gap, recent form (`ewm50`), FSRS's recall (`f_logitR`) and today's success rate (`rate_today`).
- Prediction: a small logistic regression with the user's weights **w**.
- After the review's answer is known, one gradient step updates **w**. The update is *after* the prediction, so it's still past-only.
- Weights start at "trust the GBDT exactly" (w = [1, 0, 0, 0, 0, 0]). A small pull (λ) keeps them from drifting far.
- **It's individual:** there is no global Elo. Each user's 6 numbers only see that user's reviews.

## Very detailed picture

The design vector for a review is:

$$x = \left[\, z,\; 1,\; \tfrac{\text{log\_el\_frac}}{3},\; (\text{ewm50} - 0.5)\cdot 4,\; \tfrac{\text{f\_logitR}}{3},\; (\text{rate\_today} - 0.5)\cdot 4 \,\right]$$

The prediction and the update after the answer *y* is seen:

$$p = \sigma(w \cdot x), \qquad w_j \leftarrow w_j - \eta\left[(p - y)\,x_j + \lambda\,(w_j - w_{0,j})\right]$$

with η = 0.003 (chosen on validation users 8001–8050) and λ = 0.001 (fixed).

**Worked example: one update.** Take z = 1.5, log_el_frac = 0.6, ewm50 = 0.8, f_logitR = 2.1, rate_today = 0.9. Then x = [1.5, 1, 0.2, 1.2, 0.7, 1.6].

1. **Before:** w = [1, 0, 0, 0, 0, 0], so w·x = 1.5 and p = 0.818.
2. **The learner forgets (y = 0).** The error is g = p − y = 0.818.
3. **The update** (λ term ≈ 0 at the start):
   - w₁ = 1 − 0.003 · 0.818 · 1.5 = 0.99632
   - bias w₂ = −0.00245
   - w₃ = −0.00049, w₄ = −0.00294, w₅ = −0.00172, w₆ = −0.00393
4. **The same situation again:** w·x = 1.4809, so p = **0.815**, down from 0.818.

One review moves the prediction only a little. Over hundreds of reviews the layer learns, for example, that "this learner does worse after long gaps than the GBDT thinks".

**Gain on validation:** −0.0042 log loss [−0.0063, −0.0027], the biggest single improvement after the features. It helps most for weak learners (Section 10).

# 8. Elo (used for analysis, and in the Elo-expert experiment)

## Big picture

**Elo** (from chess) gives each learner an *ability* θ and each card a *difficulty* d. The predicted recall is σ(θ − d). After each review both are nudged. We used it to **group learners by strength** and study where models fail.

## Detailed picture

$$p = \sigma(\theta_{\text{user}} - d_{\text{card}})$$

After a review with outcome *y*:

$$\theta \leftarrow \theta + 0.05\,(y - p), \qquad d \leftarrow d - \frac{0.4}{1 + 0.05\,n_{\text{card}}}\,(y - p)$$

A correct answer raises ability and makes the card look easier. A card's step size shrinks as it gets more reviews.

## Very detailed picture

**Cold start.** θ = 0 and d = 0 give p = 50%, while real recall is about 88%. With step 0.05 it takes about **100–200 reviews** for θ to settle (the average is around 2.6). This is a known limitation, and we chose not to change it mid-experiment, to keep the runs comparable.

**What the Elo bands showed** (100 validation users, split into 5 equal bands E1 = weakest to E5 = strongest):

| | E1 (weak) | E5 (strong) |
|---|---|---|
| Log loss of our model | 0.549 | 0.084 |
| Gain from the online layer | 0.0085 | 0.0016 |
| AUC | about 0.73 (E1–E4) | 0.84 |
| FSRS bias (predicted − actual recall) | **+9.2 points** (over-confident) | **−5.7 points** (too pessimistic) |

The take-home: **FSRS is over-confident for weak learners and too pessimistic for strong ones; our model corrects both.** Weak learners are also where we lose most to RWKV.

**The Elo-expert experiment** trained separate GBDTs per Elo band, with hard cut-offs or soft overlapping cut-offs, and with or without FSRS inputs. Result: *pending*. Early signs suggest the fine-tuned experts add little: their early stopping picked very few extra trees for strong bands.

# 9. How models are scored

## Big picture

Two numbers:

- **Log loss:** how good the probabilities are; lower is better.
- **AUC:** how well the model *ranks* forgotten reviews below remembered ones; higher is better, 0.5 is guessing.

## Detailed picture

**Log loss** for one review: −[y·ln p + (1 − y)·ln(1 − p)]. It's averaged over reviews, then over users.

**Log loss examples:**

| Prediction *p* | Outcome | Loss |
|---|---|---|
| 0.9 | remembered | 0.105 (small: confident and right) |
| 0.9 | forgot | 2.303 (large: confident and wrong) |
| 0.5 | anything | 0.693 (coin flip) |

**AUC example.** Four reviews, with predictions 0.95 (remembered), 0.90 (forgot), 0.80 (remembered), 0.60 (forgot). Check every (remembered, forgot) pair to see whether the remembered one got the higher *p*:

- (0.95, 0.90): yes
- (0.95, 0.60): yes
- (0.80, 0.90): **no**
- (0.80, 0.60): yes

AUC = 3/4 = **0.75**.

**Calibration** asks whether, when we say 80%, it really happens 80% of the time. Ours is calibrated within ±0.6 percentage points in every bucket. **Our remaining gap to RWKV is ranking (AUC), not calibration.**

## Very detailed picture: the benchmark protocol

- **Time-ordered splits** (TimeSeriesSplit, 5 folds per user). Each user's history is cut into consecutive chunks. A model is always tested on a later chunk, using only earlier data, so the test always lies in the future.
- **Pooled models.** The GBDT and RWKV are trained on *other* users (ours on users 2001–2300, RWKV on 5,000 users), then applied to each test user's test chunks. Per-user adaptation happens only through the history features and the online layer.
- **Two averages:**
  - **unweighted**: every user counts equally;
  - **weighted**: users count by their number of test reviews, so heavy users count more.
- **Paired bootstrap 99% intervals.** We resample *users* 1,000 times and recompute the difference between two models on the same users. If the 99% interval excludes 0, the difference is real, not luck.
- **Honesty rules:**
  - settings chosen on validation users 8001–8050 only;
  - the final model written down (pre-registered, commit 38c1199) *before* touching test users 1–1,000;
  - test users scored once.

# 10. srs-benchmark and our results

## Big picture

srs-benchmark is the open leaderboard of memory models, run by the FSRS authors, on 10,000 Anki users. On the same 1,000 test users the leaderboard uses, our GBDT is **2nd of all published models**.

## Detailed picture: the final test (users 1–1,000, 32.7M reviews)

**Stage 1: frozen GBDT alone** (done):

| Model | Log loss (weighted) | Log loss (unweighted) | AUC (weighted) | AUC (unweighted) |
|---|---|---|---|---|
| RWKV-Instant (2.76M params, GPU) | **0.2828** | **0.2860** | **0.827** | **0.832** |
| **Our GBDT** | **0.3051** | **0.3088** | **0.790** | **0.800** |
| RWKV-Curve | 0.3143 | 0.3302 | 0.771 | 0.767 |
| LSTM | 0.3279 | 0.3460 | 0.734 | 0.728 |
| FSRS-7 | 0.3342 | 0.3520 | 0.718 | 0.715 |
| FSRS-6 (optimised per user) | 0.3383 | 0.3580 | 0.706 | 0.699 |
| FSRS-6 (default weights) | 0.3567 | 0.3780 | 0.690 | 0.693 |
| HLR (Duolingo's old model) | 0.4710 | 0.5132 | 0.613 | 0.611 |
| Anki default scheduler | 0.5219 | 0.6304 | 0.587 | 0.611 |

**Paired differences, GBDT minus the other model** (weighted, 99% interval; negative means we are better):

- vs RWKV-Instant: **+0.0224** [+0.0198, +0.0253]: we lose;
- vs RWKV-Curve: **−0.0091** [−0.0141, −0.0045]: we win;
- vs FSRS-6: **−0.0332** [−0.0387, −0.0285]: we win clearly.

**Stage 2: GBDT + online layer** (the full final model): *pending; running*. On validation the online layer cut log loss by about 0.004, so expect about **0.301 weighted**, about 0.018 behind RWKV-Instant. Only the real number counts.

## Very detailed picture: how we got here (validation, gap to RWKV-Instant)

| Step | Gap |
|---|---|
| First GBDT (pre-existing features, 500 train users) | +0.032 |
| + new features (seconds, today, durations, siblings, decks) | +0.0218 |
| + user and card memory features (`umem`) | +0.0180 |
| bigger trees (63 leaves) | +0.0174 |
| **+ online layer (final model)** | **+0.0133** |
| (5× more training users, 1,500: helps about 0.002) | (+0.0151 without online) |

# 11. Why FSRS is still in the model (ablation)

## Big picture

Removing all FSRS inputs makes the model slightly worse, by 0.0016 (or 0.0009 with the online layer). FSRS costs almost nothing to compute, so it stays.

## Detailed picture

| Comparison | No-FSRS minus with-FSRS (99% interval) |
|---|---|
| GBDT alone | +0.0016 [+0.0008, +0.0025] |
| GBDT + online | +0.0009 [+0.0003, +0.0014] |

The gain is largest for **low-retention learners** (+0.0022). FSRS is under 0.3% of compute time.

## Very detailed picture

In our own Phase 7 tests the story was the same: removing FSRS cost +0.0013 (Anki), while removing the user features cost +0.014, ten times more. **The user's own history is the main signal; FSRS is a useful prior on the forgetting curve.**

# 12. Before srs-benchmark: our own tests (Phase 7)

| Model | Anki (100 test users) | Duolingo (all users) |
|---|---|---|
| **GBDT** | **0.3584** | **0.2588** |
| FSRS-6 optimised | 0.3794 | 0.3003 |
| Bayesian v1 (MCMC) | 0.4221 | 0.2682 |
| Bayesian v3 / v2 (MAP) | 0.3780 / 0.3791 | 0.2830 / 0.2892 |

The GBDT won on 82 of 100 Anki users and was the best on every metric on both datasets.

**Robustness checks:**

- other random seeds: the same result;
- validation → test: 0.3614 → 0.3584, so no overfitting to validation;
- train vs test: 0.3467 vs 0.3584, a small gap;
- Duolingo users never seen in training: still better than FSRS (0.233 vs 0.275).

**A bug we caught and disclosed.** Duolingo logs a whole session at one timestamp. Our first user statistics could see same-timestamp answers. We found it, fixed it, and kept the invalid results labelled as invalid.

# 13. What we tried that didn't help (and why)

| Idea | What happened | Why |
|---|---|---|
| Per-user boosting (a small GBDT per learner) | much worse (0.314 → 0.37) | too little data per user, so it overfits |
| Kalman filter for user drift | worse; some settings blew up | over-reacts after study breaks |
| Attention model (TAM, 0.63M params) | gap +0.0218 | needs far more data and GPU time than we had |
| Residual GRU on the user's sequence | −0.0014 (small gain) | under-trained, CPU only |
| Experts by user type (weak, mid, strong) | no gain | the global GBDT already models user type |
| Mixture-of-experts gates (incl. a 3×3 form × gap grid) | tie or slightly worse | same reason |
| Elo ratings as features | negative | ewm50 already carries that information |
| Learning FSRS weights inside the GBDT | tie or worse | FSRS was never the bottleneck |
| Elo-band experts | *pending* | — |

**The message:** we hit the ceiling of hand-made features. The remaining gap is RWKV reading the **raw sequence** at large scale.

# 14. Size, cost and speed

| | Our final model | RWKV-Instant |
|---|---|---|
| Stored numbers | about 196k (1,047 trees) + 6 per user | 2.76M parameters |
| Training data | 270 users (10.4M reviews) | 5,000 users |
| Hardware | 4 CPU cores, about 47 min | GPU |
| Time per review (local test, busy PC) | about 0.55 ms | about 156 ms (CPU, step by step) |

The speed numbers are preliminary. A clean Kaggle measurement on idle machines (with RWKV also on GPU) is *pending*. One caveat to state honestly: we compute features in a batch per user, while RWKV was stepped review by review.

**Your line:** *"Second place with about 14 times fewer stored numbers, no GPU, and less than 1/18 of the training users."*

# 15. Likely questions and short answers

**Q: Why didn't you beat RWKV-Instant?**
It's a 2.76M-parameter sequence model trained on 5,000 users on GPUs, and it reads the raw review sequence. Our gap is in *ranking* (AUC), concentrated on weak and same-day-heavy learners. Feature engineering hit a ceiling: ten different ideas gave no further gain.

**Q: Is the GBDT just FSRS plus corrections?**
No. FSRS outputs are 6 of 78 inputs. Removing them costs only 0.0016, while removing user history costs about 0.014.

**Q: Did you tune on the test set?**
No. All choices were made on validation users. The final model was written down and committed before the test, and the test users were scored once.

**Q: What is the online layer, exactly?**
Six weights per learner in a logistic regression on top of the GBDT's output, updated by one gradient step after each review. It starts as "trust the GBDT" and learns each learner's bias.

**Q: Is this "Elo"?**
It's Elo-*like*: an online rating updated by the prediction error. Proper Elo (ability minus difficulty) was used for the analysis and the expert experiment.

**Q: How do you turn ratings into a probability problem?**
Again = forgot (0); Hard, Good and Easy = remembered (1). We predict the probability of 1. This is the benchmark's own convention.

**Q: Why is weighted different from unweighted?**
Weighted counts heavy users more. Both are reported, and we are 2nd on both.

**Q: Could it be luck?**
The 99% paired bootstrap intervals exclude zero against every model except RWKV-Instant. A second fresh test set (users 1001–2000) is *pending* as an extra check.

**Q: What does it cost to run in an app?**
A CPU-only model of a few MB. Features update incrementally per review, and the online layer is 6 multiply-adds.

**Q: Limitations?**
- Elo cold start (100–200 reviews).
- srs-benchmark results are Anki-only; Duolingo was tested on our own split.
- We have not built a scheduler or run a user study: we predict, we don't schedule.

# 16. Glossary

- **Recall probability:** the chance the learner remembers the card right now.
- **Logit / margin z:** a score where p = 1/(1 + e^(−z)); z = 0 means 50%, z = 2 means 88%.
- **Stability (S):** in FSRS, the days until recall falls to 90%.
- **GBDT:** gradient-boosted decision trees, a sum of many small trees, each fixing the previous ones' errors.
- **Early stopping:** stop adding trees when held-out users stop improving.
- **EWMA (ewm50):** an exponentially weighted moving average, a running average that favours recent reviews.
- **Past-only:** a feature uses only data from before the review being predicted.
- **Pre-registration:** writing down the final model before testing, so it can't be changed after seeing results.
- **Bootstrap CI:** resampling users to measure how much a result could vary by chance.
- **RWKV:** a recurrent neural network architecture. RWKV-Instant predicts from the full sequence including timing; RWKV-Curve outputs a forgetting curve.
