# Phase 6.4 design: a new model to beat RWKV-Instant (2026-11-17)

## What the literature check found (2026-11-17)
- srs-benchmark upstream (checked: local clone = upstream HEAD 4ddb9fd, 2026-10-25): RWKV-Instant (file RWKV-P) is
  still #1 in both tables (0.2773 without / 0.2660 with same-day reviews, 9,999 users). Nothing published beats it.
- Open PR #360 (FSRS7-Instant, 2026-11-03): FSRS-7 with one Adam step per review. About -7% log loss vs FSRS-7, i.e.
  between RWKV-Curve and RWKV-Instant. Key finding: the gain is largest when predicting at review time, using every
  review up to then, including other cards reviewed since this card was scheduled (-7.1%). With parameters frozen at
  the start of the day it is -3.8%; at scheduling time about 0%. **So "Instant" models win mainly from same-day,
  cross-card, up-to-the-moment information.** Our GBDT today has almost none of it; `today` and `secs` features add it.
- Issue #350: the GRU's meta-training saw users 1001-1100's full histories (contamination). Our splits avoid this.
- Issue #226 (xLSTM) and #23 (Transformer): proposed, never benchmarked.
- Knowledge tracing (KT) literature: attention models with time-aware/forgetting biases (LefoKT AAAI 2025, TO-RoPE
  EDM 2026, recency encodings arXiv 2508.17092, FlatFormer arXiv 2512.06629). None is evaluated on srs-benchmark.
- RWKV's inputs (README): interval lengths, grades, duration, today's new cards and reviews, siblings, deck and preset
  hierarchy, day of week/month/year; card/note/deck/preset/user id embeddings; 2.76M parameters; trained on 5k users.

## Proposed model: TAM (time-aware attention memory), pooled over users
One network trained on the train-pool users, applied to each eval user's stream (like RWKV). Prediction for review t
of card c at time tau_t:

1. **Context tokens:** the user's last K reviews before t (K=256, any card) plus the card's own last M reviews
   (M=32). Each token embeds: rating (one-hot), log duration, log gap of that review, state, and RELATION flags to the
   target: same card / same note (sibling) / same deck / same preset / other. Plus log(tau_t - tau_token) in
   fractional days (relative time, the key forgetting signal).
2. **Query token:** the target's known-before-answer inputs: log gap (secs), state, card review count, FSRS-6
   default R (cheap prior), and the GBDT+ feature vector (so the net starts from what the trees know).
3. **Encoder:** 3 layers, d=128, 4 heads, causal; attention logits get a learned bias per relation type and a
   learned decay in log relative time per head (ALiBi-style forgetting bias, like LefoKT/TO-RoPE). About 0.6M params.
4. **Head:** p = sigmoid(logit R_fsrs + MLP(query output)), so the net learns a correction to the FSRS prior and is
   well-calibrated from step 0.
5. **Training:** log loss on train-pool scored rows (same definition as the benchmark: non-first, elapsed_days >= 1
   for the --short "without same-day" scoring; same-day rows still appear as context). AdamW, lr 1e-3 cosine, batch
   of targets sampled across users, early stopping on train-pool holdout users. Kaggle GPU (T4/P100).
   Every target only ever sees reviews before it, so training is leak-free by construction; tests check this.
6. **Ensemble:** final p = sigmoid(a * logit p_GBDT+ + b * logit p_TAM + c), with a, b, c fitted on the holdout
   users. Trees and attention nets make different errors, so blending usually helps.

## Why this design
- Gets the same up-to-the-moment cross-card information as RWKV-Instant, plus explicit sibling/deck relations.
- Fully parallel training (no per-step recurrence over millions of reviews), unlike a stream RNN.
- Starts from FSRS and GBDT+, so even a small net should not do worse than GBDT+.

## Order of work
GBDT+ first (6.1-6.3; cheapest, may already be enough). TAM (6.4) is built only if GBDT+ is still behind
RWKV-Instant on the validation users 8001-8300. It reuses the 6.2 harness (same users, same scoring, same jsonl).
