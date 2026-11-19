import assert from "node:assert/strict";
import test from "node:test";
import {
  answer,
  ETA,
  LAMBDA,
  logit,
  predict,
  RIYA_CARDS,
  riskQueue,
  update,
  W0,
} from "./online.ts";

const [A, ...OTHERS] = RIYA_CARDS;

test("seed data: six cards with six features and consistent margin", () => {
  assert.equal(RIYA_CARDS.length, 6);
  assert.equal(new Set(RIYA_CARDS.map((c) => c.id)).size, 6);
  for (const card of RIYA_CARDS) {
    assert.equal(card.x.length, 6);
    assert.ok(Math.abs(card.x[0] - logit(card.baseP)) < 1e-3, card.id);
    assert.equal(card.x[1], 1, `bias of ${card.id}`);
    for (const v of card.x.slice(1)) assert.ok(v >= -1 && v <= 1, card.id);
    assert.ok(
      card.bayesLow < card.baseP && card.baseP < card.bayesHigh,
      card.id,
    );
    assert.ok(card.bayesLow > 0 && card.bayesHigh < 1, card.id);
    assert.ok(card.minutes > 0);
  }
});

test("lr 0: output equals the GBDT base probability and weights do not move", () => {
  for (const card of RIYA_CARDS) {
    assert.ok(Math.abs(predict(W0, card.x) - card.baseP) < 1e-3, card.id);
    const w = answer(W0, card, false, 0);
    assert.deepEqual(w, [...W0]);
    assert.ok(Math.abs(predict(w, card.x) - card.baseP) < 1e-3, card.id);
  }
});

test("forgot lowers that card and, through the shared bias, the others", () => {
  const w = answer(W0, A, false);
  assert.ok(predict(w, A.x) < A.baseP, "forgotten card must drop");
  assert.ok(w[1] < 0, "shared bias must move down");
  const lowered = OTHERS.filter((c) => predict(w, c.x) < c.baseP).length;
  assert.ok(
    lowered >= 4,
    `expected at least 4 of 5 other cards to drop, got ${lowered}`,
  );
});

test("remembered raises that card and, through the shared bias, the others", () => {
  const w = answer(W0, A, true);
  assert.ok(predict(w, A.x) > A.baseP, "remembered card must rise");
  assert.ok(w[1] > 0, "shared bias must move up");
  const raised = OTHERS.filter((c) => predict(w, c.x) > c.baseP).length;
  assert.ok(
    raised >= 4,
    `expected at least 4 of 5 other cards to rise, got ${raised}`,
  );
});

test("update is pure: inputs are not mutated", () => {
  const w = [1, 0.2, -0.1, 0, 0.3, 0];
  const snapshot = [...w];
  const next = update(w, A.x, 0, predict(w, A.x), ETA);
  assert.deepEqual(w, snapshot);
  assert.notEqual(next, w);
});

test("L2 pull: with y equal to p the step is exactly the pull toward w0", () => {
  const w = [1.5, 0.4, 0, 0, 0, 0];
  const p = predict(w, A.x);
  const eta = 0.1;
  const lam = 0.5;
  const next = update(w, A.x, p, p, eta, lam);
  for (let j = 0; j < w.length; j++) {
    const expected = w[j] - eta * lam * (w[j] - W0[j]);
    assert.ok(Math.abs(next[j] - expected) < 1e-12, `dim ${j}`);
  }
  assert.ok(next[0] < w[0] && next[0] > W0[0]);
});

test("default lambda is small and positive", () => {
  assert.ok(LAMBDA > 0 && LAMBDA < 0.01);
});

test("risk queue is sorted ascending by predicted recall", () => {
  const w = answer(answer(W0, OTHERS[0], false), OTHERS[2], true);
  const queue = riskQueue(RIYA_CARDS, w);
  assert.equal(queue.length, 6);
  for (let i = 1; i < queue.length; i++) {
    assert.ok(queue[i - 1].p <= queue[i].p, "queue must not decrease");
  }
  assert.equal(
    queue[0].card.id,
    RIYA_CARDS.reduce((lo, c) => (predict(w, c.x) < predict(w, lo.x) ? c : lo))
      .id,
  );
});

test("predictions stay strictly inside (0, 1) after many answers", () => {
  let w = [...W0];
  for (let i = 0; i < 300; i++) {
    const card = RIYA_CARDS[i % RIYA_CARDS.length];
    w = answer(w, card, i % 3 !== 0);
    for (const c of RIYA_CARDS) {
      const p = predict(w, c.x);
      assert.ok(p > 0 && p < 1, `p=${p} at step ${i}`);
    }
  }
});
