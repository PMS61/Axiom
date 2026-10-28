import assert from "node:assert/strict";
import test from "node:test";
import { makeQueue, topics } from "./demo.ts";

const items = topics.map((_, id) => ({
  id,
  recall: (6 - id) / 7,
  low: 0,
  high: 1,
  static: 0.5,
}));

test("risk-first queue respects every study budget without duplicates", () => {
  for (let budget = 30; budget <= 120; budget += 15) {
    const queue = makeQueue(items, budget);
    assert.ok(
      queue.reduce((sum, item) => sum + topics[item.id].minutes, 0) <= budget,
    );
    assert.equal(new Set(queue.map((item) => item.id)).size, queue.length);
    assert.equal(queue[0].id, 5);
    for (let i = 1; i < queue.length; i++)
      assert.ok(queue[i - 1].recall <= queue[i].recall);
  }
});

test("fixed-order baseline ignores recall and skips sessions that do not fit", () => {
  assert.deepEqual(
    makeQueue(items, 40, false).map((item) => item.id),
    [0, 2],
  );
  assert.deepEqual(makeQueue(items, 0), []);
});
