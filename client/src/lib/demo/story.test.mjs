import assert from "node:assert/strict";
import test from "node:test";
import {
  COMMITMENTS,
  DAY_END,
  DAY_START,
  defaultTopicId,
  EXCLUSIONS,
  roadmapFor,
  scoreTopic,
  TASKS,
  TOPICS,
} from "./story.ts";

test("score formula is 100*(0.4F + 0.4E + 0.2R), rounded", () => {
  assert.equal(scoreTopic({ F: 1, E: 1, R: 1 }), 100);
  assert.equal(scoreTopic({ F: 0, E: 0, R: 0 }), 0);
  assert.equal(scoreTopic({ F: 0.5, E: 0.5, R: 0.5 }), 50);
  assert.equal(scoreTopic({ F: 1, E: 0, R: 0 }), 40);
  assert.equal(scoreTopic({ F: 0, E: 0, R: 1 }), 20);
  assert.equal(scoreTopic({ F: 0.95, E: 0.9, R: 0.92 }), 92);
  for (const t of TOPICS) {
    const s = scoreTopic(t);
    assert.ok(Number.isInteger(s) && s >= 0 && s <= 100, `${t.id} score ${s}`);
  }
});

test("default selection is the strictly top-scored topic", () => {
  const top = TOPICS.reduce((a, b) => (scoreTopic(b) > scoreTopic(a) ? b : a));
  assert.equal(defaultTopicId(), "transformers");
  assert.equal(defaultTopicId(), top.id);
  const others = TOPICS.filter((t) => t.id !== top.id);
  assert.ok(others.every((t) => scoreTopic(t) < scoreTopic(top)));
});

test("every roadmap DAG is acyclic and its edges reference existing nodes", () => {
  for (const topic of TOPICS) {
    const { nodes, edges } = roadmapFor(topic.id);
    assert.equal(nodes.length, 6, `${topic.id} has 6 nodes`);
    const ids = new Set(nodes.map((n) => n.id));
    assert.equal(ids.size, nodes.length, `${topic.id} node ids unique`);
    for (const e of edges) {
      assert.ok(ids.has(e.from), `${topic.id}: unknown edge source ${e.from}`);
      assert.ok(ids.has(e.to), `${topic.id}: unknown edge target ${e.to}`);
    }
    // Kahn's algorithm: all nodes must be removable, otherwise there is a cycle.
    const indeg = new Map([...ids].map((id) => [id, 0]));
    for (const e of edges) indeg.set(e.to, indeg.get(e.to) + 1);
    const queue = [...ids].filter((id) => indeg.get(id) === 0);
    let removed = 0;
    while (queue.length) {
      const id = queue.shift();
      removed++;
      for (const e of edges.filter((x) => x.from === id)) {
        indeg.set(e.to, indeg.get(e.to) - 1);
        if (indeg.get(e.to) === 0) queue.push(e.to);
      }
    }
    assert.equal(removed, nodes.length, `${topic.id} roadmap has a cycle`);
  }
});

test("every agent has an output for every roadmap node", () => {
  for (const node of roadmapFor("transformers").nodes) {
    for (const id of [
      "roadmap",
      "course",
      "assessment",
      "cheat-sheet",
      "flashcard",
      "short-bit",
      "story",
    ]) {
      assert.ok(node.outputs[id], `${node.id} missing ${id}`);
    }
  }
});

test("no task overlaps a commitment, an exclusion, or another task", () => {
  const fixed = [...COMMITMENTS, ...EXCLUSIONS];
  const overlaps = (a, b) => a.start < b.end && b.start < a.end;
  for (const task of TASKS) {
    assert.ok(
      task.start >= DAY_START && task.end <= DAY_END,
      `${task.id} inside the day`,
    );
    assert.ok(task.end > task.start, `${task.id} has positive length`);
    for (const f of fixed) {
      assert.ok(!overlaps(task, f), `${task.id} overlaps ${f.id}`);
    }
  }
  for (let i = 0; i < TASKS.length; i++) {
    for (let j = i + 1; j < TASKS.length; j++) {
      assert.ok(
        !overlaps(TASKS[i], TASKS[j]),
        `${TASKS[i].id} overlaps ${TASKS[j].id}`,
      );
    }
  }
});

test("every task has at least 3 reasoning steps", () => {
  for (const task of TASKS) {
    assert.ok(
      task.reasoning.length >= 3 && task.reasoning.length <= 5,
      `${task.id}`,
    );
  }
});
