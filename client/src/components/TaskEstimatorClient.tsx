"use client";

import { type FormEvent, useEffect, useState, useTransition } from "react";
import {
  type EstimatedTask,
  estimateLearningTasks,
  type TaskPlanResult,
} from "@/app/actions/task-estimation";
import type { AdaptiveProfile } from "@/lib/research/contracts";
import {
  FIXED_INTERVAL_BASELINE,
  fixedIntervalSessionCount,
} from "@/lib/research/m2-fixed-interval";
import { M3Profile } from "@/lib/research/m3-profile";

const example =
  "I want to learn SQL joins and solve practice questions. I know simple SELECT queries but have not used joins. I can study for three hours this week.";
const split = (value: string, max: number) =>
  value
    .split(",")
    .map((item) => item.trim().slice(0, max))
    .filter(Boolean)
    .slice(0, 30);

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <div className="eyebrow">{label}</div>
      <div className="metric-value">{value}</div>
    </div>
  );
}

function MemoryContext({
  context,
}: {
  context: NonNullable<TaskPlanResult["profileContext"]>;
}) {
  return (
    <section className="memory-panel">
      <h3>Memory used in this estimate</h3>
      <p className="subtle">
        This is the read-only learner context included in the M4 prompt.
      </p>
      <div className="metric-grid">
        <Metric label="Experience level" value={context.experienceLevel} />
        <Metric label="Learning style" value={context.learningStyle} />
        <Metric
          label="Topic skill"
          value={
            context.matchingTopic && context.matchingSkill !== null
              ? `${context.matchingTopic} · ${(context.matchingSkill * 100).toFixed(0)}%`
              : "No matching saved skill"
          }
        />
        <Metric
          label="Profile snapshot"
          value={`M3 local · ${context.profileVersion}`}
        />
      </div>
      <div className="memory-list">
        <div className="eyebrow">Relevant task history sent to M4</div>
        {context.taskHistory.length ? (
          <ul>
            {context.taskHistory.map((task) => (
              <li key={task.taskId}>
                <strong>{task.name}</strong> · {task.subject} · estimated{" "}
                {task.estimatedDurationMinutes} min, actual{" "}
                {task.actualDurationMinutes} min · reported difficulty{" "}
                {task.reportedDifficulty}/10
              </li>
            ))}
          </ul>
        ) : (
          <p className="subtle">
            No matching history yet; no past-task memory was sent.
          </p>
        )}
      </div>
      <details className="scheduler-context">
        <summary>Profile windows kept for M2</summary>
        <p>These profile fields were not sent to M4.</p>
        <ul>
          <li>
            Peak focus:{" "}
            {context.schedulerWindows.peakFocus.join(", ") || "not set"}
          </li>
          <li>
            Minimum focus:{" "}
            {context.schedulerWindows.minimumFocus.join(", ") || "not set"}
          </li>
          <li>
            Unavailable:{" "}
            {context.schedulerWindows.unavailable.join(", ") || "not set"}
          </li>
        </ul>
      </details>
    </section>
  );
}

function ProfileEditor({
  profile,
  onChange,
}: {
  profile: AdaptiveProfile;
  onChange: (profile: AdaptiveProfile) => void;
}) {
  const [experience, setExperience] = useState(profile.experienceLevel);
  const [style, setStyle] = useState(profile.learningStyle);
  const [peak, setPeak] = useState(profile.peakFocusWindows.join(", "));
  const [minimum, setMinimum] = useState(
    profile.minimumFocusWindows.join(", "),
  );
  const [unavailable, setUnavailable] = useState(
    profile.nonAvailabilityWindows.join(", "),
  );
  const [topic, setTopic] = useState("");
  const [skill, setSkill] = useState("50");
  const [saved, setSaved] = useState(false);

  function savePreferences(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    onChange(
      M3Profile.updatePreferences(profile, {
        experienceLevel: experience,
        learningStyle: style,
        peakFocusWindows: split(peak, 60),
        minimumFocusWindows: split(minimum, 60),
        nonAvailabilityWindows: split(unavailable, 100),
      }),
    );
    setSaved(true);
  }
  function saveSkill(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    onChange(M3Profile.updateTopicSkill(profile, topic, Number(skill) / 100));
    setTopic("");
    setSaved(true);
  }

  return (
    <details className="profile-editor">
      <summary>M3 profile · learner memory and windows</summary>
      <p className="subtle">
        Stored in this browser only. M3 is the sole profile writer.
      </p>
      <form className="profile-form" onSubmit={savePreferences}>
        <label>
          Experience level
          <select
            value={experience}
            onChange={(event) =>
              setExperience(
                event.target.value as AdaptiveProfile["experienceLevel"],
              )
            }
          >
            <option value="unknown">Unknown</option>
            <option value="beginner">Beginner</option>
            <option value="intermediate">Intermediate</option>
            <option value="advanced">Advanced</option>
          </select>
        </label>
        <label>
          Learning style
          <select
            value={style}
            onChange={(event) =>
              setStyle(event.target.value as AdaptiveProfile["learningStyle"])
            }
          >
            <option value="unknown">Unknown</option>
            <option value="visual">Visual</option>
            <option value="reading">Reading</option>
            <option value="practice">Practice</option>
            <option value="balanced">Balanced</option>
          </select>
        </label>
        <label>
          Peak focus windows
          <input
            value={peak}
            onChange={(event) => setPeak(event.target.value)}
            placeholder="weekdays 09:00–11:00"
          />
        </label>
        <label>
          Minimum focus windows
          <input
            value={minimum}
            onChange={(event) => setMinimum(event.target.value)}
            placeholder="after 20:00"
          />
        </label>
        <label>
          Non-availability windows
          <input
            value={unavailable}
            onChange={(event) => setUnavailable(event.target.value)}
            placeholder="weekdays 10:00–16:00"
          />
        </label>
        <button type="submit" className="button">
          Save profile fields
        </button>
      </form>
      <form className="skill-form" onSubmit={saveSkill}>
        <label>
          Topic skill
          <input
            value={topic}
            onChange={(event) => setTopic(event.target.value)}
            maxLength={100}
            placeholder="e.g. SQL joins"
            required
          />
        </label>
        <label>
          Self-reported skill · 0–100%
          <input
            type="number"
            min={0}
            max={100}
            value={skill}
            onChange={(event) => setSkill(event.target.value)}
          />
        </label>
        <button type="submit" className="button button-secondary">
          Save topic skill
        </button>
      </form>
      <p className="subtle">
        {saved
          ? `Saved as profile version ${profile.version}.`
          : `Profile version ${profile.version}.`}
      </p>
    </details>
  );
}

function TaskCard({
  task,
  record,
  recorded,
}: {
  task: EstimatedTask;
  record: (task: EstimatedTask, minutes: number, difficulty: number) => void;
  recorded: boolean;
}) {
  const [minutes, setMinutes] = useState(String(task.durationMinutes));
  const [difficulty, setDifficulty] = useState("5");
  const dependencies = task.prerequisiteIndexes.length
    ? task.prerequisiteIndexes.map((index) => `Task ${index}`).join(", ")
    : "None";
  return (
    <article className="research-card">
      <div className="task-title-row">
        <span className="task-number">{task.sequence}</span>
        <div>
          <h3>{task.name}</h3>
          <div className="eyebrow">
            {task.subject} · {task.type.replaceAll("_", " ")}
          </div>
        </div>
      </div>
      <p className="task-description">{task.description}</p>
      <div className="metric-grid">
        <Metric label="Difficulty · 1–10" value={`${task.difficulty} / 10`} />
        <Metric
          label="Active study time"
          value={`${task.durationMinutes} min`}
        />
        <Metric
          label="Load proxy · axioms"
          value={task.cognitiveLoadProxy.toFixed(2)}
        />
        <Metric label="Priority" value={task.priority} />
        <Metric label="Confidence" value={task.confidence} />
        <Metric label="Prerequisites" value={dependencies} />
      </div>
      <p className="task-rationale">
        <strong>Estimate:</strong> {task.rationale}
      </p>
      <form
        className="outcome-form"
        onSubmit={(event) => {
          event.preventDefault();
          record(task, Number(minutes), Number(difficulty));
        }}
      >
        <span className="eyebrow">M3 · Record task outcome</span>
        <label>
          Actual minutes
          <input
            type="number"
            min={1}
            max={1440}
            value={minutes}
            onChange={(event) => setMinutes(event.target.value)}
            required
          />
        </label>
        <label>
          Reported difficulty
          <select
            value={difficulty}
            onChange={(event) => setDifficulty(event.target.value)}
          >
            {Array.from({ length: 10 }, (_, i) => i + 1).map((level) => (
              <option key={level} value={level}>
                {level} / 10
              </option>
            ))}
          </select>
        </label>
        <button
          className="button button-secondary"
          type="submit"
          disabled={recorded}
        >
          {recorded ? "Recorded in profile" : "Save outcome to memory"}
        </button>
      </form>
    </article>
  );
}

function WeeklyReport({
  profile,
  baseline,
}: {
  profile: AdaptiveProfile;
  baseline: AdaptiveProfile;
}) {
  const weekAgo = Date.now() - 7 * 24 * 60 * 60 * 1000;
  const recent = profile.taskHistory.filter(
    (item) => Date.parse(item.recordedAt) >= weekAgo,
  );
  const averageMinutes = recent.length
    ? Math.round(
        recent.reduce((sum, item) => sum + item.actualDurationMinutes, 0) /
          recent.length,
      )
    : null;
  const averageDifficulty = recent.length
    ? (
        recent.reduce((sum, item) => sum + item.reportedDifficulty, 0) /
        recent.length
      ).toFixed(1)
    : null;
  return (
    <section className="weekly-report">
      <div className="eyebrow">M3 · last 7 days</div>
      <h3>Behaviour report</h3>
      <p>
        {profile.version === baseline.version
          ? `Profile remains at its initial version ${baseline.version}.`
          : `Profile changed from initial version ${baseline.version} to version ${profile.version}.`}
      </p>
      <div className="metric-grid">
        <Metric label="Tasks observed" value={String(recent.length)} />
        <Metric
          label="Average actual time"
          value={
            averageMinutes === null
              ? "No observations"
              : `${averageMinutes} min`
          }
        />
        <Metric
          label="Reported difficulty"
          value={
            averageDifficulty === null
              ? "No observations"
              : `${averageDifficulty} / 10`
          }
        />
      </div>
      <p className="subtle">
        The fixed-interval versus adaptive outcome comparison is pending M2's
        scheduler. This report summarizes browser-local observations and makes
        no baseline improvement claim.
      </p>
    </section>
  );
}

export default function TaskEstimatorClient() {
  const [description, setDescription] = useState("");
  const [profile, setProfile] = useState<AdaptiveProfile | null>(null);
  const [baseline, setBaseline] = useState<AdaptiveProfile | null>(null);
  const [result, setResult] = useState<TaskPlanResult | null>(null);
  const [error, setError] = useState("");
  const [recorded, setRecorded] = useState<Set<string>>(new Set());
  const [pending, startTransition] = useTransition();

  useEffect(() => {
    const current = M3Profile.read();
    setProfile(current);
    setBaseline(M3Profile.baseline());
  }, []);

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!profile) return;
    setError("");
    setResult(null);
    setRecorded(new Set());
    startTransition(async () => {
      const response = await estimateLearningTasks(description, profile);
      setResult(response);
      if (response.error) setError(response.error);
    });
  }

  function recordOutcome(
    task: EstimatedTask,
    minutes: number,
    difficulty: number,
  ) {
    if (!profile || !Number.isFinite(minutes) || minutes < 1) return;
    const updated = M3Profile.recordTaskOutcome(
      profile,
      {
        taskId: task.id,
        name: task.name,
        subject: task.subject,
        type: task.type,
        estimatedDifficulty: task.difficulty,
        estimatedDurationMinutes: task.durationMinutes,
        cognitiveLoadProxy: task.cognitiveLoadProxy,
      },
      minutes,
      difficulty,
    );
    setProfile(updated);
    setRecorded((current) => new Set(current).add(task.id));
  }

  if (!profile || !baseline)
    return (
      <main className="research-shell">Loading local research profile…</main>
    );
  const tasks = result?.tasks ?? [];
  const totalMinutes = tasks.reduce(
    (sum, task) => sum + task.durationMinutes,
    0,
  );

  return (
    <main className="research-shell">
      <header className="research-header">
        <a className="brand" href="/">
          Axiom
        </a>
        <span className="eyebrow">Adaptive scheduler · learning study</span>
        <span className="local-badge">No account · local profile</span>
      </header>
      <section className="intro">
        <div className="eyebrow">M4 · task-variable estimation</div>
        <h1>Turn a learning goal into study tasks.</h1>
        <p>
          Describe what you want to learn and what you already know. Get an
          ordered task list with estimated difficulty, active study time,
          priority, and a provisional load score.
        </p>
      </section>
      <ProfileEditor profile={profile} onChange={setProfile} />
      <form className="goal-form" onSubmit={submit}>
        <label htmlFor="learning-goal" className="eyebrow">
          Learning goal
        </label>
        <textarea
          id="learning-goal"
          value={description}
          onChange={(event) => setDescription(event.target.value)}
          placeholder="For example: learn SQL joins, explain the difference between INNER and LEFT JOIN, then solve practice questions. I have used basic SELECT queries before."
          minLength={12}
          maxLength={3000}
          required
        />
        <div className="form-row">
          <button
            className="text-button"
            type="button"
            onClick={() => setDescription(example)}
          >
            Try an example
          </button>
          <span className="eyebrow">{description.length} / 3000</span>
        </div>
        <button
          className="button"
          type="submit"
          disabled={pending || description.trim().length < 12}
        >
          {pending ? "Estimating…" : "Build my task list →"}
        </button>
      </form>
      {error && (
        <p className="error-message" role="alert">
          {error}
        </p>
      )}
      {result?.tasks && (
        <>
          <section className="results-heading" aria-live="polite">
            <div>
              <div className="eyebrow">
                {result.source === "llm" ? "LLM estimate" : "M4 fallback stub"}
              </div>
              <h2>Study task breakdown</h2>
            </div>
            <div className="eyebrow">
              {tasks.length} tasks · {Math.floor(totalMinutes / 60)}h{" "}
              {totalMinutes % 60}m
            </div>
          </section>
          <aside className="provisional-notice">
            {result.notice} Load uses{" "}
            <code>{tasks[0]?.cognitiveLoadFormula}</code>, a provisional M1
            formula stub.
          </aside>
          {result.profileContext && (
            <MemoryContext context={result.profileContext} />
          )}
          <section className="task-list">
            {tasks.map((task) => (
              <TaskCard
                key={task.id}
                task={task}
                record={recordOutcome}
                recorded={recorded.has(task.id)}
              />
            ))}
          </section>
          <section className="baseline-preview">
            <div className="eyebrow">M2 · fixed-interval baseline stub</div>
            <strong>
              {FIXED_INTERVAL_BASELINE.focusMinutes} min focus /{" "}
              {FIXED_INTERVAL_BASELINE.breakMinutes} min break
            </strong>
            <p>
              About {fixedIntervalSessionCount(totalMinutes)} fixed-interval
              sessions for the task list. M2's optimal adaptive scheduler is
              still to be implemented.
            </p>
          </section>
        </>
      )}
      <WeeklyReport profile={profile} baseline={baseline} />
      <footer className="research-footer">
        M3 writes the local profile · M4 reads a snapshot · M1 formula and M2
        adaptive optimizer are stubs
      </footer>
    </main>
  );
}
