"use client";

import { useCallback, useEffect, useState } from "react";
import { defaultTopicId } from "@/lib/demo/story";
import styles from "./demo.module.css";
import StepEvidence from "./StepEvidence";
import StepMemory from "./StepMemory";
import StepRoadmap from "./StepRoadmap";
import StepScheduler from "./StepScheduler";
import StepTrends from "./StepTrends";

const STEPS = [
  { n: 1, label: "Trends" },
  { n: 2, label: "Roadmap" },
  { n: 3, label: "Scheduler" },
  { n: 4, label: "Memory engine" },
  { n: 5, label: "Evidence" },
] as const;

const FIRST = 1;
const LAST = STEPS.length;

function clampStep(n: number): number {
  return Math.min(LAST, Math.max(FIRST, n));
}

export default function DemoShell() {
  const [step, setStep] = useState(FIRST);
  const [topicId, setTopicId] = useState(defaultTopicId);
  const [theme, setTheme] = useState<"light" | "dark">("light");

  // Restore the step from the URL hash, and follow back/forward.
  useEffect(() => {
    const readHash = () => {
      const n = Number.parseInt(window.location.hash.slice(1), 10);
      if (Number.isInteger(n)) setStep(clampStep(n));
    };
    readHash();
    window.addEventListener("hashchange", readHash);
    return () => window.removeEventListener("hashchange", readHash);
  }, []);

  // Start from the theme the page already uses (explicit attribute, else OS preference).
  useEffect(() => {
    const explicit = document.documentElement.dataset.theme;
    const dark = explicit
      ? explicit === "dark"
      : window.matchMedia("(prefers-color-scheme: dark)").matches;
    setTheme(dark ? "dark" : "light");
  }, []);

  const goTo = useCallback((n: number) => {
    const next = clampStep(n);
    setStep(next);
    window.history.replaceState(null, "", `#${next}`);
  }, []);

  // Arrow keys move between steps; number keys jump directly.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (
        target &&
        (target.isContentEditable ||
          /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName))
      )
        return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === "ArrowRight") goTo(step + 1);
      else if (e.key === "ArrowLeft") goTo(step - 1);
      else if (/^[1-5]$/.test(e.key)) goTo(Number(e.key));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [step, goTo]);

  const toggleTheme = () => {
    const next = theme === "dark" ? "light" : "dark";
    setTheme(next);
    document.documentElement.dataset.theme = next;
  };

  const current = STEPS[step - 1];

  return (
    <div className={`${styles.root} ${theme === "dark" ? "dark" : ""}`}>
      <header className={styles.topbar}>
        <div className={styles.brand}>Axiom</div>
        <nav className={styles.pills} aria-label="Demo steps">
          {STEPS.map((s) => (
            <button
              key={s.n}
              type="button"
              className={styles.pill}
              aria-current={s.n === step ? "step" : undefined}
              onClick={() => goTo(s.n)}
            >
              {s.n} {s.label}
            </button>
          ))}
        </nav>
        <div className={styles.controls}>
          <span className={styles.meta}>← → to move</span>
          <button
            type="button"
            className={styles.ctrl}
            onClick={() => goTo(step - 1)}
            disabled={step === FIRST}
            aria-label="Previous step"
          >
            ← Prev
          </button>
          <button
            type="button"
            className={styles.ctrl}
            onClick={() => goTo(step + 1)}
            disabled={step === LAST}
            aria-label="Next step"
          >
            Next →
          </button>
          <button
            type="button"
            className={styles.themeBtn}
            onClick={toggleTheme}
            aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
          >
            {theme === "dark" ? "Light" : "Dark"}
          </button>
        </div>
      </header>

      <main key={step} className={styles.stage} aria-label={current.label}>
        {step === 1 && <StepTrends topicId={topicId} onSelect={setTopicId} />}
        {step === 2 && <StepRoadmap topicId={topicId} />}
        {step === 3 && <StepScheduler />}
        {step === 4 && <StepMemory />}
        {step === 5 && <StepEvidence />}
      </main>
    </div>
  );
}
