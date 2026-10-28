"use client";

import Image from "next/image";
import { useCallback, useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  defaultConfig,
  humanize,
  type RunConfig,
  type RunDetail,
  type RunListItem,
  samplerLabels,
  samplers,
  scenarios,
  variants,
} from "@/lib/bayesian/workspace";
import s from "./workspace.module.css";

type View =
  | "Results"
  | "Diagnostics"
  | "Recovery"
  | "Plots"
  | "Report"
  | "Artifacts";
const views: View[] = [
  "Results",
  "Diagnostics",
  "Recovery",
  "Plots",
  "Report",
  "Artifacts",
];
const number = (v: number | null | undefined, digits = 3) =>
  v == null || !Number.isFinite(v) ? "—" : v.toFixed(digits);
const active = (state?: string) => state === "queued" || state === "running";
const example = [
  {
    user: 0,
    item: 0,
    r: [1, 0, 1, 1, 0, 1, 0, 1],
    dt: [1, 0.5, 1, 2, 3, 1, 4, 7],
  },
  {
    user: 0,
    item: 1,
    r: [0, 1, 0, 1, 1, 1, 0, 0],
    dt: [1, 1, 2, 1, 3, 4, 7, 14],
  },
];

export default function ResearchWorkspace() {
  const [config, setConfig] = useState<RunConfig>(defaultConfig);
  const [runs, setRuns] = useState<RunListItem[]>([]);
  const [runId, setRunId] = useState("");
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [view, setView] = useState<View>("Results");
  const [error, setError] = useState("");
  const [launching, setLaunching] = useState(false);
  const [environment, setEnvironment] = useState<{
    ready: boolean;
    versions?: Record<string, string>;
    error?: string;
  } | null>(null);
  const [uploadName, setUploadName] = useState("");
  const [filter, setFilter] = useState("");
  const [variantFilter, setVariantFilter] = useState("all");
  const [onlyFailed, setOnlyFailed] = useState(false);
  const [matchedOnly, setMatchedOnly] = useState(true);
  const [plot, setPlot] = useState("comparison.png");
  const [report, setReport] = useState("");
  const [connectionError, setConnectionError] = useState("");
  const artifact = useCallback(
    (name: string) =>
      `/api/research-runs/${runId}/artifact/${encodeURIComponent(name)}`,
    [runId],
  );

  useEffect(() => {
    fetch("/api/research-runs/environment")
      .then((r) => r.json())
      .then(setEnvironment)
      .catch(() =>
        setEnvironment({
          ready: false,
          error: "Could not check Python environment.",
        }),
      );
    fetch("/api/research-runs")
      .then((r) => r.json())
      .then((list: RunListItem[]) => {
        setRuns(list);
        const saved = localStorage.getItem("axiom-research-run");
        setRunId(
          list.some((r) => r.id === saved) ? saved || "" : list[0]?.id || "",
        );
      })
      .catch(() =>
        setConnectionError(
          "Could not load local runs. Check the Next.js server.",
        ),
      );
  }, []);

  useEffect(() => {
    if (!runId) return;
    localStorage.setItem("axiom-research-run", runId);
    setDetail(null);
    setReport("");
    setPlot("comparison.png");
    let disposed = false;
    let timer: ReturnType<typeof setTimeout>;
    async function refresh() {
      try {
        const response = await fetch(`/api/research-runs/${runId}`);
        if (!response.ok) throw new Error("Could not read this run.");
        const result: RunDetail = await response.json();
        if (disposed) return;
        setDetail(result);
        setConnectionError("");
        const list: RunListItem[] = await fetch("/api/research-runs").then(
          (r) => r.json(),
        );
        if (disposed) return;
        setRuns(list);
        if (
          active(result.status.state) ||
          list.some((r) => active(r.status.state))
        )
          timer = setTimeout(refresh, 3000);
      } catch (e) {
        if (!disposed) {
          setConnectionError(
            e instanceof Error ? e.message : "Connection failed.",
          );
          timer = setTimeout(refresh, 5000);
        }
      }
    }
    refresh();
    return () => {
      disposed = true;
      clearTimeout(timer);
    };
  }, [runId]);

  useEffect(() => {
    if (view !== "Report" || !detail?.artifacts.includes("report.md")) return;
    let cancelled = false;
    fetch(artifact("report.md"))
      .then((r) => {
        if (!r.ok) throw new Error("Report unavailable");
        return r.text();
      })
      .then((text) => {
        if (!cancelled) setReport(text);
      })
      .catch(() => {
        if (!cancelled)
          setReport("Could not load report. Try the artifact download.");
      });
    return () => {
      cancelled = true;
    };
  }, [view, detail?.artifacts, artifact]);

  function setting(key: keyof RunConfig, value: unknown) {
    setConfig((c) => ({ ...c, [key]: value }));
  }
  function toggle(key: "variants" | "scenarios", value: string) {
    setConfig((c) => ({
      ...c,
      [key]: c[key].includes(value)
        ? c[key].filter((v) => v !== value)
        : [...c[key], value],
    }));
  }
  async function launch() {
    setLaunching(true);
    setError("");
    try {
      const response = await fetch("/api/research-runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error);
      setRunId(result.id);
      setView("Results");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Launch failed.");
    } finally {
      setLaunching(false);
    }
  }
  async function cancel() {
    try {
      const response = await fetch(`/api/research-runs/${runId}`, {
        method: "DELETE",
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error);
      setError(result.message);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Cancellation failed.");
    }
  }
  const summary = detail?.summary;
  const busy =
    launching ||
    runs.some((r) => active(r.status.state)) ||
    active(detail?.status.state);
  const diagnostics =
    summary?.diagnostics.filter(
      (d) =>
        (variantFilter === "all" || d.variant === variantFilter) &&
        d.parameter.toLowerCase().includes(filter.toLowerCase()) &&
        (!onlyFailed || !d.passed),
    ) || [];
  const recovery =
    summary?.recovery?.filter(
      (r) =>
        (variantFilter === "all" || r.variant === variantFilter) &&
        r.parameter.toLowerCase().includes(filter.toLowerCase()) &&
        (!matchedOnly || r.matched),
    ) || [];
  const plots = detail?.artifacts.filter((name) => name.endsWith(".png")) || [];
  const selectedPlot = plots.includes(plot) ? plot : plots[0];
  const numericField = (
    key: keyof RunConfig,
    label: string,
    min: number,
    max: number,
    step = 1,
  ) => (
    <label className={s.field} key={key}>
      {label}
      <input
        type="number"
        min={min}
        max={max}
        step={step}
        value={config[key] as number}
        onChange={(e) => setting(key, e.target.valueAsNumber)}
      />
    </label>
  );
  const filterBar = (
    <div className={s.filters}>
      <label>
        Variant
        <select
          value={variantFilter}
          onChange={(e) => setVariantFilter(e.target.value)}
        >
          <option value="all">All variants</option>
          {variants.map((v) => (
            <option key={v} value={v}>
              {humanize(v)}
            </option>
          ))}
        </select>
      </label>
      <label>
        Parameter
        <input
          type="search"
          placeholder="e.g. lam, alpha, sigma2"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
        />
      </label>
      {view === "Diagnostics" ? (
        <label className={s.check}>
          <input
            type="checkbox"
            checked={onlyFailed}
            onChange={(e) => setOnlyFailed(e.target.checked)}
          />
          Failed gates only
        </label>
      ) : (
        <label className={s.check}>
          <input
            type="checkbox"
            checked={matchedOnly}
            onChange={(e) => setMatchedOnly(e.target.checked)}
          />
          Matched generating model only
        </label>
      )}
    </div>
  );

  return (
    <main className={s.workspace}>
      <header className={s.header}>
        <div>
          <h1>
            Axiom <span>/ Bayesian research</span>
          </h1>
          <p>Dynamic retention · diagnostics, ablations and recovery</p>
        </div>
        <div className={s.environment}>
          <span className={environment?.ready ? s.ready : s.warning}>
            {environment
              ? environment.ready
                ? "Python ready"
                : "Environment unavailable"
              : "Checking environment…"}
          </span>
          {environment?.versions && (
            <small>
              {Object.entries(environment.versions)
                .map(([k, v]) => `${k} ${v}`)
                .join(" · ")}
            </small>
          )}
        </div>
      </header>
      <div className={s.layout}>
        <aside className={s.settings}>
          <h2>Experiment configuration</h2>
          <label className={s.field}>
            Data source
            <select
              value={config.source}
              onChange={(e) =>
                setConfig((c) => ({
                  ...c,
                  source: e.target.value as RunConfig["source"],
                  scenarios:
                    e.target.value === "anki"
                      ? ["anki"]
                      : c.scenarios.filter((v) => v !== "anki").length
                        ? c.scenarios.filter((v) => v !== "anki")
                        : ["full"],
                }))
              }
            >
              <option value="synthetic">Synthetic · known ground truth</option>
              <option value="upload">Uploaded review sequences</option>
              <option value="anki">Local Anki Parquet data</option>
            </select>
          </label>
          {config.source === "synthetic" ? (
            <>
              <fieldset>
                <legend>Generating conditions</legend>
                {scenarios.map((v) => (
                  <label className={s.check} key={v}>
                    <input
                      type="checkbox"
                      checked={config.scenarios.includes(v)}
                      onChange={() => toggle("scenarios", v)}
                    />
                    {humanize(v)}
                  </label>
                ))}
              </fieldset>
              <div className={s.inputGrid}>
                {numericField("users", "Users", 2, 8)}
                {numericField("items", "Items per user", 2, 10)}
                {numericField("events", "Reviews per pair", 8, 60)}
                {numericField("replicates", "Dataset replicates", 1, 3)}
              </div>
            </>
          ) : config.source === "anki" ? (
            <div>
              <p className={s.note}>
                Reads the local AnkiFirstRun train/test Parquet files. Keeps
                their temporal partition and recorded elapsed seconds; no
                recovery truth. Fixed disjoint user cohorts select eligible
                observed card histories, without selecting on recall outcomes.
                This is a cohort pilot, not a full-dataset benchmark. Settings
                may exceed available eligible users.
              </p>
              <div className={s.inputGrid}>
                {numericField("users", "Users per cohort", 2, 8)}
                {numericField("items", "Cards per user", 2, 10)}
                {numericField("events", "Maximum history length", 8, 60)}
                {numericField("replicates", "Disjoint cohorts", 1, 3)}
              </div>
              <p className={s.note}>
                Requires at least six training and two test events per card.
                Histories that fail adjacent-gap checks or have overlapping
                partition dates are excluded and counted in selection artifacts.
                Same-day outcome order and source provenance remain unverified.
                These checks do not establish complete lifetime histories.
              </p>
            </div>
          ) : (
            <div className={s.upload}>
              <label className={s.field}>
                Review dataset (.json)
                <input
                  type="file"
                  accept=".json,application/json"
                  onChange={async (e) => {
                    const file = e.target.files?.[0];
                    if (!file) return;
                    setError("");
                    try {
                      if (file.size > 200000)
                        throw new Error("Maximum file size is 200 KB.");
                      const sequences = JSON.parse(await file.text());
                      setting("sequences", sequences);
                      setUploadName(file.name);
                    } catch (err) {
                      setError(
                        err instanceof Error ? err.message : "Invalid JSON.",
                      );
                      setting("sequences", undefined);
                      setUploadName("");
                    }
                  }}
                />
              </label>
              <p>
                {uploadName ||
                  (config.sequences
                    ? "Dataset loaded from saved run"
                    : "No dataset selected")}
              </p>
              <p>
                One record per user/item pair. Ordered binary outcomes{" "}
                <code>r</code> and equal-length gaps <code>dt</code> in days.
                8–100 events per pair; 2–120 pairs.
              </p>
              <a
                download="review-sequences-example.json"
                href={`data:application/json,${encodeURIComponent(JSON.stringify(example, null, 2))}`}
              >
                Download format example
              </a>
              <p>Recovery is unavailable for uploaded data.</p>
            </div>
          )}
          <fieldset>
            <legend>Fitted variants</legend>
            {variants.map((v) => (
              <label className={s.check} key={v}>
                <input
                  type="checkbox"
                  checked={config.variants.includes(v)}
                  disabled={v === "full"}
                  onChange={() => toggle("variants", v)}
                />
                {humanize(v)}
                {v === "full" ? " (reference)" : ""}
              </label>
            ))}
          </fieldset>
          <fieldset>
            <legend>Sampler</legend>
            <p className={s.note}>
              Four independent chains. Exact PG draws, FFBS and collapsed
              Metropolis. No thinning.
            </p>
            <label>
              Transition kernel
              <select
                value={config.sampler}
                onChange={(e) =>
                  setConfig((v) => ({
                    ...v,
                    sampler: e.target.value as RunConfig["sampler"],
                  }))
                }
              >
                {samplers.map((value) => (
                  <option key={value} value={value}>
                    {samplerLabels[value]}
                  </option>
                ))}
              </select>
            </label>
            <p className={s.note}>
              All kernels preserve the model and priors. Joint decay adapts its
              covariance and scale during warmup only; it applies when practice
              decay is fitted. Compare separate runs with identical data, seeds
              and iteration budgets. Improvements are hypotheses, not
              guarantees.
            </p>
            <div className={s.inputGrid}>
              {numericField("mh_steps", "MH proposals per sweep", 1, 20)}
              {numericField("iterations", "Total iterations", 200, 8000, 100)}
              {numericField("warmup", "Warmup", 50, 6000, 50)}
              {numericField("seed", "Seed", 0, 2147483647)}
              {numericField("workers", "Parallel workers", 1, 4)}
              {config.source !== "anki" &&
                numericField("holdout", "Held-out fraction", 0.1, 0.4, 0.05)}
            </div>
          </fieldset>
          <div className={s.launch}>
            <button
              type="button"
              className={s.primary}
              disabled={busy || !environment?.ready}
              onClick={launch}
            >
              {launching ? "Starting…" : "Run experiment"}
            </button>
            <p>
              {config.variants.length *
                4 *
                (config.source === "upload"
                  ? 1
                  : config.source === "anki"
                    ? config.replicates
                    : config.scenarios.length * config.replicates)}{" "}
              chains · {Math.max(0, config.iterations - config.warmup)} retained
              draws per chain
            </p>
          </div>
          {environment?.error && <p className={s.error}>{environment.error}</p>}
          <details className={s.protocol}>
            <summary>Model and protocol</summary>
            <p>
              Time-aware OU model from AnkiFirstRun. Full model estimates
              practice-dependent decay and hierarchical effects. Ablations keep
              training/test splits and shared priors fixed.
            </p>
            <p>
              Diagnostics: rank-normalized split R̂ &lt; 1.01, bulk and tail ESS
              ≥ 400 for every monitored parameter. A completed run can fail
              these checks.
            </p>
            <p>
              Prediction uses chronological held-out suffixes and updates with
              outcomes only after predicting them. Dynamic filtering is
              approximate.
            </p>
            <a href="/bayesian-demo/recall">Open recall-loop demo</a>
          </details>
        </aside>
        <section className={s.content} aria-label="Experiment results">
          <div className={s.runBar}>
            <label>
              Local runs
              <select
                aria-label="Local runs"
                value={runId}
                onChange={(e) => setRunId(e.target.value)}
              >
                <option value="">Select a run</option>
                {runs.map((r) => (
                  <option key={r.id} value={r.id}>
                    {r.id === "task3-study"
                      ? "Task 3 · reference study"
                      : `${r.id.startsWith("task3-") ? r.id : r.id.slice(0, 8)} · ${new Date(r.created).toLocaleString()}`}{" "}
                    · {r.status.state}
                  </option>
                ))}
              </select>
            </label>
            {detail?.config && (
              <button
                type="button"
                onClick={() => {
                  if (!detail.config) return;
                  setConfig({
                    ...defaultConfig,
                    ...detail.config,
                    scenarios:
                      detail.config.source === "upload"
                        ? ["full"]
                        : detail.config.scenarios,
                  });
                  setUploadName("");
                }}
              >
                Load settings
              </button>
            )}
            {active(detail?.status.state) && (
              <button type="button" onClick={cancel}>
                Cancel run
              </button>
            )}
          </div>
          {(error || connectionError) && (
            <div className={s.error} role="alert">
              {error || connectionError}
              <button
                type="button"
                onClick={() => {
                  setError("");
                  setConnectionError("");
                }}
                aria-label="Dismiss message"
              >
                ×
              </button>
            </div>
          )}
          {!runId ? (
            <div className={s.empty}>
              <h2>No experiment selected</h2>
              <p>
                Choose local Anki data, configure a synthetic experiment or
                upload review data, then run the sampler.
              </p>
              <p>
                Each run saves posterior draws, convergence diagnostics,
                predictive comparisons and a technical report locally.
              </p>
            </div>
          ) : (
            <>
              <div className={s.runStatus}>
                <div>
                  <strong>{detail?.status.state || "Loading"}</strong>
                  <output>
                    {detail?.status.message || "Reading local artifacts…"}
                  </output>
                </div>
                {detail?.status.total && (
                  <progress
                    aria-label="Completed chains"
                    value={detail.status.completed || 0}
                    max={detail.status.total}
                  />
                )}
                <span>
                  {detail?.status.completed ?? 0} /{" "}
                  {detail?.status.total ?? "—"} chains
                </span>
              </div>
              <div className={s.runMeta}>
                <code>{runId}</code>
                {detail?.config && (
                  <span>
                    {detail.config.source} ·{" "}
                    {samplerLabels[detail.config.sampler ?? "reference"]} ·{" "}
                    {detail.config.chains} chains · {detail.config.iterations}{" "}
                    iterations · {detail.config.warmup} warmup · seed{" "}
                    {detail.config.seed}
                  </span>
                )}
              </div>
              {summary?.results.some((r) => !r.passed) && (
                <div className={s.caution}>
                  Convergence unresolved in{" "}
                  {summary.results.filter((r) => !r.passed).length} /{" "}
                  {summary.results.length} fits. Predictive scores and recovery
                  intervals from those fits remain exploratory.
                </div>
              )}
              <nav className={s.tabs} aria-label="Research views">
                {views.map((v) => (
                  <button
                    type="button"
                    aria-pressed={view === v}
                    className={view === v ? s.selectedTab : ""}
                    onClick={() => setView(v)}
                    key={v}
                  >
                    {v}
                  </button>
                ))}
              </nav>
              {view === "Results" && (
                <>
                  <div className={s.sectionTitle}>
                    <h2>Predictive performance and cost</h2>
                    {detail?.artifacts.includes("results.csv") && (
                      <a href={artifact("results.csv")}>Export CSV</a>
                    )}
                  </div>
                  <p className={s.note}>
                    Identical held-out suffixes across variants. Lower log loss
                    and Brier are better. CPU time is summed across chains; AUC
                    is undefined for a single outcome class.
                  </p>
                  {summary?.results.length ? (
                    <div className={s.tableWrap}>
                      <table>
                        <thead>
                          <tr>
                            {[
                              "Scenario / replicate",
                              "Variant",
                              "Log loss",
                              "Brier",
                              "AUC",
                              "Max R̂",
                              "Min bulk ESS",
                              "Min tail ESS",
                              "Fit CPU s",
                              "Gate",
                            ].map((h) => (
                              <th key={h}>{h}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {summary.results.map((r) => (
                            <tr
                              key={`${r.scenario}-${r.replicate}-${r.variant}`}
                            >
                              <td>
                                {humanize(r.scenario)} / {r.replicate + 1}
                              </td>
                              <td>{humanize(r.variant)}</td>
                              <td>{number(r.log_loss, 4)}</td>
                              <td>{number(r.brier, 4)}</td>
                              <td>{number(r.auc)}</td>
                              <td className={r.max_rhat >= 1.01 ? s.fail : ""}>
                                {number(r.max_rhat)}
                              </td>
                              <td>{number(r.min_ess_bulk, 0)}</td>
                              <td>{number(r.min_ess_tail, 0)}</td>
                              <td>{number(r.fit_cpu_seconds, 1)}</td>
                              <td className={r.passed ? s.pass : s.fail}>
                                {r.passed ? "Pass" : "Fail"}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <p className={s.empty}>
                      Results appear as each variant finishes all four chains.
                    </p>
                  )}
                  <div className={s.sectionTitle}>
                    <h2>Component ablations</h2>
                    {detail?.artifacts.includes("ablations.csv") && (
                      <a href={artifact("ablations.csv")}>Export CSV</a>
                    )}
                  </div>
                  <p className={s.note}>
                    Δ log loss = ablated − full. Positive favors full. Paired
                    95% user-cluster bootstrap intervals do not include MCMC
                    uncertainty.
                  </p>
                  {summary?.comparisons?.length ? (
                    <div className={s.tableWrap}>
                      <table>
                        <thead>
                          <tr>
                            <th>Scenario / replicate</th>
                            <th>Ablation</th>
                            <th>Δ log loss</th>
                            <th>95% interval</th>
                            <th>User clusters</th>
                            <th>Both converged</th>
                          </tr>
                        </thead>
                        <tbody>
                          {summary.comparisons.map((r) => (
                            <tr
                              key={`${r.scenario}-${r.replicate}-${r.variant}`}
                            >
                              <td>
                                {humanize(r.scenario)} / {r.replicate + 1}
                              </td>
                              <td>{humanize(r.variant)}</td>
                              <td>{number(r.delta_log_loss, 4)}</td>
                              <td>
                                [{number(r.low, 4)}, {number(r.high, 4)}]
                              </td>
                              <td>{r.clusters}</td>
                              <td className={r.converged ? s.pass : s.fail}>
                                {r.converged ? "Yes" : "No"}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <p className={s.note}>
                      Comparisons become available when the study completes.
                    </p>
                  )}
                </>
              )}
              {view === "Diagnostics" && (
                <>
                  <div className={s.sectionTitle}>
                    <h2>Parameter diagnostics</h2>
                    <a href={artifact("diagnostics.csv")}>Export CSV</a>
                  </div>
                  <p className={s.note}>
                    Gate: rank R̂ &lt; 1.01 and bulk/tail ESS ≥ 400. Includes
                    group effects, scale parameters, terminal states and
                    centered contrasts. Fixed parameters are excluded.
                  </p>
                  {filterBar}
                  <p className={s.note}>{diagnostics.length} parameter rows</p>
                  <div className={s.tableWrap}>
                    <table>
                      <thead>
                        <tr>
                          {[
                            "Scenario / rep",
                            "Variant",
                            "Parameter",
                            "Mean",
                            "95% interval",
                            "R̂",
                            "Bulk ESS",
                            "Tail ESS",
                            "MCSE",
                            "ESS/CPU s",
                            "Gate",
                          ].map((h) => (
                            <th key={h}>{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {diagnostics.map((d) => (
                          <tr
                            key={`${d.scenario}-${d.replicate}-${d.variant}-${d.parameter}`}
                          >
                            <td>
                              {humanize(d.scenario)} / {d.replicate + 1}
                            </td>
                            <td>{humanize(d.variant)}</td>
                            <td>
                              <code>{d.parameter}</code>
                            </td>
                            <td>{number(d.mean)}</td>
                            <td>
                              [{number(d.low)}, {number(d.high)}]
                            </td>
                            <td>{number(d.rhat)}</td>
                            <td>{number(d.ess_bulk, 0)}</td>
                            <td>{number(d.ess_tail, 0)}</td>
                            <td>{number(d.mcse_mean, 4)}</td>
                            <td>{number(d.ess_per_cpu_second, 1)}</td>
                            <td className={d.passed ? s.pass : s.fail}>
                              {d.passed ? "Pass" : "Fail"}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </>
              )}
              {view === "Recovery" && (
                <>
                  <div className={s.sectionTitle}>
                    <h2>Synthetic parameter recovery</h2>
                    {detail?.artifacts.includes("recovery.csv") && (
                      <a href={artifact("recovery.csv")}>Export CSV</a>
                    )}
                  </div>
                  <p className={s.note}>
                    Known truth versus posterior 95% intervals. Few-replicate
                    coverage is descriptive, not calibrated coverage. A covered
                    truth does not override a failed diagnostic gate.
                  </p>
                  {detail?.config?.source !== "synthetic" ? (
                    <p className={s.empty}>
                      Ground truth is unavailable for real observations. Run a
                      synthetic experiment to assess recovery.
                    </p>
                  ) : (
                    <>
                      {filterBar}
                      <p className={s.note}>{recovery.length} parameter rows</p>
                      <div className={s.tableWrap}>
                        <table>
                          <thead>
                            <tr>
                              {[
                                "Scenario / rep",
                                "Variant",
                                "Parameter",
                                "Truth",
                                "Mean",
                                "Error",
                                "95% interval",
                                "Covered",
                                "Gate",
                              ].map((h) => (
                                <th key={h}>{h}</th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {recovery.map((r) => (
                              <tr
                                key={`${r.scenario}-${r.replicate}-${r.variant}-${r.parameter}`}
                              >
                                <td>
                                  {humanize(r.scenario)} / {r.replicate + 1}
                                </td>
                                <td>{humanize(r.variant)}</td>
                                <td>
                                  <code>{r.parameter}</code>
                                </td>
                                <td>{number(r.truth)}</td>
                                <td>{number(r.mean)}</td>
                                <td>{number(r.error)}</td>
                                <td>
                                  [{number(r.low)}, {number(r.high)}]
                                </td>
                                <td>{r.covered ? "Yes" : "No"}</td>
                                <td className={r.passed ? s.pass : s.fail}>
                                  {r.passed ? "Pass" : "Fail"}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </>
                  )}
                </>
              )}
              {view === "Plots" && (
                <>
                  <div className={s.sectionTitle}>
                    <h2>Diagnostic plots</h2>
                  </div>
                  <label className={s.plotSelect}>
                    Figure
                    <select
                      value={selectedPlot || ""}
                      onChange={(e) => setPlot(e.target.value)}
                    >
                      {plots.map((name) => (
                        <option key={name}>{name}</option>
                      ))}
                    </select>
                  </label>
                  {selectedPlot ? (
                    <div className={s.figure}>
                      <a
                        href={artifact(selectedPlot)}
                        target="_blank"
                        rel="noreferrer"
                      >
                        Open full-size figure
                      </a>
                      <Image
                        src={artifact(selectedPlot)}
                        alt={selectedPlot.replaceAll("-", " ")}
                        width={1600}
                        height={1000}
                        unoptimized
                        style={{ width: "100%", height: "auto" }}
                      />
                    </div>
                  ) : (
                    <p className={s.empty}>
                      Figures appear after each fit finishes.
                    </p>
                  )}
                </>
              )}
              {view === "Report" && (
                <>
                  <div className={s.sectionTitle}>
                    <h2>Technical report</h2>
                    {detail?.artifacts.includes("report.md") && (
                      <a href={artifact("report.md")}>Download Markdown</a>
                    )}
                  </div>
                  {report ? (
                    <article className={s.report}>
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        components={{
                          img: ({ src, alt }) =>
                            typeof src === "string" &&
                            /^[\w.-]+\.png$/.test(src) ? (
                              <Image
                                src={artifact(src)}
                                alt={alt || "Diagnostic figure"}
                                width={1600}
                                height={900}
                                unoptimized
                                style={{ width: "100%", height: "auto" }}
                              />
                            ) : null,
                        }}
                      >
                        {report}
                      </ReactMarkdown>
                    </article>
                  ) : (
                    <p className={s.empty}>
                      The report is generated after all variants and recovery
                      checks finish.
                    </p>
                  )}
                </>
              )}
              {view === "Artifacts" && (
                <>
                  <div className={s.sectionTitle}>
                    <h2>Saved run artifacts</h2>
                  </div>
                  <p className={s.note}>
                    Local files include settings, exact data, source hashes,
                    chain seeds, raw posterior/warmup draws and tables. NPZ axes
                    preserve chain × draw × component.
                  </p>
                  <ul className={s.artifacts}>
                    {detail?.artifacts.map((name) => (
                      <li key={name}>
                        <a href={artifact(name)}>{name}</a>
                      </li>
                    ))}
                  </ul>
                </>
              )}
            </>
          )}
        </section>
      </div>
    </main>
  );
}
