import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import {
  mkdir,
  open,
  readdir,
  readFile,
  stat,
  writeFile,
} from "node:fs/promises";
import path from "node:path";
import { z } from "zod";
import type { RunListItem, RunStatus } from "./workspace";

// Python environments and run artifacts are local runtime files, outside the JS bundle.
export const root = path.resolve(
  /* turbopackIgnore: true */ process.cwd(),
  "..",
);
export const runsRoot = path.join(root, ".task3-runs");
export const runSchema = z
  .object({
    source: z.enum(["synthetic", "upload", "anki"]),
    scenarios: z
      .array(
        z.enum([
          "full",
          "no_practice",
          "no_hierarchy",
          "no_dynamics",
          "short_gaps",
          "anki",
        ]),
      )
      .min(1)
      .max(5),
    variants: z
      .array(z.enum(["full", "no_practice", "no_hierarchy", "no_dynamics"]))
      .min(1)
      .max(4),
    replicates: z.number().int().min(1).max(3),
    chains: z.literal(4),
    iterations: z.number().int().min(200).max(8000),
    warmup: z.number().int().min(50).max(6000),
    seed: z.number().int().min(0).max(2147483647),
    users: z.number().int().min(2).max(8),
    items: z.number().int().min(2).max(10),
    events: z.number().int().min(8).max(60),
    holdout: z.number().min(0.1).max(0.4),
    workers: z.number().int().min(1).max(4),
    sampler: z
      .enum(["reference", "joint", "blocked", "joint_blocked"])
      .default("reference"),
    mh_steps: z.number().int().min(1).max(20).default(5),
    sequences: z
      .array(
        z
          .object({
            user: z.number().int().nonnegative(),
            item: z.number().int().nonnegative(),
            r: z
              .array(z.union([z.literal(0), z.literal(1)]))
              .min(8)
              .max(100),
            dt: z.array(z.number().min(0.01).max(365)).min(8).max(100),
          })
          .refine(
            (s) => s.r.length === s.dt.length,
            "r and dt lengths must match",
          ),
      )
      .min(2)
      .max(120)
      .optional(),
  })
  .strict()
  .superRefine((v, ctx) => {
    if (v.iterations - v.warmup < 100)
      ctx.addIssue({
        code: "custom",
        message: "Keep at least 100 retained draws per chain.",
      });
    if (!v.variants.includes("full"))
      ctx.addIssue({
        code: "custom",
        message: "Full model is required as reference.",
      });
    for (const field of ["scenarios", "variants"] as const) {
      if (new Set(v[field]).size !== v[field].length)
        ctx.addIssue({ code: "custom", message: `Duplicate ${field}.` });
    }
    if (v.source === "synthetic" && v.scenarios.includes("anki"))
      ctx.addIssue({
        code: "custom",
        message: "Choose synthetic generating conditions.",
      });
    if (v.source === "upload" && !v.sequences)
      ctx.addIssue({
        code: "custom",
        message: "Upload review sequences first.",
      });
    if (
      v.sequences &&
      new Set(v.sequences.map((s) => `${s.user}/${s.item}`)).size !==
        v.sequences.length
    )
      ctx.addIssue({
        code: "custom",
        message: "Duplicate user/item sequences.",
      });
  });

export async function pythonPath() {
  return process.env.AXIOM_DIAGNOSTICS_PYTHON || "python3";
}
export function runPath(id: string) {
  if (!/^[a-zA-Z0-9_-]{1,64}$/.test(id)) throw new Error("Invalid run ID.");
  return path.join(runsRoot, id);
}
export async function readJson(file: string) {
  try {
    return JSON.parse(await readFile(file, "utf8"));
  } catch {
    return null;
  }
}
export async function listRuns(): Promise<RunListItem[]> {
  await mkdir(runsRoot, { recursive: true });
  const entries = await readdir(runsRoot, { withFileTypes: true });
  const runs = await Promise.all(
    entries
      .filter((e) => e.isDirectory() && /^[a-zA-Z0-9_-]{1,64}$/.test(e.name))
      .map(async (e) => {
        const folder = runPath(e.name);
        const [savedStatus, savedConfig, queuedInput] = await Promise.all([
          readJson(path.join(folder, "status.json")),
          readJson(path.join(folder, "config.json")),
          readJson(path.join(folder, "input.json")),
        ]);
        // CLI investigations and exports use a different artifact schema.
        // Queued browser jobs have input.json before the worker saves config.json.
        if (!savedStatus || (!savedConfig && !queuedInput)) return null;
        let status: RunStatus = savedStatus;
        const pid = await readJson(path.join(folder, "pid.json"));
        if (pid && ["queued", "running"].includes(status.state)) {
          try {
            process.kill(pid, 0);
          } catch {
            status = {
              state: "failed",
              message:
                "Worker stopped before completion. Check worker.log and start a new run.",
            };
          }
        }
        return {
          id: e.name,
          status,
          created: (await stat(folder)).birthtime.toISOString(),
        };
      }),
  );
  return runs
    .filter((run): run is RunListItem => run !== null)
    .sort((a, b) => b.created.localeCompare(a.created));
}

export async function launchRun(config: z.infer<typeof runSchema>) {
  await mkdir(runsRoot, { recursive: true });
  // Atomic launch guard prevents concurrent requests from starting overlapping studies.
  const lock = await open(path.join(runsRoot, ".launch-lock"), "wx");
  try {
    const active = (await listRuns()).find((r) =>
      ["queued", "running"].includes(r.status.state),
    );
    if (active)
      throw new Error(
        `Run ${active.id} is still active. Wait or cancel it first.`,
      );
    const id = randomUUID();
    const folder = runPath(id);
    await mkdir(folder);
    await writeFile(
      path.join(folder, "input.json"),
      JSON.stringify(config, null, 2),
    );
    await writeFile(
      path.join(folder, "status.json"),
      JSON.stringify({ state: "queued", message: "Starting Python worker." }),
    );
    const log = await open(path.join(folder, "worker.log"), "a");
    try {
      const child = spawn(
        await pythonPath(),
        [
          "-m",
          "model_diagnostics.run",
          "--config",
          path.join(folder, "input.json"),
          "--out",
          folder,
        ],
        {
          cwd: root,
          detached: true,
          stdio: ["ignore", log.fd, log.fd],
        },
      );
      await new Promise<void>((resolve, reject) => {
        child.once("spawn", resolve);
        child.once("error", reject);
      });
      await writeFile(path.join(folder, "pid.json"), JSON.stringify(child.pid));
      child.unref();
    } catch (e) {
      await writeFile(
        path.join(folder, "status.json"),
        JSON.stringify({
          state: "failed",
          message: "Could not start Python. Check diagnostics environment.",
        }),
      );
      throw e;
    } finally {
      await log.close();
    }
    return id;
  } finally {
    await lock.close();
    const { unlink } = await import("node:fs/promises");
    await unlink(path.join(runsRoot, ".launch-lock"));
  }
}
