import { readdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { listRuns, readJson, runPath } from "@/lib/bayesian/run-server";

export const runtime = "nodejs";
export async function GET(
  _: Request,
  context: { params: Promise<{ id: string }> },
) {
  try {
    const { id } = await context.params;
    const folder = runPath(id);
    const item = (await listRuns()).find((r) => r.id === id);
    if (!item)
      return Response.json({ error: "Run not found." }, { status: 404 });
    return Response.json(
      {
        ...item,
        summary:
          (await readJson(path.join(folder, "summary.json"))) ||
          (await readJson(path.join(folder, "partial.json"))),
        config:
          (await readJson(path.join(folder, "config.json"))) ||
          (await readJson(path.join(folder, "input.json"))),
        artifacts: (await readdir(folder))
          .filter((name) => /\.(png|csv|md|json|npz|log|zip)$/.test(name))
          .sort(),
      },
      { headers: { "Cache-Control": "no-store" } },
    );
  } catch {
    return Response.json({ error: "Run not found." }, { status: 404 });
  }
}
export async function DELETE(
  _: Request,
  context: { params: Promise<{ id: string }> },
) {
  try {
    const { id } = await context.params;
    const run = (await listRuns()).find((r) => r.id === id);
    if (!run)
      return Response.json({ error: "Run not found." }, { status: 404 });
    if (!["queued", "running"].includes(run.status.state))
      return Response.json(
        { error: "Run is no longer active." },
        { status: 409 },
      );
    await writeFile(path.join(runPath(id), "CANCEL"), "cancel requested");
    return Response.json({
      message: "Cancellation requested. Workers check every 25 iterations.",
    });
  } catch {
    return Response.json({ error: "Could not cancel run." }, { status: 400 });
  }
}
