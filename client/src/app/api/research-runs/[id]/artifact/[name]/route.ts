import { readFile } from "node:fs/promises";
import path from "node:path";
import { runPath } from "@/lib/bayesian/run-server";

export const runtime = "nodejs";
export async function GET(
  _: Request,
  context: { params: Promise<{ id: string; name: string }> },
) {
  try {
    const { id, name } = await context.params;
    if (!/^[a-zA-Z0-9_.-]+\.(png|csv|md|json|npz|log|zip)$/.test(name))
      return new Response("Invalid artifact name", { status: 400 });
    const data = await readFile(path.join(runPath(id), name));
    const type = name.endsWith(".png")
      ? "image/png"
      : name.endsWith(".json")
        ? "application/json"
        : "application/octet-stream";
    return new Response(data, {
      headers: {
        "Content-Type": type,
        "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff",
        ...(type === "image/png"
          ? {}
          : { "Content-Disposition": `attachment; filename="${name}"` }),
      },
    });
  } catch {
    return new Response("Artifact not found", { status: 404 });
  }
}
