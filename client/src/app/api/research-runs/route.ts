import { launchRun, listRuns, runSchema } from "@/lib/bayesian/run-server";

export const runtime = "nodejs";
export async function GET() {
  return Response.json(await listRuns(), {
    headers: { "Cache-Control": "no-store" },
  });
}
export async function POST(request: Request) {
  try {
    const body = await request.text();
    if (body.length > 200000)
      return Response.json(
        { error: "Dataset exceeds 200 KB." },
        { status: 413 },
      );
    const result = runSchema.safeParse(JSON.parse(body));
    if (!result.success)
      return Response.json(
        { error: result.error.issues.map((i) => i.message).join(" ") },
        { status: 400 },
      );
    const id = await launchRun(result.data);
    return Response.json({ id }, { status: 202 });
  } catch (e) {
    return Response.json(
      { error: e instanceof Error ? e.message : "Could not launch study." },
      { status: 409 },
    );
  }
}
