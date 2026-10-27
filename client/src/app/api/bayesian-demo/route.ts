import { execFile } from "node:child_process";
import path from "node:path";
import { promisify } from "node:util";
import { z } from "zod";

export const runtime = "nodejs";
const execute = promisify(execFile);
const schema = z
  .array(
    z
      .array(z.union([z.literal(0), z.literal(1)]))
      .min(4)
      .max(40),
  )
  .length(6);

export async function POST(request: Request) {
  let histories: z.infer<typeof schema>;
  try {
    const body = await request.text();
    if (body.length > 4096)
      return Response.json(
        { error: "Review history is too large." },
        { status: 413 },
      );
    histories = schema.parse(JSON.parse(body));
  } catch {
    return Response.json(
      { error: "Expected six binary review histories (4–40 reviews each)." },
      { status: 400 },
    );
  }
  try {
    const { stdout } = await execute(
      process.env.AXIOM_PYTHON || "python3",
      [
        path.resolve(process.cwd(), "../bayesian_demo.py"),
        JSON.stringify(histories),
      ],
      { timeout: 30000, maxBuffer: 256 * 1024 },
    );
    return Response.json(JSON.parse(stdout));
  } catch (error) {
    console.error("Bayesian demo fit failed", error);
    return Response.json(
      {
        error:
          "Model unavailable. Run from client/ with Python 3 and NumPy installed (see BAYESIAN_DEMO.md).",
      },
      { status: 503 },
    );
  }
}
