import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { pythonPath, root } from "@/lib/bayesian/run-server";

export const runtime = "nodejs";
export async function GET() {
  try {
    const { stdout } = await promisify(execFile)(
      await pythonPath(),
      [
        "-c",
        "import json,sys,arviz,numpy,polyagamma,matplotlib,scipy,pyarrow; print(json.dumps(dict(python=sys.version.split()[0],arviz=arviz.__version__,numpy=numpy.__version__,polyagamma=polyagamma.__version__,pyarrow=pyarrow.__version__)))",
      ],
      { cwd: root, timeout: 15000 },
    );
    return Response.json({ ready: true, versions: JSON.parse(stdout) });
  } catch {
    return Response.json({
      ready: false,
      error:
        "Diagnostics dependencies unavailable. Install requirements-diagnostics.txt in .venv-diagnostics, or set AXIOM_DIAGNOSTICS_PYTHON.",
    });
  }
}
