"""Task E step 1: validation predictions of the original model (v1 = OursModel, numba backend).

  python -m fsrs_benchmark.experiments.v1_validation --dataset anki [--iters 2000 --burn 1000]
Fits on the fit part (is_train of make_validation_split), predicts the validation scored rows,
saves results/phase6_validation/pred_v1_<dataset>.parquet (event_id, p). No test rows exist in
the validation table.
"""
import argparse
import time

from fsrs_benchmark.experiments.prepare_data import build
from fsrs_benchmark.experiments.validation import make_validation_split
from fsrs_benchmark.models.ours import OursModel

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True, choices=["anki", "duolingo"])
ap.add_argument("--iters", type=int, default=2000)
ap.add_argument("--burn", type=int, default=1000)
ap.add_argument("--out", default=None)
ap.add_argument("--backend", default="numba", choices=["numba", "gpu"])
ap.add_argument("--threads", type=int, default=0, help="torch threads (gpu backend on cpu)")
ap.add_argument("--progress", type=int, default=50)
a = ap.parse_args()
if a.threads:
    import torch
    torch.set_num_threads(a.threads)
ev = make_validation_split(build(a.dataset))
t = time.perf_counter()
m = OursModel(iters=a.iters, burn=a.burn, thin=4, seed=0, backend=a.backend, device="cpu" if a.backend == "gpu" else None, progress_every=a.progress).fit(ev)
print(f"fit {time.perf_counter() - t:.0f}s", flush=True)
p = m.predict(ev)
out = a.out or f"results/phase6_validation/pred_v1_{a.dataset}.parquet"
p.rename("p").to_frame().to_parquet(out)
print(f"saved {out} ({len(p)} rows), total {time.perf_counter() - t:.0f}s", flush=True)
