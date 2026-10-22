#!/bin/bash
# Wait for the "ours" fit (pid 5746) to exit, then run ours_kappa and merge all five models.
cd /home/amal/fyp/Axiom
while kill -0 5746 2>/dev/null; do sleep 60; done
if [ ! -f results/full_u100_c1000/pred_ours.parquet ]; then
  echo "[chain $(date +%T)] ours fit exited without pred_ours.parquet; not starting ours_kappa" >> results/full_u100_c1000/run.log
  exit 1
fi
echo "[chain $(date +%T)] starting ours_kappa" >> results/full_u100_c1000/run.log
exec systemd-inhibit --what=sleep:idle --why="FYP benchmark fit (kappa)" .venv/bin/python -m fsrs_benchmark.run \
  --n_users 100 --max_cards 1000 --seed 0 --iters 2000 --burn 1000 --workers 8 \
  --run_name full_u100_c1000 --models fsrs_default,fsrs_opt,hlr,ours,ours_kappa --resume \
  >> results/full_u100_c1000/run.log 2>&1
