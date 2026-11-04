#!/bin/bash
# Host-side supervisor: keeps the v2d_jisu container and the batch workers alive (workers are idempotent).
cd /ndata/jisu/NVIDIA_CHALLENGE_claude
declare -A LISTS=( [0]="2 23 40 1 10 19 28 35" [1]="24 31 42 3 14 20 30 36" [2]="13 26 44 5 15 25 32 37" [3]="12 27 0 6 16 33 38 8" [6]="18 41 43 45 11 21 22 39" )
while true; do
  if [ "$(docker inspect -f '{{.State.Running}}' v2d_jisu 2>/dev/null)" != "true" ]; then echo "$(date) container down -> start"; docker start v2d_jisu; sleep 8; fi
  for g in 0 1 2 3 6; do
    name=s$g
    [ -f outputs/batch/done_$name ] && continue
    if ! docker exec v2d_jisu pgrep -f "name $name --episodes" >/dev/null 2>&1; then
      docker exec v2d_jisu bash -c "cd /jisu/NVIDIA_CHALLENGE_claude && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && (nohup envs/v2d/bin/python -u compose_v2d/run_all.py --gpu $g --name $name --episodes ${LISTS[$g]} >> logs/batch/worker_$name.log 2>&1 && touch outputs/batch/done_$name) > /dev/null 2>&1 &"
      echo "$(date) launched $name on GPU $g"
    fi
  done
  sleep 300
done
