#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --job-name=J8-19stage
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamJ8_19_25_28_38/ep19/logs/stage-%j.log
# usage: sbatch stage.sbatch DISTILLED_ACTOR_DIR BASE TEACHER WORLD_DIR [extra lineage files...]
set -euo pipefail
source /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamJ8_19_25_28_38/ep19/code/env.sh
A=$1; B=$2; T=$3; W=$4; shift 4; X=""; for f in "$@"; do X="$X --lineage-file $f"; done
cd /tmp && $PY -B -u $J/code/stage_ep19J.py --episode 19 --episode-dir $W/episode_target/episode_000019 --distilled $A --base $B --teacher $T \
  --teacher-note "open-loop servo reference teacher: stream F F_ep19_twin3 teacher (C_ep19_ilc lineage, twin port) re-ported / knot-CEM refined by J8 in $W; teacher only, not shipped" \
  --lineage-file $J/code/port19J.py --lineage-file $J/code/build_world19J.py --lineage-file $J/code/mk_alpha.py --lineage-file $J/code/topple.py --lineage-file $J/code/kcem19.py \
  --lineage-file $J/sweeper_alpha.npz --lineage-file $J/sweeper_P.npz --lineage-file $F/code/common19F.py --lineage-file $F/twin_poses.npz --lineage-file $W/scene/scene_change.json \
  --lineage-file /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamC_rigid_ilc/tools/actor_grid_c.py $X
