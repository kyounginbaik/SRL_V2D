#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --array=0-5%6
#SBATCH --job-name=dustpan38-margins
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/expanded_batch/logs/dustpan-margins-%A_%a.log
set -euo pipefail
cd /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
margin=$(.venv/bin/python - "$SLURM_ARRAY_TASK_ID" <<'PY'
import json,sys
i=int(sys.argv[1]);print(json.dumps({'robot_right_right_pinky_CMC':[4,6,8][i//2],'robot_right_right_index_PIP':[1.5,3][i%2]}))
PY
)
.venv/bin/python code/polish_physical_joint_margins.py \
 --source refinement_20261018/expanded_batch/dustpan_release/episode_000038/case_3 --episode-dir episodes/episode_000038 \
 --output "refinement_20261018/expanded_batch/dustpan_margins/episode_000038/case_${SLURM_ARRAY_TASK_ID}" --margins "$margin"
