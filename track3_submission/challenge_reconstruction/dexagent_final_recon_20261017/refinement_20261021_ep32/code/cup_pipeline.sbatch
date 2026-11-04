#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=03:00:00
#SBATCH --job-name=cup3233
#SBATCH --array=0-1
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep32/logs/cup-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
EPS=(32 33); TS=(/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/expanded_batch/preserved_retention/episode_000032/seed81 /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/expanded_batch/cup33_retention/seed_72)
E=${EPS[$SLURM_ARRAY_TASK_ID]}; T=${TS[$SLURM_ARRAY_TASK_ID]}; O=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep$E
cd /tmp
[ -d $O/source ] || $PY -u $O/code/build_cup_assets.py $O
$PY -u $O/code/make_cup_collision.py
$PY -u /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep32/code/retarget_teacher_cup.py $T $O/retarget $E
$PY /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep32/code/spoon_in_cup.py $E teacher:$T/execution/trajectory.npz scancup:$O/retarget/execution/trajectory.npz
