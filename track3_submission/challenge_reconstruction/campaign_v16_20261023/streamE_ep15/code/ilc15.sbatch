#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --job-name=E-ilc15
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamE_ep15/logs/ilc15-%A_%a.log
# usage: SRC=TEACHER_DIR TAG=name sbatch --array=0-7 ilc15.sbatch
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; W=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamE_ep15
G=("0.2 0.2 0.35 2.0" "0.3 0.3 0.35 2.0" "0.3 0.0 0.35 2.0" "0.5 0.5 0.35 2.0" "0.3 0.3 0.6 2.0" "0.5 0.5 0.6 2.0" "0.3 0.3 0.35 4.9" "0.6 0.6 0.35 4.9"); set -- ${G[$SLURM_ARRAY_TASK_ID]}
cd /tmp && $R/.venv/bin/python -B -u $R/refinement_20261021_ep38/code/ilc_carry_generic.py --source $SRC --out $W/ilc/${TAG}/gr$1_gp$2_s$3_t$4 --side right --object wooden_spoon --t0 $4 --t1 8.2 --t-clear 9.0 --sigma-s $3 --episode-dir /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/physics_collision_v2/episodes/episode_000015 --iterations 3 --gain-rot $1 --gain-pos $2
