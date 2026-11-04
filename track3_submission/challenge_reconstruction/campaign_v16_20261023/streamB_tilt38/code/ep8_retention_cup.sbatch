#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --job-name=sB-ep8-ret
#SBATCH --array=0-1
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamB_tilt38/logs/ep8-ret-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
W=("2.0 10.8 63" "2.5 10.8 64"); set -- ${W[$SLURM_ARRAY_TASK_ID]}
cd /tmp && /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B -u /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code/refine_versioned_retention_full.py --source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamB_tilt38/ep8/carry/teacher_ctrl --episode-dir /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamB_tilt38/ep8/calibrated_inputs/episodes/episode_000008 --output /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamB_tilt38/ep8/retention/cup_right_$1_$2_s$3 --start-s $1 --stop-s $2 --side right --protect-object white_pot --samples 96 --iterations 14 --workers 16 --seed $3
