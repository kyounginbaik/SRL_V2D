#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --job-name=margins1536
#SBATCH --array=0-6
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261022_margins/logs/search-%A_%a.log
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
set -- $(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261022_margins/runs.txt)
cd /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code
/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B -u /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261022_margins/code/search_finger_margins_gen.py --episode $1 --actor $3 --output /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261022_margins/$2_r2 --samples 96 --iterations 12 --workers 16 --bound 0.25 --seed $((4000+SLURM_ARRAY_TASK_ID))
