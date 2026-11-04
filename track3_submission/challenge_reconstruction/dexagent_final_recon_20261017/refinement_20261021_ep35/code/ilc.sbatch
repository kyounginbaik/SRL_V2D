#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --job-name=ep35-ilc
#SBATCH --array=0-3
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep35/logs/ilc-%A_%a.log
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
G=("0.6 0.6" "0.4 0.4" "0.8 0.5" "0.8 0.8"); set -- ${G[$SLURM_ARRAY_TASK_ID]}
cd /tmp && /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B -u /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep38/code/ilc_carry_generic.py --source /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/spoon_local_retention/episode_000035 --out /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep35/ilc/gr$1_gp$2 --side right --object wooden_spoon --t0 4.5 --t1 18.5 --t-clear 18.6 --episode-dir /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/wrist_chart/episode_000035/chart_0/adapter --iterations 4 --gain-rot $1 --gain-pos $2
