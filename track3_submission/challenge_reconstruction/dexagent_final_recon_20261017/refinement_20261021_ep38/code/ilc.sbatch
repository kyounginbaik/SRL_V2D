#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --job-name=ep38-ilc
#SBATCH --array=0-2
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep38/logs/ilc-%A_%a.log
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
G=("0.6 0.6" "0.4 0.4" "0.8 0.5"); set -- ${G[$SLURM_ARRAY_TASK_ID]}
cd /tmp && /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -u /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep38/code/ilc_carry_generic.py --source /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/expanded_batch/dustpan_release/episode_000038/case_3 --out /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep38/ilc/gr$1_gp$2 --side right --object mini_dust_pan --t0 2.4 --t1 10.5 --t-clear 11.8 --episode-dir /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/episodes/episode_000038 --iterations 4 --gain-rot $1 --gain-pos $2
