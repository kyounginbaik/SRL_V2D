#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --job-name=A2-ilc5
#SBATCH --array=0-4
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamA2_crotch/logs/ilc5-%A_%a.log
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
G=("0.3 0.3 0.35" "0.5 0.5 0.35" "0.3 0.0 0.35" "0.2 0.2 0.6" "0.4 0.4 0.6"); set -- ${G[$SLURM_ARRAY_TASK_ID]}
cd /tmp && $R/.venv/bin/python -B -u $R/refinement_20261021_ep38/code/ilc_carry_generic.py --source $R/refinement_20261020_pot/power_grasp/retention_full/episode_000005/near_a_iter4 --out /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamA2_crotch/ep5/ilc/gr$1_gp$2_s$3 --side right --object wooden_spoon --t0 2.0 --t1 13.3 --t-clear 14.0 --sigma-s $3 --episode-dir $R/refinement_20261018/calibrated_scene_candidates/episodes/episode_000005 --iterations 4 --gain-rot $1 --gain-pos $2
