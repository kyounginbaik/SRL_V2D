#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --job-name=sB-ep8-ilc
#SBATCH --array=0-2
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamB_tilt38/logs/ep8-ilc-%A_%a.log
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
G=("cup_right_2.5_10.8_s64 0.5 0.8" "cup_right_2.5_10.8_s64 0.6 0.6" "cup_right_2.0_10.8_s63 0.5 0.8"); set -- ${G[$SLURM_ARRAY_TASK_ID]}
cd /tmp && /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B -u /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep38/code/ilc_carry_generic.py --source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamB_tilt38/ep8/retention/$1 --out /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamB_tilt38/ep8/ilc/$1_gr$2_gp$3 --side left --object white_pot --t0 5.2 --t1 8.0 --t-clear 9.65 --episode-dir /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamB_tilt38/ep8/calibrated_inputs/episodes/episode_000008 --iterations 4 --gain-rot $2 --gain-pos $3
