#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --job-name=sB-ep3-pilc
#SBATCH --array=0-1
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamB_tilt38/logs/ep3-pilc-%A_%a.log
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
G=("0.6 0.6" "0.4 0.4"); set -- ${G[$SLURM_ARRAY_TASK_ID]}
cd /tmp && /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B -u /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep38/code/ilc_carry_generic.py --source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamB_tilt38/ep3/release_grid/mid_a_s71/open1.5_frac0.6_handle_rs0.6_rm0.04 --out /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamB_tilt38/ep3/ilc_pot/rm0.04_gr$1_gp$2 --side left --object white_pot --t0 2.75 --t1 11.6 --t-clear 12.6 --episode-dir /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamB_tilt38/ep3/calibrated_inputs/episodes/episode_000003 --iterations 3 --gain-rot $1 --gain-pos $2
