#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --job-name=G36-ret
#SBATCH --array=0-2
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamG_ep36_30/logs/ep36_ret-%A_%a.log
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; PY=$R/.venv/bin/python
G=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamG_ep36_30/ep36; EPD=$R/refinement_20261021_ep36/episode_corrected/episode_000036
export PYTHONPATH=$R/code:/data1/SRL_V2D/third_party/DexAgent:/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/code
S=("$G/rebased/seat17.5ilc seat17.5ilc 0.5 201" "$G/rebased/rm1_14.3ilc2 rm1_14.3ilc2 0.5 202" "$G/rebased/base base 0.75 203"); set -- ${S[$SLURM_ARRAY_TASK_ID]}
cd $R/code && $PY -u /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/code/retention.py --source $1 --episode-dir $EPD --output $G/retention/$2_ret13.8_18.5_s$3 --start-s 13.8 --stop-s 18.5 --protect-object wooden_spoon --seed $4 --scale $3 --samples 96 --iterations 12 --workers 16
