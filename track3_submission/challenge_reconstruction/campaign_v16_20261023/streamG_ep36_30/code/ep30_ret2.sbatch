#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --job-name=G30-ret2
#SBATCH --array=0-1
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamG_ep36_30/logs/ep30_ret2-%A_%a.log
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; PY=$R/.venv/bin/python
RS=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021; EPDIR=$RS/ep30_v8/episodes/episode_000030
export PYTHONPATH=$R/code:/data1/SRL_V2D/third_party/DexAgent:/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/code RACKW=$RS/ep30_v8
G=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamG_ep36_30/ep30/ret2
SRC=$RS/ep30_v9/search_v8scene/rel2_11.6_o1.2_l0_body0.01
S=("3.2 11.4 0.5 111" "2.6 11.4 0.75 112"); set -- ${S[$SLURM_ARRAY_TASK_ID]}
cd $R/code && $PY -u $RS/code/retention.py --source $SRC --episode-dir $EPDIR --output $G/ret_$1_$2_s$3 --start-s $1 --stop-s $2 --protect-object plastic_dish_rack --seed $4 --scale $3 --samples 96 --iterations 12 --workers 16
