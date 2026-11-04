#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --job-name=G30-ret3
#SBATCH --array=0-3
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamG_ep36_30/logs/ep30_ret3-%A_%a.log
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; PY=$R/.venv/bin/python
RS=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021; EPDIR=$RS/ep30_v8/episodes/episode_000030
export PYTHONPATH=$R/code:/data1/SRL_V2D/third_party/DexAgent:/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/code RACKW=$RS/ep30_v8
G=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamG_ep36_30/ep30
SHIP=$RS/ep30_v9/search_v8scene/rel2_11.6_o1.2_l0_body0.01
S=("$SHIP p2_2.9_11.4_s0.75 2.9 0.75 113" "$SHIP p2_3.2_11.4_s0.75 3.2 0.75 114" "$G/ret2/ret_3.2_11.4_s0.5 p3of32_3.2_11.4_s0.5 3.2 0.5 115" "$G/ret2/ret_2.6_11.4_s0.75 p3of26_3.5_11.4_s0.5 3.5 0.5 116"); set -- ${S[$SLURM_ARRAY_TASK_ID]}
cd $R/code && $PY -u $RS/code/retention.py --source $1 --episode-dir $EPDIR --output $G/ret3/$2 --start-s $3 --stop-s 11.4 --protect-object plastic_dish_rack --seed $5 --scale $4 --samples 96 --iterations 12 --workers 16
