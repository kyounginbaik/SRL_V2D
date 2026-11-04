#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --job-name=G30-rel
#SBATCH --array=0-5
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamG_ep36_30/logs/ep30_rel-%A_%a.log
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; PY=$R/.venv/bin/python
RS=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021; EPDIR=$RS/ep30_v8/episodes/episode_000030
G=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamG_ep36_30/ep30; SRC=$G/ret2/ret_3.2_11.4_s0.5
S=("11.5 1.2 0 body 0.01" "11.3 1.2 0.005 body 0.01" "11.6 0.8 0 neutral 0.02" "11.4 1.5 0.005 body 0.01" "11.5 1.2 0.01 body 0.01" "11.2 1.2 0 pinch 0.02"); set -- ${S[$SLURM_ARRAY_TASK_ID]}
cd /tmp && $PY -u $RS/ep30_v9/code/release_v9.py $SRC $EPDIR $G/rel_ret2/rel_$1_o$2_l$3_$4$5 $1 $2 $3 0.06 0.06 $4 $5
