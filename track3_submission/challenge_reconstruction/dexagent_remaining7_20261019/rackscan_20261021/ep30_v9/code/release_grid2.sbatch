#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --job-name=e30v9-rel2
#SBATCH --array=0-23
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/ep30_v9/logs/rel2-%A_%a.log
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
G=()
for r in 11.6 11.8; do for o in 0.8 1.2; do for md in "body 0.02" "pinch 0.02" "body 0.01"; do for l in 0 0.005; do G+=("$r $o $l $md"); done; done; done; done
set -- ${G[$SLURM_ARRAY_TASK_ID]}
/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -u /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/ep30_v9/code/release_v9.py /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/ep30_v8/search/pickup_s45_lift0.06/release_neutral/trial0/ret_3.2_12.0_s0.75 /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/ep30_v8/episodes/episode_000030 /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/ep30_v9/search_v8scene/rel2_$1_o$2_l$3_$4$5 $1 $2 $3 0.06 0.06 $4 $5
