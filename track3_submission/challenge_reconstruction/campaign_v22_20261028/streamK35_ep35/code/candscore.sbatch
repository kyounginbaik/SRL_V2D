#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node100
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=00:50:00
#SBATCH --job-name=K35-cand
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK35_ep35/logs/cand-%j.log
# usage: candscore.sbatch TARGET_DIR LABELPREFIX  (scores every ep35 build-v2 candidate vs TARGET via score_any.sh)
T=$1; P=$2
ls -d /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates/*/episode_000035 /home/exaFLOPs/v22_overflow_20261028/common_v2_candidates/*/episode_000035 2>/dev/null | \
  xargs -P 12 -I{} sh -c 't=$(basename $(dirname {})); bash /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/code/score_any.sh 35 {} '"$P"'_$t '"$T"' > /dev/null 2>&1; echo $t'
echo DONE
