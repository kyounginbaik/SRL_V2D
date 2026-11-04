#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node100
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --time=00:50:00
#SBATCH --job-name=K32-scan
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK32_ep32/logs/%x-%j.log
# K32 B1: kit-faithful score (score_any) of every existing ep32 build-v2 candidate vs target $1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
C=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates; C22=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028; T=$1
ls -d $C/*ep32* | xargs -n1 basename | xargs -P 3 -I{} sh -c "$C22/code/score_any.sh 32 $C/{}/episode_000032 K32scan_{}_$(basename $T) $T >/dev/null 2>&1; echo {}"
echo DONE
