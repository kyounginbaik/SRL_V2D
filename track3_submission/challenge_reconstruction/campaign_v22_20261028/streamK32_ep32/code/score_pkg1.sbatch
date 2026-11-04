#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node100
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=00:50:00
#SBATCH --job-name=K32-score
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK32_ep32/logs/score-%j.log
# usage: sbatch score_pkg.sbatch PACKAGE_DIR(all_20_episodes) TARGET_DIR OUT_DIR [EPS...] -> kit-faithful score_vs_compose per episode (4 parallel)
set -uo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PKG=$1; TGT=$2; OUT=$3; shift 3; EPS=${*:-1 3 5 6 8 10 14 15 16 19 20 25 28 30 32 33 35 36 37 38}
PY=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
S=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/compose_target/score_vs_compose.py
mkdir -p $OUT; cd /tmp
for e in $EPS; do echo $e; done | xargs -P 4 -I{} sh -c "$PY -B -u $S $PKG/track3_submission.parquet $PKG/results $TGT $OUT/ep{}.json {} > $OUT/ep{}.log 2>&1"
echo DONE
