#!/bin/bash
#SBATCH -p debug --nodelist=node01 -c 2 --mem=16G -t 00:50:00 -o /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG12/logs/pkg_%A_%a.log
G=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG12
PY=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $G/code/${JOBS}); set -- $LINE
PKG=$1; TGT=$2; LBL=$3; shift 3
mkdir -p $G/out/pkg/$LBL
for EP in "$@"; do
 [ -s $G/out/pkg/$LBL/ep$EP.json ] && continue
 cd /tmp && OMP_NUM_THREADS=2 $PY -B /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/compose_target/score_vs_compose.py $PKG/track3_submission.parquet $PKG/results $TGT $G/out/pkg/$LBL/ep$EP.json $EP 2>&1 | grep -v Warn
done
