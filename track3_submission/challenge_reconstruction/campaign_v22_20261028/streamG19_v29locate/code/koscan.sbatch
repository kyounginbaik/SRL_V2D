#!/bin/bash
#SBATCH -p debug --nodelist=node100 -c 6 --mem=24G -t 00:50:00 -J G19-koscan -o /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG19_v29locate/logs/koscan_%j.log
# usage: koscan.sbatch a|b EP SPECS   a = H13a h13scan.py ('|' specs, frames advance; negative = delay), b = H13b tadv.py (',' shifts)
set -uo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG19_v29locate/tmp
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code:/data1/SRL_V2D/third_party/DexAgent
S=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG19_v29locate; C22=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028
W=$1; E=$2; SP=$3; cd $TMPDIR
if [ $W = a ]; then source $C22/streamH13a/code/env.sh; eval A=\$A$E; eval M=\$M$E
  $PY -B -u $C22/streamH13a/code/h13scan.py $E $A $M $S/ko1/scan/a_ep$E.json "$SP" 2>&1 | grep -v -E "Warn|warn\("
else source $C22/streamH13b/code/env.sh; eval A=\$A$E; eval SS=\$S$E; M=$(ls -d $SS/components/episode_*/meshes/episode_*)
  $PY -B -u $C22/streamH13b/code/tadv.py --ep $E --actor $A --meshes $M --out $S/ko1/scan/b_ep$E --shifts=$SP --workers 6 2>&1 | grep -v -E "Warn|warn\("
fi
