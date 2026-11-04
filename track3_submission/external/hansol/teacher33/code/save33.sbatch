#!/bin/bash
#SBATCH -p debug
#SBATCH --exclude=node200
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH -J T33-save
#SBATCH -o /data1/hansol/teacher33/logs/%x-%j.log
# usage: [T33TGT=<target dir> T33EPD=<snapped episode dir>] save33.sbatch ACTOR TABLE OUT -- bake a beam table into the actor (mresH save, as retarget32 save32.sbatch), official evaluation in the snapped c18 W1s world
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data1/hansol/teacher33/tmp E31_TGT=${T33TGT:-/data1/hansol/retarget33/targets/R33_HGAc18} E31_EPD=${T33EPD:-/data1/hansol/teacher33/worlds/R33c18_W1s/ep33/episode_corrected/episode_000033} PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code:/data1/SRL_V2D/third_party/DexAgent:/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages
PY=/home/hansol/v2d/.venvs/v2d-host/bin/python; M=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates/G18_ep33_H13b1/episode_000033/meshes/episode_000033; cd /data1/hansol/teacher33/tmp
[ -f $3/actor/evaluation/result.json ] || $PY -B -u /data1/hansol/teacher33/code/mresH.py save --ep 33 --actor $1 --meshes $M --out $3 --table $2 2>&1 | grep -v -E "Warn|warn\(" | grep -E "INIT|FINAL|Error|error"
