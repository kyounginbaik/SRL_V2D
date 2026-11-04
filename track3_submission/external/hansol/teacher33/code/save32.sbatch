#!/bin/bash
#SBATCH -p debug
#SBATCH --nodelist=node100
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH -J R32-save
#SBATCH -o /data1/hansol/retarget32/logs/%x-%j.log
# usage: save32.sbatch ACTOR TABLE OUT  -- bake a beam table into the actor (mresH save), official evaluation in the snapped W1 world, then teacher_check32
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data1/hansol/retarget32/tmp E31_TGT=/data1/hansol/retarget32/targets/R32_HGA E31_EPD=/data1/hansol/retarget32/worlds/R32_HGA_W1s/ep32/episode_corrected/episode_000032 PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code:/data1/SRL_V2D/third_party/DexAgent:/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages
PY=/home/hansol/v2d/.venvs/v2d-host/bin/python; M=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates/G18_ep32_H12e1f0/episode_000032/meshes/episode_000032; cd /data1/hansol/retarget32/tmp
[ -f $3/actor/evaluation/result.json ] || $PY -B -u /data1/hansol/retarget32/code/mresH.py save --ep 32 --actor $1 --meshes $M --out $3 --table $2 2>&1 | grep -v -E "Warn|warn\(" | grep -E "INIT|FINAL"
$PY -B /data1/hansol/retarget32/code/teacher_check32.py $3/actor/evaluation 2>&1 | grep -v -i warn
