#!/bin/bash
#SBATCH -p debug --nodelist=node100 -c 16 --mem=48G -t 00:50:00 -J C30-beam -o /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamC30_compose/logs/beam_%j.log
# usage: C30_TGT=... sbatch --export=ALL beam.sbatch "<script and args>"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamC30_compose/tmp
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code:/data1/SRL_V2D/third_party/DexAgent
mkdir -p $TMPDIR; cd $TMPDIR; echo "C30_TGT=$C30_TGT H13_ADV=${H13_ADV:-}"
eval /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B -u $1 2>&1 | grep --line-buffered -v -E "Warn|warn\("
