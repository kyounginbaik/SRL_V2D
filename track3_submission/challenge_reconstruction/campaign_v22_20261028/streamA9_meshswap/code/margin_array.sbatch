#!/bin/bash
#SBATCH --job-name=A9-margin
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:50:00
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamA9_meshswap/logs/margin-%A_%a.log
# A9: Y9 twin_margin.py (scorer-side) on one item per task. ITEMS=<json> TAG=<prefix>
set -uo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamY9_mesh
O=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamA9_meshswap/out/margins; mkdir -p $O
PY=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
$PY -B -c "import json;it=json.load(open('$ITEMS'));json.dump([it[$SLURM_ARRAY_TASK_ID]],open('$O/${TAG}_item$SLURM_ARRAY_TASK_ID.in.json','w'))"
$PY -B twin_margin.py $O/${TAG}_item$SLURM_ARRAY_TASK_ID.in.json $O/${TAG}_item$SLURM_ARRAY_TASK_ID.json --seeds ${SEEDS:-16} --random ${RAND:-16}
