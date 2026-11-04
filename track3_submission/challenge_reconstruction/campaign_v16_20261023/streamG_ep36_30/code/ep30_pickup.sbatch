#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --job-name=G30-pick
#SBATCH --array=0-1
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamG_ep36_30/logs/ep30_pickup-%A_%a.log
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
PY=$R/.venv/bin/python
G=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamG_ep36_30/ep30/pickup
export PYTHONPATH=$R/code:/data1/SRL_V2D/third_party/DexAgent:/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/code
cd /tmp
if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
  $PY -B -u $R/code/physics_pickup_search_v2.py --spec $G/spec_early.json --output $G/early_resume45 --workers 16 --samples 48 --iterations 12 --seed 45 --resume /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/ep30_v8/search/pickup_s45/progress.json
else
  $PY -B -u $R/code/physics_pickup_search_v2.py --spec $G/spec_early.json --output $G/early_fresh46 --workers 16 --samples 48 --iterations 12 --seed 46
fi
