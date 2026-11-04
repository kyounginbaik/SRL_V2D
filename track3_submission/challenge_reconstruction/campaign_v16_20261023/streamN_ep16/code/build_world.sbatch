#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --job-name=N16-world
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamN_ep16/logs/build_world-%j.log
# usage: sbatch build_world.sbatch RACK_VISUAL_OBJ WORKSPACE [extra build_world.py args]
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8
PY=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
M=$1; WS=$2; shift 2
cd /tmp && $PY -u /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamN_ep16/code/build_world.py --rack-visual $M --workspace $WS "$@"
