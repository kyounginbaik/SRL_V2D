#!/bin/bash
#SBATCH -p debug
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --exclude=node200
#SBATCH -o /data1/hansol/ep33_20261102/logs/agentcrops32_%j.log
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages
PY=/home/hansol/v2d/.venvs/v2d-host/bin/python
N=$($PY -c "import cv2;print(int(cv2.VideoCapture('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical/evaluation/episode_000032/rgb.mp4').get(7)))"); echo frames $N
$PY -B /data1/hansol/ep33_20261102/agentmarks32/crops32.py /data1/hansol/ep33_20261102/agentmarks32/crops 1-$((N-1)) 3
