#!/bin/bash
#SBATCH -p debug
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:50:00
#SBATCH -o /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/h2h_other/logs/%x_%j.out
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
cd /tmp && /home/exaFLOPs/miniforge3/bin/python -I -u /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/h2h_other/code/rdepth.py $1 $2 /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/h2h_other/results/rdepth_ep$1.json
