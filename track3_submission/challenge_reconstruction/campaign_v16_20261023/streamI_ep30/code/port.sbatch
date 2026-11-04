#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:40:00
#SBATCH --job-name=I30-port
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamI_ep30/logs/port-%A_%a.log
# usage: sbatch --array=0-N port.sbatch WORLD OUT_BASE TEACHER_DIR:NAME [TEACHER_DIR:NAME ...]
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
W=$1; O=$2; shift 2; A=($@); spec=${A[$SLURM_ARRAY_TASK_ID]}; T=${spec%%:*}; N=${spec##*:}
cd /tmp && $PY -u /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamI_ep30/code/port_teacher.py $T $W $O/$N
