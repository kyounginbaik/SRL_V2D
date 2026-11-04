#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:40:00
#SBATCH --job-name=N16-port
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamN_ep16/logs/port-%A_%a.log
# usage: sbatch --array=0-N port.sbatch WORLD OUT_BASE TEACHER_DIR:NAME [TEACHER_DIR:NAME ...]   (NAME starting with 'actor@' = checkpoint path, closed-loop)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
CODE=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamN_ep16/code
W=$1; O=$2; shift 2; A=($@); spec=${A[$SLURM_ARRAY_TASK_ID]}; T=${spec%%:*}; NM=${spec##*:}
cd /tmp
case $NM in
  actor@*) $PY -u $CODE/eval_actor.py $T $W $O/${NM#actor@} ;;
  *) $PY -u $CODE/port_teacher.py $T $W $O/$NM ;;
esac
