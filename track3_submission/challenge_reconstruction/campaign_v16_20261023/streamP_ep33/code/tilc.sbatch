#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:40:00
#SBATCH --job-name=P33-tilc
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamP_ep33/logs/tilc-%A_%a.log
# usage: sbatch --array=0-N tilc.sbatch LIST (line: TEACHER OUT args...)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
line=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $1); set -- $line; T=$1; O=$2; shift 2
cd /tmp; /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B -u /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamP_ep33/code/teacher_ilc33.py $T $O "$@"
