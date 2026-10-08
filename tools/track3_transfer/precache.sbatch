#!/bin/bash
#SBATCH --job-name=t3-collision-cache
#SBATCH --partition=debug
#SBATCH --nodelist=node100
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --array=0-24%8
#SBATCH --output=/data1/SRL_V2D/slurm_logs/t3_collision_%A_%a.out
#SBATCH --error=/data1/SRL_V2D/slurm_logs/t3_collision_%A_%a.err
set -euo pipefail
sg docker -c "docker run --rm --entrypoint python --user $(id -u):$(id -g) -e HOME=/tmp -e OMP_NUM_THREADS=2 -v /data1:/data1 -v /data1/SRL_V2D/robotic_grounding/flash_chord:/workspace -w /workspace flash_chord:latest /data1/SRL_V2D/challenge_submissions/track3/transfer_hawor_chunk1s_20261009/code/precache.py $SLURM_ARRAY_TASK_ID"
