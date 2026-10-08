#!/bin/bash
#SBATCH --job-name=t3-pack-monitor
#SBATCH --partition=debug
#SBATCH --nodelist=node200
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=3-00:00:00
#SBATCH --output=/data1/SRL_V2D/slurm_logs/t3_monitor_%j.out
#SBATCH --error=/data1/SRL_V2D/slurm_logs/t3_monitor_%j.err
exec /home/exaFLOPs/miniforge3/bin/python /data1/SRL_V2D/challenge_submissions/track3/transfer_hawor_chunk1s_20261009/code/monitor.py
