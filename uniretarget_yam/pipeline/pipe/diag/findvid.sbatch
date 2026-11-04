#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=0:15:00
#SBATCH --output=/scratch2/exaFLOPs/srl_yam_run/pipe/%x-%j.log
hostname; ls -d /data* /data1/video_to_data_challenge 2>&1; df -h 2>/dev/null | awk '{print $NF}' | head -30
