#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch2/exaFLOPs/hawor_rerun/%x-%j.log
source /home/exaflops/miniconda3/etc/profile.d/conda.sh; conda activate hawor; cd /scratch2/exaFLOPs/hawor_rerun/HaWoR; python ../video_true.py $1 2>&1 | grep -E 'VIDEO|Error|Traceback' | cut -c1-200
