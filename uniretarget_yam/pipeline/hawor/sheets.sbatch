#!/usr/bin/env bash
#SBATCH --gres=gpu:RTX3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch2/exaFLOPs/hawor_rerun/%x-%j.log
source /home/exaflops/miniconda3/etc/profile.d/conda.sh; conda activate hawor; cd /scratch2/exaFLOPs/hawor_rerun/HaWoR; python ../sheets.py 16 15 20 1 3 5 6 10 14 19 28 30 2>&1 | grep -vE 'Warning|warn\(|WARNING' | tail -32 | cut -c1-160
