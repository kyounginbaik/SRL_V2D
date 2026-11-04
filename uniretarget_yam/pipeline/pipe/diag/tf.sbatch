#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:20:00
#SBATCH --output=/scratch2/exaFLOPs/srl_yam_run/pipe/%x-%j.log
source /home/exaflops/miniconda3/etc/profile.d/conda.sh; conda activate hawor; python /scratch2/exaFLOPs/srl_yam_run/pipe/diag/table_fit.py
