#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:15:00
#SBATCH --output=/scratch2/exaFLOPs/hawor_rerun/%x-%j.log
source /home/exaflops/miniconda3/etc/profile.d/conda.sh; conda activate hawor; cd /scratch2/exaFLOPs/hawor_rerun/HaWoR; for T in $1; do python ../ik_vs_hawor.py $T 2>&1 | grep -vE 'Warning|warn\(|WARNING' | tail -16 | cut -c1-220; done
