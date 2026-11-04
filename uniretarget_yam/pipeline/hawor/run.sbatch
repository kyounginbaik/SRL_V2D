#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:RTX3090:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch2/exaFLOPs/hawor_rerun/%x-%j.log
# Usage: sbatch run.sbatch "37 8 25"
source /home/exaflops/miniconda3/etc/profile.d/conda.sh; conda activate hawor; cd /scratch2/exaFLOPs/hawor_rerun/HaWoR
for N in $1; do python ../run_true.py --episode $N --out ../out 2>&1 | grep -vE 'Warning|warn\(|it/s|^$' | tail -25 | cut -c1-200; done; echo RUN_DONE
cd /scratch2/exaFLOPs/hawor_rerun/HaWoR; python ../eval_true.py $1 2>&1 | grep -vE 'Warning|warn\(|WARNING' | tail -14 | cut -c1-230
