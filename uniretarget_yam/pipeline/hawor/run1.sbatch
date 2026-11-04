#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:RTX3090:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch2/exaFLOPs/hawor_rerun/%x-%j.log
source /home/exaflops/miniconda3/etc/profile.d/conda.sh; conda activate hawor; cd /scratch2/exaFLOPs/hawor_rerun/HaWoR
for N in $1; do python ../run_true.py --episode $N --out ../out 2>&1 | grep -E 'tracks kept|HAWOR_TRUE_DONE|Error|^[A-Za-z]*Error|Traceback' | cut -c1-400; done; python ../eval_true.py $1 2>&1 | grep -E ' valid |EVAL_DONE|Error' | cut -c1-230; echo RUN_DONE
