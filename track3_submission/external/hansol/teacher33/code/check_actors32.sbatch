#!/bin/bash
#SBATCH -p debug
#SBATCH --nodelist=node100
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH -J R32-check
#SBATCH -o /data1/hansol/retarget32/logs/%x-%j.log
# per-0.5 s spoon error / grasp of official evaluation trajectories vs R32_HGA (teacher_check32 on each evaluation dir)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 E31_TGT=/data1/hansol/retarget32/targets/R32_HGA PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code:/data1/SRL_V2D/third_party/DexAgent:/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages
for E in "$@"; do echo "== $E"; /home/hansol/v2d/.venvs/v2d-host/bin/python -B /data1/hansol/retarget32/code/teacher_check32.py $E 2>&1 | grep -v -i warn; done
