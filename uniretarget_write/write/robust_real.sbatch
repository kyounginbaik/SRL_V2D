#!/usr/bin/env bash
#SBATCH --partition=dell_cpu
#SBATCH --qos=cpu_qos
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch2/exaFLOPs/srl_yam_run/ablate/oak/write/logs/%x-%j.log
cd /scratch2/exaFLOPs/srl_yam_run/ablate/oak/write; export REAL=1 OPEN=0.2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/lustre/exaflops/retarget_bench/.venv_spider/bin/python robust.py
echo ROBUST_DONE
