#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --job-name=sweeper30-contact
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/slurm/sweeper30-contact-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -u /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/code/inspect_rack_contacts.py
