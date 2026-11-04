#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --job-name=ep25-asm
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/ep25_v15/logs/asm-%j.log
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd /data1/SRL_V2D
/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B -u /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/package_code/assemble_twenty_swap.py --original-package /data1/SRL_V2D/challenge_submissions/track3_v14_20261022/all_20_episodes --component 25=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/ep25_v15/staging/common_components/episode_000025 --portable 25=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/ep25_v15/staging/portable_archives/episode_000025_portable.tar.gz --reason "test: first real episode 25 component" --output $1
