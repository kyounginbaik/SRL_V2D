#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --job-name=assemble-v10
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/logs/assemble_v10-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd /data1/SRL_V2D
/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B -u /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/package_code/assemble_twenty_swap.py --original-package /data1/SRL_V2D/challenge_submissions/track3_v10_20261021/all_20_episodes_no_32_33  --component 32=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep32/staging/common_components/episode_000032 --portable 32=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep32/staging/portable_archives/episode_000032_portable.tar.gz --video 32=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep32/videos/episode_000032  --reason "v10: episode 32 spoon re-planned with a spoon-in-cup retention objective so the bowl enters the cup during the scoops (3.3-5.3 s, 6.6-8.3 s)"  --output /data1/SRL_V2D/challenge_submissions/track3_v10_20261021/all_20_episodes --archive
