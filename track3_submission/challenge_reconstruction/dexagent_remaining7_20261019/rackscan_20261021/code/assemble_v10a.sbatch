#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --job-name=assemble-v10a
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/logs/assemble_v10a-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd /data1/SRL_V2D
/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B -u /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/package_code/assemble_twenty_swap.py --original-package /data1/SRL_V2D/challenge_submissions/track3_v9_20261021/all_20_episodes  --component 36=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep36/staging/common_components/episode_000036 --portable 36=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep36/staging/portable_archives/episode_000036_portable.tar.gz --video 36=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep36/videos/episode_000036  --component 38=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep38/staging/common_components/episode_000038 --portable 38=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep38/staging/portable_archives/episode_000038_portable.tar.gz --video 38=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep38/videos/episode_000038  --reason "v10: episode 36 spoon track roll flip at 12.55-14.65 s corrected (verified with the handle markers) and re-planned (corrected-track lineage); episode 38 dust pan carry orientation corrected (ILC on the right wrist)"  --output /data1/SRL_V2D/challenge_submissions/track3_v10_20261021/all_20_episodes_no_32_33 --archive
