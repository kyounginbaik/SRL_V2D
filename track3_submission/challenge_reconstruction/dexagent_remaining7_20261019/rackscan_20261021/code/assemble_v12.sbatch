#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --job-name=assemble-v12
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/logs/assemble_v12-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd /data1/SRL_V2D
/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B -u /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/package_code/assemble_twenty_swap.py --original-package /data1/SRL_V2D/challenge_submissions/track3_v11_20261021/all_20_episodes  --component 33=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep33/staging_v12/common_components/episode_000033 --portable 33=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep33/staging_v12/portable_archives/episode_000033_portable.tar.gz --video 33=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep33/videos_v12/episode_000033  --reason "v12: episode 33 scoops re-planned segment by segment (right hand on the spoon, left hand on the cup tilt) and the actor refitted with finger margins; bowl closer to the cup in every scoop"  --output /data1/SRL_V2D/challenge_submissions/track3_v12_20261021/all_20_episodes --archive
