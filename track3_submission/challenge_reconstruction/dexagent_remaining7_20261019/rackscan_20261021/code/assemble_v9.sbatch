#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --job-name=assemble-v9
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/logs/assemble_v9-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd /data1/SRL_V2D
/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B -u /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/package_code/assemble_twenty_swap.py --original-package /data1/SRL_V2D/challenge_submissions/track3_v9_20261021/all_20_episodes_no_ep10 --component 10=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep10/staging/common_components/episode_000010 --portable 10=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep10/staging/portable_archives/episode_000010_portable.tar.gz --video 10=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep10/videos/episode_000010 --reason "v9: episode 10 cups replaced by the scan-derived cups in our body frames with thin-wall collision; stacking re-planned so the beige cup nests at the depth measured from stereo (about 11 cm)" --output /data1/SRL_V2D/challenge_submissions/track3_v9_20261021/all_20_episodes --archive
