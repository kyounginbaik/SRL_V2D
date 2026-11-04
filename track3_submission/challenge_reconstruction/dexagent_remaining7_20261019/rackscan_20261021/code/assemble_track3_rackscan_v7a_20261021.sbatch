#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --job-name=rack-assemble
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/logs/assemble-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /data1/SRL_V2D
/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -u /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/package_code/assemble_twenty_swap.py --original-package /data1/SRL_V2D/challenge_submissions/track3_pot_fix_v6_20261020/all_20_episodes  --component 14=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/common_components/episode_000014 --portable 14=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/portable_archives/episode_000014_portable.tar.gz --component 16=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/common_components/episode_000016 --portable 16=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/portable_archives/episode_000016_portable.tar.gz --component 28=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/common_components/episode_000028 --portable 28=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/portable_archives/episode_000028_portable.tar.gz --component 30=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/common_components/episode_000030 --portable 30=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/portable_archives/episode_000030_portable.tar.gz --reason "plastic_dish_rack replaced by the scan-derived mesh (gt_best_per_object/plastic_dish_rack.ply) expressed in our rack body frame; robot re-planned in physics with the new rack (see rackscan_20261021)" --output /data1/SRL_V2D/challenge_submissions/track3_rackscan_v7a_20261021/all_20_episodes --archive
