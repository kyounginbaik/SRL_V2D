#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --job-name=ep1-hand-margins
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/logs/ep1-hand-margins-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code/search_repaired_hand_margins_v3.py --checkpoint /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/episode_000001/reconstruction_repair/patch_height094/learned_feedback/centers_1024_gain_0.03/training/policy.npz --scene /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/episode_000001/reconstruction_repair/patch_height094/learned_feedback/centers_1024_gain_0.03/evaluation/execution/scene_physics.xml --episode-dir /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/episodes/episode_000001 --baseline-report /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/episode_000001/reconstruction_repair/patch_height094/learned_feedback/centers_1024_gain_0.03/evaluation/result.json --config /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/episode_000001/reconstruction_repair/patch_height094/learned_hand_margins/config_v3.json --output /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/episode_000001/reconstruction_repair/patch_height094/learned_hand_margins/search_v3 --samples 64 --iterations 16 --workers 16 --seed 1044 --tolerance-deg 1.1
/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code/audit_full_joint_limits.py --run /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/episode_000001/reconstruction_repair/patch_height094/learned_hand_margins/search_v3/evaluation
/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code/audit_object_support.py /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/episode_000001/reconstruction_repair/patch_height094/learned_hand_margins/search_v3/evaluation
