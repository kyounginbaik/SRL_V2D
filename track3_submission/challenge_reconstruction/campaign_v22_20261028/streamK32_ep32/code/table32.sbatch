#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node100
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=K32-table
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK32_ep32/logs/%x-%j.log
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
PY=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python; T=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK32_ep32/code/table32.py
B=/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/selected20_final_v14_g17/episode_000032
J=/data1/SRL_V2D/challenge_submissions_SRL/cowork_ep32_kinematic_fixed_20261102/ep32_kinematic_fixed/ep32_fixed_object_poses.npz
cd /tmp
for P in $B/object_poses.npz $J; do
  $PY -B $T $P $B/meshes wooden_spoon 0,5,10,15,20,25,225,230,235,240,245,250,255
  $PY -B $T $P $B/meshes blue_cup 0,3,6,225,230,235,240,245,250,255
done
