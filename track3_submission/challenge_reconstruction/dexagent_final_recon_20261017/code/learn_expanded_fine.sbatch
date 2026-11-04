#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:30:00
#SBATCH --array=0-11%12
#SBATCH --job-name=expanded-learn-fine
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/expanded_batch/logs/learned-fine-%A_%a.log
set -euo pipefail
cd /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
episodes=(6 33 38)
teachers=(retention_coordinates/episode_000006 cup33_retention/seed_72 dustpan_release/episode_000038/case_3)
centers=(1024 2048)
gains=(0.01 0.1)
i=$((SLURM_ARRAY_TASK_ID / 4));j=$(((SLURM_ARRAY_TASK_ID % 4) / 2));k=$((SLURM_ARRAY_TASK_ID % 2))
name=$(printf 'episode_%06d' "${episodes[$i]}")
teacher="refinement_20261018/expanded_batch/${teachers[$i]}/execution"
out="refinement_20261018/expanded_batch/learned_fine/${name}/basis_${centers[$j]}_gain_${gains[$k]}"
.venv/bin/python code/learned_feedback_policy.py --scene "$teacher/scene_physics.xml" --trajectory "$teacher/trajectory.npz" \
 --output "$out/training" --centers "${centers[$j]}" --feedback-gain "${gains[$k]}"
.venv/bin/python code/evaluate_learned_feedback.py --episode "${episodes[$i]}" --scene "$teacher/scene_physics.xml" \
 --checkpoint "$out/training/policy.npz" --output "$out/evaluation"
.venv/bin/python code/audit_joint_limits.py --scene "$out/evaluation/execution/scene_physics.xml" \
 --trajectory "$out/evaluation/execution/trajectory.npz" --output "$out/evaluation/actual_joint_limits.json"
