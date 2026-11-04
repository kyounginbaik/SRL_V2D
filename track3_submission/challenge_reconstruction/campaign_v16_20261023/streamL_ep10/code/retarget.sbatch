#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:45:00
#SBATCH --job-name=L10-retarget
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamL_ep10/logs/retarget-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; S=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamL_ep10
PY=$R/.venv/bin/python; T=$R/refinement_20261018/episode_000010/calibrated_nest_release4
cd /tmp
$PY -u $S/code/retarget_recon.py $T $T $S/retarget/nest4_teacher
$PY -u $S/code/retarget_recon.py $T $R/refinement_20261021_ep10/search/ret_right_3.5_8.0_s0.5 $S/retarget/v17_teacher_ref
$PY -u $S/code/retarget_recon.py $T $R/refinement_20261021_ep10/search/ret_right_3.5_5.5_s0.5 $S/retarget/v17b_teacher_ref
$PY $S/code/insertion_recon.py $T/execution/trajectory.npz $S/retarget/nest4_teacher/execution/trajectory.npz $S/retarget/v17_teacher_ref/execution/trajectory.npz $S/retarget/v17b_teacher_ref/execution/trajectory.npz
