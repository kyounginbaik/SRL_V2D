#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node100
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --job-name=R33-svst
#SBATCH --output=/data1/hansol/retarget33/logs/%x-%j.log
# R33 stage + G18 mesh swap + build-v2 of an already-saved actor (copy of the team svst.sbatch, save step done separately). usage: sbatch r33svst.sbatch NAME SAVE_DIR
set -euo pipefail
R=/data1/hansol/retarget33; N=$1; SV=$2; E=33; NM=episode_000033; B=$R/staging/${TG:-R33_HGA7b}_ep33_$N; TAG=${TG:-R33_HGA7b}_$N
export E31_TGT=$R/targets/${TG:-R33_HGA7b} OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=$R/tmp PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code:/data1/SRL_V2D/third_party/DexAgent:/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages
NOTE="R33: v29 ep33 actor (H13_ep33_b1) + window-beam phase residuals (team K33 HGA_c9 table, then R33 R1_c1 + R2 passes), retargeted vs ${TG:-R33_HGA7b} = our GT-free ep33 recon (S1_HGA)"
bash $R/code/stage_a_r33.sh $E $SV/actor $B "$NOTE"
/home/hansol/v2d/.venvs/v2d-host/bin/python -B $R/code/g18swap.py $E $B/components/$NM
bash $R/code/r33run_one.sh $E $B/components/$NM $TAG
/home/hansol/v2d/.venvs/v2d-host/bin/python -c "import json;d=json.load(open('$R/candidates/$TAG/$NM/completed.json'));print('BUILD exact_previous_physical_states_actions =', d.get('exact_previous_physical_states_actions'))"
