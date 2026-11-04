#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --job-name=D9-save
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamD9_tight/logs/save-%j.log
# usage: sbatch save.sbatch EP TABLE.npz OUT   (bake the residual table into the base actor's phase weights + official evaluate_policy_component)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
source /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamD9_tight/code/env.sh
eval A=\$A$1; eval M=\$M$1
cd $D9/tmp; /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B -u $D9/code/mres9.py save --ep $1 --actor $A --meshes $M --table $2 --out $3 2>&1 | grep -v -E "Warn|warn\("
