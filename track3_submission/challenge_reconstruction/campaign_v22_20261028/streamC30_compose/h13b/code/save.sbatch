#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node100
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --job-name=H13b-save
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamC30_compose/h13b/logs/save-%j.log
# usage: sbatch save.sbatch EP ACTOR TABLE.npz OUT   (bake the residual table into ACTOR's phase weights + official evaluate_policy_component, mresH)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
source /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamC30_compose/h13b/code/env.sh
eval S=\$S$1; M=$(ls -d $S/components/episode_*/meshes/episode_*)
cd $H13/tmp; $PY -B -u $H13/code/mresH.py save --ep $1 --actor $2 --meshes $M --table $3 --out $4 2>&1 | grep -v -E "Warn|warn\("
