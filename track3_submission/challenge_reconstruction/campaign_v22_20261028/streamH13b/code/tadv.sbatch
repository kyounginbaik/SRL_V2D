#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=00:50:00
#SBATCH --job-name=H13b-tadv
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamH13b/logs/tadv-%j.log
# usage: sbatch tadv.sbatch EP OUT [SHIFTS]    or   sbatch tadv.sbatch EP OUT save SHIFT
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
source /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamH13b/code/env.sh
E=$1; eval A=\$A$E; eval S=\$S$E; M=$(ls -d $S/components/episode_*/meshes/episode_*)
cd $H13/tmp
if [ "${3:-}" = save ]; then $PY -B -u $H13/code/tadv.py --ep $E --actor ${ACTOR:-$A} --meshes $M --out $2 --save $4 2>&1 | grep -v -E "Warn|warn\("
else $PY -B -u $H13/code/tadv.py --ep $E --actor ${ACTOR:-$A} --meshes $M --out $2 --workers 12 --shifts ${3:-0,0.5,0.75,0.9,1.0,1.1,1.25,1.5,1.75,2.0,2.5,3.0} 2>&1 | grep -v -E "Warn|warn\("; fi
