#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:50:00
#SBATCH --job-name=A-ilc
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamA_ep35/logs/ilc-%j.log
# usage: sbatch ilc.sbatch EP SRC OUT SIDE OBJECT T0 T1 TCLEAR GP SIGMA ITERS   (streamK ilc_axis.py, position-only, compose episode dir)
source /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamA_ep35/code/env.sh
EPD=$ST/ep$1/world/episode_compose/$(printf episode_%06d $1)
cd /tmp && $PY -B -u $C/streamK_ep35/code/ilc_axis.py --source $2 --out $3 --side $4 --object $5 --t0 $6 --t1 $7 --t-clear $8 --sigma-s ${10} --episode-dir $EPD --iterations ${11} --gain-rot 0 --gain-pos $9 --rot-axis none
