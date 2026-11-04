#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:50:00
#SBATCH --job-name=v19-grid
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamV_C19/logs/grid-%j.log
# usage: sbatch grid.sbatch TEACHER OUT EPISODE_DIR GRID...
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
T=$1; O=$2; E=$3; shift 3
cd /tmp && /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B -u /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamC_rigid_ilc/tools/actor_grid_c.py --episode 19 --teacher $T --output $O --episode-dir $E --workers 4 --grid "$@"
