#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:50:00
#SBATCH --job-name=F19-distill
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamF_margins/ep19/logs/F19-distill-%j.log
# usage: sbatch distill.sbatch TEACHER OUT EPISODE_DIR [SEED] GRID...   (streamC_rigid_ilc/tools/actor_grid_c.py, unchanged)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
T=$1; O=$2; E=$3; S=$4; shift 4
cd /tmp && /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B -u /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamC_rigid_ilc/tools/actor_grid_c.py --episode 19 --teacher $T --output $O --episode-dir $E --workers 4 --seed $S --grid "$@"
