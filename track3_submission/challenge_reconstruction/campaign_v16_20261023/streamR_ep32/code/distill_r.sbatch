#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:50:00
#SBATCH --job-name=R32-distill
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamR_ep32/logs/distill-%j.log
# env: TEACHER OUT EPD [GRID] [SEED=44]
source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamJ_ep3233/code/common.sh
: "${TEACHER:?}" "${OUT:?}" "${EPD:?}"; cd /tmp
$PY -B -u /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamC_rigid_ilc/tools/actor_grid_c.py --episode 32 --teacher $TEACHER --output $OUT --episode-dir $EPD --workers 8 --seed ${SEED:-44} ${GRID:+--grid $GRID}
