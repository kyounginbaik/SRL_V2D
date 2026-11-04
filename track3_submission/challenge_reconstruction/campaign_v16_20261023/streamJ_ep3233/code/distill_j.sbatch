#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:50:00
#SBATCH --job-name=J-distill
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamJ_ep3233/logs/distill-%j.log
# env: EP TEACHER OUT EPD [GRID="c,p,v ..."] [SEED=44]
source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamJ_ep3233/code/common.sh
: "${EP:?}" "${TEACHER:?}" "${OUT:?}" "${EPD:?}"
cd /tmp; $PY -B -u /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamC_rigid_ilc/tools/actor_grid_c.py --episode $EP --teacher $TEACHER --output $OUT --episode-dir $EPD --workers 8 --seed ${SEED:-44} ${GRID:+--grid $GRID}
