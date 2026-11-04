#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --job-name=G18v29-v2
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG18v29/logs/v2-%j.log
# usage: sbatch v2.sbatch EP COMPONENT_DIR TAG -> build-v2 (common_v2_candidates/run_one.sbatch body)
set -euo pipefail
B=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates
EP=$1; COMP=$2; TAG=$3; NAME=$(printf episode_%06d $EP)
[ -e $B/$TAG ] && { echo "exists $B/$TAG"; exit 1; }
bash $B/run_one.sbatch $EP $COMP $TAG
python3 -c "import json;d=json.load(open('$B/$TAG/$NAME/completed.json'));print('EXACT',d['exact_previous_physical_states_actions'],'SPOON',d['meshes']['wooden_spoon'])"
