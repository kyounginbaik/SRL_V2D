#!/bin/bash
# B9: port (scene binding + report scene) then save the actor with a residual table. usage: save_chain.sh SRC_ACTOR PORT_OUT TABLE SAVE_OUT
set -e; B=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamB9_ep32; PY=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
M=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamV8_ep32/staging/ep32_wb4r2/components/episode_000032/meshes/episode_000032
[ -d $2 ] || $PY -B -u $B/code/port.py $1 $B/W6 $2 $M
$PY -B -u $B/code/mres.py save --ep 32 --actor $2 --meshes $M --out $4 --table $3
