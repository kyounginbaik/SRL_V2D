#!/bin/bash
# H9: save a beam table on the ported actor (mres.py save, official evaluator in W7). usage: save_chain.sh TABLE SAVE_OUT
set -e; H=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamH9_ep32v7; PY=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
M=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamB9_ep32/staging/ep32_s6f0/components/episode_000032/meshes/episode_000032
$PY -B -u $H/code/mres.py save --ep 32 --actor $H/runs/port_s6f0 --meshes $M --out $2 --table $1
