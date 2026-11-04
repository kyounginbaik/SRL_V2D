#!/bin/bash
# usage: ilc_chain.sh SRC OUT GR GP NITER_POT NITER_CUP [cup t1] -> pot ILC (left) then cup ILC (right, symmetric axis)
set -euo pipefail
U=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamU_C8; PY=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
EPD=$U/worldC2/world/episodes/episode_000008; SRC=$1; OUT=$2; GR=$3; GP=$4; NP=$5; NC=$6; CT1=${7:-9.4}
if [ "$NP" -gt 0 ]; then $PY -B -u $U/code/ilc_carry_sym.py --source $SRC --out $OUT/pot --side left --object white_pot --t0 5.0 --t1 8.0 --t-clear 9.65 --episode-dir $EPD --iterations $NP --gain-rot $GR --gain-pos $GP --fresh; SRC=$OUT/pot/iter$((NP-1)); fi
$PY -B -u $U/code/ilc_carry_sym.py --source $SRC --out $OUT/cup --side right --object beige_cup --t0 2.2 --t1 $CT1 --t-clear 10.8 --episode-dir $EPD --iterations $NC --gain-rot $GR --gain-pos $GP --sym-axis 0.00668,-0.01054,0.99992 --fresh
