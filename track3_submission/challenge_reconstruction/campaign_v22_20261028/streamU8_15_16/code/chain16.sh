#!/bin/bash
# U8 ep16: step 1 = one forced rotation ILC step (D=8 weighting so it is accepted), step 2 = pos/full ILC with goal hinge in the same window
U=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamU8_15_16; source $U/code/eps.sh
NAME=$1 T0=$2 T1=$3 MODE=$4 AL=$5 SG=$6 TB=$7
PY=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
CAPL='{"plastic_dish_rack":[0.45,2.6],"wooden_spoon":[1.6,16]}'
CAP='{"plastic_dish_rack":[0.372,2.36],"wooden_spoon":[1.30,15.1]}'
$PY -B -u $U/code/mres.py ilc --ep 16 --actor $A16 --meshes $M16 --table $TB --out $U/runs/ep16/${NAME}_r --t0 $T0 --t1 $T1 --cap "$CAPL" --P 1.3 --D 8 --alphas $AL --sigmas $SG --ilc-modes $MODE --iterations 1 --workers 30 --no-save
$PY -B -u $U/code/mres.py ilc --ep 16 --actor $A16 --meshes $M16 --table $U/runs/ep16/${NAME}_r/best_table.npz --out $U/runs/ep16/${NAME}_p --t0 $T0 --t1 $T1 --cap "$CAP" --P 1.3 --D 15 --goal 1.25,14.5 --alphas 0.15,0.3,0.5,0.8,1.0 --sigmas 0.1,0.3 --ilc-modes pos,full --iterations 8 --workers 30
