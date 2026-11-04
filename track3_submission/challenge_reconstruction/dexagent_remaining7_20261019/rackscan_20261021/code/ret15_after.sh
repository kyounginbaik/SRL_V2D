#!/bin/bash
W=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021; n=episode_000015
python3 -c "import json;print(json.load(open('$W/retarget/$n/result.json'))['reference'])" | xargs -I{} cp {} $W/retarget/$n/reference.npz
for spec in "15 2.8 4.6 0.5 91" "15 3.0 5.5 0.5 92" "15 3.2 4.4 0.75 93" "15 3.0 8.0 0.75 94"; do set -- $spec; EP=$1 SRC=$W/retarget/$n OUT=$W/search/$n/ret_${2}_${3}_s$4 T0=$2 T1=$3 SCALE=$4 SEED=$5 sbatch --export=ALL --job-name=rr$1_$2 $W/code/retention.sbatch; done
