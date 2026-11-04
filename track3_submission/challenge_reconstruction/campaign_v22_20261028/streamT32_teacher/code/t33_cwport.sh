#!/bin/bash
# port the coworker's firmer-grip teacher (FN_base, world R33c18_W1) into our ep33 world W_RP33b (rg0 and rg1), then launch fcem3 two-hand CEMs from both bases
set -euo pipefail
C22=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028; T=$C22/streamT32_teacher; DX=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; PY=$DX/.venv/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$DX/code:/data1/SRL_V2D/third_party/DexAgent
W=$T/ep33/W_RP33b; NEW=$W/episode_corrected/episode_000033; SC=$W/scene/scene_physics.xml; OLD=/data1/hansol/teacher33/worlds/R33c18_W1/ep33/episode_corrected/episode_000033
SEED=/data1/hansol/teacher33/teacher/FN_base/reference.npz; [ -f $SEED ] || SEED=$T/ep33/cw_FN_base/reference.npz
cd /tmp; for rg in 0 1; do B=$T/ep33/port/cwFN_rg${rg}_b0.5; [ -f $B/result.json ] || $PY -B -u $T/code/delta_port.py $SEED $B $SC $OLD $NEW --blend 0.5 --right-gain $rg 2>&1 | grep -vi warn | tail -n 1 | cut -c1-200; done
FS=$T/ep33/fcem2_fpb/execution/scene_physics.xml; : > $T/ep33/fcem3_33.list
for rg in 0 1; do B=$T/ep33/port/cwFN_rg${rg}_b0.5
 echo "33 $B $NEW $T/ep33/fcem3_33_cwrg${rg}_off15 --t-lock 3.6 --t-off 15.2 --ramp 0.15 --knots 10 --samples 128 --iterations 60 --workers 48 --seed 7$rg --scene $FS" >> $T/ep33/fcem3_33.list
 echo "33 $B $NEW $T/ep33/fcem3_33_cwrg${rg}_off17 --t-on 2.5 --t-lock 3.6 --t-off 17.0 --ramp 0.15 --knots 12 --samples 128 --iterations 60 --workers 48 --seed 8$rg --scene $FS" >> $T/ep33/fcem3_33.list
done
J=$(sbatch --parsable --export=ALL,FC=fcem3.py --array=0-3 --nodelist=node01 -c 48 --time=03:00:00 -J T33-fcem3 -o $T/logs/fcem3_33_%a.log $T/code/fcem.sbatch $T/ep33/fcem3_33.list); echo "fcem3_33 array $J"
