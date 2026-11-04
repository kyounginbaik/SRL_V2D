#!/bin/bash
# runs on the login node (no compute): wait for tgt_e3s ep3 (written atomically via rename), then launch the ep3 chain on it
R=/data1/hansol/spoon3537/rt; D=/data1/hansol/spoon3537/targets/tgt_e3s/episode_000003
until [ -f $D/wooden_spoon_trajectory.npz ] && [ -f $D/white_pot_trajectory.npz ]; do sleep 20; done
sleep 30; echo "tgt_e3s ready $(date +%T): $(ls $D | tr '\n' ' ')"
I=""; [ -f $R/runs/ep3/fix/final_0.npz ] && [ -f $R/runs/ep3/fix/beam_w11.npz ] && I=$R/runs/ep3/fix/beam_w11.npz
echo "init: ${I:-none (FINAL b2 table)}"
rm -rf $R/runs/ep3/e3s
INIT=$I PKGLABEL=track3_FINALc_ep3recon_e3s OW='{"wooden_spoon":2}' WSTEP=0.25 WK=16 TL=01:00:00 bash /data1/hansol/spoon3537/code/s3run.sh 3 e3s tgt_e3s 5.25:11.25 | sed 's/windows=[^ ]* //'
