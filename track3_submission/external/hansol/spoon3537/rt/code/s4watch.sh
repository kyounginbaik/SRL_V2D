#!/bin/bash
# runs on the login node (no compute): wait for tgt_s4 ep36 trajectory to exist and be unchanged for 90 s, then launch the s4 chain
R=/data1/hansol/spoon3537/rt; T=/data1/hansol/spoon3537/targets/tgt_s4/episode_000036/wooden_spoon_trajectory.npz
B37=$R/packages/track3_FINALc_ep37s1a_tgt_s1/all_20_episodes/track3_submission.parquet
until [ -f $T ] && [ -f /data1/hansol/spoon3537/targets/tgt_s4/episode_000036/wooden_spoon_mesh_metric.obj ] && [ $(( $(date +%s) - $(stat -c %Y $T) )) -gt 90 ]; do sleep 20; done
echo "tgt_s4 ready $(date +%T) sha $(sha256sum $T | cut -c1-16)"
rm -rf $R/runs/ep36/s4
INIT=$R/runs/ep36/s1d/beam_w03.npz WK=16 TL=00:50:00 BASEP=$B37 PKGLABEL=track3_FINALc_ep36s4_ep37s1a_tgt_s4 bash /data1/hansol/spoon3537/code/s3run.sh 36 s4 tgt_s4 12.5:20.25 | sed 's/windows=[^ ]* //'
