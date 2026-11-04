#!/bin/bash
# login-node watcher (no compute): wait (until 05:50) for tgt_s6 ep36 (atomic rename), then launch ep36 s6 from the FINAL b2 table; package on 8b6fa2b9
R=/data1/hansol/spoon3537/rt; D=/data1/hansol/spoon3537/targets/tgt_s6/episode_000036
until [ -f $D/wooden_spoon_trajectory.npz ]; do [ $(date +%H%M) -ge 0550 ] && { echo "tgt_s6 never appeared by 05:50; nothing launched"; exit 0; }; sleep 20; done
sleep 20; echo "tgt_s6 ready $(date +%T) sha $(sha256sum $D/wooden_spoon_trajectory.npz | cut -c1-16)"
rm -rf $R/runs/ep36/s6
PKGLABEL=track3_FINALc_ep36s6 WK=16 TL=00:50:00 bash /data1/hansol/spoon3537/code/s3run.sh 36 s6 tgt_s6 12.5:20.25 | sed 's/windows=[^ ]* //'
