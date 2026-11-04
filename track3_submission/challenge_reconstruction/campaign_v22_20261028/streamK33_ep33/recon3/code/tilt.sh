#!/bin/bash
PY=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
for AX in mid short; do
  AXIS=$AX FRAMES=$(seq -s, 0 2 347) $PY -B -u /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK33_ep33/recon3/code/spoontilt.py /data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/selected20_final_v14_g17/episode_000033 /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK33_ep33/recon3/out/tilt_${AX}_ep33.npz 2 33
  for ep in 36 5 37 6; do N=$($PY -c "import numpy as np; print(len(np.load('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/selected20_final_v14_g17/episode_0000$(printf %02d $ep)/object_poses.npz')['frame_indices']))")
    AXIS=$AX FRAMES=$(seq -s, 0 6 $((N-1))) $PY -B -u /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK33_ep33/recon3/code/spoontilt.py /data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/selected20_final_v14_g17/episode_0000$(printf %02d $ep) /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK33_ep33/recon3/out/tilt_${AX}_ep$ep.npz 2 $ep; done
done
