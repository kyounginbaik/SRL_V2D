#!/bin/bash
# K33-recon3 control: same full-circle spoon scan on PASSING spoon episodes (every 5th frame) - if the depth cue prefers ~0 there, it is a
# valid detector; if it prefers ~-160 there too, it is a mesh/renderer bias.
PY=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
for ep in 36 5 37 3 6 15; do
  N=$($PY -c "import numpy as np; print(len(np.load('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/selected20_final_v14_g17/episode_0000$(printf %02d $ep)/object_poses.npz')['frame_indices']))")
  FRAMES=$(seq -s, 0 5 $((N-1))) $PY -B -u /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK33_ep33/recon3/code/spoon360.py /data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/selected20_final_v14_g17/episode_0000$(printf %02d $ep) /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK33_ep33/recon3/out/ctrl_s360_ep$ep.npz 10 $ep
done
FRAMES=$(seq -s, 0 5 420) $PY -B -u /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK33_ep33/recon3/code/spoon360.py /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK35_ep35/bundles/K35_H2joint/episode_000035 /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK33_ep33/recon3/out/ctrl_s360_ep35K35.npz 10 35
