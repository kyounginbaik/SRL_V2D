#!/bin/bash
# usage: mk_w.sh EP TAG DIOU MINLEN
set -e
E=$1; TAG=$2; R=$(python3 -c "print('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical/'+('public' if $E in [11,12,18,21,23,24,31,39,41,44] else 'public_heldout')+'/episode_%06d'%$E)")
X=$R/exports/g15_cam; P=$R/exports/${TAG}_pre
mkdir $P; for f in $X/*; do b=$(basename $f); [ "$b" = object_poses.npz ] || [ "$b" = public_score.json ] || cp -p $f $P/; done
./py.sh winswap.py public $E $X/object_poses.npz $X/meshes.json $P/object_poses.npz --diou $3 --minlen $4
/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG15/code/py.sh retime.py $P $R/exports/$TAG public_$E
cd /tmp && OMP_NUM_THREADS=2 /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B /data1/SRL_V2D/tools/reconstruction_audit/score_bundle.py $R/exports/$TAG --metrics
