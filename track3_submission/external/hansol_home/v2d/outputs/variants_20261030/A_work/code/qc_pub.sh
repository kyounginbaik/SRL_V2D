#!/bin/bash
# usage: qc_pub.sh EXPORT_TAG... ; team qc.py (unchanged, no reference data) on our public/held-out mirror exports,
# to calibrate how much image-evidence QC moves for a given public score change.
W=/home/hansol/v2d/outputs/variants_20261030/A_work
CAN=/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONNOUSERSITE=1
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages
PY=/home/hansol/v2d/.venvs/v2d-host/bin/python
for t in "$@"; do ls -d $W/pub/*/episode_*/exports/$t; done | xargs -P 4 -I{} sh -c 'e=$(dirname $(dirname {})); r=$0/$(basename $(dirname $e))/$(basename $e); [ -e {}/qc.json ] || $1 -B /data1/SRL_V2D/tools/reconstruction_audit/qc.py $r {} > {}/qc.log 2>&1; echo "QCDONE {}"' $CAN $PY
echo ALLDONE
