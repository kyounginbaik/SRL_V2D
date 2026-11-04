#!/bin/bash
# usage: qc_tag.sh TAG [EPISODES...]; GT-free eval QC (team qc.py unchanged + ballqc.py) of A_work/eval/TAG/episode_*.
# TAG=base copies v8_g10b poses first. 4 episodes at a time.
W=/home/hansol/v2d/outputs/variants_20261030/A_work
RR=/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026
B=$RR/selected20_final_v8_g10b
TAG=$1; shift
EPS=${@:-1 3 5 6 8 10 14 15 16 19 20 25 28 30 33 35 36 37 38}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONNOUSERSITE=1
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages
PY=/home/hansol/v2d/.venvs/v2d-host/bin/python
for e in $EPS; do
  n=$(printf episode_%06d $e); f=$W/eval/$TAG/$n; mkdir -p $f
  [ "$TAG" = base ] && [ ! -e $f/object_poses.npz ] && cp -p $B/$n/object_poses.npz $f/
  $PY -c "import json,sys;m=json.load(open('$B/$n/meshes.json'));json.dump({k:'$B/$n/'+v['file'] for k,v in m.items()},open('$f/meshes.json','w'),indent=1)"
  echo $f
done | xargs -P 4 -I{} sh -c 'r=$0/$(basename {}); [ -e {}/qc.json ] || $1 -B /data1/SRL_V2D/tools/reconstruction_audit/qc.py $r {} > {}/qc.log 2>&1; $1 -B $2/code/ballqc.py $r {} >> {}/qc.log 2>&1; echo "QCDONE {}"' $RR/canonical/evaluation $PY $W
echo ALLDONE
