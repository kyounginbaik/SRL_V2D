#!/bin/bash
# unchanged team qc.py + ballqc.py on A_work/eval14/TAG (TAG=base copies v14_g17 poses), 4 at a time
W=/home/hansol/v2d/outputs/variants_20261030/A_work
RR=/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026
B=$RR/selected20_final_v14_g17
TAG=$1
for e in 1 3 5 6 8 10 14 15 16 19 20 25 28 30 33 35 36 37 38; do
  n=$(printf episode_%06d $e); f=$W/eval14/$TAG/$n; mkdir -p $f
  [ "$TAG" = base ] && [ ! -e $f/object_poses.npz ] && cp -p $B/$n/object_poses.npz $f/
  /home/hansol/v2d/.venvs/v2d-host/bin/python -c "import json;m=json.load(open('$B/$n/meshes.json'));json.dump({k:'$B/$n/'+v['file'] for k,v in m.items()},open('$f/meshes.json','w'),indent=1)"
  echo $f
done | xargs -P 4 -I{} sh -c 'r=$0/$(basename {}); [ -e {}/qc.json ] || $1/code/genv.sh /data1/SRL_V2D/tools/reconstruction_audit/qc.py $r {} > {}/qc.log 2>&1; $1/code/genv.sh $1/code/ballqc.py $r {} >> {}/qc.log 2>&1; echo "QCDONE {}"' $RR/canonical/evaluation $W
echo ALLDONE
