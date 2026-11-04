set -e
CAN=/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical
W=/home/hansol/v2d/outputs/variants_20261030/CD_work
cp /home/hansol/v2d/outputs/variants_20261030/A_work/code/score.sh $W/code/score.sh
for se in public/episode_000021 public_heldout/episode_000022; do
  M=$W/pub/$se; mkdir -p $M/exports
  ln -sfn $CAN/$se/prepared.json $M/prepared.json; ln -sfn $CAN/$se/objects $M/objects
  for t in r8_base g15_camrt; do
    mkdir -p $M/exports/$t; cp $CAN/$se/exports/$t/object_poses.npz $CAN/$se/exports/$t/meshes.json $M/exports/$t/
  done
done
cd $W
nohup bash -c 'for se in public/episode_000021 public_heldout/episode_000022; do for t in r8_base g15_camrt; do bash code/score.sh pub/$se/exports/$t; done; done; echo DONE' > $W/logs_rescore.txt 2>&1 &
echo started
