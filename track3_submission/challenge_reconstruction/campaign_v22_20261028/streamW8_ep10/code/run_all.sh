for d in /data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical/public/episode_* /data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical/public_heldout/episode_*; do
  [ -f $d/exports/w8_nest/object_poses.npz ] || python nest_rule.py $d $d/exports/r8_base $d/exports/w8_nest --report ../out/w8_nest_$(basename $(dirname $d))_$(basename $d).json >> ../logs/rule_all.log 2>&1
done
for e in /data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/selected20_final_v5/episode_*; do d=/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical/evaluation/$(basename $e)
  [ -f $d/exports/w8_nest/object_poses.npz ] || python nest_rule.py $d $e $d/exports/w8_nest --report ../out/w8_nest_evaluation_$(basename $d).json >> ../logs/rule_all.log 2>&1
done
echo ALLDONE >> ../logs/rule_all.log
