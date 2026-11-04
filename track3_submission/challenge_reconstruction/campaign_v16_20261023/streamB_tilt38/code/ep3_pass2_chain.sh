#!/bin/bash
set -uo pipefail
B=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamB_tilt38; R0=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; PY=$R0/.venv/bin/python
RET=$B/ep3/retention/mid_a_s71_pass2_s73; SPEC=$B/ep3/power_grasp/search/episode_000003/mid_a/result.json; E3=$B/ep3/calibrated_inputs/episodes/episode_000003
j=$(sbatch --parsable --export=ALL,SOURCE=$RET,SPEC=$SPEC,OUTROOT=$B/ep3/release_grid/pass2_s73 $B/code/ep3_release_grid.sbatch); echo "release grid $j"
until ! squeue -h -j $j -o %T | grep -q .; do sleep 30; done
best=$($PY - <<PY
import json,glob
rows=[]
for f in glob.glob('$B/ep3/release_grid/pass2_s73/*/result.json'):
    r=json.load(open(f)); o=r['metrics']['official_style_reference_proxy']['objects']
    if not r['execution']['execution_completed']: continue
    rows.append((max(o['white_pot']['mean_position_error_cm'][0]/3,o['white_pot']['mean_orientation_error_degrees'][0]/30,o['wooden_spoon']['mean_position_error_cm'][0]/3,o['wooden_spoon']['mean_orientation_error_degrees'][0]/30), f.rsplit('/result.json',1)[0]))
rows.sort(); print(' '.join(p for _,p in rows[:2]))
PY
)
echo "best release teachers: $best"; jobs=""
for t in $best; do jj=$(sbatch --parsable --export=ALL,EP=3,TEACHER=$t,EPDIR=$E3,OUT=$B/ep3/distilled/pass2_$(basename $t) $B/code/distill.sbatch); jobs="$jobs $jj"; done
for jj in $jobs; do until ! squeue -h -j $jj -o %T | grep -q .; do sleep 30; done; done
cand=$($PY - <<PY
import json,glob
rows=[]
for f in glob.glob('$B/ep3/distilled/pass2_*/attempts.json'):
    for r in json.load(open(f)):
        if r.get('passed'): rows.append((r['maximum_ratio'], r['checkpoint'].rsplit('/training/policy.npz',1)[0], r['object_errors']))
rows.sort(); print(json.dumps(rows[:3]))
print('BEST', rows[0][1] if rows else 'NONE', rows[0][0] if rows else 99)
PY
)
echo "$cand"
bestdir=$(echo "$cand" | grep ^BEST | cut -d' ' -f2); ratio=$(echo "$cand" | grep ^BEST | cut -d' ' -f3)
if [ "$bestdir" != "NONE" ] && $PY -c "import sys; sys.exit(0 if float('$ratio') < 0.689 else 1)"; then
  s=$(sbatch --parsable --export=ALL,EP=3,DISTILLED=$bestdir $(sed "s#ep\$EP/staging#ep\$EP/staging_v2#" $B/code/stage.sbatch > $B/code/stage_v2.sbatch; echo $B/code/stage_v2.sbatch)); echo "staging_v2 job $s"
  until ! squeue -h -j $s -o %T | grep -q .; do sleep 30; done
  ls $B/ep3/staging_v2/components/episode_000003/staged.json && echo "STAGED_V2 $bestdir ratio $ratio" || echo "STAGE_V2_FAILED"
else echo "NO_IMPROVEMENT (best ratio $ratio vs staged 0.689)"; fi
echo CHAIN_DONE
