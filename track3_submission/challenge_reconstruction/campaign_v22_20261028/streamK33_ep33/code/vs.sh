#!/bin/bash
# usage: vs.sh EP TAG V -> build-v2 exactness, archive, meshes cmp vs the v29 component, score_any vs target_E14_HGA, target_E14_HGB, target_final_v14_g17 (scores/cand/TAG[_A|_B|_v14])
C22=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028; B=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates
E=$1; T=$2; V=$3; NM=$(printf episode_%06d $E); D=$B/$T/$NM
M=$B/$(python3 -c "import json;print(json.load(open('$C22/selection_v29.json'))['$E'])")/$NM/meshes/$NM
X=$(python3 -c "import json;print(json.load(open('$D/completed.json')).get('exact_previous_physical_states_actions'))" 2>/dev/null || echo missing)
LC=$(python3 -c "import json;print(json.load(open('$D/completed.json'))['legacy_component'])" 2>/dev/null)
AR=$(ls $(dirname $(dirname $LC))/portable_archives/${NM}_portable.tar.gz 2>/dev/null | head -1)
MS=ok; for f in $M/*.obj; do cmp -s $f $D/meshes/$NM/$(basename $f) || MS="DIFF:$(basename $f)"; done; [ $(ls $D/meshes/$NM | wc -l) = $(ls $M | wc -l) ] || MS=COUNT_DIFF
for TG in $V:streamK33_ep33/targets/$V H4:streamK33_ep33/targets/K33_H4settle H3:streamK33_ep33/targets/K33_H3img v14:target_final_v14_g17; do
  K=${TG%%:*}; TD=${TG#*:}; TT=${T}_$K
  $C22/code/score_any.sh $E $D $TT $C22/$TD >/dev/null 2>&1
  python3 -c "import json;d=json.load(open('$C22/scores/cand/$TT/score.json'))['$E'];o=d['objects'];s=d['scores'];print('VS $TD $T exact=$X meshes=$MS archive=${AR:+yes}', ' '.join(f'{k} {v[0]:.3f}/{v[1]:.2f}' for k,v in o.items()), f\"MPPE {s['MPPE']:.3f} AUC {s['AUC']:.4f} MP-SR {s['MP-SR']} SP-SR {s.get('SP-SR')}\")"
done
