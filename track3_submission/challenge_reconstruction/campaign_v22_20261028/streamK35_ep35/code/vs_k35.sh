#!/bin/bash
# usage: vs_k35.sh TAG HYP -> build-v2 exactness, meshes cmp vs v29 component, score_any vs K35 target HYP and target_final_v14_g17
C22=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028; B=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates
T=$1; V=$2; E=35; NM=episode_000035; D=$B/$T/$NM; M=$B/G18_ep35_H13b2/$NM/meshes/$NM
X=$(python3 -c "import json;print(json.load(open('$D/completed.json')).get('exact_previous_physical_states_actions'))" 2>/dev/null || echo missing)
MS=ok; for f in $M/*.obj; do cmp -s $f $D/meshes/$NM/$(basename $f) || MS="DIFF:$(basename $f)"; done
for TG in $V:streamK35_ep35/targets/$V v14:target_final_v14_g17; do
  K=${TG%%:*}; TD=${TG#*:}; TT=${T}_vs_$K
  $C22/code/score_any.sh $E $D $TT $C22/$TD >/dev/null 2>&1
  python3 -c "import json;d=json.load(open('$C22/scores/cand/$TT/score.json'))['$E'];o=d['objects'];s=d['scores'];print('VS $TD $T exact=$X meshes=$MS', ' '.join(f'{k} {v[0]:.3f}/{v[1]:.2f}' for k,v in o.items()), f\"MPPE {s['MPPE']:.3f} AUC {s['AUC']:.4f} MP-SR {s['MP-SR']} SP-SR {s.get('SP-SR')}\")"
done
