#!/bin/bash
# usage: vs.sh EP TAG -> exactness + score_any vs target_final_v6 (writes scores/cand/TAG)
C=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028; B=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates
E=$1; T=$2; D=$B/$T/$(printf episode_%06d $E)
X=$(python3 -c "import json;print(json.load(open('$D/completed.json')).get('exact_previous_physical_states_actions'))" 2>/dev/null || echo missing)
$C/code/score_any.sh $E $D $T $C/target_final_v6 >/dev/null 2>&1
python3 -c "import json;d=json.load(open('$C/scores/cand/$T/score.json'))['$E'];o=d['objects'];s=d['scores'];print('$T exact=$X', ' '.join(f'{k} {v[0]:.2f}/{v[1]:.1f}' for k,v in o.items()), f\"MPPE {s['MPPE']:.3f} AUC {s['AUC']:.4f} MP-SR {s['MP-SR']}\")"
