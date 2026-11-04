#!/bin/bash
# usage: vs.sh EP TAG -> build-v2 exactness, archive, meshes cmp vs the v27 component, score_any vs target_final_v12b_g15 (scores/cand/TAG)
source /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamC30_compose/h13a/code/env.sh
B=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates
E=$1; T=$2; NM=$(printf episode_%06d $E); D=$B/$T/$NM; eval M=\$M$E
X=$(python3 -c "import json;print(json.load(open('$D/completed.json')).get('exact_previous_physical_states_actions'))" 2>/dev/null || echo missing)
LC=$(python3 -c "import json;print(json.load(open('$D/completed.json'))['legacy_component'])" 2>/dev/null)
AR=$(ls $(dirname $(dirname $LC))/portable_archives/${NM}_portable.tar.gz 2>/dev/null | head -1)
MS=ok; for f in $M/*.obj; do cmp -s $f $D/meshes/$NM/$(basename $f) || MS="DIFF:$(basename $f)"; done; [ $(ls $D/meshes/$NM | wc -l) = $(ls $M | wc -l) ] || MS=COUNT_DIFF
$C22/code/score_any.sh $E $D $T $C30_TGT >/dev/null 2>&1
python3 -c "import json;d=json.load(open('$C22/scores/cand/$T/score.json'))['$E'];o=d['objects'];s=d['scores'];print('VS $T exact=$X meshes=$MS archive=${AR:+yes}', ' '.join(f'{k} {v[0]:.3f}/{v[1]:.2f}' for k,v in o.items()), f\"MPPE {s['MPPE']:.3f} AUC {s['AUC']:.4f} MP-SR {s['MP-SR']} SP-SR {s.get('SP-SR')}\")"
