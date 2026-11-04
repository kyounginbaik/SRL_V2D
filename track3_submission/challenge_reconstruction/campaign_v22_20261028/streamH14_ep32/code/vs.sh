#!/bin/bash
# usage: vs.sh TAG -> build-v2 exactness, archive, meshes cmp vs H12_ep32_e1f0 component, score_any vs target_final_v12b_g15 (scores/cand/TAG) and vs target_final_v13_g16 if it exists (scores/cand/TAG_g16)
C22=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028; B=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates
T=$1; NM=episode_000032; D=$B/$T/$NM; M=$C22/streamH12_ep32/staging/ep32_e1f0/components/episode_000032/meshes/$NM
X=$(python3 -c "import json;print(json.load(open('$D/completed.json')).get('exact_previous_physical_states_actions'))" 2>/dev/null || echo missing)
LC=$(python3 -c "import json;print(json.load(open('$D/completed.json'))['legacy_component'])" 2>/dev/null)
AR=$(ls $(dirname $(dirname $LC))/portable_archives/${NM}_portable.tar.gz 2>/dev/null | head -1)
MS=ok; for f in $M/*.obj; do cmp -s $f $D/meshes/$NM/$(basename $f) || MS="DIFF:$(basename $f)"; done; [ $(ls $D/meshes/$NM | wc -l) = $(ls $M | wc -l) ] || MS=COUNT_DIFF
for TG in target_final_v12b_g15 target_final_v13_g16; do
  [ -d $C22/$TG ] || continue; TT=$T; [ $TG = target_final_v13_g16 ] && TT=${T}_g16
  $C22/code/score_any.sh 32 $D $TT $C22/$TG >/dev/null 2>&1
  python3 -c "import json;d=json.load(open('$C22/scores/cand/$TT/score.json'))['32'];o=d['objects'];s=d['scores'];print('VS $TG $T exact=$X meshes=$MS archive=${AR:+yes}', ' '.join(f'{k} {v[0]:.3f}/{v[1]:.2f}' for k,v in o.items()), f\"MPPE {s['MPPE']:.3f} AUC {s['AUC']:.4f} MP-SR {s['MP-SR']} SP-SR {s.get('SP-SR')}\")"
done
