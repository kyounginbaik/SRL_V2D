#!/bin/bash
# usage: verify.sh EP TAG  -> exactness (build-v2 completed.json), score_any vs target_final_v6 (label = TAG), submitted meshes cmp vs the
# current component's build-v2 meshes, portable archive at legacy_component.parents[1]/portable_archives, comparison vs the current scores.
C22=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028; CC=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates
E=$1; TAG=$2; NM=$(printf episode_%06d $E)
declare -A CUR=([8]=N8_ep08_klr512 [20]=N8_ep20_k3c1024 [28]=J8_ep28_w2k3c2048 [38]=J8_ep38_ilc36c2048 [30]=A9_ep30_sweeper)
declare -A CURS=([8]=N8_ep08_klr512 [20]=N8_ep20_k3c1024 [28]=J8_ep28_w2k3c2048 [38]=J8_ep38_ilc36c2048 [30]=A9v6_A9_ep30_sweeper)
D=$CC/$TAG/$NM
EX=$(python3 -c "import json;print(json.load(open('$D/completed.json'))['exact_previous_physical_states_actions'])")
LC=$(python3 -c "import json;print(json.load(open('$D/completed.json'))['legacy_component'])")
AR=$(ls $(dirname $(dirname $LC))/portable_archives/*.tar.gz 2>/dev/null | head -1)
M=ok; for f in $CC/${CUR[$E]}/$NM/meshes/$NM/*.obj; do cmp -s $f $D/meshes/$NM/$(basename $f) || M="DIFF:$(basename $f)"; done
[ $(ls $D/meshes/$NM | wc -l) = $(ls $CC/${CUR[$E]}/$NM/meshes/$NM | wc -l) ] || M="COUNT_DIFF"
$C22/code/score_any.sh $E $D $TAG $C22/target_final_v6 > /dev/null
python3 - $E $TAG ${CURS[$E]} "$EX" "$M" "$AR" <<'PY'
import json, sys
e, tag, cur, ex, m, ar = sys.argv[1:]
C = '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/scores/cand/'
n = json.load(open(C + tag + '/score.json'))[e]; o = json.load(open(C + cur + '/score.json'))[e]
worse = [k for k in o['objects'] if n['objects'][k][0] > o['objects'][k][0] or n['objects'][k][1] > o['objects'][k][1]]
if n['scores']['MPPE'] > o['scores']['MPPE']: worse.append('MPPE')
if n['scores']['AUC'] < o['scores']['AUC']: worse.append('AUC')
f = lambda s: ', '.join(f"{k} {v[0]:.2f}/{v[1]:.1f}" for k, v in s['objects'].items()) + f"; MP-SR {s['scores']['MP-SR']:.0f} MPPE {s['scores']['MPPE']:.3f} AUC {s['scores']['AUC']:.3f}"
print(f"{tag} exact={ex} meshes_vs_{cur.replace('A9v6_','')}={m} archive={'yes' if ar else 'MISSING'} | {f(n)} | current {f(o)} | worse_than_current={worse or 'none'}")
PY
