#!/bin/bash
# ANCHOR post: after svst of A20_ep<EP>_<NAME> -> donor (a20splice: v29 + EP rows) -> local scores of EP vs our corrected target V and vs tgt_v32
# -> full package = base track3_v33r16c3_ep32r10 spec with EP replaced (v32build) -> verify only EP rows differ -> dry-run.
# usage: ancpost.sh V EP NAME
set -euo pipefail
C22=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028; S=$C22/streamALL20K; A=$C22/streamANCHOR; R=/data1/SRL_V2D/challenge_submissions_SRL
PY=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
V=$1; EP=$2; N=$3; TAG=A20_ep${EP}_$N
CAND=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates/$TAG/$(printf episode_%06d $EP)
python3 -I -c "import json;c=json.load(open('$CAND/completed.json'));print('exact',c.get('exact_previous_physical_states_actions'));assert c.get('exact_previous_physical_states_actions')"
DON=$R/anchor_20261103/donors/$TAG/all_20_episodes
[ -d $DON ] || (cd /tmp && $PY -B $S/code/a20splice.py $EP $TAG $DON $TAG)
for T in $V tgt_v32; do
  [ -f $A/out/score_${TAG}_$T.json ] || (cd /tmp && $PY -B $S/code/score_all.py $DON/track3_submission.parquet $DON/results $S/targets/$T $A/out/score_${TAG}_$T.json $EP >/dev/null 2>&1)
  python3 -I -c "import json;d=json.load(open('$A/out/score_${TAG}_$T.json'))['$EP'];print('vs $T',d['objects'],{k:round(v,4) for k,v in d['scores'].items()})"
done
BASE=$R/track3_v33r16c3_ep32r10_20261102/all_20_episodes
SP=$A/out/spec_${TAG}.json
python3 -I -c "
import json;d=json.load(open('$BASE/V32_BUILD.json'));s={k:v['donor'] for k,v in d['episodes_replaced'].items()};s['$EP']='$DON';json.dump(s,open('$SP','w'),indent=1)"
PKG=$R/anchor_20261103/ep${EP}_${N}/all_20_episodes
[ -d $PKG ] || (cd /tmp && $PY -B $S/code/v32build.py $SP $PKG anc_ep${EP}_$N >/dev/null)
python3 -I -c "
import pandas as pd
a=pd.read_parquet('$PKG/track3_submission.parquet').set_index('row_id'); b=pd.read_parquet('$BASE/track3_submission.parquet').set_index('row_id')
d=(a!=b)&~(a.isna()&b.isna()); r=d.any(axis=1); eps=sorted(set(i.split('/')[1] for i in a.index[r]))
print('VERIFY rows differ',int(r.sum()),'eps',eps,'cols',list(d.columns[d.any()])); assert eps==['e%06d'%$EP], eps"
cd $PKG/submission_kit && $R/.venv/bin/python submit/v2d_submit.py ../track3_submission.parquet --track track_3 -m dry --dry-run 2>&1 | grep -E "looks valid|nothing submitted|rror"
echo "PKG $PKG sha $(sha256sum $PKG/track3_submission.parquet | cut -c1-8)"
