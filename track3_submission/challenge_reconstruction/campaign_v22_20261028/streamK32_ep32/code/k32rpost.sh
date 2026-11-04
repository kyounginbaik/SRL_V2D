#!/bin/bash
# K32-roll post: after svst of K32_<V>_<NAME> -> donor (k32splice) -> full package (v32build: spec_v33a_no_ep33 + "33": K33_H5r16_c3 donor
# + "32": donor) -> verify only ep32 rows differ from track3_v33a_ep33r16c3 (sha 0b3eb865) -> dry-run. usage: k32rpost.sh V(K32_R10) NAME(R10_c1) SUF(ep32r10)
set -euo pipefail
C22=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028; S=$C22/streamK32_ep32; R=/data1/SRL_V2D/challenge_submissions_SRL
PY=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
V=$1; N=$2; SUF=$3; TAG=K32_${V}_$N
DON=$R/ep32_fix_20261102/${V}_$N/all_20_episodes
[ -d $DON ] || (cd /tmp && $PY -B $S/code/k32splice.py $TAG $DON ${V}_$N)
SP=$S/out/spec_v33r16c3_$SUF.json
python3 -I -c "
import json;s=json.load(open('$C22/streamALL20K/out/spec_v33a_no_ep33.json'));assert '33' not in s and '32' not in s
s['33']='$R/ep33_fix_20261102/K33_H5r16_c3/all_20_episodes';s['32']='$DON';json.dump(s,open('$SP','w'),indent=1)"
REF=$R/track3_v33a_ep33r16c3_20261102/all_20_episodes
PKG=$R/track3_v33r16c3_${SUF}_20261102/all_20_episodes
cd /tmp && $PY -B $C22/streamALL20K/code/v32build.py $SP $PKG v33r16c3_$SUF
python3 -I -c "
import pandas as pd
a=pd.read_parquet('$PKG/track3_submission.parquet').set_index('row_id'); b=pd.read_parquet('$REF/track3_submission.parquet').set_index('row_id')
d=(a!=b)&~(a.isna()&b.isna()); r=d.any(axis=1); eps=sorted(set(i.split('/')[1] for i in a.index[r]))
print('VERIFY rows differ',int(r.sum()),'eps',eps,'cols',list(d.columns[d.any()])); assert eps==['e000032'], eps"
cd $PKG/submission_kit && $R/.venv/bin/python submit/v2d_submit.py ../track3_submission.parquet --track track_3 -m dry --dry-run 2>&1 | grep -E "looks valid|nothing submitted|rror"
sha256sum $PKG/track3_submission.parquet | cut -c1-16
