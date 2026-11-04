#!/bin/bash
# K33-HG clean ablation package: spec_v33a_no_ep33 + "33": donor -> verify only ep33 rows differ from v33a_pre_final (sha 3c3bd95c) -> dry-run (never submits).
# usage: hgpost2.sh V(cwHGAfix) NAME(HGA_c8) OUTNAME(v33a_ep33HGA_nospin)
set -euo pipefail
C22=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028; S=$C22/streamK33_ep33; R=/data1/SRL_V2D/challenge_submissions_SRL
PY=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
V=$1; N=$2; OUT=$3; TAG=K33_K33_${V}_$N; DON=$R/ep33_fix_20261102/K33_${V}_$N/all_20_episodes
[ -d $DON ] || (cd /tmp && $PY -B $S/code/k33splice.py $TAG $DON K33_${V}_$N)
SP=$S/cw/spec_$OUT.json
python3 -I -c "
import json;s=json.load(open('$C22/streamALL20K/out/spec_v33a_no_ep33.json'));assert '33' not in s;s['33']='$DON';json.dump(s,open('$SP','w'),indent=1)"
REF=$R/track3_v33a_pre_final_20261102/all_20_episodes; PKG=$R/ablation_ep33only_20261103/$OUT/all_20_episodes
cd /tmp && $PY -B $C22/streamALL20K/code/v32build.py $SP $PKG $OUT
python3 -I -c "
import pandas as pd
a=pd.read_parquet('$PKG/track3_submission.parquet').set_index('row_id'); b=pd.read_parquet('$REF/track3_submission.parquet').set_index('row_id')
d=(a!=b)&~(a.isna()&b.isna()); r=d.any(axis=1); eps=sorted(set(i.split('/')[1] for i in a.index[r]))
print('VERIFY rows differ',int(r.sum()),'eps',eps,'cols',list(d.columns[d.any()])); assert eps==['e000033'], eps"
cd $PKG/submission_kit && $R/.venv/bin/python submit/v2d_submit.py ../track3_submission.parquet --track track_3 -m dry --dry-run 2>&1 | grep -E "looks valid|nothing submitted|rror"
sha256sum $PKG/track3_submission.parquet | cut -c1-16
