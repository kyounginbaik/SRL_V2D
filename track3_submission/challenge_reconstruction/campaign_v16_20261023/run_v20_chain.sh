#!/bin/bash
# v20 chain: assemble -> postprocess (no ep15 mesh swap) -> compare vs v17 -> SRL folder -> knockouts. Nothing uploaded.
set -uo pipefail
C=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023
PY=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
OUT=/data1/SRL_V2D/challenge_submissions/track3_v20_20261024
SRL=/data1/SRL_V2D/challenge_submissions_SRL
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# drop --video lines whose mp4 is missing
for ep in 14 15 16 25 28 30 32 33; do n=$(printf episode_%06d $ep); if ! ls $C/videos_v20/$n/*.mp4 >/dev/null 2>&1; then echo "video missing for ep$ep, dropping"; sed -i "/--video $ep=/d" $C/assemble_v20.args; fi; done
echo "== assemble $(date +%T)"; bash $C/assemble_v20.sh > $C/logs/assemble_v20.log 2>&1 || { echo "ASSEMBLE FAILED"; tail -20 $C/logs/assemble_v20.log; exit 1; }
tail -4 $C/logs/assemble_v20.log
echo "== postprocess $(date +%T)"; cd /tmp && $PY -B -u $C/package_code/postprocess_v16.py $OUT/all_20_episodes $OUT/all_20_episodes_post --skip-ep15-mesh > $C/logs/postprocess_v20.log 2>&1 || { echo "POSTPROCESS FAILED"; tail -20 $C/logs/postprocess_v20.log; exit 1; }
grep -v FutureWarning $C/logs/postprocess_v20.log | tail -4
echo "== compare vs v17 $(date +%T)"; $PY - <<'PYEOF'
import pandas as pd, re, hashlib
a=pd.read_parquet('/data1/SRL_V2D/challenge_submissions/track3_v19_20261024/all_20_episodes_post/track3_submission.parquet')
b=pd.read_parquet('/data1/SRL_V2D/challenge_submissions/track3_v20_20261024/all_20_episodes_post/track3_submission.parquet')
print('rows', len(a), len(b), 'cols equal', list(a.columns)==list(b.columns))
ep=lambda s: s.str.extract(r'/e(\d+)/')[0].astype(int)
a['ep']=ep(a.row_id); b['ep']=ep(b.row_id)
m=a.merge(b,on='row_id',suffixes=('_a','_b'))
pose=[c for c in a.columns if c not in ('row_id','ep','object_mesh_b64') and a[c].dtype!=object]
changed=set(); meshchanged=set()
for e,g in m.groupby('ep_a'):
    for c in pose:
        if not (g[c+'_a'].astype(str).values==g[c+'_b'].astype(str).values).all(): changed.add(e); break
    if 'object_mesh_b64' in a.columns and not (g['object_mesh_b64_a'].values==g['object_mesh_b64_b'].values).all(): meshchanged.add(e)
print('episodes with changed rows:', sorted(changed)); print('episodes with changed meshes:', sorted(meshchanged))
for c in a.columns:
    if a[c].dtype==object and c not in ('row_id','object_mesh_b64'): print(c, 'uniform' if b[c].nunique()==1 else f'{b[c].nunique()} values')
print('sha256 v20 post parquet', hashlib.sha256(open('/data1/SRL_V2D/challenge_submissions/track3_v20_20261024/all_20_episodes_post/track3_submission.parquet','rb').read()).hexdigest()[:16])
PYEOF
echo "== SRL folder $(date +%T)"; cd $SRL && $PY -B -u $SRL/make_srl_folder.py $OUT/all_20_episodes_post $SRL/track3_v20_20261024/all_20_episodes $C/srl_note_v20.md > $C/logs/srl_v20.log 2>&1 || { echo "SRL FAILED"; tail -20 $C/logs/srl_v20.log; exit 1; }
tail -3 $C/logs/srl_v20.log; echo "files: $(find $SRL/track3_v20_20261024/all_20_episodes -type f | wc -l)"; echo "dexagent mentions: $(grep -ril dexagent $SRL/track3_v20_20261024/all_20_episodes | wc -l)"
mkdir -p $SRL/ablation_knockout_v20_20261024; echo "== knockouts $(date +%T)"; cd /tmp && $PY -B -u $SRL/ablation_knockout_20261020/make_knockouts.py --source $SRL/track3_v20_20261024/all_20_episodes/track3_submission.parquet --out $SRL/ablation_knockout_v20_20261024 --episodes 1 3 5 6 8 10 14 15 16 19 20 25 28 30 32 33 35 36 37 38 --group likelypass9=6,8,10,14,16,20,28,30,38 --group changed8=14,15,16,25,28,30,32,33 --group rack5=14,15,16,28,30 --group spoon10=3,5,6,15,16,32,33,35,36,37 > $C/logs/knockouts_v20.log 2>&1 || { echo "KNOCKOUTS FAILED"; tail -20 $C/logs/knockouts_v20.log; }
tail -2 $C/logs/knockouts_v20.log; ls $SRL/ablation_knockout_v20_20261024 | wc -l
echo "== done $(date +%T)"
