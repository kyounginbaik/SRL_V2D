#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node100
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --job-name=E31-cwv2
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamE31_cowork2/logs/cwv2-%A_%a.log
# usage: sbatch --array=1-N cw_v2.sbatch LIST (lines "EP TAG")  -> stage_cw.py (CWTAG = CW_<TAG>) + build-v2 + exactness / mesh / rollout-identity check
set -euo pipefail
read EP TAG < <(sed -n "${SLURM_ARRAY_TASK_ID}p" $1)
E31=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamE31_cowork2; B=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates
CT=CW_$TAG; NM=$(printf episode_%06d $EP); PY=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
[ -d $E31/staging_cw/$CT ] || $PY -B $E31/code/stage_cw.py $EP $TAG $CT
[ -f $B/$CT/$NM/completed.json ] || bash $B/run_one.sbatch $EP $E31/staging_cw/$CT/components/$NM $CT
$PY -B - <<PY
import json, numpy as np, hashlib
from pathlib import Path
B=Path('$B'); a=B/'$TAG'/'$NM'; b=B/'$CT'/'$NM'
d=json.loads((b/'completed.json').read_text()); ta=json.loads((a/'backend/evaluation/result.json').read_text())['execution']['trajectory']; tb=json.loads((b/'backend/evaluation/result.json').read_text())['execution']['trajectory']
x,y=np.load(ta,allow_pickle=False),np.load(tb,allow_pickle=False)
same=sorted(x.files)==sorted(y.files) and all(np.array_equal(x[k],y[k]) for k in x.files if k!='scene_sha256')
sha=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
cw=Path('$E31/cwmesh_v5/$NM'); ms=all(sha(p)==sha(b/'meshes'/'$NM'/p.name) for p in cw.glob('*.obj'))
print('CW $EP $TAG -> $CT exact', d['exact_previous_physical_states_actions'], 'rollout_identical', same, 'meshes_cw', ms)
PY
for TG in A:target_E14_HGA B:target_E14_HGB v14:target_final_v14_g17; do K=${TG%%:*}; TD=${TG#*:}
  /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/code/score_any.sh $EP $B/$CT/$NM ${CT}_$K /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/$TD >/dev/null 2>&1
  python3 -c "import json;d=json.load(open('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/scores/cand/${CT}_$K/score.json'))['$EP'];o=d['objects'];s=d['scores'];print('VS $TD $CT', ' '.join(f'{k} {v[0]:.3f}/{v[1]:.2f}' for k,v in o.items()), f\"MPPE {s['MPPE']:.3f} AUC {s['AUC']:.4f} MP-SR {s['MP-SR']}\")"
done
