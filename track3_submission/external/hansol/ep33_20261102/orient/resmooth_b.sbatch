#!/bin/bash
#SBATCH -p debug
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH -o /data1/hansol/ep33_20261102/logs/resmoothb_%j.log
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages OMP_NUM_THREADS=1
PY="/home/hansol/v2d/.venvs/v2d-host/bin/python -B"; W=/data1/hansol/ep33_20261102; O=$W/orient/fit2; V=/home/hansol/v2d/outputs/variants_20261030; E=episode_000033; T=$W/k33x
AR=$W/code/spoon_annot_roll2.py; ANN=$W/orient/ep33_marks_merged_20261103b.json
for m in HGA HGB; do (
  B=$V/recon_E14_best_B2A_S1_${m}_prev_p33d; N=recon_E14_best_B2A_S1_${m}Ybc; rm -rf $O/$N; mkdir -p $O/$N
  $PY $W/code/ep33_smooth_corr.py $B/$E $O/${m}Yb/bundle/$E $O/$N/$E --sigma 1.5 > $O/smooth_${m}bc.txt 2>&1
  $PY $AR $O/rep_${m}bc 33 $ANN 150 290 --base $O/$N --face-thr 0.8 --report-only > $O/rep_${m}Ybc.txt 2>&1
  $PY $W/code/ep33_facing_sheet.py $O/$N $O/sheet_${m}Ybc_f240_280.jpg 240 280 2 $ANN --label "$m refit 6b (back-side cap -0.25 at f254-260)" > /dev/null 2>&1
) & done; wait
$PY $W/code/ep33_rollcurve_target.py $O/recon_E14_best_B2A_S1_HGAYbc/$E $T/targets/HGAYbc/$E "0:0" > /dev/null
C=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028; S=/data1/SRL_V2D/challenge_submissions_SRL; SC=$C/../campaign_v16_20261023/compose_target/score_vs_compose.py
declare -A P=([v26]=$S/track3_v26_20261029/all_20_episodes [v29]=$S/track3_v29_20261030/all_20_episodes [v31a]=$S/track3_v31a_HGA_20261101/all_20_episodes [H1rest_b2]=$S/ep33_fix_20261102/K33_H1rest_b2/all_20_episodes)
cd /tmp; for r in v26 v29 v31a H1rest_b2; do $PY $SC ${P[$r]}/track3_submission.parquet ${P[$r]}/results $T/targets/HGAYbc $T/scores/HGAYbc__${r}.json 33 > $T/scores/HGAYbc__${r}.log 2>&1 & done; wait
$PY - <<'PYX'
import json, re
import numpy as np
T='/data1/hansol/ep33_20261102/k33x/scores'; O='/data1/hansol/ep33_20261102/orient/fit2'; V='/home/hansol/v2d/outputs/variants_20261030'
cells=[]
for r in ('v26','v29','v31a','H1rest_b2'):
    j=json.load(open(f'{T}/HGAYbc__{r}.json')); j=j.get('33', j.get(33, {})); o=j['objects']; cells.append(f"{r}: MP{j['scores']['MP-SR']:.0f} spoon {o['wooden_spoon'][0]}/{o['wooden_spoon'][1]}")
print('HGAYbc', ' | '.join(cells))
def fac(p):
    d={}
    for l in open(p):
        m=re.match(r"(\d+) .*'facing': (-?[\d.]+), 'arrow_err': ([\d.]+|None), 'ok': (\w+)",l)
        if m: d[int(m[1])]=(float(m[2]), m[4]=='True')
    return d
for m in ('HGA','HGB'):
    E='episode_000033'; Z=lambda p: np.load(p)
    A=Z(f'{O}/recon_E14_best_B2A_S1_{m}Ys15/{E}/object_poses.npz'); B=Z(f'{O}/recon_E14_best_B2A_S1_{m}Ybc/{E}/object_poses.npz'); C=Z(f'{O}/recon_E14_best_B2A_S1_{m}Ybs15/{E}/object_poses.npz')
    k=[str(n) for n in A['names']].index('wooden_spoon')
    dd=lambda X,Y: np.degrees(np.arccos(np.clip((np.einsum('fij,fij->f', X['world_from_object'][:,k,:3,:3], Y['world_from_object'][:,k,:3,:3])-1)/2,-1,1)))
    d5=dd(A,B); db=dd(C,B)
    print(m, 'vs (5) stored f<225 max %.3f deg, changed frames %s' % (d5[:225].max(), np.where(d5>0.01)[0][[0,-1]].tolist()), '| vs 6b(as shown) max %.1f deg' % db.max())
    f=fac(f'{O}/rep_{m}Ybc.txt'); print('  ', ' '.join(f'f{x}:{f[x][0]:+.2f}' for x in range(246, 270, 2)), '| ok 150-290 (cap rule):', sum(v[1] for x,v in f.items() if 150<=x<=290), '/', sum(1 for x in f if 150<=x<=290))
PYX
