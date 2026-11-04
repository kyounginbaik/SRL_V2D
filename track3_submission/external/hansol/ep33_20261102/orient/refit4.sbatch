#!/bin/bash
#SBATCH -p debug
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH -o /data1/hansol/ep33_20261102/logs/orientfit4_%j.log
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages OMP_NUM_THREADS=1
PY="/home/hansol/v2d/.venvs/v2d-host/bin/python -B"; W=/data1/hansol/ep33_20261102; O=$W/orient/fit2; V=/home/hansol/v2d/outputs/variants_20261030; E=episode_000033; T=$W/k33x
AR=$W/code/spoon_annot_roll2.py
$PY - <<'PY'
import json
src = json.load(open('/data1/hansol/ep33_20261102/orient/ep33_marks_merged_20261103b.json'))
for v, F in (('c1', -0.25), ('c2', -0.5)):
    d = dict(src)
    for f in (184, 186, 188, 190):
        d[str(f)] = list(src[str(f)][:2]) + [F]
    json.dump(d, open(f'/data1/hansol/ep33_20261102/orient/ep33_marks_merged_20261103{v}.json', 'w'), indent=0)
    print(v, {f: d[str(f)] for f in (182, 184, 186, 188, 190, 192, 254, 260)})
PY
for v in c1 c2; do for m in HGA HGB; do (
  B=$V/recon_E14_best_B2A_S1_${m}_prev_p33d; ANN=$W/orient/ep33_marks_merged_20261103$v.json; N=recon_E14_best_B2A_S1_${m}Y$v
  $PY $AR $O/${m}Y${v}_w1 33 $ANN 150 230 --base $B --face-thr 0.8 --iou 160,226,1 > $O/${m}Y${v}_w1.txt 2>&1
  $PY $AR $O/${m}Y$v 33 $ANN 236 290 --base $O/${m}Y${v}_w1/bundle --face-thr 0.8 --iou 240,284,1 > $O/${m}Y${v}_w2.txt 2>&1
  rm -rf $O/$N; mkdir -p $O/$N; $PY $W/code/ep33_smooth_corr.py $B/$E $O/${m}Y$v/bundle/$E $O/$N/$E --sigma 1.5 > $O/smooth_${m}$v.txt 2>&1
  $PY $AR $O/rep_${m}$v 33 $ANN 150 290 --base $O/$N --face-thr 0.8 --report-only > $O/rep_${m}Y$v.txt 2>&1
  $PY $W/code/ep33_facing_sheet.py $O/$N $O/sheet_${m}Y${v}_f176_224.jpg 176 224 2 $ANN --label "$m refit $v (back-side cap at f184-190 and f254-260)" > /dev/null 2>&1
) & done; done; wait
for v in c1 c2; do $PY $W/code/ep33_rollcurve_target.py $O/recon_E14_best_B2A_S1_HGAY$v/$E $T/targets/HGAY$v/$E "0:0" > /dev/null; done
C=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028; S=/data1/SRL_V2D/challenge_submissions_SRL; SC=$C/../campaign_v16_20261023/compose_target/score_vs_compose.py
declare -A P=([v26]=$S/track3_v26_20261029/all_20_episodes [v29]=$S/track3_v29_20261030/all_20_episodes [v31a]=$S/track3_v31a_HGA_20261101/all_20_episodes [H1rest_b2]=$S/ep33_fix_20261102/K33_H1rest_b2/all_20_episodes)
cd /tmp; for tg in HGAYc1 HGAYc2; do for r in v26 v29 v31a H1rest_b2; do $PY $SC ${P[$r]}/track3_submission.parquet ${P[$r]}/results $T/targets/$tg $T/scores/${tg}__${r}.json 33 > $T/scores/${tg}__${r}.log 2>&1 & done; done; wait
for m in HGA HGB; do for v in c1 c2; do echo "== $m $v w1"; tail -1 $O/${m}Y${v}_w1.txt | cut -c1-330; done; done
$PY - <<'PYX'
import json, re
import numpy as np
T='/data1/hansol/ep33_20261102/k33x/scores'; O='/data1/hansol/ep33_20261102/orient/fit2'; V='/home/hansol/v2d/outputs/variants_20261030'; E='episode_000033'
for tg in ('HGAYbc', 'HGAYc1', 'HGAYc2'):
    cells=[]
    for r in ('v26','v29','v31a','H1rest_b2'):
        j=json.load(open(f'{T}/{tg}__{r}.json')); j=j.get('33', j.get(33, {})); o=j['objects']; cells.append(f"{r}: MP{j['scores']['MP-SR']:.0f} spoon {o['wooden_spoon'][0]}/{o['wooden_spoon'][1]}")
    print(f'{tg:7s}', ' | '.join(cells))
def fac(p):
    d={}
    for l in open(p):
        m=re.match(r"(\d+) .*'facing': (-?[\d.]+), 'arrow_err': ([\d.]+|None)",l)
        if m: d[int(m[1])]=(float(m[2]), m[3])
    return d
for m in ('HGA','HGB'):
    cur=np.load(f'{V}/recon_E14_best_B2A_S1_{m}/{E}/object_poses.npz'); k=[str(n) for n in cur['names']].index('wooden_spoon')
    f0=fac(f'{O}/rep_{m}Ybc.txt')
    for v in ('c1','c2'):
        new=np.load(f'{O}/recon_E14_best_B2A_S1_{m}Y{v}/{E}/object_poses.npz')
        d=np.degrees(np.arccos(np.clip((np.einsum('fij,fij->f', cur['world_from_object'][:,k,:3,:3], new['world_from_object'][:,k,:3,:3])-1)/2,-1,1)))
        ch=np.where(d>0.01)[0]; f=fac(f'{O}/rep_{m}Y{v}.txt')
        print(m, v, 'vs installed: changed stored', (ch.min(), ch.max()) if len(ch) else None, 'max %.1f deg, stored f>=235 max %.3f' % (d.max(), d[235:].max()))
        print('   ', ' '.join(f"f{x}:{f0[x][0]:+.2f}->{f[x][0]:+.2f}" for x in range(178, 202, 2)))
PYX
