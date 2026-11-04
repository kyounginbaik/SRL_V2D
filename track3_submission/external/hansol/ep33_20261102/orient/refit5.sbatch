#!/bin/bash
#SBATCH -p debug
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH -o /data1/hansol/ep33_20261102/logs/orientfit5_%j.log
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages OMP_NUM_THREADS=1
PY="/home/hansol/v2d/.venvs/v2d-host/bin/python -B"; W=/data1/hansol/ep33_20261102; O=$W/orient/fit3; V=/home/hansol/v2d/outputs/variants_20261030; E=episode_000033; T=$W/k33x
AR=$W/code/spoon_annot_roll2.py
$PY - <<'PY'
import json
d = json.load(open('/data1/hansol/ep33_20261102/orient/ep33_marks_merged_20261103c2.json'))
u = {k: v for k, v in json.load(open('/data1/hansol/ep33_20261102/orient/ep33_marks_20261103_user_f96_150.json')).items() if not k.startswith('_')}
d = {k: v for k, v in d.items() if not 96 <= int(k) <= 150}
for k, v in u.items(): d[k] = v if isinstance(v, str) else list(v) + [0.3]
d = {str(k): d[str(k)] for k in sorted(map(int, d))}; json.dump(d, open('/data1/hansol/ep33_20261102/orient/ep33_marks_merged_20261103d.json', 'w'), indent=0); print('marks', len(d))
PY
ANN=$W/orient/ep33_marks_merged_20261103d.json
for m in HGA HGB; do (
  B=$V/recon_E14_best_B2A_S1_$m; N=recon_E14_best_B2A_S1_${m}Z
  $PY $AR $O/${m}Zw 33 $ANN 88 148 --base $B --face-thr 0.8 --iou 96,146,1 > $O/${m}Zw.txt 2>&1
  rm -rf $O/$N; mkdir -p $O/$N; $PY $W/code/ep33_smooth_corr.py $B/$E $O/${m}Zw/bundle/$E $O/$N/$E --sigma 1.5 > $O/smooth_$m.txt 2>&1
  $PY $AR $O/rep_${m}_new 33 $ANN 60 160 --base $O/$N --face-thr 0.8 --report-only > $O/rep_${m}_new.txt 2>&1
  $PY $AR $O/rep_${m}_old 33 $ANN 60 160 --base $B --face-thr 0.8 --report-only > $O/rep_${m}_old.txt 2>&1
  $PY $W/code/ep33_facing_sheet.py $O/$N $O/sheet_${m}Z_f96_150.jpg 96 150 2 $ANN --label "$m refit f88-148 (user marks f96-150, 11/03 night)" > /dev/null 2>&1
) & done; wait
$PY $W/code/ep33_rollcurve_target.py $O/recon_E14_best_B2A_S1_HGAZ/$E $T/targets/HGAZ/$E "0:0" > /dev/null
C=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028; S=/data1/SRL_V2D/challenge_submissions_SRL; SC=$C/../campaign_v16_20261023/compose_target/score_vs_compose.py
declare -A P=([v26]=$S/track3_v26_20261029/all_20_episodes [v29]=$S/track3_v29_20261030/all_20_episodes [v31a]=$S/track3_v31a_HGA_20261101/all_20_episodes [H1rest_b2]=$S/ep33_fix_20261102/K33_H1rest_b2/all_20_episodes [HGA_c9]=$S/ablation_ep33only_20261103/v33a_ep33HGA_nospin/all_20_episodes [R2]=/data1/hansol/retarget33/packages/track3_FINAL_ep33R33R2_20261103/all_20_episodes)
cd /tmp; for tg in HGAYc2 HGAZ; do for r in v26 v29 v31a H1rest_b2 HGA_c9 R2; do [ -f $T/scores/${tg}__${r}.json ] || $PY $SC ${P[$r]}/track3_submission.parquet ${P[$r]}/results $T/targets/$tg $T/scores/${tg}__${r}.json 33 > $T/scores/${tg}__${r}.log 2>&1 & done; done; wait
$PY - <<'PYX'
import json, re, numpy as np
T='/data1/hansol/ep33_20261102/k33x/scores'; O='/data1/hansol/ep33_20261102/orient/fit3'; V='/home/hansol/v2d/outputs/variants_20261030'
for tg in ('HGAYc2', 'HGAZ'):
    cells=[]
    for r in ('v26','v29','v31a','H1rest_b2','HGA_c9','R2'):
        j=json.load(open(f'{T}/{tg}__{r}.json')); j=j.get('33', j.get(33, {})); o=j['objects']; mp=j['scores']['MP-SR']
        cells.append(f"{r}:{'PASS' if mp>0.5 else 'fail'} sp {o['wooden_spoon'][0]}/{o['wooden_spoon'][1]} cup {o['blue_cup'][0]}/{o['blue_cup'][1]}")
    print(f'{tg:7s}', ' | '.join(cells))
def fac(p):
    d={}
    for l in open(p):
        m=re.match(r"(\d+) (\S+(?: \S+)?(?: \S+)?) \{'facing': (-?[\d.]+), 'arrow_err': ([\d.]+|None)",l)
        if m: d[int(m[1])]=(m[2], float(m[3]), None if m[4]=='None' else float(m[4]))
    return d
def ok(lab,f,a):
    if lab=='out': return f>0.3
    if lab=='in': return f<-0.3
    cap = re.findall(r'-?[\d.]+', lab); cap = float(cap[2]) if len(cap) > 2 else None
    return (a is not None and a<45) and (abs(f) < 0.65 if cap is None else f <= cap + 0.05)
for m in ('HGA','HGB'):
    o=fac(f'{O}/rep_{m}_old.txt'); n=fac(f'{O}/rep_{m}_new.txt'); ks=[k for k in n if 96<=k<=150]
    print(m, 'f96-150 agree old', sum(ok(*o[k]) for k in ks), 'new', sum(ok(*n[k]) for k in ks), '/', len(ks))
    print('  ', ' '.join(f"f{k}:{o[k][1]:+.2f}->{n[k][1]:+.2f}" for k in ks if k % 4 == 0))
    A=np.load(f'{V}/recon_E14_best_B2A_S1_{m}/episode_000033/object_poses.npz'); B=np.load(f'{O}/recon_E14_best_B2A_S1_{m}Z/episode_000033/object_poses.npz'); k=[str(x) for x in A['names']].index('wooden_spoon')
    d=np.degrees(np.arccos(np.clip((np.einsum('fij,fij->f',A['world_from_object'][:,k,:3,:3],B['world_from_object'][:,k,:3,:3])-1)/2,-1,1))); ch=np.where(d>0.01)[0]
    print('   changed stored', (int(ch.min()), int(ch.max())) if len(ch) else None, 'max %.1f deg; f<80 max %.3f, f>=155 max %.3f' % (d.max(), d[:80].max(), d[155:].max()))
    print('  ', open(f'{O}/{m}Zw.txt').read().strip().splitlines()[-1][:330])
PYX
