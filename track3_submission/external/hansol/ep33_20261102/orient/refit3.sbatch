#!/bin/bash
#SBATCH -p debug
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH -o /data1/hansol/ep33_20261102/logs/orientfit3_%j.log
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages OMP_NUM_THREADS=1
PY="/home/hansol/v2d/.venvs/v2d-host/bin/python -B"; W=/data1/hansol/ep33_20261102; O=$W/orient/fit2; V=/home/hansol/v2d/outputs/variants_20261030; E=episode_000033
AR=$W/code/spoon_annot_roll2.py; T=$W/k33x
$PY - <<'PY'
import json
src = json.load(open('/data1/hansol/ep33_20261102/orient/ep33_marks_merged_20261103.json'))
for v, F in (('a', 0.0), ('b', -0.25)):
    d = dict(src)
    for f in (254, 256, 258, 260):
        d[str(f)] = [0, 1, F]
    json.dump(d, open(f'/data1/hansol/ep33_20261102/orient/ep33_marks_merged_20261103{v}.json', 'w'), indent=0)
PY
for v in a b; do for m in HGA HGB; do (
  B=$V/recon_E14_best_B2A_S1_$m; ANN=$W/orient/ep33_marks_merged_20261103$v.json; N=recon_E14_best_B2A_S1_${m}Y${v}s15
  $PY $AR $O/${m}Y$v 33 $ANN 236 290 --base $O/${m}Y_w1/bundle --face-thr 0.8 --iou 240,284,1 > $O/${m}Y${v}_w2.txt 2>&1
  mkdir -p $O/$N; $PY $W/code/ep33_smooth_corr.py $B/$E $O/${m}Y$v/bundle/$E $O/$N/$E --sigma 1.5 > $O/smooth_${m}$v.txt 2>&1
  $PY $AR $O/rep_${m}$v 33 $ANN 150 290 --base $O/$N --face-thr 0.8 --report-only > $O/rep_${m}Y$v.txt 2>&1
  $PY $AR $O/rep_${m}5 33 $ANN 150 290 --base $O/recon_E14_best_B2A_S1_${m}Ys15 --face-thr 0.8 --report-only > $O/rep_${m}Y5_vs$v.txt 2>&1
  $PY $W/code/ep33_facing_sheet.py $O/$N $O/sheet_${m}Y${v}_f240_280.jpg 240 280 2 $ANN --label "$m refit $v (facing cap at f254-260)" > /dev/null 2>&1
) & done; done; wait
for v in a b; do $PY $W/code/ep33_rollcurve_target.py $O/recon_E14_best_B2A_S1_HGAY${v}s15/$E $T/targets/HGAY$v/$E "0:0" > /dev/null; done
C=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028; S=/data1/SRL_V2D/challenge_submissions_SRL; SC=$C/../campaign_v16_20261023/compose_target/score_vs_compose.py
declare -A P=([v26]=$S/track3_v26_20261029/all_20_episodes [v29]=$S/track3_v29_20261030/all_20_episodes [v31a]=$S/track3_v31a_HGA_20261101/all_20_episodes [H1rest_b2]=$S/ep33_fix_20261102/K33_H1rest_b2/all_20_episodes)
cd /tmp; for tg in HGAYa HGAYb; do for r in v26 v29 v31a H1rest_b2; do $PY $SC ${P[$r]}/track3_submission.parquet ${P[$r]}/results $T/targets/$tg $T/scores/${tg}__${r}.json 33 > $T/scores/${tg}__${r}.log 2>&1 & done; done; wait
for m in HGA HGB; do for v in a b; do echo "== $m $v"; tail -1 $O/${m}Y${v}_w2.txt | cut -c1-330; done; done
$PY - <<'PYX'
import json, re
T='/data1/hansol/ep33_20261102/k33x/scores'
for tg in ('HGAY', 'HGAYa', 'HGAYb'):
    cells=[]
    for r in ('v26','v29','v31a','H1rest_b2'):
        j=json.load(open(f'{T}/{tg}__{r}.json')); j=j.get('33', j.get(33, {})); o=j['objects']; mp=j['scores']['MP-SR']
        cells.append(f"{r}: MP{mp:.0f} spoon {o['wooden_spoon'][0]}/{o['wooden_spoon'][1]}")
    print(f'{tg:6s}', ' | '.join(cells))
O='/data1/hansol/ep33_20261102/orient/fit2'
def load(p):
    d={}
    for l in open(p):
        m=re.match(r"(\d+) (\S+(?: \S+)?(?: \S+)?) \{'facing': (-?[\d.]+), 'arrow_err': ([\d.]+|None), 'ok': (\w+)",l)
        if m: d[int(m[1])]=(m[2],float(m[3]),m[4],m[5])
    return d
for m in ('HGA','HGB'):
    for v in ('a','b'):
        Y5=load(f'{O}/rep_{m}Y5_vs{v}.txt'); Yv=load(f'{O}/rep_{m}Y{v}.txt')
        print(m, v, ' '.join(f"f{k}:{Y5[k][1]:+.2f}->{Yv[k][1]:+.2f}{'' if Yv[k][3]=='True' else '!'}" for k in sorted(Yv) if 246<=k<=268))
        print('   ok in 236-290 (cap rule):', sum(Yv[k][3]=='True' for k in Yv if 236<=k<=290), '/', sum(1 for k in Yv if 236<=k<=290), ' f150-234 unchanged:', all(abs(Yv[k][1]-Y5[k][1])<0.005 for k in Yv if k<=234))
PYX
