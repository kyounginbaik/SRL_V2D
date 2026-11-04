#!/bin/bash
#SBATCH -p debug
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH -o /data1/hansol/ep33_20261102/logs/orycons_%j.log
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages OMP_NUM_THREADS=1
PY="/home/hansol/v2d/.venvs/v2d-host/bin/python -B"; W=/data1/hansol/ep33_20261102; T=$W/k33x; O=$W/orient/fit2
$PY $W/code/ep33_rollcurve_target.py $O/recon_E14_best_B2A_S1_HGAYs15/episode_000033 $T/targets/HGAY/episode_000033 "0:0"
C=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028; S=/data1/SRL_V2D/challenge_submissions_SRL; SC=$C/../campaign_v16_20261023/compose_target/score_vs_compose.py
declare -A P=([v26]=$S/track3_v26_20261029/all_20_episodes [v29]=$S/track3_v29_20261030/all_20_episodes [v31a]=$S/track3_v31a_HGA_20261101/all_20_episodes [H1rest_b2]=$S/ep33_fix_20261102/K33_H1rest_b2/all_20_episodes)
cd /tmp; for r in v26 v29 v31a H1rest_b2; do $PY $SC ${P[$r]}/track3_submission.parquet ${P[$r]}/results $T/targets/HGAY $T/scores/HGAY__${r}.json 33 > $T/scores/HGAY__${r}.log 2>&1 & done; wait
$PY - <<'PYX'
import json, os
T='/data1/hansol/ep33_20261102/k33x/scores'
for tg in ('HGARF_c', 'HGAX', 'HGAY'):
    cells=[]
    for r in ('v26','v29','v31a','H1rest_b2'):
        j=json.load(open(f'{T}/{tg}__{r}.json')); j=j.get('33', j.get(33, {})); o=j['objects']; mp=j['scores']['MP-SR']
        cells.append(f"{r}: MP{mp:.0f} cup {o['blue_cup'][0]}/{o['blue_cup'][1]} spoon {o['wooden_spoon'][0]}/{o['wooden_spoon'][1]}")
    print(f'{tg:8s}', ' | '.join(cells))
PYX
