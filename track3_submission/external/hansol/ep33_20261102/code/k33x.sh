#!/bin/bash
# targets: our HGA (before/after roll fix) plain and with K33's f0 rest roll; robots: v26, v29, v31a (HGA), K33_H1rest_b2, K33_H4settle_b3
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages OMP_NUM_THREADS=1
PY="/home/hansol/v2d/.venvs/v2d-host/bin/python -B"; W=/data1/hansol/ep33_20261102; T=$W/k33x
HGA0=/home/hansol/v2d/outputs/variants_20261030/recon_E14_best_B2A_S1_HGA/episode_000033; HGAR=$W/roll/recon_E14_best_B2A_S1_HGAR8s15/episode_000033
$PY $W/code/ep33_restroll_target.py $HGA0 $T/targets/HGA0/episode_000033 0 25 40
$PY $W/code/ep33_restroll_target.py $HGA0 $T/targets/HGA0_rest/episode_000033 rest 25 40
$PY $W/code/ep33_restroll_target.py $HGAR $T/targets/HGAR/episode_000033 0 25 40
$PY $W/code/ep33_restroll_target.py $HGAR $T/targets/HGAR_rest/episode_000033 rest 25 40
C=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028
ln -sfn $C/target_final_v14_g17 $T/targets/v14; ln -sfn $C/streamK33_ep33/targets/K33_H1rest $T/targets/K33_H1rest
S=/data1/SRL_V2D/challenge_submissions_SRL; SC=$C/../campaign_v16_20261023/compose_target/score_vs_compose.py
declare -A P=([v26]=$S/track3_v26_20261029/all_20_episodes [v29]=$S/track3_v29_20261030/all_20_episodes [v31a]=$S/track3_v31a_HGA_20261101/all_20_episodes [H1rest_b2]=$S/ep33_fix_20261102/K33_H1rest_b2/all_20_episodes)
cd /tmp
for tg in v14 K33_H1rest HGA0 HGA0_rest HGAR HGAR_rest; do for r in v26 v29 v31a H1rest_b2; do
  $PY $SC ${P[$r]}/track3_submission.parquet ${P[$r]}/results $T/targets/$tg $T/scores/${tg}__${r}.json 33 > $T/scores/${tg}__${r}.log 2>&1 &
done; done; wait
$PY - <<'PYX'
import json, glob, os
T='/data1/hansol/ep33_20261102/k33x/scores'; rows={}
for f in sorted(glob.glob(T+'/*.json')):
    tg, r = os.path.basename(f)[:-5].split('__'); j=json.load(open(f)).get('33', json.load(open(f)).get(33, {}))
    o=j.get('objects', {}); mp=j.get('scores', {}).get('MP-SR')
    rows.setdefault(tg, {})[r]=f"MP{mp:.0f} cup {o.get('blue_cup',['-','-'])[0]}/{o.get('blue_cup',['-','-'])[1]} spoon {o.get('wooden_spoon',['-','-'])[0]}/{o.get('wooden_spoon',['-','-'])[1]}" if mp is not None else 'ERR'
print('official: every robot below FAILS ep33 (v26, v29, v31a, K33_H1rest_b2)')
for tg, d in rows.items(): print(f'{tg:11s}', ' | '.join(f'{r}: {d.get(r)}' for r in ('v26','v29','v31a','H1rest_b2')))
PYX
