#!/bin/bash
# ep33: export current best two-hand teacher -> distill -> both-hands beam on every distilled actor that holds the spoon (proxy pass or spoon < 4 cm)
set -euo pipefail
C22=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028; T=$C22/streamT32_teacher; K33=$C22/streamK33_ep33; DX=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; PY=$DX/.venv/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$DX/code:/data1/SRL_V2D/third_party/DexAgent
NEW=$T/ep33/W_RP33b/episode_corrected/episode_000033; B=$T/ep33/port/pg_rg0_b0.5b; FS=$T/ep33/fcem2_fpb/execution/scene_physics.xml; SRC=$1; OUT=$2
cp $SRC/progress.json $T/ep33/${OUT}_progress.json; cd /tmp
$PY -B -u $T/code/fcem2.py 33 $B $NEW $T/ep33/$OUT --t-lock 3.2 --t-off 15.0 --ramp 0.2 --knots 10 --samples 4 --iterations 0 --workers 2 --seed 3 --scene $FS --init $T/ep33/${OUT}_progress.json 2>&1 | grep -E "^RESULT|Traceback" | cut -c1-220
$PY -B -u /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamC_rigid_ilc/tools/actor_grid_c.py --episode 33 --teacher $T/ep33/$OUT --output $T/ep33/distill_$OUT --episode-dir $NEW --workers 8 --seed 44 2>&1 | grep -vE "Warn|warn" | tail -n 1 | cut -c1-120
python3 - <<P
import json,glob,os
rows=[]
for d in sorted(glob.glob('$T/ep33/distill_$OUT/actor_*')):
    f=d+'/evaluation/result.json'
    if not os.path.exists(f):continue
    r=json.load(open(f));m=r.get('metrics',{});o=m.get('official_style_reference_proxy',{}).get('objects',{})
    try:rows.append((d,m.get('reference_proxy_mp_sr'),o['blue_cup']['mean_position_error_cm'][0],o['blue_cup']['mean_orientation_error_degrees'][0],o['wooden_spoon']['mean_position_error_cm'][0],o['wooden_spoon']['mean_orientation_error_degrees'][0]))
    except Exception:pass
rows.sort(key=lambda r:(-(r[1] or 0),r[4]+r[5]/10))
for r in rows:print('DISTILL',os.path.basename(r[0]),'proxy',r[1],'cup %.2f/%.1f spoon %.2f/%.1f'%r[2:])
open('$T/ep33/distill_$OUT/beam_list.txt','w').write('\n'.join(r[0] for r in rows if (r[1]==1.0 or (r[2]<3.0 and r[4]<7.0)))[:2000])
P
cd $K33; i=0; export HANDS=wooden_spoon:right,blue_cup:left RS=right,left ADVS=right,both
for A in $(head -3 $T/ep33/distill_$OUT/beam_list.txt); do i=$((i+1)); o=$(bash $K33/code/k33chain_t33.sh K33_RP T33_${OUT}_a${i}_b1 $A none none 0.6,8 6,1.0 1.00 16 2 '{"blue_cup":2}'); echo "BEAM T33_${OUT}_a${i}_b1 <- $(basename $A): $(echo "$o" | grep -oE 'g[12] job=[0-9]+' | tr '\n' ' ')"; done
