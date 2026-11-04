#!/usr/bin/env bash
# When ep38's 16-part run has its videos, keep them for comparison and rerun it on the 64-part GT shape.
cd /scratch2/exaFLOPs/srl_yam_run/pipe; H=/scratch2/exaFLOPs/SRL_V2D/uniretarget_yam; R=/scratch2/exaFLOPs/srl_yam_run; E=recon_000038
for i in $(seq 1 590); do squeue -u $USER -h -o %j | grep -qE '^(s38s42|sa38|v38)$' || break; sleep 60; done
python3 board.py | grep -A6 '^ep38 ' | grep test | cut -c1-90; python3 -c "
import json
for r in json.load(open('$R/comparison/results/upper_test_${E}_s42.json')):print(r['steps']//10**6,r['kinds'],round(r['cm'],2),round(r['deg'],1),round(r['object_max_rise_cm'],1))"
./collect.sh 38; [ -d $R/videos/successes/ep38 ] && mv $R/videos/successes/ep38 $R/videos/successes/.parts16_ep38
mv $R/search/$E $R/search/.parts16_$E; for d in episodes objects out; do mv $H/$d/$E $H/$d/.parts16_$E; done; mv $R/videos/$E $R/videos/.parts16_$E; mv $R/comparison/results/upper_test_${E}_s42.json $R/comparison/results/upper_test_${E}_s42.json.parts16; mkdir -p oldlogs; mv v38-*.log oldlogs/; ./pilot.sh 38 | tail -1; date
