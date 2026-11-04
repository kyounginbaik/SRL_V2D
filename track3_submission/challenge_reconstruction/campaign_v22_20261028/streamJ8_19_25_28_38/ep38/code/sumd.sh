#!/bin/bash
# usage: sumd.sh DISTILLED_DIR...  -> per-actor internal proxy (cm/deg/max cm)
for d in "$@"; do for a in $d/actor_*; do f=$a/evaluation/result.json; [ -f $f ] || continue
python3 -c "
import json,sys;r=json.load(open('$f'));o=r['metrics']['official_style_reference_proxy']['objects']['mini_dust_pan']
print('$(basename $d)/$(basename $a)', r['metrics']['reference_proxy_mp_sr'], round(o['mean_position_error_cm'][0],2), round(o['mean_orientation_error_degrees'][0],1), round(o['max_position_error_m'][0]*100,1))"; done; done
