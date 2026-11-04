"""Print per-object errors (+ last-2s tail) for result dirs. usage: summarize.py DIR [DIR ...]"""
import sys,json,numpy as np
from scipy.spatial.transform import Rotation as R
from pathlib import Path
for d in sys.argv[1:]:
    p=Path(d)/'result.json'
    if not p.exists(): print(d,'(no result)'); continue
    r=json.load(open(p)); m=r['metrics']['official_style_reference_proxy']['objects']
    print(f"{d}: "+'  '.join(f"{k[-7:]} {v['mean_position_error_cm'][0]:.2f}/{v['mean_orientation_error_degrees'][0]:.1f}" for k,v in m.items()), 'mp_sr',r['metrics']['official_style_reference_proxy']['mp_sr'])
