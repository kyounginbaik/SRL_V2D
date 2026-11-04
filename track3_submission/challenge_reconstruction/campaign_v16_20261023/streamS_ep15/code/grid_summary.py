"""Summarise an actor_grid_c.py output dir: per actor rack/spoon cm/deg and pass. usage: grid_summary.py DIST_DIR"""
import json, sys
from pathlib import Path
for d in sorted(Path(sys.argv[1]).glob('actor_*')):
    p = d / 'evaluation/result.json'
    if not p.exists(): print(d.name, 'no evaluation'); continue
    r = json.loads(p.read_text()); m = r['metrics']; o = m['official_style_reference_proxy']['objects']
    print(f"{d.name:28s} mp_sr {m['reference_proxy_mp_sr']}", '  '.join(f"{k[:5]} {v['mean_position_error_cm'][0]:.3f}/{v['mean_orientation_error_degrees'][0]:.2f} ({v['max_position_error_m'][0]*100:.1f})" for k, v in o.items()), 'verified', r['execution'].get('physics_verified'))
