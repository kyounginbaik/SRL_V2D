"""like r33endjit.py but for an eval.npz (key Q) + scene xml; also prints spoon/cup error per 0.5 s from the eval npz. usage: r33endjit2.py EVAL_NPZ SCENE_XML"""
import sys, json
from pathlib import Path
ev, scene = sys.argv[1], sys.argv[2]
import numpy as np
z = np.load(ev); Q = z['Q']; open('/tmp/_q.npz', 'wb').close()
np.savez(str(Path(ev).with_name('_endjit_q.npz')), qpos=Q); json.dump({'execution': {'scene': scene, 'trajectory': str(Path(ev).with_name('_endjit_q.npz'))}}, open(str(Path(ev).with_name('_endjit.json')), 'w'))
sp, sd, cp, cd = z['wooden_spoon_pos'], z['wooden_spoon_deg'], z['blue_cup_pos'], z['blue_cup_deg']
print(f'mean cup {cp.mean():.2f}/{cd.mean():.1f} spoon {sp.mean():.2f}/{sd.mean():.1f}; per 0.5 s from 14.5 s (spoon cm/deg | cup cm):',
      ' '.join(f'{a/20:.1f}:{sp[a:a+10].mean():.2f}/{sd[a:a+10].mean():.0f}|{cp[a:a+10].mean():.2f}' for a in range(290, len(sp), 10)))
sys.argv = ['x', str(Path(ev).with_name('_endjit.json'))]; exec(open('/data1/hansol/retarget33/code/r33endjit.py').read())
