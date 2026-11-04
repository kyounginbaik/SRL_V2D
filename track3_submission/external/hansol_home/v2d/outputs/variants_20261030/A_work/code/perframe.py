"""Per-frame dump for one eval episode/object: base vs TAG IoU, |z|, ball px, change, base steps. usage: perframe.py TAG EP OBJ"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import slib
W = Path('/home/hansol/v2d/outputs/variants_20261030/A_work')
tag, ep, obj = sys.argv[1], int(sys.argv[2]), sys.argv[3]
n = f'episode_{ep:06d}'
b, a = W / 'eval' / 'base' / n, W / 'eval' / tag / n
qb, qa = json.loads((b / 'qc.json').read_text())['objects'][obj]['per_frame'], json.loads((a / 'qc.json').read_text())['objects'][obj]['per_frame']
db, da = np.load(b / 'object_poses.npz'), np.load(a / 'object_poses.npz')
k = [str(x) for x in db['names']].index(obj)
info = slib.mesh_info(obj, json.loads((a / 'meshes.json').read_text())[obj])
Tb, Ta = db['world_from_object'][:, k], da['world_from_object'][:, k]
dr = slib.rel_angle_deg(Ta[:, :3, :3], Tb[:, :3, :3], info); dp = np.linalg.norm(slib.centers(Ta, info) - slib.centers(Tb, info), axis=-1) * 100
st = slib.motion_stats(Tb, info)
obs = db['object_observed'][:, k]
rep = json.loads((W / 'eval' / tag / 'smooth_report.json').read_text())['eval'][n][obj]
low = set(rep['low_weight_frames'])
bb = ba = None
if (b / 'ballqc.json').exists():
    bb = json.loads((b / 'ballqc.json').read_text()).get('per_frame'); ba = json.loads((a / 'ballqc.json').read_text()).get('per_frame')
f = lambda x: 'nan' if x is None else f'{x:.2f}'
print('cuts', rep['cuts'], 'low', sorted(low))
for t in range(len(Tb)):
    di = (qa['iou'][t] or np.nan) - (qb['iou'][t] or np.nan)
    if '-a' in sys.argv or abs(di) > 0.08 or t in low or dr[t] > 8:
        print(f"f{t:3d} obs {int(obs[t])} low {int(t in low)} chg {dr[t]:5.1f}deg {dp[t]:4.2f}cm | step_in {st['dr'][t-1] if t else 0:5.1f}deg {st['dp'][t-1] if t else 0:4.2f}cm "
              f"| IoU {f(qb['iou'][t])}->{f(qa['iou'][t])} z {f(qb['median_z_mm'][t])}->{f(qa['median_z_mm'][t])}" + (f" ball {f(bb[t])}->{f(ba[t])}" if bb else ''))
