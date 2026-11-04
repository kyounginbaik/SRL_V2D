"""scan rollouts (any dir with result.json + execution/trajectory.npz, or evaluation/...) for ep35 errors vs target_final_v5 (tlibK, scorer-identical)
in windows. usage: scan35.py OUT_JSON ROOT [ROOT...]"""
import sys, json, glob, os, numpy as np, mujoco
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import tlibK as T
from scipy.spatial.transform import Rotation as R
out = sys.argv[1]; rows = []; adr = {}
for root in sys.argv[2:]:
    for rj in glob.glob(root + '/**/result.json', recursive=True):
        try:
            r = json.load(open(rj)); tr = r['execution']['trajectory']; sc = r['execution']['scene']
            if not os.path.exists(tr): continue
            z = np.load(tr); q = z['qpos']
            if len(q) != 421: continue
            if sc not in adr: m = mujoco.MjModel.from_xml_path(sc); adr[sc] = int(m.jnt_qposadr[m.joint('wooden_spoon_free').id])
            s = adr[sc]; Ta = np.tile(np.eye(4), (len(q), 1, 1)); Ta[:, :3, :3] = R.from_quat(q[:, s + 3:s + 7], scalar_first=True).as_matrix(); Ta[:, :3, 3] = q[:, s:s + 3]
            pe, re = T.errors(35, Ta)
            rows.append(dict(path=os.path.dirname(rj), full=[float(pe.mean()), float(re.mean())], early=[float(pe[:190].mean()), float(re[:190].mean())], e60_190=[float(pe[60:190].mean()), float(re[60:190].mean())], late=[float(pe[190:].mean()), float(re[190:].mean())]))
        except Exception as e: print('skip', rj, repr(e)[:100], file=sys.stderr)
json.dump(rows, open(out, 'w'), indent=0)
rows.sort(key=lambda r: r['early'][1] / 12 + r['early'][0] / 1.2)
for r in rows[:40]: print('early %.2f/%.1f  3-9.5s %.2f/%.1f  late %.2f/%.1f  full %.2f/%.1f  %s' % (*r['early'], *r['e60_190'], *r['late'], *r['full'], r['path'].replace('/data1/SRL_V2D/challenge_reconstruction/', '')))
print(len(rows), 'rollouts')
