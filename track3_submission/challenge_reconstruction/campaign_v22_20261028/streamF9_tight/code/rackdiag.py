"""F9 diag: object displacement from frame 0 (sim vs target, physics world) per 0.5 s.  usage: rackdiag.py EP ACTOR MESHES Q.npz OBJ"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mresF9 as mres
from scipy.spatial.transform import Rotation as R
ep, actor, meshes, qf, obj = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
env = mres.Env(dict(ep=ep, actor=str(Path(actor).resolve()), meshes=meshes)); Q = np.load(qf)['Q']
P = env.poses(Q); _, A = mres.L.per_frame(ep, P, env.reg, env.tgt, want_A=True); tp = mres.L.target_phys(ep, env.reg, A, env.tgt)
ca = env.reg[obj][0]
a = P[obj][:, :3, 3] + np.einsum('nij,j->ni', P[obj][:, :3, :3], ca); t = tp[obj][:len(Q), :3, 3] + np.einsum('nij,j->ni', tp[obj][:len(Q), :3, :3], ca)
for k in range(0, len(Q), 10):
    da = (a[k] - a[0]) * 100; dt = (t[k] - t[0]) * 100; ra = R.from_matrix(P[obj][k, :3, :3] @ P[obj][0, :3, :3].T).magnitude() * 57.3; rt = R.from_matrix(tp[obj][k, :3, :3] @ tp[obj][0, :3, :3].T).magnitude() * 57.3
    print(f'{k/env.fps:5.1f}s sim d {np.round(da,2)} rot {ra:5.1f} | tgt d {np.round(dt,2)} rot {rt:5.1f} | err {np.round(da-dt,2)}')
