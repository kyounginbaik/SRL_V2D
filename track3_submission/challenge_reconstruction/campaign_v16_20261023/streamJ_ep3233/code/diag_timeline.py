"""Stream J: per-window object error timeline + spoon-bowl-in-cup geometry + wrist distances for an executed trajectory.
usage: diag_timeline.py TRAJECTORY_NPZ EPISODE_DIR SCENE_XML [--step FRAMES] [--spoon-mesh OBJ --cup-mesh OBJ]
Errors are body-origin errors vs the episode dir tracking reference (diagnostic, not the kit metric)."""
import sys, json, argparse, numpy as np, mujoco, trimesh
from pathlib import Path
from scipy.spatial.transform import Rotation as R
sys.path[:0] = ['/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code', '/data1/SRL_V2D/third_party/DexAgent']
from tool_library.common import tracking as TR
ap = argparse.ArgumentParser(); ap.add_argument('traj'); ap.add_argument('epdir'); ap.add_argument('scene'); ap.add_argument('--step', type=int, default=10)
ap.add_argument('--spoon-mesh'); ap.add_argument('--cup-mesh'); a = ap.parse_args()
m = mujoco.MjModel.from_xml_path(a.scene); ref = TR.load_reference(str(Path(a.epdir) / 'tracking_reference.npz'), m)['qpos']; tr = np.load(a.traj)
names = [str(x) for x in tr['object_names']]; fps = 20.; n = min(len(ref), len(tr['qpos'])); d = mujoco.MjData(m)
man = json.loads((Path(a.epdir) / 'input_manifest.json').read_text()); src = Path(man['source'])
spoon_mesh = a.spoon_mesh or str(src / 'assets/wooden_spoon.obj'); cup_mesh = a.cup_mesh or str(src / 'assets/blue_cup.obj')
def spoon_geometry(spoon_mesh, cup_mesh):
    sv = np.asarray(trimesh.load(spoon_mesh, force='mesh').vertices); s = sv - sv.mean(0); w, U = np.linalg.eigh(s.T @ s); t = s @ U[:, -1]
    thick = lambda mk: np.ptp(s[mk] @ U[:, 1]); hi, lo = t > np.percentile(t, 85), t < np.percentile(t, 15); bowl = sv[hi] if thick(hi) > thick(lo) else sv[lo]
    cv = np.asarray(trimesh.load(cup_mesh, force='mesh').vertices); c = cv - cv.mean(0); w, U = np.linalg.eigh(c.T @ c)
    g = [abs(w[k]-w[(k+1)%3]) + abs(w[k]-w[(k+2)%3]) - abs(w[(k+1)%3]-w[(k+2)%3]) for k in range(3)]; ax = U[:, int(np.argmax(g))]; h = c @ ax
    rad = np.linalg.norm(c - np.outer(h, ax), axis=1)
    if rad[h > np.percentile(h, 92)].min() < rad[h < np.percentile(h, 8)].min(): ax, h = -ax, -h
    rim_r = np.median(rad[h > h.max() - 0.01]); return bowl.mean(0), cv.mean(0), ax, h.max(), rim_r
tip, cc, cax, rim_h, rim_r = spoon_geometry(spoon_mesh, cup_mesh)
def wrist_pos(q):
    d.qpos[:] = q; mujoco.mj_kinematics(m, d); return {s: d.body(f'robot_{s}_{s}_base_yaw').xpos.copy() for s in ('left', 'right')}
W = [wrist_pos(tr['qpos'][k]) for k in range(n)]
P = {}; Q = {}; Pr = {}; Qr = {}
for i, o in enumerate(names):
    oc = int(m.joint(o + '_free').qposadr[0]); Pr[o] = ref[:n, oc:oc+3]; Qr[o] = R.from_quat(ref[:n, oc+3:oc+7], scalar_first=True)
    P[o] = tr['object_positions'][:n, i]; Q[o] = R.from_quat(tr['object_quaternions_wxyz'][:n, i][:, [1, 2, 3, 0]])
def incup(Pc, Qc, Ps, Qs):
    tw = Qs.apply(tip) + Ps; loc = Qc.inv().apply(tw - Pc) - cc; hh = loc @ cax; rr = np.linalg.norm(loc - np.outer(hh, cax), axis=1); return rim_h - hh, rr
dep_a, rad_a = incup(P['blue_cup'], Q['blue_cup'], P['wooden_spoon'], Q['wooden_spoon']); dep_r, rad_r = incup(Pr['blue_cup'], Qr['blue_cup'], Pr['wooden_spoon'], Qr['wooden_spoon'])
# bowl point in the achieved cup frame vs reference bowl point in the reference cup frame (the spoon-in-cup term)
la = Q['blue_cup'].inv().apply(Q['wooden_spoon'].apply(tip) + P['wooden_spoon'] - P['blue_cup']); lr = Qr['blue_cup'].inv().apply(Qr['wooden_spoon'].apply(tip) + Pr['wooden_spoon'] - Pr['blue_cup'])
bowl_rel = np.linalg.norm(la - lr, axis=1) * 100
E = {o: (np.linalg.norm(P[o] - Pr[o], axis=1) * 100, np.degrees((Q[o] * Qr[o].inv()).magnitude())) for o in names}
F = tr['hand_object_force_n'][:n] if 'hand_object_force_n' in tr.files else np.zeros((n, 2))
print(f'frames {n}; cup rim radius {rim_r*100:.1f} cm; ' + '; '.join(f'{o}: mean {E[o][0].mean():.2f} cm / {E[o][1].mean():.1f} deg (max {E[o][0].max():.1f} / {E[o][1].max():.0f})' for o in names))
print('   t   | cup cm  deg | spoon cm  deg | bowl depth ach/ref cm | radial ach/ref cm | bowl rel err | zcup ach-ref | R-spoon L-cup wrist dist | hand force L R | ref spoon speed')
for s in range(0, n, a.step):
    sl = slice(s, min(n, s + a.step)); sp = np.linalg.norm(np.diff(Pr['wooden_spoon'][sl], axis=0), axis=1).mean() * fps * 100 if sl.stop - sl.start > 1 else 0
    dr = np.mean([np.linalg.norm(W[k]['right'] - P['wooden_spoon'][k]) for k in range(sl.start, sl.stop)]) * 100; dl = np.mean([np.linalg.norm(W[k]['left'] - P['blue_cup'][k]) for k in range(sl.start, sl.stop)]) * 100
    print(f'{s/fps:5.1f}s | {E["blue_cup"][0][sl].mean():5.2f} {E["blue_cup"][1][sl].mean():5.1f} | {E["wooden_spoon"][0][sl].mean():5.2f} {E["wooden_spoon"][1][sl].mean():5.1f} | {dep_a[sl].mean()*100:6.1f} {dep_r[sl].mean()*100:6.1f} | {rad_a[sl].mean()*100:5.1f} {rad_r[sl].mean()*100:5.1f} | {bowl_rel[sl].mean():5.1f} | {(P["blue_cup"][sl,2]-Pr["blue_cup"][sl,2]).mean()*100:5.1f} | {dr:5.1f} {dl:5.1f} | {F[sl,0].mean():5.1f} {F[sl,1].mean():5.1f} | {sp:5.1f}')
