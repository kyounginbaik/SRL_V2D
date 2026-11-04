"""Iterative learning control of the spoon (held by one hand) relative to the cup it works in.
Each iteration: replay the reference in fresh physics, compute the spoon pose error w.r.t. the target
S*(t) = C_ach(t) C_ref(t)^-1 S_ref(t) (reference spoon pose expressed relative to the ACHIEVED cup, so the bowl goes
where the video puts it inside the cup), smooth it, and correct the holding wrist by rotating about the spoon origin and
translating (rigid-grasp approximation), gated to [t0, t1] with 0.3 s ramps. Object targets/clock untouched; robot only.
usage: spoon_ilc.py SOURCE_RUN OUT --t0 --t1 [--side right --iterations 4 --gain-rot .5 --gain-pos .6 --sigma-s .15]"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import argparse, json, sys
from pathlib import Path
import numpy as np, mujoco
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R
LEG = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); sys.path[:0] = [str(LEG / 'code'), '/data1/SRL_V2D/third_party/DexAgent']
from tool_library.common import hand as H, tracking as TR
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files
from wrist_reference import continuous_euler_xyz
smooth = lambda u: np.clip(u, 0, 1) ** 3 * (10 - 15 * np.clip(u, 0, 1) + 6 * np.clip(u, 0, 1) ** 2)
def dump(p, o): Path(p).write_text(json.dumps(o, indent=1, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)) + '\n')

def one(source, out, a, it):
    prev = json.loads((source / 'result.json').read_text()); scene = prev['execution']['scene']; epdir = Path(a.episode_dir) if a.episode_dir else Path(prev['input_dir']) if 'input_dir' in prev else Path(prev['spec']['manifest']).parent
    man = json.loads((epdir / 'input_manifest.json').read_text()); src = Path(man['source'])
    m = mujoco.MjModel.from_xml_path(scene); ref = TR.load_reference(source / 'reference.npz', m); Q0 = ref['qpos'].copy(); fps = ref['fps']; T = len(Q0); t = np.arange(T) / fps
    rig = H.HandRig.build(m, f'robot_{a.side}_', a.side); wc = np.array([rig.wrist_qpos[k] for k in ('x', 'y', 'z', 'rx', 'ry', 'rz')])
    z = np.load(src / 'object_poses.npz'); names = [str(x) for x in z['names']]; W = z['world_from_object']
    F = np.asarray(man.get('setup_from_source', np.eye(4)), float); W = np.einsum('ij,nojk->noik', F, W)   # source world -> physics (setup) world
    Cr, Sr = W[:T, names.index(a.cup)], W[:T, names.index(a.obj)]
    tr = np.load(prev['execution']['trajectory']); on = list(tr['object_names']); n = min(T, len(tr['object_positions']))
    def mk(j):
        M = np.tile(np.eye(4), (n, 1, 1)); M[:, :3, :3] = R.from_quat(tr['object_quaternions_wxyz'][:n, j][:, [1, 2, 3, 0]]).as_matrix(); M[:, :3, 3] = tr['object_positions'][:n, j]; return M
    Ca, Sa = mk(on.index(a.cup)), mk(on.index(a.obj))
    Ss = np.einsum('nij,njk,nkl->nil', Ca, np.linalg.inv(Cr[:n]), Sr[:n]) if a.relative else Sr[:n]
    sig = a.sigma_s * fps
    e_rot = gaussian_filter1d((R.from_matrix(Ss[:, :3, :3]) * R.from_matrix(Sa[:, :3, :3]).inv()).as_rotvec(), sig, axis=0, mode='nearest')
    e_pos = gaussian_filter1d(Ss[:, :3, 3] - Sa[:, :3, 3], sig, axis=0, mode='nearest')
    acc = prev.get('spoon_ilc', {})
    rot_tab = np.asarray(acc.get('rotation_table', np.zeros((T, 3))), float); pos_tab = np.asarray(acc.get('translation_table', np.zeros((T, 3))), float)
    base = np.load(acc['base_qpos'])['qpos'] if 'base_qpos' in acc else Q0.copy()
    gate = smooth((t - a.t0) / .3) * (1 - smooth((t - a.t1) / .3)); win = gate > 0
    rot_tab[:n][win[:n]] = (R.from_rotvec(a.gain_rot * e_rot[win[:n]]) * R.from_rotvec(rot_tab[:n][win[:n]])).as_rotvec()
    pos_tab[:n][win[:n]] += a.gain_pos * e_pos[win[:n]]
    nr = np.linalg.norm(rot_tab, axis=1, keepdims=True); rot_tab = rot_tab * np.minimum(1, np.radians(a.max_rot_deg) / np.maximum(nr, 1e-9))
    npn = np.linalg.norm(pos_tab, axis=1, keepdims=True); pos_tab = pos_tab * np.minimum(1, a.max_pos_m / np.maximum(npn, 1e-9))
    # rotation pivot: the reference spoon origin moved by the accumulated correction (approx. achieved spoon origin)
    piv = np.zeros((T, 3)); piv[:n] = Sa[:, :3, 3]; piv[n:] = piv[n - 1]
    Q = base.copy()
    for k in range(T):
        g = gate[k]
        if g <= 0: continue
        pos, ang = Q[k, wc[:3]].copy(), Q[k, wc[3:]].copy(); Rc = R.from_rotvec(g * rot_tab[k]); c = piv[k]
        Q[k, wc[:3]] = c + Rc.apply(pos - c) + g * pos_tab[k]
        Q[k, wc[3:]] = continuous_euler_xyz(R.from_matrix((Rc * R.from_euler('XYZ', ang)).as_matrix()[None]), Q[k - 1, wc[3:]] if k else ang)[0]
    lim = m.jnt_range[m.actuator_trnid[:, 0]]; qadr = m.jnt_qposadr[m.actuator_trnid[:, 0]]; Q[:, qadr] = np.clip(Q[:, qadr], lim[:, 0], lim[:, 1])
    out.mkdir(parents=True, exist_ok=True); np.savez_compressed(out / 'base_qpos.npz', qpos=base)
    refp = TR.save_reference(out / 'reference.npz', m, Q, ref['labels'], fps, ref['segment'], ref['segments'], source='spoon ILC relative to the achieved cup; robot commands only; immutable object targets/clock', arms_coupled=False, floating_runtime=True)
    ex = execute_reference(scene, refp, out / 'execution')
    met = evaluate_files(src, ex['trajectory'], epdir / 'input_manifest.json', ex['scene'], epdir / 'metric_geometry.json')
    objs = {k: (round(v['mean_position_error_cm'][0], 2), round(v['mean_orientation_error_degrees'][0], 1)) for k, v in met['official_style_reference_proxy']['objects'].items()}
    res = dict(prev, reference=str(refp), execution=ex, metrics=met, source=str(source), spoon_ilc=dict(iteration=int(acc.get('iteration', 0)) + 1, rotation_table=rot_tab, translation_table=pos_tab,
               base_qpos=str(out / 'base_qpos.npz'), t0=a.t0, t1=a.t1, gain_rot=a.gain_rot, gain_pos=a.gain_pos, sigma_s=a.sigma_s, relative=a.relative, side=a.side,
               window_residual_pos_cm=float(100 * np.linalg.norm(e_pos[win[:n]], axis=1).mean()), window_residual_rot_deg=float(np.degrees(np.linalg.norm(e_rot[win[:n]], axis=1)).mean())))
    dump(out / 'result.json', res)
    print(json.dumps(dict(iteration=it, out=str(out), mp_sr=met['reference_proxy_mp_sr'], objects=objs, prev_window_residual_cm=round(res['spoon_ilc']['window_residual_pos_cm'], 2), prev_window_residual_deg=round(res['spoon_ilc']['window_residual_rot_deg'], 1))), flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('source', type=Path); p.add_argument('out', type=Path)
    p.add_argument('--t0', type=float, required=True); p.add_argument('--t1', type=float, required=True); p.add_argument('--side', default='right')
    p.add_argument('--obj', default='wooden_spoon'); p.add_argument('--cup', default='blue_cup'); p.add_argument('--iterations', type=int, default=4)
    p.add_argument('--gain-rot', type=float, default=.3); p.add_argument('--gain-pos', type=float, default=.3); p.add_argument('--sigma-s', type=float, default=.15)
    p.add_argument('--absolute', dest='relative', action='store_false'); p.add_argument('--episode-dir', default=None)
    p.add_argument('--max-pos-m', type=float, default=.04); p.add_argument('--max-rot-deg', type=float, default=25.)
    a = p.parse_args(); s = a.source
    for k in range(a.iterations):
        d = a.out / f'iter{k}'
        if not (d / 'result.json').exists(): one(s, d, a, k)
        s = d
